"""
src/train_final_v2.py
=====================

Turkce otel yorumlari (3 sinif: negatif / notr / pozitif) icin
son jenerasyon, uretim kalitesine yakin, akademik olarak savunulabilir
ve tekrarlanabilir tek-dosya egitim pipeline'i.

Kullanim (minimum):
    python src/train_final_v2.py \
        --csv_path data/processed/Final_Dataset_PlusAI.csv \
        --run_name baseline_plusai_v1

Kullanim (onerilen, akademik karsilastirma icin):
    python src/train_final_v2.py \
        --csv_path data/processed/Final_Dataset_PlusAI.csv \
        --run_name ablation_ai_excluded_v1 \
        --exclude_ai_generated \
        --group_split_by_hotel \
        --seeds 42,43,44 \
        --quarantine_label_mismatch

Tasarim ilkeleri:
  1) Tek giris noktasi. Her deney kendi dizinini olusturur, config.json + run_log.json
     + data_quality_report.json + split hash'leri ile tamamen tekrarlanabilir.
  2) Veri kalitesi birinci sinif vatandas. Butun temizlik adimlari sayi verir;
     sessiz kayip yok.
  3) Label/rating celiskisi, duplicate, kisa yorum, CSV parse hatasi,
     AI-generated overload, hotel-name leakage gibi bilinen sorunlar BIRER BIRER
     tespit edilir ve raporlanir.
  4) Stratified split + dedup ON + (opsiyonel) hotel bazli group split.
     Leakage hosgorusuz: bulunursa egitim baslamaz.
  5) Class weights SADECE train'den, validation'a bakilmadan.
  6) Per-class ve macro/weighted F1 + normalized confusion matrix + hata analizi dump'i.
  7) HuggingFace Trainer son API (eval_strategy) ile uyumlu. fp16, pad_to_multiple_of=8.
  8) Deterministik run; fakat deterministiklik sinirlari (fp16, cuDNN) actikca loglanir.

Gereksinimler:
    pip install transformers>=4.41 datasets>=2.19 torch pandas scikit-learn numpy
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import os
import random
import re
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
from torch import nn

from datasets import Dataset
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)
from sklearn.model_selection import StratifiedKFold, train_test_split
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
    set_seed,
)


# =============================================================================
# 1) SABITLER
# =============================================================================

# Etiket sirasi tum pipeline icin SABIT. Asla degistirme: id -> etiket eslesmesi
# modelin config.json'una gomulur, checkpoint'ler bu sirayla kaydedilir.
LABELS: list[str] = ["negatif", "nötr", "pozitif"]
LABEL2ID: dict[str, int] = {lbl: i for i, lbl in enumerate(LABELS)}
ID2LABEL: dict[int, str] = {i: lbl for lbl, i in LABEL2ID.items()}

# Kabul edilen label varyantlari -> kanonik form.
# BUGUNLERDE gordugumuz hatali kombinasyonlar buraya ekleniyor.
LABEL_ALIAS: dict[str, str] = {
    "notr": "nötr",
    "ntr": "nötr",
    "neutral": "nötr",
    "neg": "negatif",
    "negative": "negatif",
    "pos": "pozitif",
    "positive": "pozitif",
    # tam Turkce karakter varyantlari
    "nötr": "nötr",
    "pozitif": "pozitif",
    "negatif": "negatif",
}

VALID_LABELS_SET = set(LABELS)


# =============================================================================
# 2) KONFIGURASYON
# =============================================================================

@dataclass
class ExpConfig:
    # --- Data ---
    csv_path: str = "data/processed/Final_Dataset_PlusAI.csv"
    text_col: str = "comment_text"
    label_col: str = "sentiment_label"
    source_col: str = "source"
    rating_col: str = "rating_auto"
    hotel_col: str = "hotel_name"

    # --- Cleaning ---
    min_words: int = 3
    drop_duplicates: bool = True
    strip_html: bool = True
    lower_case_labels: bool = True
    # Rating/label celiskisini karantinaya al (egitimden cikar)
    quarantine_label_mismatch: bool = False
    # AI generated ornekleri tamamen dislar. Akademik savunulabilirlik icin ONERILIR.
    exclude_ai_generated: bool = False

    # --- Split ---
    test_size: float = 0.10
    val_size: float = 0.10  # toplamdan
    stratify: bool = True
    # Hotel bazli group split. friend_hotel gibi devasa gruplar varsa
    # group split etkisini kaybeder; o zaman uyari ile fallback yapar.
    group_split_by_hotel: bool = False

    # --- Model ---
    model_name: str = "dbmdz/bert-base-turkish-cased"
    max_len: int = 384

    # --- Training ---
    lr: float = 2e-5
    epochs: int = 5
    train_bs: int = 16
    eval_bs: int = 32
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    gradient_accumulation_steps: int = 1
    fp16: bool = True
    bf16: bool = False
    gradient_checkpointing: bool = False
    max_grad_norm: float = 1.0

    # --- Loss ---
    loss_type: str = "wce"  # "ce" | "wce" | "wce_smooth" | "focal"
    smooth_alpha: float = 0.25
    focal_gamma: float = 2.0

    # --- Early stop / checkpointing ---
    early_stop: bool = True
    early_patience: int = 3
    early_threshold: float = 1e-4
    save_total_limit: int = 2
    metric_for_best_model: str = "f1_macro"  # "f1_macro" | "f1_notr" | "accuracy"

    # --- Reproducibility ---
    seed: int = 42
    deterministic: bool = True  # slow but reproducible
    cublas_workspace_config: str = ":4096:8"  # set env for cuBLAS determinism

    # --- Output ---
    base_outdir: str = "models/final_v2"
    run_name: str = "run"

    # --- Misc ---
    limit_rows: Optional[int] = None  # hizli debug icin
    log_level: str = "INFO"


# =============================================================================
# 3) LOGLAMA
# =============================================================================

def setup_logger(out_dir: Path, level: str = "INFO") -> logging.Logger:
    log_path = out_dir / "run_log.txt"
    logger = logging.getLogger("train_final_v2")
    logger.setLevel(getattr(logging, level.upper()))
    logger.handlers.clear()

    fmt = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-7s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(sh)

    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    return logger


# =============================================================================
# 4) REPRODUCIBILITY
# =============================================================================

def seed_everything(seed: int, deterministic: bool, cublas_cfg: str) -> None:
    """
    Tam deterministiklik FP16 + cuDNN altinda her zaman garanti edilemez.
    Yine de en iyi caba gosterilir ve sinirlari loglanir.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = cublas_cfg

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    set_seed(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        try:
            # PyTorch 1.12+
            torch.use_deterministic_algorithms(True, warn_only=True)
        except Exception:
            pass
    else:
        torch.backends.cudnn.benchmark = True


# =============================================================================
# 5) VERI YUKLEME VE VALIDASYON
# =============================================================================

HTML_RE = re.compile(r"<.*?>")
MULTISPACE_RE = re.compile(r"\s+")
ZERO_WIDTH_RE = re.compile(r"[​‌‍﻿]")


def _clean_text(text, strip_html: bool) -> str:
    if text is None or (isinstance(text, float) and np.isnan(text)):
        return ""
    s = str(text)
    s = ZERO_WIDTH_RE.sub(" ", s)
    if strip_html:
        s = HTML_RE.sub(" ", s)
    s = s.replace("\\n", " ").replace("\n", " ").replace("\r", " ").replace("\t", " ")
    s = MULTISPACE_RE.sub(" ", s).strip()
    return s


def _normalize_label(lbl) -> Optional[str]:
    if lbl is None or (isinstance(lbl, float) and np.isnan(lbl)):
        return None
    s = str(lbl).strip().lower()
    s = LABEL_ALIAS.get(s, s)
    return s if s in VALID_LABELS_SET else None


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_and_validate(cfg: ExpConfig, logger: logging.Logger, out_dir: Path) -> tuple[pd.DataFrame, dict]:
    """
    Tek CSV yukler, kapsamli kalite raporu cikarir, bozuk/celiskili
    satirlari karantinaya alir, temiz DataFrame dondurur.
    """
    path = Path(cfg.csv_path)
    if not path.exists():
        raise FileNotFoundError(f"CSV bulunamadi: {path.resolve()}")

    quality: dict = {}
    quality["csv_path"] = str(path.resolve())
    quality["csv_sha256"] = file_sha256(path)
    quality["csv_size_bytes"] = path.stat().st_size

    # CSV okuma: bozuk satirlari yakalamak icin on_bad_lines="warn" yerine
    # sessiz birakmayip, bozuk satirlari try/except ile yakalamaya calisiyoruz.
    # python engine CSV_QUOTE gevsekligi icin daha dayaniklidir.
    df = pd.read_csv(path, engine="python", quoting=csv.QUOTE_MINIMAL, on_bad_lines="warn")

    quality["rows_raw"] = int(len(df))
    quality["columns"] = list(df.columns)

    # Zorunlu kolonlar
    required = {cfg.text_col, cfg.label_col}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"CSV'de eksik kolon(lar): {sorted(missing)}. Mevcut: {list(df.columns)}")

    # Opsiyonel kolonlar (varsa kullanilir)
    has_source = cfg.source_col in df.columns
    has_rating = cfg.rating_col in df.columns
    has_hotel = cfg.hotel_col in df.columns

    # --- Metin temizligi ---
    df[cfg.text_col] = df[cfg.text_col].map(lambda x: _clean_text(x, cfg.strip_html))
    empty_mask = df[cfg.text_col].str.len() == 0
    quality["rows_empty_text"] = int(empty_mask.sum())
    df = df.loc[~empty_mask].copy()

    # --- Label normalizasyonu ---
    df["_label_norm"] = df[cfg.label_col].map(_normalize_label)
    invalid_label_mask = df["_label_norm"].isna()
    quality["rows_invalid_label"] = int(invalid_label_mask.sum())

    # Gecersiz label satirlari BOZUK CSV isareti olabilir - karantinaya yaz
    if invalid_label_mask.any():
        q = df.loc[invalid_label_mask].copy()
        q_path = out_dir / "quarantine_invalid_label.csv"
        q.to_csv(q_path, index=False, encoding="utf-8", quoting=csv.QUOTE_ALL)
        logger.warning(
            f"[VERI] {int(invalid_label_mask.sum())} satirda gecersiz/anormal etiket var. "
            f"Karantinaya alindi: {q_path}"
        )
    df = df.loc[~invalid_label_mask].copy()
    df[cfg.label_col] = df["_label_norm"]
    df = df.drop(columns=["_label_norm"])

    # --- Kelime sayisi filtresi ---
    wc = df[cfg.text_col].str.split().map(len)
    quality["word_count_describe"] = {
        k: float(v) for k, v in wc.describe().to_dict().items()
    }
    short_mask = wc < cfg.min_words
    quality["rows_short_under_min_words"] = int(short_mask.sum())
    if short_mask.any():
        logger.info(
            f"[VERI] {int(short_mask.sum())} yorum {cfg.min_words} kelimeden kisa; cikariliyor."
        )
    df = df.loc[~short_mask].copy()

    # --- Duplicate kontrolu (raporlanir, opsiyonel silinir) ---
    norm_text = df[cfg.text_col].str.strip().str.lower()
    dup_mask = norm_text.duplicated(keep="first")
    quality["rows_duplicated_text"] = int(dup_mask.sum())
    if cfg.drop_duplicates and dup_mask.any():
        logger.info(f"[VERI] {int(dup_mask.sum())} duplicate yorum cikariliyor.")
        df = df.loc[~dup_mask].copy()

    # --- Celisen etiket kontrolu (ayni text, farkli label) ---
    # Dedup kapali ise bile bu ayri bir kalite uyarisidir.
    grp = df.groupby(cfg.text_col)[cfg.label_col].nunique()
    conflicting = grp[grp > 1]
    quality["texts_with_conflicting_labels"] = int(len(conflicting))
    if len(conflicting) > 0:
        logger.warning(
            f"[VERI] {len(conflicting)} adet ayni metin farkli etiket aliyor - riskli."
        )

    # --- AI generated handling ---
    ai_count = 0
    if has_source:
        ai_mask = df[cfg.source_col].astype(str).str.strip().str.lower().eq("ai_generated")
        ai_count = int(ai_mask.sum())
        quality["rows_ai_generated"] = ai_count
        if ai_count > 0:
            ai_label_dist = df.loc[ai_mask, cfg.label_col].value_counts().to_dict()
            quality["ai_generated_label_distribution"] = {
                str(k): int(v) for k, v in ai_label_dist.items()
            }
            # Tum AI ornekleri tek sinifta mi?
            if len(ai_label_dist) == 1:
                only_label = list(ai_label_dist.keys())[0]
                logger.warning(
                    f"[VERI][KRITIK] {ai_count} AI-generated yorumun TAMAMI '{only_label}' "
                    f"etiketinde. Bu, {only_label} sinifini yapay olarak sisiriyor. "
                    f"Akademik savunulabilirlik icin --exclude_ai_generated kullanin."
                )
        if cfg.exclude_ai_generated and ai_count > 0:
            logger.info(f"[VERI] --exclude_ai_generated aktif: {ai_count} satir cikariliyor.")
            df = df.loc[~ai_mask].copy()

    # --- Rating/Label celiskisi ---
    mismatch_count = 0
    if has_rating:
        rating_num = pd.to_numeric(df[cfg.rating_col], errors="coerce")
        # heuristik esik:
        #   rating <=2 ve label != negatif  -> celiski
        #   rating >=4 ve label == negatif  -> celiski
        mm_low = (rating_num <= 2) & (df[cfg.label_col] != "negatif")
        mm_high = (rating_num >= 4) & (df[cfg.label_col] == "negatif")
        mismatch_mask = (mm_low | mm_high).fillna(False)
        mismatch_count = int(mismatch_mask.sum())
        quality["rows_rating_label_mismatch"] = mismatch_count
        if mismatch_count > 0:
            q_path = out_dir / "quarantine_label_mismatch.csv"
            df.loc[mismatch_mask].to_csv(q_path, index=False, encoding="utf-8", quoting=csv.QUOTE_ALL)
            logger.warning(
                f"[VERI] {mismatch_count} satirda rating-label celiskisi tespit edildi. "
                f"Ornek dosya: {q_path}"
            )
            if cfg.quarantine_label_mismatch:
                df = df.loc[~mismatch_mask].copy()
                logger.info("[VERI] --quarantine_label_mismatch aktif: bu satirlar egitimden cikarildi.")

    # --- Hotel_name dagilimi (group split feasibility) ---
    if has_hotel:
        hotel_counts = df[cfg.hotel_col].value_counts()
        quality["unique_hotels"] = int(hotel_counts.shape[0])
        top_hotel = hotel_counts.index[0] if len(hotel_counts) > 0 else None
        top_hotel_share = float(hotel_counts.iloc[0] / len(df)) if len(df) > 0 else 0.0
        quality["top_hotel"] = str(top_hotel)
        quality["top_hotel_share"] = top_hotel_share
        if top_hotel_share > 0.3:
            logger.warning(
                f"[VERI] '{top_hotel}' tek basina verinin %{top_hotel_share*100:.1f}'ini "
                f"olusturuyor. group_split_by_hotel, bu veri icin etkili olmayacak."
            )

    # --- Son filtre: labels sinifi icinde mi ---
    df = df[df[cfg.label_col].isin(LABELS)].copy()

    # --- Sinif dagilimi ---
    class_dist = df[cfg.label_col].value_counts().to_dict()
    quality["rows_clean"] = int(len(df))
    quality["class_distribution_clean"] = {str(k): int(v) for k, v in class_dist.items()}

    # --- Debug limit ---
    if cfg.limit_rows is not None and cfg.limit_rows > 0 and len(df) > cfg.limit_rows:
        df = df.sample(n=cfg.limit_rows, random_state=cfg.seed).reset_index(drop=True)
        quality["rows_after_limit"] = int(len(df))
        logger.warning(f"[VERI] --limit_rows={cfg.limit_rows} aktif; sadece alt kume kullaniliyor.")

    df["labels"] = df[cfg.label_col].map(LABEL2ID).astype(int)
    df = df.reset_index(drop=True)

    # Kalite raporunu kaydet
    (out_dir / "data_quality_report.json").write_text(
        json.dumps(quality, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    logger.info(f"[VERI] data_quality_report.json yazildi.")

    return df, quality


# =============================================================================
# 6) SPLIT
# =============================================================================

def stratified_split(
    df: pd.DataFrame,
    label_col: str,
    seed: int,
    test_size: float,
    val_size: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Iki asamali stratified split:
      toplam -> (train+val) + test
      (train+val) -> train + val
    val_size: toplam icinden oran (orn 0.10)
    """
    y = df[label_col]
    train_val, test = train_test_split(
        df, test_size=test_size, random_state=seed, stratify=y
    )
    val_rel = val_size / (1.0 - test_size)
    train, val = train_test_split(
        train_val, test_size=val_rel, random_state=seed, stratify=train_val[label_col]
    )
    return (
        train.reset_index(drop=True),
        val.reset_index(drop=True),
        test.reset_index(drop=True),
    )


def group_split_by_hotel(
    df: pd.DataFrame,
    hotel_col: str,
    label_col: str,
    seed: int,
    test_size: float,
    val_size: float,
    logger: logging.Logger,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Otel isimlerine gore grouplanmis split.
    Top-hotel %30'dan fazla ise stratified fallback yapilir (cunku GroupSplit
    dengesiz grup boyutlarinda test setini tek otele sikistirir).
    """
    hotel_counts = df[hotel_col].value_counts()
    if len(hotel_counts) == 0:
        logger.warning("[SPLIT] hotel_col bos; stratified split'e dusulur.")
        return stratified_split(df, label_col, seed, test_size, val_size)

    top_share = hotel_counts.iloc[0] / len(df)
    if top_share > 0.3:
        logger.warning(
            f"[SPLIT] top_hotel %{top_share*100:.1f} buyukluge sahip; "
            "group_split etkisiz, stratified split'e geciliyor."
        )
        return stratified_split(df, label_col, seed, test_size, val_size)

    # Basit group split: otelleri karistir, sirayla test/val/train bucket'lara koy
    rng = np.random.RandomState(seed)
    hotels = list(hotel_counts.index)
    rng.shuffle(hotels)

    target_test = int(round(len(df) * test_size))
    target_val = int(round(len(df) * val_size))

    test_hotels, val_hotels = set(), set()
    running = 0
    for h in hotels:
        n = hotel_counts[h]
        if running < target_test:
            test_hotels.add(h)
            running += n
        else:
            break

    running = 0
    for h in hotels:
        if h in test_hotels:
            continue
        n = hotel_counts[h]
        if running < target_val:
            val_hotels.add(h)
            running += n
        else:
            break

    test_df = df[df[hotel_col].isin(test_hotels)].copy()
    val_df = df[df[hotel_col].isin(val_hotels)].copy()
    train_df = df[~df[hotel_col].isin(test_hotels | val_hotels)].copy()
    return (
        train_df.reset_index(drop=True),
        val_df.reset_index(drop=True),
        test_df.reset_index(drop=True),
    )


def assert_no_leakage(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    text_col: str,
    logger: logging.Logger,
) -> None:
    tr = set(train_df[text_col].str.strip().str.lower())
    va = set(val_df[text_col].str.strip().str.lower())
    te = set(test_df[text_col].str.strip().str.lower())
    tv = len(tr & va)
    tt = len(tr & te)
    vt = len(va & te)
    total = tv + tt + vt
    logger.info(f"[SPLIT] leakage kontrolu: train-val={tv}, train-test={tt}, val-test={vt}")
    if total > 0:
        raise RuntimeError(
            f"Split'ler arasinda text-bazli leakage var (toplam {total}). "
            "Egitim baslamiyor. Veri setinde duplicate text olabilir."
        )


# =============================================================================
# 7) CLASS WEIGHTS
# =============================================================================

def compute_class_weights_balanced(
    train_df: pd.DataFrame, label_col: str
) -> torch.Tensor:
    """
    sklearn 'balanced':
        w_c = n_samples / (n_classes * n_samples_c)
    Hepsi LABELS sirasinda dondurulur. Herhangi bir sinif train'de yoksa hata.
    """
    counts = train_df[label_col].value_counts().to_dict()
    n_samples = len(train_df)
    n_classes = len(LABELS)

    missing = [lbl for lbl in LABELS if lbl not in counts]
    if missing:
        raise ValueError(
            f"[CLASS_WEIGHTS] Train setinde eksik sinif(lar) var: {missing}. "
            "Split stratified degil veya veri yetersiz."
        )

    weights = []
    for lbl in LABELS:
        w = n_samples / (n_classes * counts[lbl])
        weights.append(w)
    return torch.tensor(weights, dtype=torch.float)


def smooth_weights(weights: torch.Tensor, alpha: float) -> torch.Tensor:
    w = weights.clone().float()
    m = w.mean()
    return (1.0 - alpha) * w + alpha * m


# =============================================================================
# 8) LOSS FONKSIYONLARI
# =============================================================================

class FocalLoss(nn.Module):
    """
    Multi-class focal loss with optional class weights.
    CE uzerinden (1-pt)^gamma * ce ile standart formul.
    """

    def __init__(self, gamma: float = 2.0, weight: Optional[torch.Tensor] = None):
        super().__init__()
        self.gamma = gamma
        self.weight = weight

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce = nn.functional.cross_entropy(
            logits, targets, weight=self.weight, reduction="none"
        )
        pt = torch.exp(-ce)
        focal = ((1.0 - pt) ** self.gamma) * ce
        return focal.mean()


class CustomLossTrainer(Trainer):
    """
    Secilebilir loss fonksiyonu ile Trainer.
    compute_loss imzasi HF 4.41+ ile uyumlu (num_items_in_batch eklendi).
    """

    def __init__(
        self,
        loss_type: str,
        class_weights: Optional[torch.Tensor],
        focal_gamma: float,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.loss_type = loss_type
        self.class_weights = class_weights
        self.focal_gamma = focal_gamma

    def compute_loss(
        self, model, inputs, return_outputs: bool = False, num_items_in_batch: Optional[int] = None
    ):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits if hasattr(outputs, "logits") else outputs.get("logits")

        cw = self.class_weights.to(logits.device) if self.class_weights is not None else None

        if self.loss_type == "ce":
            loss_fct = nn.CrossEntropyLoss()
        elif self.loss_type in ("wce", "wce_smooth"):
            loss_fct = nn.CrossEntropyLoss(weight=cw)
        elif self.loss_type == "focal":
            loss_fct = FocalLoss(gamma=self.focal_gamma, weight=cw)
        else:
            raise ValueError(f"Bilinmeyen loss_type: {self.loss_type}")

        loss = loss_fct(logits, labels)
        return (loss, outputs) if return_outputs else loss


# =============================================================================
# 9) METRIK
# =============================================================================

def compute_metrics(eval_pred) -> dict:
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)

    acc = accuracy_score(labels, preds)
    f1_macro = f1_score(labels, preds, average="macro", zero_division=0)
    f1_weighted = f1_score(labels, preds, average="weighted", zero_division=0)

    # per-class F1 (LABELS sirasiyla)
    prec, rec, f1_per, _ = precision_recall_fscore_support(
        labels, preds, labels=list(range(len(LABELS))), zero_division=0
    )
    out = {
        "accuracy": float(acc),
        "f1_macro": float(f1_macro),
        "f1_weighted": float(f1_weighted),
    }
    for i, lbl in enumerate(LABELS):
        key = "f1_" + (lbl.replace("ö", "o").replace("ı", "i").replace("ü", "u")
                        .replace("ç", "c").replace("ş", "s").replace("ğ", "g"))
        out[key] = float(f1_per[i])
    return out


# =============================================================================
# 10) ANA CALISMA
# =============================================================================

def _make_out_dir(cfg: ExpConfig) -> Path:
    tag = (
        f"{Path(cfg.model_name).name}"
        f"_len{cfg.max_len}"
        f"_ep{cfg.epochs}"
        f"_lr{cfg.lr}"
        f"_loss{cfg.loss_type}"
        f"_seed{cfg.seed}"
    )
    out = Path(cfg.base_outdir) / cfg.run_name / tag
    out.mkdir(parents=True, exist_ok=True)
    return out


def run_one(cfg: ExpConfig) -> dict:
    # 1) Seed
    seed_everything(cfg.seed, cfg.deterministic, cfg.cublas_workspace_config)

    # 2) Cikti dizini + logger
    out_dir = _make_out_dir(cfg)
    logger = setup_logger(out_dir, cfg.log_level)
    logger.info("=" * 80)
    logger.info(f"RUN_NAME   : {cfg.run_name}")
    logger.info(f"OUT_DIR    : {out_dir}")
    logger.info(f"MODEL      : {cfg.model_name}")
    logger.info(f"SEED       : {cfg.seed}")
    logger.info(f"FP16       : {cfg.fp16}")
    logger.info(f"DET        : {cfg.deterministic} (FP16 determinism tam degildir)")
    logger.info("=" * 80)

    # 3) Config dondur
    (out_dir / "config.json").write_text(
        json.dumps(asdict(cfg), ensure_ascii=False, indent=2), encoding="utf-8"
    )

    # 4) Veri yukle ve validasyon
    df, quality = load_and_validate(cfg, logger, out_dir)
    logger.info(
        f"[VERI] Clean satir: {len(df)} | dagilim: {df[cfg.label_col].value_counts().to_dict()}"
    )

    # 5) Split
    if cfg.group_split_by_hotel and cfg.hotel_col in df.columns:
        train_df, val_df, test_df = group_split_by_hotel(
            df, cfg.hotel_col, cfg.label_col, cfg.seed,
            cfg.test_size, cfg.val_size, logger,
        )
    else:
        train_df, val_df, test_df = stratified_split(
            df, cfg.label_col, cfg.seed, cfg.test_size, cfg.val_size
        )

    # 6) Leakage kontrolu (HATA ATAR, egitim baslamaz)
    assert_no_leakage(train_df, val_df, test_df, cfg.text_col, logger)

    # 7) Split'leri kaydet (reproducibility icin kritik)
    splits_dir = out_dir / "splits"
    splits_dir.mkdir(exist_ok=True)
    for name, d in [("train", train_df), ("val", val_df), ("test", test_df)]:
        out_p = splits_dir / f"{name}.csv"
        d.to_csv(out_p, index=False, encoding="utf-8", quoting=csv.QUOTE_ALL)
    logger.info(
        f"[SPLIT] train={len(train_df)}  val={len(val_df)}  test={len(test_df)}"
    )
    logger.info(f"[SPLIT] train dist: {train_df[cfg.label_col].value_counts().to_dict()}")
    logger.info(f"[SPLIT] val   dist: {val_df[cfg.label_col].value_counts().to_dict()}")
    logger.info(f"[SPLIT] test  dist: {test_df[cfg.label_col].value_counts().to_dict()}")

    # 8) Class weights (SADECE TRAIN'den)
    base_w = compute_class_weights_balanced(train_df, cfg.label_col)
    if cfg.loss_type == "wce_smooth":
        cw = smooth_weights(base_w, cfg.smooth_alpha)
    elif cfg.loss_type == "ce":
        cw = None
    else:
        cw = base_w
    logger.info(f"[WEIGHTS] LABELS={LABELS}  base={base_w.tolist()}")
    if cw is not None and cfg.loss_type == "wce_smooth":
        logger.info(f"[WEIGHTS] smoothed (alpha={cfg.smooth_alpha})={cw.tolist()}")

    # 9) Tokenizer + dataset
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)

    def _to_hf(df_: pd.DataFrame) -> Dataset:
        return Dataset.from_pandas(
            df_[[cfg.text_col, "labels"]].rename(columns={cfg.text_col: "text"}),
            preserve_index=False,
        )

    ds_train = _to_hf(train_df)
    ds_val = _to_hf(val_df)
    ds_test = _to_hf(test_df)

    def _tok(batch):
        return tokenizer(batch["text"], truncation=True, max_length=cfg.max_len)

    ds_train = ds_train.map(_tok, batched=True, remove_columns=["text"])
    ds_val = ds_val.map(_tok, batched=True, remove_columns=["text"])
    ds_test = ds_test.map(_tok, batched=True, remove_columns=["text"])

    collator = DataCollatorWithPadding(tokenizer=tokenizer, pad_to_multiple_of=8 if cfg.fp16 else None)

    # 10) Model
    model = AutoModelForSequenceClassification.from_pretrained(
        cfg.model_name,
        num_labels=len(LABELS),
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )
    if cfg.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    # 11) Training arguments
    args = TrainingArguments(
        output_dir=str(out_dir),
        seed=cfg.seed,
        data_seed=cfg.seed,

        eval_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="steps",
        logging_steps=50,

        load_best_model_at_end=True,
        metric_for_best_model=(
            cfg.metric_for_best_model
            if cfg.metric_for_best_model.startswith("eval_")
            else f"eval_{cfg.metric_for_best_model}"
        ),
        greater_is_better=True,
        save_total_limit=cfg.save_total_limit,

        learning_rate=cfg.lr,
        num_train_epochs=cfg.epochs,
        per_device_train_batch_size=cfg.train_bs,
        per_device_eval_batch_size=cfg.eval_bs,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        weight_decay=cfg.weight_decay,
        warmup_ratio=cfg.warmup_ratio,
        max_grad_norm=cfg.max_grad_norm,

        fp16=cfg.fp16 and torch.cuda.is_available(),
        bf16=cfg.bf16 and torch.cuda.is_available(),

        report_to="none",
        disable_tqdm=False,
        remove_unused_columns=True,
    )

    callbacks = []
    if cfg.early_stop:
        callbacks.append(
            EarlyStoppingCallback(
                early_stopping_patience=cfg.early_patience,
                early_stopping_threshold=cfg.early_threshold,
            )
        )

    trainer = CustomLossTrainer(
        loss_type=cfg.loss_type,
        class_weights=cw,
        focal_gamma=cfg.focal_gamma,
        model=model,
        args=args,
        train_dataset=ds_train,
        eval_dataset=ds_val,
        data_collator=collator,
        tokenizer=tokenizer,  # eski HF ile uyum
        compute_metrics=compute_metrics,
        callbacks=callbacks,
    )

    # 12) Train
    t0 = time.time()
    logger.info("[TRAIN] basladi")
    train_output = trainer.train()
    elapsed = time.time() - t0
    logger.info(f"[TRAIN] bitti - sure: {elapsed:.1f}s")

    # 13) Best model'i ayri klasore kaydet
    best_dir = out_dir / "best_model"
    trainer.save_model(str(best_dir))
    tokenizer.save_pretrained(str(best_dir))

    # 14) Evaluate: val ve test
    val_metrics = trainer.evaluate(ds_val, metric_key_prefix="val")
    test_pred = trainer.predict(ds_test, metric_key_prefix="test")
    y_true = test_pred.label_ids
    y_pred = np.argmax(test_pred.predictions, axis=-1)

    # Classification report + confusion matrix
    report_txt = classification_report(y_true, y_pred, target_names=LABELS, digits=4)
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(LABELS))))
    # Normalized (row-wise, recall) confusion matrix
    cm_row_sum = cm.sum(axis=1, keepdims=True)
    cm_row_sum[cm_row_sum == 0] = 1
    cm_norm = cm / cm_row_sum

    # Test summary
    acc = accuracy_score(y_true, y_pred)
    f1_macro = f1_score(y_true, y_pred, average="macro", zero_division=0)
    f1_weighted = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    f1_per = f1_score(y_true, y_pred, average=None, labels=list(range(len(LABELS))), zero_division=0)

    report_header = (
        f"RUN_NAME          : {cfg.run_name}\n"
        f"MODEL_NAME        : {cfg.model_name}\n"
        f"MAX_LEN           : {cfg.max_len}\n"
        f"EPOCHS            : {cfg.epochs}\n"
        f"LR                : {cfg.lr}\n"
        f"SEED              : {cfg.seed}\n"
        f"CSV_PATH          : {cfg.csv_path}\n"
        f"CSV_SHA256        : {quality['csv_sha256']}\n"
        f"CLEAN_ROWS        : {quality['rows_clean']}\n"
        f"TRAIN/VAL/TEST    : {len(train_df)}/{len(val_df)}/{len(test_df)}\n"
        f"LOSS_TYPE         : {cfg.loss_type}\n"
        f"CLASS_WEIGHTS     : {base_w.tolist()}\n"
        f"AI_EXCLUDED       : {cfg.exclude_ai_generated}\n"
        f"LABEL_MISMATCH_QA : {cfg.quarantine_label_mismatch}\n"
        f"FP16              : {cfg.fp16}\n"
        f"DETERMINISTIC     : {cfg.deterministic}\n"
        f"\n"
    )
    test_report_str = (
        report_header
        + report_txt
        + "\n\nConfusion matrix (counts):\n" + str(cm)
        + "\n\nConfusion matrix (row-normalized):\n"
        + np.array2string(cm_norm, precision=4, suppress_small=True)
    )
    (out_dir / "test_report.txt").write_text(test_report_str, encoding="utf-8")

    # 15) Hata analizi dump'i (test setindeki yanlislar)
    mis_df = test_df.copy().reset_index(drop=True)
    mis_df["pred_id"] = y_pred
    mis_df["pred_label"] = [ID2LABEL[i] for i in y_pred]
    mis_df = mis_df[mis_df[cfg.label_col] != mis_df["pred_label"]].copy()
    mis_df.to_csv(out_dir / "test_misclassifications.csv", index=False, encoding="utf-8", quoting=csv.QUOTE_ALL)
    logger.info(f"[EVAL] test misclassification sayisi: {len(mis_df)}")

    # 16) JSON rapor (makine okunabilir)
    run_report = {
        "run_name": cfg.run_name,
        "out_dir": str(out_dir),
        "config": asdict(cfg),
        "data_quality": quality,
        "split_sizes": {
            "train": int(len(train_df)),
            "val": int(len(val_df)),
            "test": int(len(test_df)),
        },
        "class_weights": base_w.tolist(),
        "elapsed_sec": elapsed,
        "val_metrics": {k: float(v) if isinstance(v, (int, float, np.floating)) else v
                         for k, v in val_metrics.items()},
        "test_metrics": {
            "accuracy": float(acc),
            "f1_macro": float(f1_macro),
            "f1_weighted": float(f1_weighted),
            "f1_per_class": {LABELS[i]: float(f1_per[i]) for i in range(len(LABELS))},
        },
        "confusion_matrix_counts": cm.tolist(),
        "confusion_matrix_normalized": cm_norm.tolist(),
    }
    (out_dir / "run_report.json").write_text(
        json.dumps(run_report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    logger.info(f"[DONE] test_acc={acc:.4f}  f1_macro={f1_macro:.4f}  f1_weighted={f1_weighted:.4f}")

    return run_report


# =============================================================================
# 11) CLI
# =============================================================================

def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="train_final_v2: temiz, deterministik, raporlu egitim.")
    # Data
    p.add_argument("--csv_path", type=str, default=ExpConfig.csv_path)
    p.add_argument("--text_col", type=str, default=ExpConfig.text_col)
    p.add_argument("--label_col", type=str, default=ExpConfig.label_col)
    p.add_argument("--source_col", type=str, default=ExpConfig.source_col)
    p.add_argument("--rating_col", type=str, default=ExpConfig.rating_col)
    p.add_argument("--hotel_col", type=str, default=ExpConfig.hotel_col)
    # Cleaning
    p.add_argument("--min_words", type=int, default=ExpConfig.min_words)
    p.add_argument("--no_dedup", action="store_true")
    p.add_argument("--quarantine_label_mismatch", action="store_true")
    p.add_argument("--exclude_ai_generated", action="store_true")
    # Split
    p.add_argument("--test_size", type=float, default=ExpConfig.test_size)
    p.add_argument("--val_size", type=float, default=ExpConfig.val_size)
    p.add_argument("--no_stratify", action="store_true")
    p.add_argument("--group_split_by_hotel", action="store_true")
    # Model & training
    p.add_argument("--model_name", type=str, default=ExpConfig.model_name)
    p.add_argument("--max_len", type=int, default=ExpConfig.max_len)
    p.add_argument("--lr", type=float, default=ExpConfig.lr)
    p.add_argument("--epochs", type=int, default=ExpConfig.epochs)
    p.add_argument("--train_bs", type=int, default=ExpConfig.train_bs)
    p.add_argument("--eval_bs", type=int, default=ExpConfig.eval_bs)
    p.add_argument("--weight_decay", type=float, default=ExpConfig.weight_decay)
    p.add_argument("--warmup_ratio", type=float, default=ExpConfig.warmup_ratio)
    p.add_argument("--grad_accum", type=int, default=ExpConfig.gradient_accumulation_steps)
    p.add_argument("--no_fp16", action="store_true")
    p.add_argument("--bf16", action="store_true")
    p.add_argument("--gradient_checkpointing", action="store_true")
    # Loss
    p.add_argument("--loss_type", type=str, default=ExpConfig.loss_type,
                   choices=["ce", "wce", "wce_smooth", "focal"])
    p.add_argument("--smooth_alpha", type=float, default=ExpConfig.smooth_alpha)
    p.add_argument("--focal_gamma", type=float, default=ExpConfig.focal_gamma)
    # Early stop
    p.add_argument("--no_early_stop", action="store_true")
    p.add_argument("--early_patience", type=int, default=ExpConfig.early_patience)
    p.add_argument("--metric_for_best_model", type=str, default=ExpConfig.metric_for_best_model)
    # Reproducibility
    p.add_argument("--seeds", type=str, default=str(ExpConfig.seed),
                   help="virgulle ayrilmis seed listesi: '42' veya '42,43,44'")
    p.add_argument("--no_deterministic", action="store_true")
    # Output
    p.add_argument("--base_outdir", type=str, default=ExpConfig.base_outdir)
    p.add_argument("--run_name", type=str, default=ExpConfig.run_name)
    # Misc
    p.add_argument("--limit_rows", type=int, default=None)
    p.add_argument("--log_level", type=str, default=ExpConfig.log_level)
    return p


def main():
    args = build_argparser().parse_args()
    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]

    all_reports = []
    for sd in seeds:
        cfg = ExpConfig(
            csv_path=args.csv_path,
            text_col=args.text_col,
            label_col=args.label_col,
            source_col=args.source_col,
            rating_col=args.rating_col,
            hotel_col=args.hotel_col,
            min_words=args.min_words,
            drop_duplicates=(not args.no_dedup),
            quarantine_label_mismatch=args.quarantine_label_mismatch,
            exclude_ai_generated=args.exclude_ai_generated,
            test_size=args.test_size,
            val_size=args.val_size,
            stratify=(not args.no_stratify),
            group_split_by_hotel=args.group_split_by_hotel,
            model_name=args.model_name,
            max_len=args.max_len,
            lr=args.lr,
            epochs=args.epochs,
            train_bs=args.train_bs,
            eval_bs=args.eval_bs,
            weight_decay=args.weight_decay,
            warmup_ratio=args.warmup_ratio,
            gradient_accumulation_steps=args.grad_accum,
            fp16=(not args.no_fp16),
            bf16=args.bf16,
            gradient_checkpointing=args.gradient_checkpointing,
            loss_type=args.loss_type,
            smooth_alpha=args.smooth_alpha,
            focal_gamma=args.focal_gamma,
            early_stop=(not args.no_early_stop),
            early_patience=args.early_patience,
            metric_for_best_model=args.metric_for_best_model,
            seed=sd,
            deterministic=(not args.no_deterministic),
            base_outdir=args.base_outdir,
            run_name=args.run_name,
            limit_rows=args.limit_rows,
            log_level=args.log_level,
        )
        rep = run_one(cfg)
        all_reports.append(rep)

    # Multi-seed ozeti
    if len(all_reports) >= 2:
        rows = []
        for r in all_reports:
            rows.append(
                {
                    "seed": r["config"]["seed"],
                    "accuracy": r["test_metrics"]["accuracy"],
                    "f1_macro": r["test_metrics"]["f1_macro"],
                    "f1_weighted": r["test_metrics"]["f1_weighted"],
                    "f1_negatif": r["test_metrics"]["f1_per_class"]["negatif"],
                    "f1_notr": r["test_metrics"]["f1_per_class"]["nötr"],
                    "f1_pozitif": r["test_metrics"]["f1_per_class"]["pozitif"],
                }
            )
        summary = pd.DataFrame(rows)
        summary_path = Path(args.base_outdir) / args.run_name / "summary_multiseed.csv"
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary.to_csv(summary_path, index=False, encoding="utf-8")

        mean = summary.drop(columns=["seed"]).mean()
        std = summary.drop(columns=["seed"]).std(ddof=1)
        print("\n=== MULTI-SEED SUMMARY ===")
        for col in ["accuracy", "f1_macro", "f1_weighted", "f1_negatif", "f1_notr", "f1_pozitif"]:
            print(f"  {col:14s}: {mean[col]:.4f} ± {std[col]:.4f}")
        print(f"summary -> {summary_path}")


if __name__ == "__main__":
    main()
