"""
src/train_optimized_v3.py
=========================

Final_Dataset_Optimized_v3.csv icin nihai, tek-dosya egitim pipeline'i.

Bu dosya 'train_final_v2.py'nin saglam altyapisini korur fakat
*Optimized_v3* veri setine ozel kontroller ekler:

  - Varsayilan CSV: data/processed/Final_Dataset_Optimized_v3.csv
  - AI-generated yorumlarin train/val/test'teki dagilimini RAPORLAR.
  - Stratified split + (opsiyonel) source-stratified split: AI ornekleri
    sinif-icinde de orantili dagilsin diye.
  - AI orneklerinin TEST setine sizmasini opsiyonel olarak ENGELLER
    (--ai_only_in_train), boylece test seti 100% gercek kullanici yorumu olur.
  - Leakage koruma: text bazli kesisim varsa egitim baslatmaz.
  - Class weights SADECE train'den, sklearn 'balanced' formulu.
  - Loss: ce | wce | wce_smooth | focal
  - Tam reproducibility: seed, deterministik backend, CSV sha256, split hash.
  - HF Trainer 4.41+ uyumlu (eval_strategy, num_items_in_batch).
  - Cikti: config.json, run_log.txt, data_quality_report.json, splits/*.csv,
           best_model/, test_report.txt, run_report.json,
           test_misclassifications.csv, test_predictions.csv

Onerilen calistirma (akademik):
    python src/train_optimized_v3.py \
        --run_name v3_baseline \
        --seeds 42,43,44

Onerilen calistirma (test seti tamamen gercek):
    python src/train_optimized_v3.py \
        --run_name v3_real_test \
        --ai_only_in_train \
        --seeds 42

Gereksinimler:
    pip install "transformers>=4.41" "datasets>=2.19" torch pandas scikit-learn numpy
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
from dataclasses import asdict, dataclass
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
from sklearn.model_selection import train_test_split
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

LABELS: list[str] = ["negatif", "nötr", "pozitif"]
LABEL2ID: dict[str, int] = {lbl: i for i, lbl in enumerate(LABELS)}
ID2LABEL: dict[int, str] = {i: lbl for lbl, i in LABEL2ID.items()}

LABEL_ALIAS: dict[str, str] = {
    "notr": "nötr", "ntr": "nötr", "neutral": "nötr",
    "neg": "negatif", "negative": "negatif",
    "pos": "pozitif", "positive": "pozitif",
    "nötr": "nötr", "pozitif": "pozitif", "negatif": "negatif",
}
VALID_LABELS_SET = set(LABELS)


# =============================================================================
# 2) KONFIGURASYON
# =============================================================================

@dataclass
class ExpConfig:
    # --- Data ---
    csv_path: str = "data/processed/Final_Dataset_Optimized_v3.csv"
    text_col: str = "comment_text"
    label_col: str = "sentiment_label"
    source_col: str = "source"
    rating_col: str = "rating_auto"
    hotel_col: str = "hotel_name"

    # --- Cleaning (Optimized_v3 zaten temiz; defansif ek kontroller) ---
    min_words: int = 3
    drop_duplicates: bool = True   # zaten kalkti, savunma amacli
    strip_html: bool = True

    # --- AI handling (Optimized_v3'te kalitelendirilmis AI ornekleri var) ---
    # AI'larin train haricinde val/test'e dusmesini engelle.
    # Akademik raporlamada test setinin 100% gercek olmasini istiyorsak True.
    ai_only_in_train: bool = False

    # --- Split ---
    test_size: float = 0.10
    val_size: float = 0.10  # toplamdan
    # source-stratified: hem label hem source birlikte dengeli dagilir.
    source_stratified: bool = True
    # Nötr test support'unu büyüt: True ise gerçek nötr havuzunun %20'si test'e,
    # diğer sınıflar için test_size varsayılan (%10). Amac azinlik sinifinda
    # ölçüm varyansini düsürmek. Pozitif/negatif test boyutu değismez.
    notr_test_boost: bool = False
    # notr_test_boost aktifken gerçek nötr havuzundan test'e/val'e ayrilacak oran.
    notr_test_ratio: float = 0.20
    notr_val_ratio: float = 0.10

    # --- Model ---
    model_name: str = "dbmdz/bert-base-turkish-cased"
    max_len: int = 256

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
    deterministic: bool = True
    cublas_workspace_config: str = ":4096:8"

    # --- Output ---
    base_outdir: str = "models/optimized_v3"
    run_name: str = "run"

    # --- Misc ---
    limit_rows: Optional[int] = None
    log_level: str = "INFO"


# =============================================================================
# 3) LOG, REPRO
# =============================================================================

def setup_logger(out_dir: Path, level: str = "INFO") -> logging.Logger:
    log_path = out_dir / "run_log.txt"
    logger = logging.getLogger("train_optimized_v3")
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


def seed_everything(seed: int, deterministic: bool, cublas_cfg: str) -> None:
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
            torch.use_deterministic_algorithms(True, warn_only=True)
        except Exception:
            pass
    else:
        torch.backends.cudnn.benchmark = True


# =============================================================================
# 4) VERI YUKLE + KALITE RAPORU
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
    path = Path(cfg.csv_path)
    if not path.exists():
        raise FileNotFoundError(f"CSV bulunamadi: {path.resolve()}")

    quality: dict = {
        "csv_path": str(path.resolve()),
        "csv_sha256": file_sha256(path),
        "csv_size_bytes": path.stat().st_size,
    }
    df = pd.read_csv(path, encoding="utf-8")
    quality["rows_raw"] = int(len(df))
    quality["columns"] = list(df.columns)

    required = {cfg.text_col, cfg.label_col}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"CSV'de eksik kolon(lar): {sorted(missing)}. Mevcut: {list(df.columns)}")

    # Metin temizligi (Optimized_v3 zaten temizlendi; defansif)
    df[cfg.text_col] = df[cfg.text_col].map(lambda x: _clean_text(x, cfg.strip_html))
    empty_mask = df[cfg.text_col].str.len() == 0
    quality["rows_empty_text"] = int(empty_mask.sum())
    df = df.loc[~empty_mask].copy()

    # Label normalizasyonu + invalid karantinasi (sayisi 0 olmali)
    df["_lbl_norm"] = df[cfg.label_col].map(_normalize_label)
    invalid_label_mask = df["_lbl_norm"].isna()
    quality["rows_invalid_label"] = int(invalid_label_mask.sum())
    if invalid_label_mask.any():
        q = df.loc[invalid_label_mask].copy()
        q.to_csv(out_dir / "quarantine_invalid_label.csv", index=False, encoding="utf-8")
        logger.warning(f"[VERI] {int(invalid_label_mask.sum())} satir invalid label - karantina.")
    df = df.loc[~invalid_label_mask].copy()
    df[cfg.label_col] = df["_lbl_norm"]
    df = df.drop(columns=["_lbl_norm"])

    # Min words
    wc = df[cfg.text_col].str.split().map(len)
    quality["word_count_describe"] = {k: float(v) for k, v in wc.describe().to_dict().items()}
    short_mask = wc < cfg.min_words
    quality["rows_short_under_min_words"] = int(short_mask.sum())
    if short_mask.any():
        logger.info(f"[VERI] {int(short_mask.sum())} yorum < {cfg.min_words} kelime - cikariliyor.")
    df = df.loc[~short_mask].copy()

    # Duplicate (Optimized_v3'te 0 olmali)
    norm_text = df[cfg.text_col].str.strip().str.lower()
    dup_mask = norm_text.duplicated(keep="first")
    quality["rows_duplicated_text"] = int(dup_mask.sum())
    if cfg.drop_duplicates and dup_mask.any():
        logger.info(f"[VERI] {int(dup_mask.sum())} duplicate cikariliyor.")
        df = df.loc[~dup_mask].copy()

    # AI raporu (Optimized_v3'te ~2500 AI bekleniyor)
    has_source = cfg.source_col in df.columns
    if has_source:
        ai_mask = df[cfg.source_col].astype(str).str.strip().str.lower().eq("ai_generated")
        ai_count = int(ai_mask.sum())
        quality["rows_ai_generated"] = ai_count
        if ai_count > 0:
            ai_label_dist = df.loc[ai_mask, cfg.label_col].value_counts().to_dict()
            quality["ai_generated_label_distribution"] = {
                str(k): int(v) for k, v in ai_label_dist.items()
            }
            ai_wc = df.loc[ai_mask, cfg.text_col].str.split().map(len)
            quality["ai_word_count"] = {
                "mean": float(ai_wc.mean()),
                "median": float(ai_wc.median()),
                "q05": float(ai_wc.quantile(0.05)),
                "q95": float(ai_wc.quantile(0.95)),
            }
            logger.info(
                f"[VERI] AI satir = {ai_count} | label dist = {ai_label_dist} | "
                f"wc median = {quality['ai_word_count']['median']:.0f}"
            )

    # Hotel dagilimi (group_split mumkun mu)
    if cfg.hotel_col in df.columns:
        hc = df[cfg.hotel_col].value_counts()
        quality["unique_hotels"] = int(hc.shape[0])
        quality["top_hotel"] = str(hc.index[0]) if len(hc) else None
        quality["top_hotel_share"] = float(hc.iloc[0] / len(df)) if len(df) else 0.0

    df = df[df[cfg.label_col].isin(LABELS)].copy()
    quality["rows_clean"] = int(len(df))
    quality["class_distribution_clean"] = {
        str(k): int(v) for k, v in df[cfg.label_col].value_counts().to_dict().items()
    }

    if cfg.limit_rows and cfg.limit_rows > 0 and len(df) > cfg.limit_rows:
        df = df.sample(n=cfg.limit_rows, random_state=cfg.seed).reset_index(drop=True)
        quality["rows_after_limit"] = int(len(df))
        logger.warning(f"[VERI] --limit_rows={cfg.limit_rows} aktif.")

    df["labels"] = df[cfg.label_col].map(LABEL2ID).astype(int)
    df = df.reset_index(drop=True)

    (out_dir / "data_quality_report.json").write_text(
        json.dumps(quality, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    logger.info("[VERI] data_quality_report.json yazildi.")
    return df, quality


# =============================================================================
# 5) SPLIT
# =============================================================================

def stratified_split(
    df: pd.DataFrame, label_col: str, seed: int, test_size: float, val_size: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
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


def source_stratified_split(
    df: pd.DataFrame,
    label_col: str,
    source_col: str,
    seed: int,
    test_size: float,
    val_size: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    label x source kombinasyonuna gore stratify.
    Bu, AI orneklerinin nötr sinifinda train/val/test arasinda orantili
    dagilmasini saglar; aksi takdirde nötr metrikleri AI yapaylik tarafindan
    carpitilir.
    """
    if source_col not in df.columns:
        return stratified_split(df, label_col, seed, test_size, val_size)
    # Basit ve saglam strata: label x is_ai (iki-kategori). Amac AI orneklerinin
    # train/val/test'te orantili dagilmasi; 'neredekal' gibi kucuk insan kaynaklarini
    # genel insan grubuyla birlestirmek istatistiksel olarak guvenli.
    is_ai = df[source_col].astype(str).str.strip().str.lower().eq("ai_generated")
    strata = df[label_col].astype(str) + "::" + np.where(is_ai, "ai", "human")

    train_val, test = train_test_split(
        df, test_size=test_size, random_state=seed, stratify=strata
    )
    strata_tv = strata.loc[train_val.index]
    val_rel = val_size / (1.0 - test_size)
    train, val = train_test_split(
        train_val, test_size=val_rel, random_state=seed, stratify=strata_tv
    )
    return (
        train.reset_index(drop=True),
        val.reset_index(drop=True),
        test.reset_index(drop=True),
    )


def per_class_split(
    df: pd.DataFrame,
    label_col: str,
    source_col: str,
    seed: int,
    test_size: float,
    val_size: float,
    notr_test_ratio: float,
    notr_val_ratio: float,
    logger: logging.Logger,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Nötr sınıfının azınlık olması ve test ölçüm varyansını bozması nedeniyle
    sınıf bazlı farklı test/val oranları uygular.

    Nötr sınıfı: notr_test_ratio (varsayılan 0.20) test'e, notr_val_ratio (0.10) val'e.
    Pozitif/negatif: test_size (varsayılan 0.10), val_size (0.10).

    AI örnekleri source_stratified prensibiyle test/val'den uzak tutulacaksa
    (ai_only_in_train aşaması split'ten sonra yine çağrılacaktır), burada AI
    dahil tüm nötr havuzundan örnekleme yapılır; `move_ai_to_train` sonradan
    AI'ları train'e kaydırır. Sonuçta test setindeki nötr adedi:
        round(len(real_nötr) * notr_test_ratio)
    olacaktır — çünkü AI'lar test'ten çıkarılır.

    Leakage, assert_no_leakage tarafından kontrol edilir.
    """
    rng = np.random.RandomState(seed)

    # Sınıf bazlı alt-df'ler
    sub_frames = {lbl: df[df[label_col] == lbl].copy() for lbl in LABELS}

    train_parts, val_parts, test_parts = [], [], []
    for lbl, sub in sub_frames.items():
        n = len(sub)
        if n == 0:
            logger.warning(f"[SPLIT][per_class] sınıf bos: {lbl}")
            continue

        if lbl == "nötr":
            t_ratio = notr_test_ratio
            v_ratio = notr_val_ratio
        else:
            t_ratio = test_size
            v_ratio = val_size

        # Sınıf içinde shuffle
        idx = np.arange(n)
        rng.shuffle(idx)
        sub_shuf = sub.iloc[idx].reset_index(drop=True)

        n_test = int(round(n * t_ratio))
        n_val = int(round(n * v_ratio))
        n_train = n - n_test - n_val
        if n_train <= 0:
            raise ValueError(
                f"[SPLIT][per_class] sinif {lbl} icin train bos (n={n}, test={n_test}, val={n_val})"
            )

        test_parts.append(sub_shuf.iloc[:n_test])
        val_parts.append(sub_shuf.iloc[n_test : n_test + n_val])
        train_parts.append(sub_shuf.iloc[n_test + n_val :])

        logger.info(
            f"[SPLIT][per_class] {lbl}: n={n}  train={n_train}  val={n_val}  test={n_test}"
        )

    train_df = pd.concat(train_parts, axis=0, ignore_index=True).sample(
        frac=1.0, random_state=seed
    ).reset_index(drop=True)
    val_df = pd.concat(val_parts, axis=0, ignore_index=True).sample(
        frac=1.0, random_state=seed
    ).reset_index(drop=True)
    test_df = pd.concat(test_parts, axis=0, ignore_index=True).sample(
        frac=1.0, random_state=seed
    ).reset_index(drop=True)

    return train_df, val_df, test_df


def assert_no_leakage(
    train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame,
    text_col: str, logger: logging.Logger,
) -> None:
    tr = set(train_df[text_col].str.strip().str.lower())
    va = set(val_df[text_col].str.strip().str.lower())
    te = set(test_df[text_col].str.strip().str.lower())
    tv, tt, vt = len(tr & va), len(tr & te), len(va & te)
    logger.info(f"[SPLIT] leakage: train-val={tv} train-test={tt} val-test={vt}")
    if (tv + tt + vt) > 0:
        raise RuntimeError(
            f"Leakage var (toplam {tv+tt+vt}). Veri setinde duplicate text kalmis olabilir."
        )


def move_ai_to_train(
    train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame,
    source_col: str, logger: logging.Logger,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    --ai_only_in_train: val/test'te bulunan ai_generated ornekleri train'e tasi.
    Test setinin %100 gercek kullanici yorumu olmasini saglar.
    """
    def _is_ai(s):
        return s.astype(str).str.strip().str.lower().eq("ai_generated")
    val_ai = val_df[_is_ai(val_df[source_col])]
    test_ai = test_df[_is_ai(test_df[source_col])]
    if len(val_ai) == 0 and len(test_ai) == 0:
        logger.info("[SPLIT] ai_only_in_train: val/test'te AI ornek yok, taşinacak bir sey yok.")
        return train_df, val_df, test_df
    logger.info(
        f"[SPLIT] ai_only_in_train: val'den {len(val_ai)}, test'ten {len(test_ai)} AI ornek train'e tasiniyor."
    )
    train_df = pd.concat([train_df, val_ai, test_ai], axis=0, ignore_index=True)
    val_df = val_df.drop(val_ai.index).reset_index(drop=True)
    test_df = test_df.drop(test_ai.index).reset_index(drop=True)
    return train_df.reset_index(drop=True), val_df, test_df


# =============================================================================
# 6) CLASS WEIGHTS + LOSS
# =============================================================================

def compute_class_weights_balanced(train_df: pd.DataFrame, label_col: str) -> torch.Tensor:
    counts = train_df[label_col].value_counts().to_dict()
    n_samples = len(train_df)
    n_classes = len(LABELS)
    missing = [lbl for lbl in LABELS if lbl not in counts]
    if missing:
        raise ValueError(
            f"[CLASS_WEIGHTS] Train'de eksik sinif: {missing}. Veri/split kontrol et."
        )
    return torch.tensor(
        [n_samples / (n_classes * counts[lbl]) for lbl in LABELS], dtype=torch.float
    )


def smooth_weights(weights: torch.Tensor, alpha: float) -> torch.Tensor:
    w = weights.clone().float()
    return (1.0 - alpha) * w + alpha * w.mean()


class FocalLoss(nn.Module):
    def __init__(self, gamma: float = 2.0, weight: Optional[torch.Tensor] = None):
        super().__init__()
        self.gamma = gamma
        self.weight = weight

    def forward(self, logits, targets):
        ce = nn.functional.cross_entropy(logits, targets, weight=self.weight, reduction="none")
        pt = torch.exp(-ce)
        return (((1.0 - pt) ** self.gamma) * ce).mean()


class CustomLossTrainer(Trainer):
    def __init__(self, loss_type, class_weights, focal_gamma, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.loss_type = loss_type
        self.class_weights = class_weights
        self.focal_gamma = focal_gamma

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
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
# 7) METRIK
# =============================================================================

def compute_metrics(eval_pred) -> dict:
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    acc = accuracy_score(labels, preds)
    f1_macro = f1_score(labels, preds, average="macro", zero_division=0)
    f1_weighted = f1_score(labels, preds, average="weighted", zero_division=0)
    _, _, f1_per, _ = precision_recall_fscore_support(
        labels, preds, labels=list(range(len(LABELS))), zero_division=0
    )
    out = {"accuracy": float(acc), "f1_macro": float(f1_macro), "f1_weighted": float(f1_weighted)}
    for i, lbl in enumerate(LABELS):
        key = "f1_" + lbl.translate(str.maketrans("öıüçşğ", "oiucsg"))
        out[key] = float(f1_per[i])
    return out


# =============================================================================
# 8) ANA CALISMA
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


def run_one(cfg: ExpConfig, external_splits: Optional[tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]] = None) -> dict:
    """
    Tek konfigurasyonu (cfg) ucundan ucuna kosturur.

    external_splits: Opsiyonel. Eger verilirse internal split'i bypass eder ve
    verilen (train_df, val_df, test_df) ile devam eder. Cross-validation gibi
    harici split senaryolarinda kullanilir. Verilen df'lerde 'labels' kolonu
    zaten olabilir; yoksa olusturulur. AI handling (ai_only_in_train) external
    mod'da YINE de uygulanir (savunma amacli).
    """
    seed_everything(cfg.seed, cfg.deterministic, cfg.cublas_workspace_config)
    out_dir = _make_out_dir(cfg)
    logger = setup_logger(out_dir, cfg.log_level)
    logger.info("=" * 80)
    logger.info(f"RUN_NAME       : {cfg.run_name}")
    logger.info(f"OUT_DIR        : {out_dir}")
    logger.info(f"CSV            : {cfg.csv_path}")
    logger.info(f"MODEL          : {cfg.model_name}")
    logger.info(f"SEED           : {cfg.seed}")
    logger.info(f"LOSS           : {cfg.loss_type}")
    logger.info(f"AI_ONLY_TRAIN  : {cfg.ai_only_in_train}")
    logger.info(f"SRC_STRATIFIED : {cfg.source_stratified}")
    logger.info(f"NOTR_TEST_BOOST: {cfg.notr_test_boost} (notr test={cfg.notr_test_ratio}, val={cfg.notr_val_ratio})")
    logger.info(f"EXTERNAL_SPLIT : {external_splits is not None}")
    logger.info("=" * 80)

    (out_dir / "config.json").write_text(
        json.dumps(asdict(cfg), ensure_ascii=False, indent=2), encoding="utf-8"
    )

    if external_splits is not None:
        # Harici split: load_and_validate atlanir; quality bilgisi minimal.
        train_df, val_df, test_df = external_splits
        # 'labels' kolonu garantile
        for d in (train_df, val_df, test_df):
            if "labels" not in d.columns:
                d["labels"] = d[cfg.label_col].map(LABEL2ID).astype(int)
        # quality minimal
        quality = {
            "csv_path": str(cfg.csv_path),
            "csv_sha256": "external_splits",
            "rows_clean": int(len(train_df) + len(val_df) + len(test_df)),
            "class_distribution_clean": {
                str(k): int(v)
                for k, v in pd.concat([train_df, val_df, test_df])[cfg.label_col]
                    .value_counts().to_dict().items()
            },
            "external_splits": True,
        }
        (out_dir / "data_quality_report.json").write_text(
            json.dumps(quality, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        logger.info(f"[VERI][external] train={len(train_df)} val={len(val_df)} test={len(test_df)}")
    else:
        df, quality = load_and_validate(cfg, logger, out_dir)
        logger.info(f"[VERI] clean satir={len(df)}  dist={df[cfg.label_col].value_counts().to_dict()}")

        if cfg.notr_test_boost:
            train_df, val_df, test_df = per_class_split(
                df, cfg.label_col, cfg.source_col, cfg.seed,
                cfg.test_size, cfg.val_size,
                cfg.notr_test_ratio, cfg.notr_val_ratio, logger,
            )
        elif cfg.source_stratified and cfg.source_col in df.columns:
            train_df, val_df, test_df = source_stratified_split(
                df, cfg.label_col, cfg.source_col, cfg.seed, cfg.test_size, cfg.val_size
            )
        else:
            train_df, val_df, test_df = stratified_split(
                df, cfg.label_col, cfg.seed, cfg.test_size, cfg.val_size
            )

    if cfg.ai_only_in_train and cfg.source_col in train_df.columns:
        train_df, val_df, test_df = move_ai_to_train(
            train_df, val_df, test_df, cfg.source_col, logger
        )

    assert_no_leakage(train_df, val_df, test_df, cfg.text_col, logger)

    splits_dir = out_dir / "splits"
    splits_dir.mkdir(exist_ok=True)
    for name, d in [("train", train_df), ("val", val_df), ("test", test_df)]:
        d.to_csv(splits_dir / f"{name}.csv", index=False, encoding="utf-8", quoting=csv.QUOTE_ALL)

    logger.info(f"[SPLIT] train={len(train_df)} val={len(val_df)} test={len(test_df)}")
    for name, d in [("train", train_df), ("val", val_df), ("test", test_df)]:
        logger.info(f"[SPLIT][{name}] dist={d[cfg.label_col].value_counts().to_dict()}")
        if cfg.source_col in d.columns:
            ai_n = int((d[cfg.source_col] == "ai_generated").sum())
            logger.info(f"[SPLIT][{name}] ai_generated count={ai_n}")

    base_w = compute_class_weights_balanced(train_df, cfg.label_col)
    if cfg.loss_type == "wce_smooth":
        cw = smooth_weights(base_w, cfg.smooth_alpha)
    elif cfg.loss_type == "ce":
        cw = None
    else:
        cw = base_w
    logger.info(f"[WEIGHTS] LABELS={LABELS}  base={base_w.tolist()}")

    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)

    def _to_hf(d):
        return Dataset.from_pandas(
            d[[cfg.text_col, "labels"]].rename(columns={cfg.text_col: "text"}),
            preserve_index=False,
        )

    ds_train = _to_hf(train_df)
    ds_val = _to_hf(val_df)
    ds_test = _to_hf(test_df)

    def _tok(b):
        return tokenizer(b["text"], truncation=True, max_length=cfg.max_len)

    ds_train = ds_train.map(_tok, batched=True, remove_columns=["text"])
    ds_val = ds_val.map(_tok, batched=True, remove_columns=["text"])
    ds_test = ds_test.map(_tok, batched=True, remove_columns=["text"])

    collator = DataCollatorWithPadding(
        tokenizer=tokenizer, pad_to_multiple_of=8 if cfg.fp16 else None
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        cfg.model_name, num_labels=len(LABELS), id2label=ID2LABEL, label2id=LABEL2ID,
    )
    if cfg.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    args = TrainingArguments(
        output_dir=str(out_dir),
        seed=cfg.seed, data_seed=cfg.seed,
        eval_strategy="epoch", save_strategy="epoch",
        logging_strategy="steps", logging_steps=50,
        load_best_model_at_end=True,
        metric_for_best_model=(
            cfg.metric_for_best_model
            if cfg.metric_for_best_model.startswith("eval_")
            else f"eval_{cfg.metric_for_best_model}"
        ),
        greater_is_better=True,
        save_total_limit=cfg.save_total_limit,
        learning_rate=cfg.lr, num_train_epochs=cfg.epochs,
        per_device_train_batch_size=cfg.train_bs,
        per_device_eval_batch_size=cfg.eval_bs,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        weight_decay=cfg.weight_decay, warmup_ratio=cfg.warmup_ratio,
        max_grad_norm=cfg.max_grad_norm,
        fp16=cfg.fp16 and torch.cuda.is_available(),
        bf16=cfg.bf16 and torch.cuda.is_available(),
        report_to="none", disable_tqdm=False, remove_unused_columns=True,
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
        loss_type=cfg.loss_type, class_weights=cw, focal_gamma=cfg.focal_gamma,
        model=model, args=args,
        train_dataset=ds_train, eval_dataset=ds_val, data_collator=collator,
        tokenizer=tokenizer, compute_metrics=compute_metrics, callbacks=callbacks,
    )

    t0 = time.time()
    logger.info("[TRAIN] basladi")
    trainer.train()
    elapsed = time.time() - t0
    logger.info(f"[TRAIN] bitti - sure: {elapsed:.1f}s")

    best_dir = out_dir / "best_model"
    trainer.save_model(str(best_dir))
    tokenizer.save_pretrained(str(best_dir))

    val_metrics = trainer.evaluate(ds_val, metric_key_prefix="val")
    test_pred = trainer.predict(ds_test, metric_key_prefix="test")
    y_true = test_pred.label_ids
    y_pred = np.argmax(test_pred.predictions, axis=-1)

    report_txt = classification_report(y_true, y_pred, target_names=LABELS, digits=4)
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(LABELS))))
    cm_row_sum = cm.sum(axis=1, keepdims=True)
    cm_row_sum[cm_row_sum == 0] = 1
    cm_norm = cm / cm_row_sum

    acc = accuracy_score(y_true, y_pred)
    f1_macro = f1_score(y_true, y_pred, average="macro", zero_division=0)
    f1_weighted = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    f1_per = f1_score(y_true, y_pred, average=None, labels=list(range(len(LABELS))), zero_division=0)

    header = (
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
        f"AI_ONLY_IN_TRAIN  : {cfg.ai_only_in_train}\n"
        f"SOURCE_STRATIFIED : {cfg.source_stratified}\n"
        f"FP16              : {cfg.fp16}\n"
        f"DETERMINISTIC     : {cfg.deterministic}\n\n"
    )
    (out_dir / "test_report.txt").write_text(
        header + report_txt
        + "\n\nConfusion matrix (counts):\n" + str(cm)
        + "\n\nConfusion matrix (row-normalized):\n"
        + np.array2string(cm_norm, precision=4, suppress_small=True),
        encoding="utf-8",
    )

    # Tum test tahminleri
    pred_df = test_df.copy().reset_index(drop=True)
    pred_df["pred_id"] = y_pred
    pred_df["pred_label"] = [ID2LABEL[i] for i in y_pred]
    pred_df["correct"] = (pred_df[cfg.label_col] == pred_df["pred_label"]).astype(int)
    pred_df.to_csv(out_dir / "test_predictions.csv", index=False, encoding="utf-8", quoting=csv.QUOTE_ALL)

    mis = pred_df[pred_df["correct"] == 0].copy()
    mis.to_csv(out_dir / "test_misclassifications.csv", index=False, encoding="utf-8", quoting=csv.QUOTE_ALL)
    logger.info(f"[EVAL] test misclassification = {len(mis)} / {len(pred_df)}")

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
        "split_class_dist": {
            "train": {str(k): int(v) for k, v in train_df[cfg.label_col].value_counts().to_dict().items()},
            "val": {str(k): int(v) for k, v in val_df[cfg.label_col].value_counts().to_dict().items()},
            "test": {str(k): int(v) for k, v in test_df[cfg.label_col].value_counts().to_dict().items()},
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
# 9) CLI
# =============================================================================

def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="train_optimized_v3: Optimized_v3 icin nihai egitim.")
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
    # AI
    p.add_argument("--ai_only_in_train", action="store_true",
                   help="AI ornekleri sadece train setinde kalsin (val/test'ten cikar).")
    # Split
    p.add_argument("--test_size", type=float, default=ExpConfig.test_size)
    p.add_argument("--val_size", type=float, default=ExpConfig.val_size)
    p.add_argument("--no_source_stratified", action="store_true",
                   help="source_stratified split'i kapat, sadece label-stratified kullan.")
    p.add_argument("--notr_test_boost", action="store_true",
                   help="Nötr sınıf için test/val oranlarını yükselt (per-class split kullanılır).")
    p.add_argument("--notr_test_ratio", type=float, default=ExpConfig.notr_test_ratio,
                   help="notr_test_boost aktifken nötr havuzundan test'e ayrilacak oran.")
    p.add_argument("--notr_val_ratio", type=float, default=ExpConfig.notr_val_ratio,
                   help="notr_test_boost aktifken nötr havuzundan val'e ayrilacak oran.")
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
    p.add_argument("--seeds", type=str, default=str(ExpConfig.seed))
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
            text_col=args.text_col, label_col=args.label_col,
            source_col=args.source_col, rating_col=args.rating_col,
            hotel_col=args.hotel_col,
            min_words=args.min_words, drop_duplicates=(not args.no_dedup),
            ai_only_in_train=args.ai_only_in_train,
            test_size=args.test_size, val_size=args.val_size,
            source_stratified=(not args.no_source_stratified),
            notr_test_boost=args.notr_test_boost,
            notr_test_ratio=args.notr_test_ratio,
            notr_val_ratio=args.notr_val_ratio,
            model_name=args.model_name, max_len=args.max_len,
            lr=args.lr, epochs=args.epochs,
            train_bs=args.train_bs, eval_bs=args.eval_bs,
            weight_decay=args.weight_decay, warmup_ratio=args.warmup_ratio,
            gradient_accumulation_steps=args.grad_accum,
            fp16=(not args.no_fp16), bf16=args.bf16,
            gradient_checkpointing=args.gradient_checkpointing,
            loss_type=args.loss_type, smooth_alpha=args.smooth_alpha,
            focal_gamma=args.focal_gamma,
            early_stop=(not args.no_early_stop),
            early_patience=args.early_patience,
            metric_for_best_model=args.metric_for_best_model,
            seed=sd, deterministic=(not args.no_deterministic),
            base_outdir=args.base_outdir, run_name=args.run_name,
            limit_rows=args.limit_rows, log_level=args.log_level,
        )
        all_reports.append(run_one(cfg))

    if len(all_reports) >= 2:
        rows = []
        for r in all_reports:
            rows.append({
                "seed": r["config"]["seed"],
                "accuracy": r["test_metrics"]["accuracy"],
                "f1_macro": r["test_metrics"]["f1_macro"],
                "f1_weighted": r["test_metrics"]["f1_weighted"],
                "f1_negatif": r["test_metrics"]["f1_per_class"]["negatif"],
                "f1_notr": r["test_metrics"]["f1_per_class"]["nötr"],
                "f1_pozitif": r["test_metrics"]["f1_per_class"]["pozitif"],
            })
        summary = pd.DataFrame(rows)
        sp = Path(args.base_outdir) / args.run_name / "summary_multiseed.csv"
        sp.parent.mkdir(parents=True, exist_ok=True)
        summary.to_csv(sp, index=False, encoding="utf-8")
        mean = summary.drop(columns=["seed"]).mean()
        std = summary.drop(columns=["seed"]).std(ddof=1)
        print("\n=== MULTI-SEED SUMMARY ===")
        for col in ["accuracy", "f1_macro", "f1_weighted", "f1_negatif", "f1_notr", "f1_pozitif"]:
            print(f"  {col:14s}: {mean[col]:.4f} +/- {std[col]:.4f}")
        print(f"summary -> {sp}")


if __name__ == "__main__":
    main()
