# src/train_berturk_wce_temiz-kisa-yorum.py

from __future__ import annotations
from pathlib import Path
from dataclasses import dataclass, asdict
import argparse
import json
import os
import random
import re

import numpy as np
import pandas as pd
import torch
from torch import nn

from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
    EarlyStoppingCallback,
    set_seed,
)

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix


# -----------------------------
# DEFAULTS
# -----------------------------
MODEL_NAME_DEFAULT = "dbmdz/bert-base-turkish-cased"
CSV_DEFAULT = "data/processed/manuel_corrected_dataset.csv"
BASE_OUTDIR_DEFAULT = "models/experiments_wce_temiz_kisa_yorum"

LABELS = ["negatif", "nötr", "pozitif"]
label2id = {l: i for i, l in enumerate(LABELS)}
id2label = {i: l for l, i in label2id.items()}


# -----------------------------
# Reproducibility helpers
# -----------------------------
def seed_everything(seed: int) -> None:
    """Maksimum determinism için."""
    set_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    os.environ["PYTHONHASHSEED"] = str(seed)


# -----------------------------
# Config
# -----------------------------
@dataclass
class ExpConfig:
    model_name: str = MODEL_NAME_DEFAULT
    max_len: int = 384
    lr: float = 2e-5
    epochs: int = 5
    train_bs: int = 16
    eval_bs: int = 32
    weight_decay: float = 0.01
    seed: int = 42

    # loss: "wce" | "wce_smooth" | "focal"
    loss_type: str = "wce"
    smooth_alpha: float = 0.25
    focal_gamma: float = 2.0

    # early stopping
    early_stop: bool = True
    early_patience: int = 2
    early_threshold: float = 1e-4

    # data
    csv_path: str = CSV_DEFAULT
    text_col: str = "comment_text"
    label_col: str = "sentiment_label"

    # split
    test_size: float = 0.10
    val_size: float = 0.10  # toplam verinin %10'u validation
    stratify: bool = True

    # cleaning
    do_basic_clean: bool = True
    drop_duplicates: bool = True
    min_words: int = 3  # "kısa yorum" filtresi (3 kelime altını at)

    # outputs
    base_outdir: str = BASE_OUTDIR_DEFAULT
    save_total_limit: int = 2

    # training extras
    fp16: bool = True


# -----------------------------
# Cleaning utilities
# -----------------------------
HTML_RE = re.compile(r"<.*?>")
MULTISPACE_RE = re.compile(r"\s+")

def basic_clean(text: str) -> str:
    if text is None:
        return ""
    text = str(text)
    text = text.replace("\u200b", " ").replace("\ufeff", " ")
    text = re.sub(HTML_RE, " ", text)
    text = text.replace("\\n", " ").replace("\n", " ").replace("\r", " ")
    text = re.sub(MULTISPACE_RE, " ", text).strip()
    return text

def word_count(text: str) -> int:
    if not text:
        return 0
    return len([w for w in text.split(" ") if w.strip()])


# -----------------------------
# Data loading (single CSV)
# -----------------------------
def load_df(path: Path, text_col: str, label_col: str, do_basic_clean: bool) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"CSV bulunamadı: {path}")

    df = pd.read_csv(path)

    if text_col not in df.columns or label_col not in df.columns:
        raise ValueError(
            f"{path} içinde '{text_col}' veya '{label_col}' kolonu yok.\n"
            f"Mevcut kolonlar: {list(df.columns)}"
        )

    df[text_col] = df[text_col].astype(str).fillna("").str.strip()
    df = df[df[text_col] != ""].copy()

    if do_basic_clean:
        df[text_col] = df[text_col].map(basic_clean)
        df = df[df[text_col] != ""].copy()

    df[label_col] = df[label_col].astype(str).str.strip().str.lower()

    # normalize label variants (TR + EN + ascii)
    df[label_col] = df[label_col].replace({
        "notr": "nötr",
        "ntr": "nötr",
        "neutral": "nötr",
        "neg": "negatif",
        "negative": "negatif",
        "pos": "pozitif",
        "positive": "pozitif",
        "nötr": "nötr",
        "pozitif": "pozitif",
        "negatif": "negatif",
    })

    df = df[df[label_col].isin(LABELS)].copy()
    df["labels"] = df[label_col].map(label2id).astype(int)

    return df


def stratified_split(
    df: pd.DataFrame,
    seed: int,
    test_size: float,
    val_size: float,
    stratify: bool = True
):
    """
    toplam -> (train+val) + test
    sonra (train+val) -> train + val
    val_size toplamdan oran (örn 0.10)
    """
    y = df["labels"] if stratify else None

    train_val_df, test_df = train_test_split(
        df,
        test_size=test_size,
        random_state=seed,
        stratify=y
    )

    # val oranını train_val üzerinde ayarlıyoruz:
    # toplamda val_size olacak şekilde:
    val_rel = val_size / (1.0 - test_size)
    y_tv = train_val_df["labels"] if stratify else None

    train_df, val_df = train_test_split(
        train_val_df,
        test_size=val_rel,
        random_state=seed,
        stratify=y_tv
    )
    return train_df.reset_index(drop=True), val_df.reset_index(drop=True), test_df.reset_index(drop=True)


# -----------------------------
# Tokenize
# -----------------------------
def tokenize_fn(examples, tokenizer, max_len: int):
    return tokenizer(
        examples["comment_text"],
        truncation=True,
        max_length=max_len,
    )


# -----------------------------
# Class weights + smoothing
# -----------------------------
def compute_class_weights_balanced(train_df: pd.DataFrame) -> torch.Tensor:
    """
    sklearn 'balanced':
    w_c = n_samples / (n_classes * n_samples_c)
    LABELS sırası: [negatif, nötr, pozitif]
    """
    n_samples = len(train_df)
    n_classes = len(LABELS)
    counts = train_df["sentiment_label"].value_counts().to_dict()

    weights = []
    for lbl in LABELS:
        n_c = counts.get(lbl, 1)
        w_c = n_samples / (n_classes * n_c)
        weights.append(w_c)

    return torch.tensor(weights, dtype=torch.float)


def smooth_weights(weights: torch.Tensor, alpha: float = 0.25) -> torch.Tensor:
    """
    w' = (1-alpha)*w + alpha*mean(w)
    """
    w = weights.clone().float()
    m = w.mean()
    return (1 - alpha) * w + alpha * m


# -----------------------------
# Losses
# -----------------------------
class FocalLoss(nn.Module):
    def __init__(self, gamma: float = 2.0, weight: torch.Tensor | None = None):
        super().__init__()
        self.gamma = gamma
        self.weight = weight

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce = nn.functional.cross_entropy(logits, targets, weight=self.weight, reduction="none")
        pt = torch.exp(-ce)
        focal = ((1 - pt) ** self.gamma) * ce
        return focal.mean()


class CustomLossTrainer(Trainer):
    def __init__(self, loss_type: str, class_weights: torch.Tensor, focal_gamma: float, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.loss_type = loss_type
        self.class_weights = class_weights
        self.focal_gamma = focal_gamma

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.get("labels")
        outputs = model(**{k: v for k, v in inputs.items() if k != "labels"})
        logits = outputs.get("logits")

        cw = self.class_weights.to(logits.device)

        if self.loss_type in ("wce", "wce_smooth"):
            loss_fct = nn.CrossEntropyLoss(weight=cw)
            loss = loss_fct(logits, labels)
        elif self.loss_type == "focal":
            loss_fct = FocalLoss(gamma=self.focal_gamma, weight=cw)
            loss = loss_fct(logits, labels)
        else:
            raise ValueError(f"Unknown loss_type: {self.loss_type}")

        return (loss, outputs) if return_outputs else loss


# -----------------------------
# Metrics
# -----------------------------
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    acc = accuracy_score(labels, preds)
    f1_macro = f1_score(labels, preds, average="macro")
    f1_weighted = f1_score(labels, preds, average="weighted")
    return {"accuracy": acc, "f1_macro": f1_macro, "f1_weighted": f1_weighted}


# -----------------------------
# Run single experiment
# -----------------------------
def run_one(cfg: ExpConfig) -> dict:
    seed_everything(cfg.seed)

    df = load_df(Path(cfg.csv_path), cfg.text_col, cfg.label_col, cfg.do_basic_clean)

    # cleaning: short comments
    if cfg.min_words and cfg.min_words > 0:
        df["__wc"] = df[cfg.text_col].map(word_count)
        df = df[df["__wc"] >= cfg.min_words].drop(columns=["__wc"]).copy()

    # cleaning: duplicates
    if cfg.drop_duplicates:
        df = df.drop_duplicates(subset=[cfg.text_col]).copy()

    df = df.reset_index(drop=True)

    # split
    train_df, val_df, test_df = stratified_split(
        df, seed=cfg.seed, test_size=cfg.test_size, val_size=cfg.val_size, stratify=cfg.stratify
    )

    # class weights from TRAIN only
    base_w = compute_class_weights_balanced(train_df)
    if cfg.loss_type == "wce_smooth":
        cw = smooth_weights(base_w, alpha=cfg.smooth_alpha)
    else:
        cw = base_w

    # output dir naming
    tag = (
        f"{Path(cfg.model_name).name}"
        f"_len{cfg.max_len}"
        f"_ep{cfg.epochs}"
        f"_lr{cfg.lr}"
        f"_loss{cfg.loss_type}"
        f"_seed{cfg.seed}"
        f"_minw{cfg.min_words}"
        f"_dedup{int(cfg.drop_duplicates)}"
    )
    out_dir = Path(cfg.base_outdir) / tag
    out_dir.mkdir(parents=True, exist_ok=True)

    # save splits (debug + reproducibility)
    splits_dir = out_dir / "splits"
    splits_dir.mkdir(parents=True, exist_ok=True)
    train_df[[cfg.text_col, cfg.label_col]].to_csv(splits_dir / "train.csv", index=False, encoding="utf-8")
    val_df[[cfg.text_col, cfg.label_col]].to_csv(splits_dir / "val.csv", index=False, encoding="utf-8")
    test_df[[cfg.text_col, cfg.label_col]].to_csv(splits_dir / "test.csv", index=False, encoding="utf-8")

    print("\n==============================")
    print("[CONFIG]")
    print(json.dumps(asdict(cfg), ensure_ascii=False, indent=2))
    print(f"\n[INFO] Loaded total after cleaning: {len(df)}")
    print("\n[DEBUG] Train distribution:\n", train_df[cfg.label_col].value_counts())
    print("\n[DEBUG] Val distribution:\n", val_df[cfg.label_col].value_counts())
    print("\n[DEBUG] Test distribution:\n", test_df[cfg.label_col].value_counts())
    print("\n[DEBUG] CLASS_WEIGHTS base (negatif,nötr,pozitif):", base_w.tolist())
    if cfg.loss_type == "wce_smooth":
        print(f"[DEBUG] CLASS_WEIGHTS smoothed (alpha={cfg.smooth_alpha}):", cw.tolist())
    if cfg.loss_type == "focal":
        print(f"[DEBUG] FOCAL_GAMMA: {cfg.focal_gamma}")
    print("==============================\n")

    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)

    # rename to fixed keys expected by tokenize_fn
    train_tmp = train_df.rename(columns={cfg.text_col: "comment_text", cfg.label_col: "sentiment_label"})
    val_tmp = val_df.rename(columns={cfg.text_col: "comment_text", cfg.label_col: "sentiment_label"})
    test_tmp = test_df.rename(columns={cfg.text_col: "comment_text", cfg.label_col: "sentiment_label"})

    train_ds = Dataset.from_pandas(train_tmp[["comment_text", "labels"]], preserve_index=False)
    val_ds = Dataset.from_pandas(val_tmp[["comment_text", "labels"]], preserve_index=False)
    test_ds = Dataset.from_pandas(test_tmp[["comment_text", "labels"]], preserve_index=False)

    train_ds = train_ds.map(lambda x: tokenize_fn(x, tokenizer, cfg.max_len), batched=True)
    val_ds = val_ds.map(lambda x: tokenize_fn(x, tokenizer, cfg.max_len), batched=True)
    test_ds = test_ds.map(lambda x: tokenize_fn(x, tokenizer, cfg.max_len), batched=True)

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    model = AutoModelForSequenceClassification.from_pretrained(
        cfg.model_name,
        num_labels=len(LABELS),
        id2label=id2label,
        label2id=label2id,
    )

    args = TrainingArguments(
        output_dir=str(out_dir),
        seed=cfg.seed,

        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        greater_is_better=True,

        learning_rate=cfg.lr,
        per_device_train_batch_size=cfg.train_bs,
        per_device_eval_batch_size=cfg.eval_bs,
        num_train_epochs=cfg.epochs,
        weight_decay=cfg.weight_decay,

        logging_steps=50,
        report_to="none",

        save_total_limit=cfg.save_total_limit,
        fp16=cfg.fp16,
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
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=callbacks,
    )

    print("[INFO] Training started...")
    trainer.train()

    print("\n[INFO] Evaluating on TEST...")
    preds = trainer.predict(test_ds)
    y_true = preds.label_ids
    y_pred = np.argmax(preds.predictions, axis=-1)

    report = classification_report(y_true, y_pred, target_names=LABELS, digits=4)
    cm = confusion_matrix(y_true, y_pred)

    header = (
        f"MODEL_NAME: {cfg.model_name}\n"
        f"MAX_LEN: {cfg.max_len}\n"
        f"EPOCHS: {cfg.epochs}\n"
        f"LR: {cfg.lr}\n"
        f"SEED: {cfg.seed}\n"
        f"CSV_PATH: {cfg.csv_path}\n"
        f"TEXT_COL: {cfg.text_col}\n"
        f"LABEL_COL: {cfg.label_col}\n"
        f"SPLIT(test/val): {cfg.test_size}/{cfg.val_size}\n"
        f"LOSS_TYPE: {cfg.loss_type}\n"
        f"MIN_WORDS: {cfg.min_words}\n"
        f"DROP_DUPLICATES: {cfg.drop_duplicates}\n"
        f"CLASS_WEIGHTS base: {base_w.tolist()}\n"
    )
    if cfg.loss_type == "wce_smooth":
        header += f"CLASS_WEIGHTS smoothed (alpha={cfg.smooth_alpha}): {cw.tolist()}\n"
    if cfg.loss_type == "focal":
        header += f"FOCAL_GAMMA: {cfg.focal_gamma}\n"
    header += "\n"

    (out_dir / "test_report.txt").write_text(
        header + report + "\n\nConfusion matrix:\n" + str(cm),
        encoding="utf-8",
    )
    (out_dir / "config.json").write_text(
        json.dumps(asdict(cfg), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    f1_per_class = f1_score(y_true, y_pred, average=None, labels=[0, 1, 2])
    result = {
        "out_dir": str(out_dir),
        "model_name": cfg.model_name,
        "max_len": cfg.max_len,
        "epochs": cfg.epochs,
        "lr": cfg.lr,
        "loss_type": cfg.loss_type,
        "seed": cfg.seed,
        "min_words": int(cfg.min_words),
        "drop_duplicates": int(cfg.drop_duplicates),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro")),
        "f1_weighted": float(f1_score(y_true, y_pred, average="weighted")),
        "f1_negatif": float(f1_per_class[0]),
        "f1_notr": float(f1_per_class[1]),
        "f1_pozitif": float(f1_per_class[2]),
    }

    print("\n[OK] Saved ->", out_dir / "test_report.txt")
    return result


# -----------------------------
# Main: multi-seed run + summary
# -----------------------------
def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_name", type=str, default=MODEL_NAME_DEFAULT)
    parser.add_argument("--max_len", type=int, default=384)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=2e-5)

    parser.add_argument("--loss_type", type=str, default="wce", choices=["wce", "wce_smooth", "focal"])
    parser.add_argument("--smooth_alpha", type=float, default=0.25)
    parser.add_argument("--focal_gamma", type=float, default=2.0)

    parser.add_argument("--seeds", type=str, default="42")  # "42,43,44,45,46"

    parser.add_argument("--csv_path", type=str, default=CSV_DEFAULT)
    parser.add_argument("--text_col", type=str, default="comment_text")
    parser.add_argument("--label_col", type=str, default="sentiment_label")

    parser.add_argument("--test_size", type=float, default=0.10)
    parser.add_argument("--val_size", type=float, default=0.10)
    parser.add_argument("--no_stratify", action="store_true")

    parser.add_argument("--min_words", type=int, default=3)
    parser.add_argument("--no_clean", action="store_true")
    parser.add_argument("--no_dedup", action="store_true")

    parser.add_argument("--base_outdir", type=str, default=BASE_OUTDIR_DEFAULT)

    parser.add_argument("--fp16", action="store_true", help="fp16 aç (GPU destekliyorsa)")
    parser.add_argument("--no_fp16", action="store_true", help="fp16 kapat")

    args = parser.parse_args()

    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]

    fp16 = True
    if args.fp16:
        fp16 = True
    if args.no_fp16:
        fp16 = False

    all_results = []
    for sd in seeds:
        cfg = ExpConfig(
            model_name=args.model_name,
            max_len=args.max_len,
            lr=args.lr,
            epochs=args.epochs,
            loss_type=args.loss_type,
            smooth_alpha=args.smooth_alpha,
            focal_gamma=args.focal_gamma,
            seed=sd,
            csv_path=args.csv_path,
            text_col=args.text_col,
            label_col=args.label_col,
            test_size=args.test_size,
            val_size=args.val_size,
            stratify=(not args.no_stratify),
            min_words=args.min_words,
            do_basic_clean=(not args.no_clean),
            drop_duplicates=(not args.no_dedup),
            base_outdir=args.base_outdir,
            fp16=fp16,
        )
        res = run_one(cfg)
        all_results.append(res)

    df = pd.DataFrame(all_results)

    outdir = Path(args.base_outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    summary_path = outdir / (
        f"summary_{Path(args.model_name).name}"
        f"_len{args.max_len}"
        f"_ep{args.epochs}"
        f"_loss{args.loss_type}"
        f"_minw{args.min_words}"
        f"_dedup{int(not args.no_dedup)}"
        f".csv"
    )
    df.to_csv(summary_path, index=False, encoding="utf-8")

    print("\n==============================")
    print("[SUMMARY] Mean ± Std over seeds")
    for col in ["accuracy", "f1_macro", "f1_weighted", "f1_negatif", "f1_notr", "f1_pozitif"]:
        mean = df[col].mean()
        std = df[col].std(ddof=1) if len(df) > 1 else 0.0
        print(f"{col:12s}: {mean:.4f} ± {std:.4f}")
    print("[OK] Summary saved ->", summary_path)
    print("==============================\n")


if __name__ == "__main__":
    main()