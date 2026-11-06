# src/train_berturk_3class_no_kaggle_experiments.py

from __future__ import annotations
from pathlib import Path
from dataclasses import dataclass, asdict
import argparse
import json
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
)
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix


# -----------------------------
# Sabitler
# -----------------------------
MODEL_NAME_DEFAULT = "dbmdz/bert-base-turkish-cased"

TRAIN_CSV_DEFAULT = "data/processed/splits3_no_kaggle/train.csv"
VAL_CSV_DEFAULT   = "data/processed/splits3_no_kaggle/val.csv"
TEST_CSV_DEFAULT  = "data/processed/splits3_no_kaggle/test.csv"

BASE_OUTDIR_DEFAULT = "models/experiments_no_kaggle"

LABELS = ["negatif", "nötr", "pozitif"]
label2id = {l: i for i, l in enumerate(LABELS)}
id2label = {i: l for l, i in label2id.items()}


# -----------------------------
# Konfig
# -----------------------------
@dataclass
class ExpConfig:
    model_name: str = MODEL_NAME_DEFAULT
    max_len: int = 512                 # <-- iyileştirme #1
    lr: float = 2e-5
    epochs: int = 5                    # <-- iyileştirme #2
    train_bs: int = 16
    eval_bs: int = 32
    weight_decay: float = 0.01
    seed: int = 42

    # loss seçenekleri: "wce" | "wce_smooth" | "focal"
    loss_type: str = "wce"
    smooth_alpha: float = 0.25         # <-- iyileştirme #3a (weight smoothing)
    focal_gamma: float = 2.0           # <-- iyileştirme #3b (focal)
    focal_alpha: float | None = None   # None -> class weights kullanılır

    # early stopping
    early_stop: bool = True
    early_patience: int = 2
    early_threshold: float = 1e-4

    # paths
    train_csv: str = TRAIN_CSV_DEFAULT
    val_csv: str = VAL_CSV_DEFAULT
    test_csv: str = TEST_CSV_DEFAULT
    base_outdir: str = BASE_OUTDIR_DEFAULT


# -----------------------------
# Veri yükleme
# -----------------------------
def load_df(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"CSV bulunamadı: {path}")

    df = pd.read_csv(path)

    if "comment_text" not in df.columns or "sentiment_label" not in df.columns:
        raise ValueError(f"{path} içinde comment_text veya sentiment_label kolonu yok.")

    df["comment_text"] = df["comment_text"].astype(str).fillna("").str.strip()
    df = df[df["comment_text"] != ""].copy()

    df["sentiment_label"] = df["sentiment_label"].astype(str).str.strip().str.lower()

    df["sentiment_label"] = df["sentiment_label"].replace({
        "notr": "nötr",
        "neutral": "nötr",
        "neg": "negatif",
        "negative": "negatif",
        "pos": "pozitif",
        "positive": "pozitif",
    })

    df = df[df["sentiment_label"].isin(LABELS)].copy()
    df["labels"] = df["sentiment_label"].map(label2id).astype(int)
    return df


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
    Aşırı ağırlıkları yumuşatmak için basit ama etkili yöntem:
    w' = (1-alpha)*w + alpha*mean(w)
    alpha 0..1
    """
    w = weights.clone().float()
    m = w.mean()
    return (1 - alpha) * w + alpha * m


# -----------------------------
# Loss'lar
# -----------------------------
class FocalLoss(nn.Module):
    """
    Multi-class focal loss.
    - gamma: odaklanma parametresi
    - alpha: None ise class_weights kullan (CE weight).
    """
    def __init__(self, gamma: float = 2.0, weight: torch.Tensor | None = None):
        super().__init__()
        self.gamma = gamma
        self.weight = weight

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # CE per-sample
        ce = nn.functional.cross_entropy(logits, targets, weight=self.weight, reduction="none")
        pt = torch.exp(-ce)  # pt = softmax prob of true class (approximately)
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
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    train_df = load_df(Path(cfg.train_csv))
    val_df = load_df(Path(cfg.val_csv))
    test_df = load_df(Path(cfg.test_csv))

    # class weights
    base_w = compute_class_weights_balanced(train_df)
    if cfg.loss_type == "wce_smooth":
        cw = smooth_weights(base_w, alpha=cfg.smooth_alpha)
    else:
        cw = base_w

    # output dir naming
    tag = f"{Path(cfg.model_name).name}_len{cfg.max_len}_ep{cfg.epochs}_lr{cfg.lr}_loss{cfg.loss_type}_seed{cfg.seed}"
    out_dir = Path(cfg.base_outdir) / tag
    out_dir.mkdir(parents=True, exist_ok=True)

    print("\n==============================")
    print("[CONFIG]")
    print(json.dumps(asdict(cfg), ensure_ascii=False, indent=2))
    print("\n[DEBUG] Train distribution:\n", train_df["sentiment_label"].value_counts())
    print("\n[DEBUG] Val distribution:\n", val_df["sentiment_label"].value_counts())
    print("\n[DEBUG] Test distribution:\n", test_df["sentiment_label"].value_counts())
    print("\n[DEBUG] CLASS_WEIGHTS base (negatif,nötr,pozitif):", base_w.tolist())
    if cfg.loss_type == "wce_smooth":
        print("[DEBUG] CLASS_WEIGHTS smoothed:", cw.tolist())
    print("==============================\n")

    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)

    train_ds = Dataset.from_pandas(train_df[["comment_text", "labels"]], preserve_index=False)
    val_ds = Dataset.from_pandas(val_df[["comment_text", "labels"]], preserve_index=False)
    test_ds = Dataset.from_pandas(test_df[["comment_text", "labels"]], preserve_index=False)

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
    )

    callbacks = []
    if cfg.early_stop:
        callbacks.append(EarlyStoppingCallback(
            early_stopping_patience=cfg.early_patience,
            early_stopping_threshold=cfg.early_threshold,
        ))

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

    # write report
    header = (
        f"MODEL_NAME: {cfg.model_name}\n"
        f"MAX_LEN: {cfg.max_len}\n"
        f"SEED: {cfg.seed}\n"
        f"TRAIN_CSV: {cfg.train_csv}\n"
        f"VAL_CSV: {cfg.val_csv}\n"
        f"TEST_CSV: {cfg.test_csv}\n"
        f"LOSS_TYPE: {cfg.loss_type}\n"
        f"CLASS_WEIGHTS base: {base_w.tolist()}\n"
    )
    if cfg.loss_type == "wce_smooth":
        header += f"CLASS_WEIGHTS smoothed (alpha={cfg.smooth_alpha}): {cw.tolist()}\n"
    if cfg.loss_type == "focal":
        header += f"FOCAL_GAMMA: {cfg.focal_gamma}\n"
    header += "\n"

    (out_dir / "test_report.txt").write_text(
        header + report + "\n\nConfusion matrix:\n" + str(cm),
        encoding="utf-8"
    )

    # collect key metrics (macro/weighted/acc + per-class f1)
    # parse from sklearn outputs is messy; compute directly:
    f1_per_class = f1_score(y_true, y_pred, average=None, labels=[0, 1, 2])
    result = {
        "out_dir": str(out_dir),
        "model_name": cfg.model_name,
        "max_len": cfg.max_len,
        "epochs": cfg.epochs,
        "lr": cfg.lr,
        "loss_type": cfg.loss_type,
        "seed": cfg.seed,
        "accuracy": accuracy_score(y_true, y_pred),
        "f1_macro": f1_score(y_true, y_pred, average="macro"),
        "f1_weighted": f1_score(y_true, y_pred, average="weighted"),
        "f1_negatif": float(f1_per_class[0]),
        "f1_notr": float(f1_per_class[1]),
        "f1_pozitif": float(f1_per_class[2]),
    }

    # also dump config
    (out_dir / "config.json").write_text(json.dumps(asdict(cfg), ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n[OK] Saved ->", out_dir / "test_report.txt")
    return result


# -----------------------------
# Main: multi-seed run + summary
# -----------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default=MODEL_NAME_DEFAULT)
    parser.add_argument("--max_len", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--loss_type", type=str, default="wce", choices=["wce", "wce_smooth", "focal"])
    parser.add_argument("--smooth_alpha", type=float, default=0.25)
    parser.add_argument("--focal_gamma", type=float, default=2.0)
    parser.add_argument("--seeds", type=str, default="42,43,44")  # <-- iyileştirme #4
    parser.add_argument("--train_csv", type=str, default=TRAIN_CSV_DEFAULT)
    parser.add_argument("--val_csv", type=str, default=VAL_CSV_DEFAULT)
    parser.add_argument("--test_csv", type=str, default=TEST_CSV_DEFAULT)
    parser.add_argument("--base_outdir", type=str, default=BASE_OUTDIR_DEFAULT)

    args = parser.parse_args()
    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]

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
            train_csv=args.train_csv,
            val_csv=args.val_csv,
            test_csv=args.test_csv,
            base_outdir=args.base_outdir,
        )
        res = run_one(cfg)
        all_results.append(res)

    df = pd.DataFrame(all_results)
    summary_path = Path(args.base_outdir) / f"summary_{Path(args.model_name).name}_len{args.max_len}_ep{args.epochs}_loss{args.loss_type}.csv"
    df.to_csv(summary_path, index=False, encoding="utf-8")

    # print mean/std for presentation
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
