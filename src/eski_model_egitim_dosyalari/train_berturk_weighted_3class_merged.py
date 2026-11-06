# src/train_berturk_weighted_3class_merged.py

from pathlib import Path
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
)
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix


# ---- Ayarlar ----
MODEL_NAME = "dbmdz/bert-base-turkish-cased"
MAX_LEN = 256
SEED = 42

TRAIN_CSV = Path("data/processed/splits3/train.csv")
VAL_CSV = Path("data/processed/splits3/val.csv")
TEST_CSV = Path("data/processed/splits3/test.csv")

OUT_DIR = Path("models/berturk_3class_weighted_merged")

# Etiket sırası SABİT olmalı
LABELS = ["negatif", "nötr", "pozitif"]
label2id = {l: i for i, l in enumerate(LABELS)}
id2label = {i: l for l, i in label2id.items()}

# Yeni train setine göre class weights (compute_class_weights_3class_splits3.py çıktısı)
CLASS_WEIGHTS = torch.tensor([2.3291, 1.6331, 0.5106], dtype=torch.float)


def load_df(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"CSV bulunamadı: {path}")

    df = pd.read_csv(path)

    if "comment_text" not in df.columns or "sentiment_label" not in df.columns:
        raise ValueError(f"{path} içinde comment_text veya sentiment_label kolonu yok.")

    df["comment_text"] = df["comment_text"].astype(str).fillna("").str.strip()
    df = df[df["comment_text"] != ""].copy()

    df["sentiment_label"] = df["sentiment_label"].astype(str).str.strip().str.lower()

    # Sadece 3 label kalsın
    df = df[df["sentiment_label"].isin(LABELS)].copy()

    df["labels"] = df["sentiment_label"].map(label2id).astype(int)
    return df


def tokenize_fn(examples, tokenizer):
    return tokenizer(
        examples["comment_text"],
        truncation=True,
        max_length=MAX_LEN,
    )


class WeightedTrainer(Trainer):
    def __init__(self, class_weights: torch.Tensor, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.get("labels")
        outputs = model(**{k: v for k, v in inputs.items() if k != "labels"})
        logits = outputs.get("logits")

        loss_fct = nn.CrossEntropyLoss(weight=self.class_weights.to(logits.device))
        loss = loss_fct(logits, labels)
        return (loss, outputs) if return_outputs else loss


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)

    acc = accuracy_score(labels, preds)
    f1_macro = f1_score(labels, preds, average="macro")
    f1_weighted = f1_score(labels, preds, average="weighted")
    return {"accuracy": acc, "f1_macro": f1_macro, "f1_weighted": f1_weighted}


def main():
    train_df = load_df(TRAIN_CSV)
    val_df = load_df(VAL_CSV)
    test_df = load_df(TEST_CSV)

    print("\n[DEBUG] Train distribution:")
    print(train_df["sentiment_label"].value_counts())
    print("\n[DEBUG] Val distribution:")
    print(val_df["sentiment_label"].value_counts())
    print("\n[DEBUG] Test distribution:")
    print(test_df["sentiment_label"].value_counts())

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    train_ds = Dataset.from_pandas(train_df[["comment_text", "labels"]], preserve_index=False)
    val_ds = Dataset.from_pandas(val_df[["comment_text", "labels"]], preserve_index=False)
    test_ds = Dataset.from_pandas(test_df[["comment_text", "labels"]], preserve_index=False)

    train_ds = train_ds.map(lambda x: tokenize_fn(x, tokenizer), batched=True)
    val_ds = val_ds.map(lambda x: tokenize_fn(x, tokenizer), batched=True)
    test_ds = test_ds.map(lambda x: tokenize_fn(x, tokenizer), batched=True)

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=len(LABELS),
        id2label=id2label,
        label2id=label2id,
    )

    # transformers sürüm uyumluluğu için eval_strategy kullanıyoruz
    args = TrainingArguments(
        output_dir=str(OUT_DIR),
        seed=SEED,

        eval_strategy="epoch",
        save_strategy="epoch",

        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        greater_is_better=True,

        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=32,
        num_train_epochs=3,
        weight_decay=0.01,

        logging_steps=50,
        report_to="none",
    )

    trainer = WeightedTrainer(
        class_weights=CLASS_WEIGHTS,
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    print("\n[INFO] Training started...")
    trainer.train()

    print("\n[INFO] Evaluating on TEST...")
    preds = trainer.predict(test_ds)
    y_true = preds.label_ids
    y_pred = np.argmax(preds.predictions, axis=-1)

    report = classification_report(y_true, y_pred, target_names=LABELS, digits=4)
    cm = confusion_matrix(y_true, y_pred)

    print("\n[REPORT] Classification report:\n")
    print(report)

    print("\n[REPORT] Confusion matrix:\n")
    print(cm)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    report_path = OUT_DIR / "test_report.txt"
    report_path.write_text(report + "\n\nConfusion matrix:\n" + str(cm), encoding="utf-8")
    print(f"\n[OK] Test report saved -> {report_path}")


if __name__ == "__main__":
    main()
