# src/make_splits_3class.py

from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

SEED = 42

IN_CSV = Path("data/processed/final_dataset_merged_3class.csv")
OUT_DIR = Path("data/processed/splits3")

TRAIN_PATH = OUT_DIR / "train.csv"
VAL_PATH = OUT_DIR / "val.csv"
TEST_PATH = OUT_DIR / "test.csv"

REQUIRED_COLS = [
    "hotel_name",
    "comment_text",
    "source_url",
    "source",
    "rating_auto",
    "sentiment_label",
]

def main():
    if not IN_CSV.exists():
        raise FileNotFoundError(f"Girdi bulunamadı: {IN_CSV}")

    df = pd.read_csv(IN_CSV)

    missing = set(REQUIRED_COLS) - set(df.columns)
    if missing:
        raise ValueError(f"Eksik kolon(lar): {sorted(missing)}")

    df = df[REQUIRED_COLS].copy()

    # Temizlik
    df["comment_text"] = df["comment_text"].astype(str).fillna("").str.strip()
    df = df[df["comment_text"] != ""].copy()

    df["sentiment_label"] = df["sentiment_label"].astype(str).str.strip().str.lower()

    # Stratified split: önce test (%10), sonra val (%10)
    train_val, test = train_test_split(
        df,
        test_size=0.10,
        random_state=SEED,
        stratify=df["sentiment_label"],
    )

    train, val = train_test_split(
        train_val,
        test_size=0.1111111111,  # train_val'ın %11.11'i => toplamda %10 val
        random_state=SEED,
        stratify=train_val["sentiment_label"],
    )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    train.to_csv(TRAIN_PATH, index=False, encoding="utf-8-sig")
    val.to_csv(VAL_PATH, index=False, encoding="utf-8-sig")
    test.to_csv(TEST_PATH, index=False, encoding="utf-8-sig")

    print(f"[OK] Train: {len(train)} -> {TRAIN_PATH}")
    print(f"[OK] Val  : {len(val)} -> {VAL_PATH}")
    print(f"[OK] Test : {len(test)} -> {TEST_PATH}")

    print("\n[INFO] Train dist:")
    print(train["sentiment_label"].value_counts())
    print("\n[INFO] Val dist:")
    print(val["sentiment_label"].value_counts())
    print("\n[INFO] Test dist:")
    print(test["sentiment_label"].value_counts())

if __name__ == "__main__":
    main()
