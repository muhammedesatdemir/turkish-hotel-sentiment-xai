from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

INPUT = Path("data/processed/final_dataset_tr.csv")

OUT_DIR = Path("data/processed/splits")
TRAIN_OUT = OUT_DIR / "train.csv"
VAL_OUT = OUT_DIR / "val.csv"
TEST_OUT = OUT_DIR / "test.csv"

SEED = 42

def main():
    df = pd.read_csv(INPUT)

    # Stratify: 5 sınıfı korumak için sentiment_label
    strat = df["sentiment_label"].astype(str)

    train_df, temp_df = train_test_split(
        df,
        test_size=0.2,
        random_state=SEED,
        stratify=strat,
    )

    # temp'in stratify'ı yeniden alınır
    temp_strat = temp_df["sentiment_label"].astype(str)

    val_df, test_df = train_test_split(
        temp_df,
        test_size=0.5,
        random_state=SEED,
        stratify=temp_strat,
    )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    train_df.to_csv(TRAIN_OUT, index=False, encoding="utf-8-sig")
    val_df.to_csv(VAL_OUT, index=False, encoding="utf-8-sig")
    test_df.to_csv(TEST_OUT, index=False, encoding="utf-8-sig")

    print("[OK] Train:", len(train_df))
    print("[OK] Val  :", len(val_df))
    print("[OK] Test :", len(test_df))
    print("\n[INFO] Train class dist:\n", train_df["sentiment_label"].value_counts())
    print("\n[INFO] Val class dist:\n", val_df["sentiment_label"].value_counts())
    print("\n[INFO] Test class dist:\n", test_df["sentiment_label"].value_counts())

if __name__ == "__main__":
    main()
