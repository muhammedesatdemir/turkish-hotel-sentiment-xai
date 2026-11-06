# src/veri_seti_ayarlama_dosyalari/make_splits_no_kaggle.py

from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


# -------------------------
# Ayarlar
# -------------------------
SEED = 42

IN_CSV = Path("data/processed/final_dataset_no_kaggle_3class.csv")
OUT_DIR = Path("data/processed/splits3_no_kaggle")

TEXT_COL = "comment_text"
LABEL_COL = "sentiment_label"

# Etiket sırası (eğitim scriptinizle birebir aynı kalmalı)
LABELS = ["negatif", "nötr", "pozitif"]

# Split oranları:
# Eğer sizde eski splits3 oranları farklıysa burayı değiştirin.
TEST_SIZE = 0.15  # toplamın %15'i test
VAL_SIZE = 0.15   # toplamın %15'i val (kalan train)


def load_df(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"CSV bulunamadı: {path.resolve()}")

    df = pd.read_csv(path)

    if TEXT_COL not in df.columns or LABEL_COL not in df.columns:
        raise ValueError(f"{path} içinde '{TEXT_COL}' veya '{LABEL_COL}' kolonu yok.")

    df[TEXT_COL] = df[TEXT_COL].astype(str).fillna("").str.strip()
    df = df[df[TEXT_COL] != ""].copy()

    df[LABEL_COL] = df[LABEL_COL].astype(str).str.strip().str.lower()

    # Olası normalize (gerekirse genişletilebilir)
    df[LABEL_COL] = df[LABEL_COL].replace({
        "notr": "nötr",
        "neutral": "nötr",
        "neg": "negatif",
        "pos": "pozitif",
        "positive": "pozitif",
        "negative": "negatif",
    })

    # Sadece 3 sınıf kalsın
    df = df[df[LABEL_COL].isin(LABELS)].copy()

    return df.reset_index(drop=True)


def main():
    np.random.seed(SEED)

    df = load_df(IN_CSV)

    print(f"[INFO] Input rows: {len(df)}")
    print("[INFO] Label distribution (full):")
    print(df[LABEL_COL].value_counts())

    # Stratify için y
    y = df[LABEL_COL]

    # 1) Test ayır
    train_val_df, test_df = train_test_split(
        df,
        test_size=TEST_SIZE,
        random_state=SEED,
        stratify=y,
    )

    # 2) Train/Val ayır (val toplamın VAL_SIZE olacak şekilde)
    # train_val içindeki val oranı:
    val_ratio_in_trainval = VAL_SIZE / (1.0 - TEST_SIZE)

    train_df, val_df = train_test_split(
        train_val_df,
        test_size=val_ratio_in_trainval,
        random_state=SEED,
        stratify=train_val_df[LABEL_COL],
    )

    # Kaydet
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    train_path = OUT_DIR / "train.csv"
    val_path = OUT_DIR / "val.csv"
    test_path = OUT_DIR / "test.csv"

    train_df.to_csv(train_path, index=False, encoding="utf-8")
    val_df.to_csv(val_path, index=False, encoding="utf-8")
    test_df.to_csv(test_path, index=False, encoding="utf-8")

    print("\n[OK] Splits created ✅")
    print(f"-> {train_path} ({len(train_df)})")
    print(f"-> {val_path}   ({len(val_df)})")
    print(f"-> {test_path}  ({len(test_df)})")

    print("\n[DEBUG] Train distribution:")
    print(train_df[LABEL_COL].value_counts())
    print("\n[DEBUG] Val distribution:")
    print(val_df[LABEL_COL].value_counts())
    print("\n[DEBUG] Test distribution:")
    print(test_df[LABEL_COL].value_counts())

    # Küçük bir kontrol: satırlar toplamı aynı mı?
    total = len(train_df) + len(val_df) + len(test_df)
    if total != len(df):
        raise RuntimeError(f"Split toplamı ({total}) input sayısına ({len(df)}) eşit değil!")

    # Kopya kontrolü (opsiyonel ama faydalı)
    # Aynı satır birden fazla split'e düşmüş mü?
    # Burada kaba bir hash ile bakıyoruz.
    def hash_rows(d):
        return pd.util.hash_pandas_object(d[[TEXT_COL, LABEL_COL]], index=False)

    h_train = set(hash_rows(train_df).tolist())
    h_val = set(hash_rows(val_df).tolist())
    h_test = set(hash_rows(test_df).tolist())

    if (h_train & h_val) or (h_train & h_test) or (h_val & h_test):
        print("\n[WARN] Bazı satırlar split'ler arasında aynı görünüyor (tam metin/etiket aynı olabilir).")
        print("      Eğer veri içinde kopya yorumlar varsa bu normal olabilir.")
    else:
        print("\n[OK] No overlap detected between splits (by text+label hash).")


if __name__ == "__main__":
    main()
