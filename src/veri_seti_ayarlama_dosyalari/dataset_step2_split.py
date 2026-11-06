# ============================================================
# dataset_step2_split.py
# ADIM 2: Stratified train/val/test split + dağılım raporu
# ============================================================

import json
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

# ------------------------------------------------------------
# GİRDİ / ÇIKTI YOLLARI (SENİN PROJE DİZİNİNE GÖRE)
# ------------------------------------------------------------
INPUT_CSV = r"C:\Users\Monster\Desktop\python\turkce_otel_yorum\data\processed\real_final_dataset_cleaned.csv"
OUT_DIR = Path(r"C:\Users\Monster\Desktop\python\turkce_otel_yorum\data\processed\splits")

TRAIN_PATH = OUT_DIR / "train.csv"
VAL_PATH = OUT_DIR / "val.csv"
TEST_PATH = OUT_DIR / "test.csv"
REPORT_PATH = OUT_DIR / "split_report.json"

# ------------------------------------------------------------
# KOLONLAR
# ------------------------------------------------------------
LABEL_COL = "sentiment_label"
REQUIRED_COLUMNS = [
    "hotel_name",
    "comment_text",
    "source_url",
    "source",
    "rating_auto",
    "sentiment_label"
]

# ------------------------------------------------------------
# SPLIT AYARLARI
# ------------------------------------------------------------
RANDOM_STATE = 42
TRAIN_RATIO = 0.80
VAL_RATIO = 0.10
TEST_RATIO = 0.10

# Basit kontrol: oranlar 1 etmeli
assert abs((TRAIN_RATIO + VAL_RATIO + TEST_RATIO) - 1.0) < 1e-9, "Split oranları toplamı 1 olmalı."


def class_distribution(df: pd.DataFrame) -> pd.DataFrame:
    dist = df[LABEL_COL].value_counts().to_frame("count")
    dist["percent"] = (dist["count"] / len(df) * 100).round(2)

    order = ["negatif", "nötr", "pozitif"]
    idx = [x for x in order if x in dist.index]
    dist = dist.loc[idx]
    return dist


def print_split_stats(name: str, df: pd.DataFrame) -> None:
    print("\n" + "=" * 70)
    print(f"{name}")
    print("=" * 70)
    print(f"Toplam: {len(df):,}")
    print(class_distribution(df).to_string())


def dist_to_dict(df: pd.DataFrame) -> dict:
    dist_df = class_distribution(df)
    return {
        idx: {"count": int(row["count"]), "percent": float(row["percent"])}
        for idx, row in dist_df.iterrows()
    }


def main():
    print("\n📥 Cleaned dataset okunuyor...")
    df = pd.read_csv(INPUT_CSV)

    missing_cols = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Eksik kolon(lar) var: {missing_cols}")

    if LABEL_COL not in df.columns:
        raise ValueError(f"Etiket kolonu bulunamadı: {LABEL_COL}")

    # --------------------------------------------------------
    # 1) Önce train ve temp ayır (train=80%, temp=20%)
    # --------------------------------------------------------
    y = df[LABEL_COL]
    train_df, temp_df = train_test_split(
        df,
        test_size=(1.0 - TRAIN_RATIO),  # 0.20
        stratify=y,
        random_state=RANDOM_STATE
    )

    # --------------------------------------------------------
    # 2) temp'i val ve test olarak ikiye böl (val=10%, test=10%)
    # temp toplam %20, bunun yarısı %10/%10
    # --------------------------------------------------------
    y_temp = temp_df[LABEL_COL]
    val_df, test_df = train_test_split(
        temp_df,
        test_size=(TEST_RATIO / (VAL_RATIO + TEST_RATIO)),  # 0.5
        stratify=y_temp,
        random_state=RANDOM_STATE
    )

    # --------------------------------------------------------
    # 3) Çıktıları kaydet
    # --------------------------------------------------------
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    train_df.to_csv(TRAIN_PATH, index=False, encoding="utf-8")
    val_df.to_csv(VAL_PATH, index=False, encoding="utf-8")
    test_df.to_csv(TEST_PATH, index=False, encoding="utf-8")

    # --------------------------------------------------------
    # 4) Konsola istatistik
    # --------------------------------------------------------
    print_split_stats("TRAIN SPLIT", train_df)
    print_split_stats("VAL SPLIT", val_df)
    print_split_stats("TEST SPLIT", test_df)

    # --------------------------------------------------------
    # 5) JSON rapor
    # --------------------------------------------------------
    report = {
        "input_csv": INPUT_CSV,
        "random_state": RANDOM_STATE,
        "ratios": {
            "train": TRAIN_RATIO,
            "val": VAL_RATIO,
            "test": TEST_RATIO
        },
        "sizes": {
            "total": int(len(df)),
            "train": int(len(train_df)),
            "val": int(len(val_df)),
            "test": int(len(test_df))
        },
        "class_distribution": {
            "train": dist_to_dict(train_df),
            "val": dist_to_dict(val_df),
            "test": dist_to_dict(test_df)
        },
        "outputs": {
            "train_csv": str(TRAIN_PATH),
            "val_csv": str(VAL_PATH),
            "test_csv": str(TEST_PATH)
        }
    }

    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print("\n✅ Split dosyaları kaydedildi:")
    print(TRAIN_PATH)
    print(VAL_PATH)
    print(TEST_PATH)

    print("\n✅ Split raporu kaydedildi:")
    print(REPORT_PATH)

    # Son kontrol: toplamlar uyuşuyor mu?
    total_check = len(train_df) + len(val_df) + len(test_df)
    if total_check != len(df):
        raise RuntimeError(f"Toplam uyuşmuyor! train+val+test={total_check}, total={len(df)}")

    print("\n🎯 Kontrol OK: train+val+test = total")


if __name__ == "__main__":
    main()
