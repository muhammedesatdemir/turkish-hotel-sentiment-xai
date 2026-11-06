# ============================================================
# ADIM 1
# - Duplicate temizliği (comment_text üzerinden)
# - Çok kısa yorumları filtreleme
# - Temizlik öncesi / sonrası sınıf dağılımını teyit etme
# ============================================================

import pandas as pd
import re
import json
from pathlib import Path

# ------------------------------------------------------------
# DOSYA YOLLARI (SENİN VERDİĞİN ŞEKLİYLE)
# ------------------------------------------------------------
INPUT_CSV = r"C:\Users\Monster\Desktop\python\turkce_otel_yorum\data\processed\real_final_dataset.csv"
OUTPUT_CSV = r"C:\Users\Monster\Desktop\python\turkce_otel_yorum\data\processed\real_final_dataset_cleaned.csv"

# ------------------------------------------------------------
# KOLON İSİMLERİ
# ------------------------------------------------------------
TEXT_COL = "comment_text"
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
# FİLTRE PARAMETRELERİ
# ------------------------------------------------------------
MIN_WORDS = 3
MIN_CHARS = 10


# ------------------------------------------------------------
# YARDIMCI FONKSİYONLAR
# ------------------------------------------------------------
def normalize_text(text: str) -> str:
    """Duplicate yakalamak için basit normalize"""
    if pd.isna(text):
        return ""
    text = str(text).strip()
    text = re.sub(r"\s+", " ", text)
    return text


def word_count(text: str) -> int:
    text = normalize_text(text)
    return len(text.split()) if text else 0


def class_distribution(df: pd.DataFrame) -> pd.DataFrame:
    dist = df[LABEL_COL].value_counts().to_frame("count")
    dist["percent"] = (dist["count"] / len(df) * 100).round(2)

    order = ["negatif", "nötr", "pozitif"]
    idx = [x for x in order if x in dist.index]
    dist = dist.loc[idx]

    return dist


def print_distribution(title: str, df: pd.DataFrame):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)
    print(f"Toplam satır: {len(df):,}")
    print("\nSınıf dağılımı:")
    print(class_distribution(df).to_string())


# ------------------------------------------------------------
# ANA AKIŞ
# ------------------------------------------------------------
def main():
    print("\n📥 Veri okunuyor...")
    df = pd.read_csv(INPUT_CSV)

    # Kolon kontrolü
    missing_cols = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Eksik kolon(lar) var: {missing_cols}")

    # Normalize edilmiş metin kolonu (duplicate için)
    df["_comment_norm"] = df[TEXT_COL].apply(normalize_text)

    # --------------------------------------------------------
    # TEMİZLİK ÖNCESİ
    # --------------------------------------------------------
    print_distribution("TEMİZLİK ÖNCESİ", df)

    report = {}
    report["rows_before"] = len(df)

    # --------------------------------------------------------
    # DUPLICATE TEMİZLİĞİ
    # --------------------------------------------------------
    dup_mask = df.duplicated(subset="_comment_norm", keep="first")
    duplicate_count = int(dup_mask.sum())

    df_no_dup = df.loc[~dup_mask].copy()
    report["duplicates_removed"] = duplicate_count
    report["rows_after_dedup"] = len(df_no_dup)

    # --------------------------------------------------------
    # KISA YORUM FİLTRELEME
    # --------------------------------------------------------
    wc = df_no_dup["_comment_norm"].apply(word_count)
    cc = df_no_dup["_comment_norm"].str.len()

    short_mask = (wc < MIN_WORDS) | (cc < MIN_CHARS)
    short_removed = int(short_mask.sum())

    df_clean = df_no_dup.loc[~short_mask].copy()
    report["short_comments_removed"] = short_removed
    report["rows_after_short_filter"] = len(df_clean)

    # Yardımcı kolon sil
    df_clean.drop(columns=["_comment_norm"], inplace=True)

    # --------------------------------------------------------
    # TEMİZLİK SONRASI
    # --------------------------------------------------------
    print("\n" + "-" * 70)
    print(f"Silinen duplicate sayısı     : {duplicate_count:,}")
    print(f"Silinen kısa yorum sayısı    : {short_removed:,}")
    print("-" * 70)

    print_distribution("TEMİZLİK SONRASI", df_clean)

    # --------------------------------------------------------
    # KAYDET
    # --------------------------------------------------------
    output_path = Path(OUTPUT_CSV)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_clean.to_csv(output_path, index=False, encoding="utf-8")

    # Rapor JSON
    dist_after = class_distribution(df_clean)
    report["class_distribution_after"] = {
        idx: {
            "count": int(row["count"]),
            "percent": float(row["percent"])
        }
        for idx, row in dist_after.iterrows()
    }

    report_path = output_path.with_suffix(".report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print("\n✅ Temizlenmiş veri kaydedildi:")
    print(output_path)

    print("\n✅ Özet rapor kaydedildi:")
    print(report_path)


if __name__ == "__main__":
    main()
