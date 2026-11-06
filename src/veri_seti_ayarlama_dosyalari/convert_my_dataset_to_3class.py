# src/convert_my_dataset_to_3class.py

from pathlib import Path
import pandas as pd

IN_CSV = Path("data/processed/final_dataset_tr.csv")
OUT_CSV = Path("data/processed/final_dataset_tr_3class.csv")

# 5-class -> 3-class mapping
SENTIMENT_5_TO_3 = {
    "çok olumsuz": "negatif",
    "olumsuz": "negatif",
    "nötr": "nötr",
    "olumlu": "pozitif",
    "çok olumlu": "pozitif",
}

def map_rating_to_3level(r):
    """1-5 rating -> 1/3/5 (3-level)"""
    if pd.isna(r):
        return None
    try:
        r = int(r)
    except Exception:
        return None

    if r <= 2:
        return 1
    if r == 3:
        return 3
    return 5  # 4-5


def main():
    if not IN_CSV.exists():
        raise FileNotFoundError(f"Girdi bulunamadı: {IN_CSV}")

    df = pd.read_csv(IN_CSV)

    # Kolon kontrolü
    needed = {"hotel_name", "comment_text", "source_url", "source", "rating_auto", "sentiment_label"}
    missing = needed - set(df.columns)
    if missing:
        raise ValueError(f"Eksik kolon(lar): {missing}")

    # Temizlik
    df["comment_text"] = df["comment_text"].astype(str).fillna("").str.strip()
    df = df[df["comment_text"] != ""].copy()

    df["source_url"] = df["source_url"].astype(str).fillna("").str.strip()
    df.loc[df["source_url"] == "", "source_url"] = "-"

    df["sentiment_label"] = df["sentiment_label"].astype(str).str.strip().str.lower()

    # 5 -> 3 map
    df["sentiment_label"] = df["sentiment_label"].map(SENTIMENT_5_TO_3)

    # Map sonrası boş kalan (beklenmedik etiket) satırları düşür
    df = df[df["sentiment_label"].notna()].copy()

    # rating_auto'yu 3 seviyeye indir (1/3/5)
    df["rating_auto"] = df["rating_auto"].apply(map_rating_to_3level)

    # rating_auto boş kaldıysa sentiment_label üzerinden doldur
    # (nadiren olur)
    def fill_rating_from_label(row):
        if pd.notna(row["rating_auto"]):
            return int(row["rating_auto"])
        lab = row["sentiment_label"]
        if lab == "negatif":
            return 1
        if lab == "nötr":
            return 3
        return 5

    df["rating_auto"] = df.apply(fill_rating_from_label, axis=1).astype(int)

    # Sadece istenen kolonlar ve sırayla
    out_df = df[["hotel_name", "comment_text", "source_url", "source", "rating_auto", "sentiment_label"]].copy()

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")

    print(f"[OK] 3-class dataset yazıldı -> {OUT_CSV} ({len(out_df)} satır)")
    print("[INFO] 3-class dağılım:")
    print(out_df["sentiment_label"].value_counts())


if __name__ == "__main__":
    main()
