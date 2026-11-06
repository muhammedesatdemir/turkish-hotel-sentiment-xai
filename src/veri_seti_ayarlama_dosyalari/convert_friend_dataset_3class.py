# src/convert_friend_dataset_3class.py

from __future__ import annotations

from pathlib import Path
import math
import re
import pandas as pd


# ---- Paths ----
INPUT_CSV = Path("data/processed/otel_yorumlari.csv")
OUTPUT_CSV = Path("data/processed/otel_yorumlari_3class.csv")


def normalize_text(s: str) -> str:
    """Basic cleanup for user text."""
    if s is None or pd.isna(s):
        return ""
    s = str(s)
    s = s.replace("\u00a0", " ")          # non-breaking space
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    s = re.sub(r"[ \t]+", " ", s)         # collapse spaces/tabs
    s = re.sub(r"\n{2,}", "\n", s)        # collapse many newlines
    s = s.strip()
    return s


def to_int_safe(x) -> int | None:
    """Parse rating from 1..10 safely."""
    if pd.isna(x):
        return None
    s = str(x).strip()
    if s == "":
        return None
    s = s.replace(",", ".")
    try:
        f = float(s)
        return int(round(f))
    except Exception:
        return None


def map_label_3(score_10: int) -> str:
    """
    Friend mapping:
    1-5  -> negatif
    6-7  -> nötr
    8-10 -> pozitif
    """
    if score_10 <= 5:
        return "negatif"
    if score_10 <= 7:
        return "nötr"
    return "pozitif"


def score10_to_rating5(score_10: int) -> int:
    """Map 1-10 -> 1-5 using ceil(score/2)."""
    r = int(math.ceil(score_10 / 2))
    return max(1, min(5, r))


def main():
    if not INPUT_CSV.exists():
        raise FileNotFoundError(
            f"Girdi dosyası bulunamadı: {INPUT_CSV} "
            f"(Beklenen: data/processed/otel_yorumlari.csv)"
        )

    df = pd.read_csv(INPUT_CSV)

    required = {"Yorum", "Puan", "Duygu"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"Girdi dosyasında eksik sütun(lar): {sorted(missing)}. "
            f"Bulunan sütunlar: {list(df.columns)}"
        )

    before = len(df)

    # Normalize & parse
    df["comment_text"] = df["Yorum"].apply(normalize_text)
    df["rating_10"] = df["Puan"].apply(to_int_safe)

    # Drop invalid rows
    df = df[df["comment_text"] != ""].copy()
    df = df[df["rating_10"].notna()].copy()
    df["rating_10"] = df["rating_10"].astype(int).clip(1, 10)

    after_clean = len(df)

    # 3-class label + 1-5 rating
    df["sentiment_label"] = df["rating_10"].apply(map_label_3)
    df["rating_auto"] = df["rating_10"].apply(score10_to_rating5)

    # Static metadata to match YOUR master format
    df["hotel_name"] = "friend_hotel"
    df["source"] = "friend_dataset"

    # IMPORTANT: avoid empty source_url so CSV doesn't show ",,"
    # Use "-" placeholder
    df["source_url"] = "-"

    # Final columns EXACTLY as you want
    out = df[
        ["hotel_name", "comment_text", "source_url", "source", "rating_auto", "sentiment_label"]
    ].copy()

    # Deduplicate
    out_before = len(out)
    out = out.drop_duplicates(subset=["comment_text"]).copy()
    out_after = len(out)

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")

    print(f"[OK] Girdi satır sayısı: {before}")
    print(f"[OK] Temiz sonrası satır: {after_clean}")
    print(f"[OK] Duplicate sonrası: {out_after} (kaldırılan: {out_before - out_after})")
    print(f"[OK] Yazıldı -> {OUTPUT_CSV}")

    print("\n[INFO] 3-class dağılımı:")
    print(out["sentiment_label"].value_counts())


if __name__ == "__main__":
    main()
