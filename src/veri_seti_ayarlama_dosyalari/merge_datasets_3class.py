# src/merge_datasets_3class.py

from __future__ import annotations

from pathlib import Path
import re
import pandas as pd


# ---- Inputs / Output ----
MY_CSV = Path("data/processed/final_dataset_tr_3class.csv")
FRIEND_CSV = Path("data/processed/otel_yorumlari_3class.csv")
OUT_CSV = Path("data/processed/final_dataset_merged_3class.csv")

REQUIRED_COLS = [
    "hotel_name",
    "comment_text",
    "source_url",
    "source",
    "rating_auto",
    "sentiment_label",
]

VALID_LABELS = {"negatif", "nötr", "pozitif"}


def normalize_text(s: str) -> str:
    """Normalize comment text for better deduping and cleanliness."""
    if s is None or pd.isna(s):
        return ""
    s = str(s)
    s = s.replace("\u00a0", " ")  # non-breaking space
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    s = re.sub(r"[ \t]+", " ", s)     # collapse spaces/tabs
    s = re.sub(r"\n{2,}", "\n", s)    # collapse multiple newlines
    return s.strip()


def load_and_validate(path: Path, name: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"{name} dosyası bulunamadı: {path}")

    df = pd.read_csv(path)

    missing = set(REQUIRED_COLS) - set(df.columns)
    if missing:
        raise ValueError(f"{name} dosyasında eksik kolon(lar): {sorted(missing)}")

    # Keep only required columns in correct order
    df = df[REQUIRED_COLS].copy()

    # Clean fields
    df["hotel_name"] = df["hotel_name"].astype(str).fillna("").str.strip()
    df["source"] = df["source"].astype(str).fillna("").str.strip()

    df["source_url"] = df["source_url"].astype(str).fillna("").str.strip()
    df.loc[df["source_url"] == "", "source_url"] = "-"

    df["comment_text"] = df["comment_text"].apply(normalize_text)
    df = df[df["comment_text"] != ""].copy()

    # Normalize labels
    df["sentiment_label"] = df["sentiment_label"].astype(str).fillna("").str.strip().str.lower()
    df = df[df["sentiment_label"].isin(VALID_LABELS)].copy()

    # rating_auto to int
    df["rating_auto"] = pd.to_numeric(df["rating_auto"], errors="coerce")
    df = df[df["rating_auto"].notna()].copy()
    df["rating_auto"] = df["rating_auto"].astype(int)

    # keep rating_auto within {1,3,5} ideally, but don't crash if not
    df.loc[~df["rating_auto"].isin([1, 3, 5]), "rating_auto"] = df["sentiment_label"].map(
        {"negatif": 1, "nötr": 3, "pozitif": 5}
    )

    print(f"[OK] {name}: {len(df)} satır (temizlenmiş)")
    print(df["sentiment_label"].value_counts())
    print()
    return df


def main():
    my_df = load_and_validate(MY_CSV, "MY")
    friend_df = load_and_validate(FRIEND_CSV, "FRIEND")

    combined = pd.concat([my_df, friend_df], ignore_index=True)

    before = len(combined)

    # Dedup by normalized comment_text
    # (Optionally also include source to keep same text from different sources,
    #  but you said comment_text alone is enough.)
    combined = combined.drop_duplicates(subset=["comment_text"]).copy()

    after = len(combined)

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")

    print(f"[OK] Birleştirildi: {before} -> {after} (duplicate silinen: {before - after})")
    print(f"[OK] Çıktı -> {OUT_CSV}")

    print("\n[INFO] Final dağılım:")
    print(combined["sentiment_label"].value_counts())


if __name__ == "__main__":
    main()
