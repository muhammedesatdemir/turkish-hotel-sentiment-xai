from pathlib import Path
import pandas as pd

# Girdiler
TRIP_TR = Path("data/processed/tripadvisor_hotel_reviews_tr_5000.csv")
NEREDEKAL = Path("data/processed/neredekal_misafirler_ne_dedi_labeled.csv")

# Çıktılar
OUT_FINAL = Path("data/processed/final_dataset_tr.csv")
OUT_STATS = Path("data/processed/final_dataset_stats.txt")
OUT_TRIP_EN = Path("data/processed/tripadvisor_comment_text_en_backup.csv")

MIN_WORDS = 1

def clean_text(s: str) -> str:
    s = str(s)
    s = s.replace("\n", " ").replace("\r", " ").strip()
    # fazla boşlukları indir
    s = " ".join(s.split())
    return s

def main():
    # TripAdvisor
    df_t = pd.read_csv(TRIP_TR)
    # backup: İngilizceyi ayrı kaydedelim
    if "comment_text_en" in df_t.columns:
        df_t[["comment_text_en"]].to_csv(OUT_TRIP_EN, index=False, encoding="utf-8-sig")

    # Şema: sadece gerekli kolonlar
    df_t = df_t[
        ["hotel_name", "comment_text", "source_url", "source", "rating_auto", "sentiment_label"]
    ].copy()

    # Neredekal
    df_n = pd.read_csv(NEREDEKAL)
    df_n = df_n[
        ["hotel_name", "comment_text", "source_url", "source", "rating_auto", "sentiment_label"]
    ].copy()

    # Temizlik
    for df in (df_t, df_n):
        df["comment_text"] = df["comment_text"].apply(clean_text)

    # boş ve çok kısa yorumları at
    def filter_df(df):
        df = df[df["comment_text"].astype(str).str.strip() != ""].copy()
        df = df[df["comment_text"].str.split().str.len() >= MIN_WORDS].copy()
        return df

    df_t = filter_df(df_t)
    df_n = filter_df(df_n)

    # Birleştir
    df = pd.concat([df_t, df_n], ignore_index=True)

    # Duplicate yorumları temizle (aynı comment_text + hotel_name + source)
    df = df.drop_duplicates(subset=["hotel_name", "comment_text", "source"]).reset_index(drop=True)

    # İstatistik raporu
    stats_lines = []
    stats_lines.append(f"TripAdvisor (temizlenmiş): {len(df_t)}")
    stats_lines.append(f"Neredekal (temizlenmiş): {len(df_n)}")
    stats_lines.append(f"Birleşik (duplicate sonrası): {len(df)}")
    stats_lines.append("\nRating dağılımı:")
    stats_lines.append(str(df["rating_auto"].value_counts().sort_index()))
    stats_lines.append("\nSentiment dağılımı:")
    stats_lines.append(str(df["sentiment_label"].value_counts()))

    OUT_FINAL.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_FINAL, index=False, encoding="utf-8-sig")

    OUT_STATS.write_text("\n".join(stats_lines), encoding="utf-8")

    print(f"[OK] Final dataset -> {OUT_FINAL} ({len(df)} satır)")
    print(f"[OK] Stats -> {OUT_STATS}")
    print(f"[OK] TripAdvisor EN backup -> {OUT_TRIP_EN}")

if __name__ == "__main__":
    main()
