import math
import time
from pathlib import Path

import pandas as pd
from deep_translator import GoogleTranslator

# Girdi: İngilizce yorumlu, 10k satırlık dosya
INPUT_CSV = Path("data/processed/tripadvisor_hotel_reviews_processed.csv")

# Çıktı: comment_text Türkçe olan dosya
OUTPUT_CSV = Path("data/processed/tripadvisor_hotel_reviews_tr.csv")

# deep_translator / Google translate için limitler
MAX_CHARS = 4800       # 5000 limitine takılmamak için güvenli sınır
SLEEP_SEC = 0.2        # her çeviri arasında küçük bekleme (sn)
PROGRESS_EVERY = 100   # her 100 satırda bir log bas


def translate_text(text: str, translator: GoogleTranslator) -> str:
    """Tek bir metni İngilizce'den Türkçe'ye çevirir; hata olursa boş döner."""
    # NaN, None vb. kontrolü
    if isinstance(text, float) and math.isnan(text):
        return ""
    t = str(text).strip()
    if not t:
        return ""

    # Çok uzun metinleri kes
    if len(t) > MAX_CHARS:
        t = t[:MAX_CHARS]

    try:
        tr = translator.translate(t)
    except Exception as e:
        # Hata durumunda programı durdurmayalım, sadece uyarı basıp boş geçelim
        print(f"[WARN] Çeviri hatası, boş geçiliyor: {e}")
        tr = ""

    return tr


def main():
    df = pd.read_csv(INPUT_CSV)
    print(f"[INFO] Girdi satır sayısı: {len(df)}")

    # Orijinal İngilizce yorumları korumak için ayrı kolona alalım
    if "comment_text_en" not in df.columns:
        df = df.rename(columns={"comment_text": "comment_text_en"})
    texts = df["comment_text_en"].astype(str).tolist()

    translator = GoogleTranslator(source="en", target="tr")

    translated = []

    for i, t in enumerate(texts):
        if i % PROGRESS_EVERY == 0:
            print(f"[INFO] Çeviri ilerleme: {i}/{len(texts)}")

        tr_txt = translate_text(t, translator)
        translated.append(tr_txt)

        # ufak bekleme
        time.sleep(SLEEP_SEC)

        # Ara kayıt (örneğin her 500 satırda bir)
        if (i + 1) % 500 == 0:
            df.loc[:i, "comment_text"] = translated
            tmp_path = OUTPUT_CSV.with_suffix(".partial.csv")
            df.to_csv(tmp_path, index=False, encoding="utf-8-sig")
            print(f"[INFO] Ara kayıt alındı -> {tmp_path}")

    # Döngü bittiğinde tüm Türkçe yorumlar hazır
    df["comment_text"] = translated

    # Kolon sırasını istediğin formata göre düzenleyelim
    base_cols = [
        "hotel_name",
        "comment_text",        # Türkçe yorum
        "source_url",
        "source",
        "rating_auto",
        "sentiment_label",
    ]

    # İngilizce orijinali de ekstra kolon olarak en sona ekleyelim (istersen sonra silersin)
    if "comment_text_en" in df.columns:
        out_cols = base_cols + ["comment_text_en"]
    else:
        out_cols = base_cols

    df = df[out_cols]

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")
    print(f"[OK] {len(df)} satır kaydedildi -> {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
