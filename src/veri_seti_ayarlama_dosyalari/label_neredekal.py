import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
RAW_DIR = BASE_DIR / "data" / "raw"
PROCESSED_DIR = BASE_DIR / "data" / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

INPUT_CSV = RAW_DIR / "neredekal_misafirler_ne_dedi.csv"
OUTPUT_CSV = PROCESSED_DIR / "neredekal_misafirler_ne_dedi_labeled.csv"


# ---- Kelime listeleri: istediğin gibi genişletebilirsin ----

STRONG_NEG = [
    "çok kötü", "kötü bir tatil", "rezalet", "berbat", "felaket",
    "temizlik kötü", "temizlik iyi değil", "odalar iyi değil",
    "atmosfer çok kötü", "olanaklar kötü", "internet yok",
]

MEDIUM_NEG = [
    "kalite normal altı", "kalitesiz", "idare eder", "sıradan",
    "orta kaliteli", "ortalama bir otel", "plaj idare eder",
    "konforsuz", "yetersiz", "memnun kalmadık", "hayal kırıklığı",
    "fena değil", "kötü sayılmaz", "kabul edilir düzeyde",
]

# “... güzel değil / iyi değil / beklentileri karşılamıyor” gibi
NEGATION_PHRASES = [
    "güzel değil", "çok güzel değil", "iyi değil",
    "beklentileri karşılamıyor", "bağlantısı zayıf",
]

STRONG_POS = [
    "olağanüstü", "muhteşem", "mükemmel", "harika",
    "harika bir yer", "muhteşem bir tatildi",
    "her şey çok güzel", "5 yıldız",
    "wellness & spor tesisleri mükemmel",
]

MEDIUM_POS = [
    "iyi", "gayet güzel", "gayet başarılı", "konforlu",
    "konforlu bir otel", "memnun kaldık", "çok memnun kaldık",
    "tavsiye ediyorum", "tavsiye ederim", "kesinlikle tavsiye ediyorum",
    "şirin bir yer", "çok şirin bir yer", "keyifli",
    "güler yüzlü", "personel çok ilgili", "personel son derece güler yüzlü",
    "çalışanlar çok iyi", "beğendik",
    "fiyatına göre çok iyi",
    "rahat bir tatil",
    "aile işletmesi", "harika bir aile işletmesi",
    "her yer tertemiz", "odalar temiz ve bakımlı",
    "odalar geniş ve ferah",
    "kaliteli hizmet", "her şey kaliteli",
    "kafa dinlemek için birebir",
]

EXTRA_POS = [
    "yemekler çok lezzetli",
    "kahvaltı muhteşem",
]


def score_comment(text: str) -> int:
    """
    1: çok olumsuz
    2: olumsuz
    3: nötr
    4: olumlu
    5: çok olumlu
    """
    if not isinstance(text, str):
        return 3

    t = text.lower().strip()

    # 0) Negatif ifade + olumluyu bozan kalıplar
    for kw in NEGATION_PHRASES:
        if kw in t:
            # çok sert bir şeyse 1, değilse 2
            return 2

    # 1) Çok olumsuz
    for kw in STRONG_NEG:
        if kw in t:
            return 1

    # 2) Orta olumsuz
    for kw in MEDIUM_NEG:
        if kw in t:
            return 2

    # 3) Çok olumlu
    for kw in STRONG_POS:
        if kw in t:
            return 5

    # 4) Orta olumlu
    for kw in MEDIUM_POS + EXTRA_POS:
        if kw in t:
            return 4

    # 5) Hiçbiri değilse nötr
    return 3



def label_to_text(score: int) -> str:
    mapping = {
        1: "çok olumsuz",
        2: "olumsuz",
        3: "nötr",
        4: "olumlu",
        5: "çok olumlu",
    }
    return mapping.get(int(score), "bilinmiyor")


def main():
    if not INPUT_CSV.exists():
        print(f"[HATA] Girdi dosyası yok: {INPUT_CSV}")
        return

    df = pd.read_csv(INPUT_CSV)

    if "comment_text" not in df.columns:
        print("[HATA] CSV içinde 'comment_text' kolonu bulunamadı.")
        print(f"Kolonlar: {list(df.columns)}")
        return

    # NaN'leri boş string yapalım ki fonksiyon patlamasın
    df["comment_text"] = df["comment_text"].fillna("")

    # Otomatik rating
    df["rating_auto"] = df["comment_text"].apply(score_comment)
    df["sentiment_label"] = df["rating_auto"].apply(label_to_text)

    # 🔥 BURADA rating kolonunu siliyoruz
    if "rating" in df.columns:
        df = df.drop(columns=["rating"])

    # CSV yazımı
    df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")
    print(f"[OK] {len(df)} satır etiketlendi -> {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
