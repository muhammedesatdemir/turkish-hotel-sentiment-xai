import time
import random
from pathlib import Path

import requests
from bs4 import BeautifulSoup
import pandas as pd

# --- HTTP ayarları ---

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "tr-TR,tr;q=0.9,en-US;q=0.8,en;q=0.7",
    "Connection": "keep-alive",
    "Referer": "https://www.tripadvisor.com.tr/",
}

# Tek bir session kullanmak (cookie vs. korumak) daha doğal görünmeyi sağlar
SESSION = requests.Session()
SESSION.headers.update(HEADERS)

OUTPUT_CSV = Path("data/tripadvisor_raw.csv")


def get_page(url: str) -> str | None:
    """Verilen URL'den HTML içeriğini çeker. Başarısız olursa None döner."""
    try:
        resp = SESSION.get(url, timeout=20)
        print(f"[DEBUG] GET {url} -> {resp.status_code}")
    except Exception as e:
        print(f"[HATA] {url} istek atılamadı: {e}")
        return None

    if resp.status_code != 200:
        print(f"[HATA] {url} status={resp.status_code}")
        return None

    return resp.text


def parse_reviews_from_html(html: str, url: str):
    """Tek bir TripAdvisor otel yorum sayfasındaki tüm yorumları çözer."""
    soup = BeautifulSoup(html, "html.parser")

    # Otel adı (sayfanın tepesindeki h1 genelde bu)
    hotel_h1 = soup.select_one("h1")
    hotel_name = hotel_h1.get_text(strip=True) if hotel_h1 else ""

    rows = []

    # Her yorum kartı
    cards = soup.find_all("div", attrs={"data-test-target": "HR_CC_CARD"})
    for card in cards:
        # --- Yorum metni ---
        # Genelde <q> ya da içinde QewHA geçen bir span/div'te oluyor
        text_tag = card.find("q")
        if text_tag is None:
            text_tag = card.find(
                ["span", "div"],
                attrs={"class": lambda c: c and "QewHA" in c},
            )

        comment_text = ""
        if text_tag:
            comment_text = text_tag.get_text(" ", strip=True)

        # --- Puan (1–5) ---
        rating = None

        # Eski tasarım: bubble_50, bubble_40 ...
        bubble = card.find("span", class_=lambda c: c and "ui_bubble_rating" in c)
        if bubble and bubble.get("class"):
            for cls in bubble["class"]:
                if cls.startswith("bubble_"):
                    # bubble_50 -> 5, bubble_40 -> 4 ...
                    rating = int(cls.split("_")[-1]) // 10

        # Yeni tasarımda text içinde "5 baloncuk üzerinden 5" gibi bir şey var
        if rating is None:
            import re

            rating_text = card.get_text(" ", strip=True)
            m = re.search(r"baloncuk üzerinden\s*([1-5])", rating_text)
            if m:
                rating = int(m.group(1))

        if not comment_text:
            # metin yoksa bu kartı atla
            continue

        rows.append(
            {
                "hotel_name": hotel_name,
                "comment_text": comment_text,
                "source_url": url,
                "source": "tripadvisor",
                "rating": rating,
            }
        )

    return rows


def build_page_url(base_url: str, page_idx: int) -> str:
    """
    page_idx = 0 -> or yok (ilk sayfa)
    page_idx = 1 -> -or5-
    page_idx = 2 -> -or10-
    ...
    """
    if page_idx == 0:
        return base_url

    offset = page_idx * 5  # TripAdvisor genelde sayfa başına 5 yorum
    # ...Reviews- ile ...-The_Story_Hotel_Pera- arasına -orX- sıkıştırıyoruz
    if "Reviews-" not in base_url:
        return base_url  # güvenlik

    left, right = base_url.split("Reviews-", 1)
    return f"{left}Reviews-or{offset}-{right}"


def scrape_hotel(base_url: str, max_pages: int = 50):
    """Tek bir otel için tüm sayfalardaki yorumları çeker."""
    all_rows = []

    for page_idx in range(max_pages):
        page_url = build_page_url(base_url, page_idx)
        print(f"[INFO] Sayfa {page_idx} -> {page_url}")

        html = get_page(page_url)
        if html is None:
            # 403, 500 vs. durumlarda kırıyoruz
            break

        rows = parse_reviews_from_html(html, page_url)
        print(f"  {len(rows)} yorum bulundu")

        if not rows:
            # artık yorum yok => kır
            break

        all_rows.extend(rows)

        # Ban riskini azaltmak için istekler arasında bekle
        time.sleep(random.uniform(3, 7))

    return all_rows


def main():
    # ŞİMDİLİK: Tek bir otel ile başla (ekleyip çoğaltacağız)
    base_url = (
        "https://www.tripadvisor.com.tr/Hotel_Review-"
        "g293974-d2049041-Reviews-The_Story_Hotel_Pera-Istanbul.html"
    )

    rows = scrape_hotel(base_url)

    if not rows:
        print("[HATA] Hiç yorum çekilemedi.")
        return

    df = pd.DataFrame(rows)
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")
    print(f"[OK] {len(df)} yorum kaydedildi -> {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
