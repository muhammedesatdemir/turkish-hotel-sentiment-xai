import time
import re
import requests
from bs4 import BeautifulSoup
import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
RAW_DIR = BASE_DIR / "data" / "raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/129.0 Safari/537.36"
    )
}


def fetch_html(url: str) -> str | None:
    """Verilen URL'den HTML döndürür."""
    try:
        resp = requests.get(url, headers=HEADERS, timeout=15)
        # Türkçe karakterler bozulmasın
        resp.encoding = resp.apparent_encoding
        resp.raise_for_status()
        return resp.text
    except Exception as e:
        print(f"[HATA] {url} alınamadı -> {e}")
        return None


def extract_misafirler_ne_dedi(html: str) -> dict:
    soup = BeautifulSoup(html, "html.parser")

    # --- Otel adı ---
    h1 = soup.find("h1")
    hotel_name = h1.get_text(strip=True) if h1 else None

    # --- Rating: sayfada geçen ilk "x.x/5" ifadesi ---
    rating = None
    for text in soup.stripped_strings:
        m = re.search(r"\b\d(?:\.\d)?/5\b", text)
        if m:
            rating = m.group(0)
            break

    # --- Yorum sayısı: "sayı + yorum" içeren metinleri ara ---
    review_count = None
    yorum_candidates = [
        t.strip()
        for t in soup.stripped_strings
        if re.search(r"\d+\s*y?orum", t.lower())  # örn: "3 yorum", "12 yorum"
    ]
    if yorum_candidates:
        # ilk eşleşeni al
        review_count = yorum_candidates[0]
    else:
        # fallback: bazı sayfalarda sadece "Yorum / Değerlendirme" yazıyor
        fallback = [
            t.strip()
            for t in soup.stripped_strings
            if "yorum / değerlendirme" in t.lower()
        ]
        if fallback:
            review_count = fallback[0]

    # --- Misafirler Ne Dedi? kutuları ---
    phrases: list[str] = []

    container = soup.find("div", class_="css-qge2j7")
    if container:
        for div in container.find_all("div"):
            txt = div.get_text(" ", strip=True)
            if not txt:
                continue

            lower = txt.lower()

            if "misafirler ne dedi" in lower:
                continue
            if "değerlendirmeleri gör" in lower:
                continue
            if "yorum puanı" in lower:
                continue

            if len(txt) < 3 or len(txt) > 80:
                continue

            cleaned = txt.strip(" '\"")
            if cleaned:
                phrases.append(cleaned)

    phrases = list(dict.fromkeys(phrases))

    return {
        "hotel_name": hotel_name,
        "rating": rating,
        "phrases": phrases,
    }


def main():
    # Birden fazla otel URL'si ile çalışalım
    hotel_urls = [
        "https://www.neredekal.com/anemon-galata-hotel-fiyatlari/",
        "https://www.neredekal.com/crystal-hotels-admiral-resort-fiyatlari/",
        "https://www.neredekal.com/susesi-de-luxe-resort-spa-golf-hotel-fiyatlari/",
        "https://www.neredekal.com/rixos-premium-bodrum-fiyatlari/",
        "https://www.neredekal.com/palm-wings-ephesus-beach-resort-fiyatlari/",
        "https://www.neredekal.com/la-blanche-island-bodrum-fiyatlari/",
        "https://www.neredekal.com/orange-country-resort-hotel-alanya-fiyatlari/",
        "https://www.neredekal.com/aria-claros-beach-spa-resort-fiyatlari/",
        "https://www.neredekal.com/aydinbey-kings-palace-fiyatlari/",
        "https://www.neredekal.com/boyalik-beach-hotel-spa-fiyatlari/", 
        "https://www.neredekal.com/d-resort-gocek-fiyatlari/",
        "https://www.neredekal.com/karbel-hotel-fiyatlari/",
        "https://www.neredekal.com/orange-county-resort-hotel-fiyatlari/",
        "https://www.neredekal.com/leodikya-resort-fiyatlari/",
        "https://www.neredekal.com/ng-afyon-fiyatlari/",
        "https://www.neredekal.com/agaoglu-my-mountain-fiyatlari/",
        "https://www.neredekal.com/kaya-artemis-resort-casino-fiyatlari/",
        "https://www.neredekal.com/sukrubey-konagi-fiyatlari/",
        "https://www.neredekal.com/suw-hotel-fiyatlari/",
        "https://www.neredekal.com/crystal-tat-beach-golf-resort-spa-hotel-fiyatlari/",
        "https://www.neredekal.com/sueno-hotels-golf-belek-fiyatlari/",
        "https://www.neredekal.com/paddy-s-aparts-fiyatlari/",
        "https://www.neredekal.com/crystal-waterworld-resort-spa-fiyatlari/",
        "https://www.neredekal.com/cappadocia-estates-fiyatlari/",
        "https://www.neredekal.com/minia-cave-hotel-ortahisar-fiyatlari/",
        "https://www.neredekal.com/castle-cave-hotel-fiyatlari/",
        "https://www.neredekal.com/agarta-cave-hotel-fiyatlari/",
        "https://www.neredekal.com/valley-inn-cave-hotel-fiyatlari/",
        "https://www.neredekal.com/terracota-hotel-fiyatlari/",
        "https://www.neredekal.com/avanos-suites-fiyatlari/",
        "https://www.neredekal.com/roca-cappadocia-fiyatlari/",
        "https://www.neredekal.com/mira-cappadocia-hotel-fiyatlari/",
        "https://www.neredekal.com/taskonaklar-fiyatlari/",
        "https://www.neredekal.com/sakli-konak-cappadocia-fiyatlari/",
        "https://www.neredekal.com/turkish-cave-house-fiyatlari/",
        "https://www.neredekal.com/nar-cave-house-fiyatlari/",
        "https://www.neredekal.com/crystal-kaymakli-hotel-spa-fiyatlari/",
        "https://www.neredekal.com/the-arkin-colony-hotel-fiyatlari/",
        "https://www.neredekal.com/rocks-hotel-fiyatlari/",
        "https://www.neredekal.com/kaya-artemis-resort-casino-fiyatlari/",
        "https://www.neredekal.com/concorde-luxury-resort-casino-convention-spa-fiyatlari/",
        "https://www.neredekal.com/bare-hill-holiday-village-fiyatlari/",
        "https://www.neredekal.com/merit-lefkosa-otel-fiyatlari/",
        "https://www.neredekal.com/arkin-palm-beach-hotel-fiyatlari/",
        "https://www.neredekal.com/dome-otel-kibris-fiyatlari/",
        "https://www.neredekal.com/oscar-resort-hotel-fiyatlari/",
        "https://www.neredekal.com/park-palace-hotel-kyrenia-fiyatlari/",
        "https://www.neredekal.com/the-olive-tree-fiyatlari/",
        "https://www.neredekal.com/vuni-palace-fiyatlari/",
        "https://www.neredekal.com/dedeman-palandoken-ski-lodge-fiyatlari/",
        "https://www.neredekal.com/aslibey-konagi-fiyatlari/",
        "https://www.neredekal.com/sapanca-villa-suite-butik-otel-fiyatlari/",
        "https://www.neredekal.com/villa-kirkpinar-fiyatlari/",
        "https://www.neredekal.com/ng-sapanca-wellness-convention-fiyatlari/",
        "https://www.neredekal.com/wind-hotel-fiyatlari/",
        "https://www.neredekal.com/egemen-butik-otel-fiyatlari/",
        "https://www.neredekal.com/richmond-nua-wellness-spa-sapanca-fiyatlari/",
        "https://www.neredekal.com/yanki-otel-fiyatlari/",
        "https://www.neredekal.com/room-room-motel-fiyatlari/",
        "https://www.neredekal.com/elite-world-sapanca-fiyatlari/",
        "https://www.neredekal.com/habitat-mesire-yedigoller-bungalov-evler-fiyatlari/",
        "https://www.neredekal.com/mekan-termal-apart-fiyatlari/",
        "https://www.neredekal.com/hampton-by-hilton-bolu-fiyatlari/",
        "https://www.neredekal.com/doga-kosku-abant-pansiyon-fiyatlari/",
        "https://www.neredekal.com/bolu-hotel-fiyatlari/",
        "https://www.neredekal.com/bolu-termal-otel-fiyatlari/",
        "https://www.neredekal.com/von-club-abant-fiyatlari/",
        "https://www.neredekal.com/koru-hotel-fiyatlari/",
        "https://www.neredekal.com/abant-lotus-otel-fiyatlari/",
        "https://www.neredekal.com/buyuk-abant-otel-fiyatlari/",
        "https://www.neredekal.com/kadifeli-konak-boutique-hotel-fiyatlari/",
        "https://www.neredekal.com/alex-village-life-fiyatlari/",
        "https://www.neredekal.com/kartepe-kosku-fiyatlari/",
        "https://www.neredekal.com/rivada-hotel-fiyatlari/",
        "https://www.neredekal.com/el-garden-hotel-residence-fiyatlari/",
        "https://www.neredekal.com/latanya-city-otel-antalya-fiyatlari/",
        "https://www.neredekal.com/hotel-atici-fiyatlari/",
        "https://www.neredekal.com/mera-park-hotel-fiyatlari/",
        "https://www.neredekal.com/cender-hotel-fiyatlari/",
        "https://www.neredekal.com/the-corner-park-hotel-fiyatlari/",
        "https://www.neredekal.com/villa-sonata-fiyatlari/",
        "https://www.neredekal.com/berrak-su-hotel-fiyatlari/",
        "https://www.neredekal.com/en-vie-beach-boutique-hotel-fiyatlari/",
        "https://www.neredekal.com/cemal-bey-apart-otel-fiyatlari/",
        "https://www.neredekal.com/arsi-enfi-city-beach-hotel-fiyatlari/",
        "https://www.neredekal.com/premium-park-apart-otel-fiyatlari/",
        "https://www.neredekal.com/best-western-khan-otel-fiyatlari/",
        "https://www.neredekal.com/adrasan-ziya-apart-evleri-fiyatlari/",
        "https://www.neredekal.com/el-vino-hotel-fiyatlari/",
        "https://www.neredekal.com/spektr-hotel-fiyatlari/",
        "https://www.neredekal.com/can-naz-apart-restaurant-fiyatlari/",
        "https://www.neredekal.com/elista-hotel-spa-fiyatlari/",
        "https://www.neredekal.com/yali-apart-residence-fiyatlari/",
        "https://www.neredekal.com/sea-palm-otel-yalikavak-fiyatlari/",
        "https://www.neredekal.com/liman-motel-cafe-fiyatlari/",
        "https://www.neredekal.com/ece-hotel-bodrum-fiyatlari/",
        "https://www.neredekal.com/casa-del-mar-bodrum-fiyatlari/",
        "https://www.neredekal.com/hola-bodrum-fiyatlari/",
        "https://www.neredekal.com/club-shark-otel-fiyatlari/",
        "https://www.neredekal.com/can-pansiyon-fiyatlari/",
        "https://www.neredekal.com/moonshine-apart-otel-fiyatlari/",
        "https://www.neredekal.com/merih-hotel-fiyatlari/",
        "https://www.neredekal.com/hakan-otel-fiyatlari/",
        "https://www.neredekal.com/turabdin-hotel-fiyatlari/",
        "https://www.neredekal.com/kaya-ninova-otel-fiyatlari/",
        "https://www.neredekal.com/mardius-tarihi-konak-fiyatlari/",
        "https://www.neredekal.com/arif-bey-konagi-fiyatlari/",
        "https://www.neredekal.com/zeynep-hanim-konaklari-fiyatlari/",
        "https://www.neredekal.com/divan-gaziantep-fiyatlari/",
        "https://www.neredekal.com/ugurlu-fiyatlari/",
        "https://www.neredekal.com/buyuk-velic-hotel-gaziantep-fiyatlari/",
        "https://www.neredekal.com/kule-hotel-spa-fiyatlari/",
        "https://www.neredekal.com/garni-hotel-gaziantep-fiyatlari/",
        "https://www.neredekal.com/ugurlu-thermal-resort-spa-fiyatlari/",
        "https://www.neredekal.com/yilmazoglu-park-otel-fiyatlari/",
        # buraya istediğin kadar yeni URL ekleyebilirsin
    ]

    records = []

    for url in hotel_urls:
        print(f"[INFO] Çekiliyor: {url}")
        html = fetch_html(url)
        if not html:
            continue

        data = extract_misafirler_ne_dedi(html)

        # phrases listesi -> ayrı satırlar
        if data["phrases"]:
            for phrase in data["phrases"]:
                records.append(
                    {
                        "hotel_name": data["hotel_name"],
                        "rating": data["rating"],
                        "comment_text": phrase,
                        "source_url": url,
                    }
                )
        else:
            # hiç phrase bulamazsak yine de bir satır kaydedelim
            records.append(
                {
                    "hotel_name": data["hotel_name"],
                    "rating": data["rating"],
                    "comment_text": None,
                    "source_url": url,
                }
            )

        # Her otel arasında biraz bekleyelim (siteyi yormamak için)
        time.sleep(1)

    if not records:
        print("[WARN] Hiç veri toplanamadı.")
        return

    df = pd.DataFrame(records)

    # 1) comment_text'i temizle (boşlukları at)
    df["comment_text"] = df["comment_text"].astype(str).str.strip()

    # 2) comment_text'i boş olan satırları at
    df = df[df["comment_text"].notna() & (df["comment_text"] != "")]

    # 3) Aynı otel + aynı yorum tekrarlarını sil
    df = df.drop_duplicates(subset=["hotel_name", "comment_text"])

    # 4) İstersen kaynak kolonu ekle
    df["source"] = "neredekal"

    out_path = RAW_DIR / "neredekal_misafirler_ne_dedi.csv"
    df.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"[OK] {len(df)} satır kaydedildi -> {out_path}")




if __name__ == "__main__":
    main()
