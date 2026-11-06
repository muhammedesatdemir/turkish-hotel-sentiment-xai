# -*- coding: utf-8 -*-
"""
yeni.txt -> Final_Dataset_Optimized_v3.csv genisletme.

Adimlar:
1) yeni.txt'i bos satirla ayrilmis yorum bloklarina parse et.
2) Cok satirli yorumlari (cok kisa ardisik satirlar) tek paragrafa birlestir.
3) Filtreler: wc<8 (birlestirme sonrasi), birebir duplicate (kendi icinde + v3 ile).
4) Supheli pozitif/negatif tespiti (anahtar kelime) -> suspicious_reviews.csv.
5) v3 format: hotel_name=friend_hotel, comment_text, source_url=-, source=manuel_toplama,
   rating_auto=3, sentiment_label=notr.
6) final_genisletilmis_dataset_optimized_v3.csv yaz.
7) Rapor.
"""
import os, re, json
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
YENI = os.path.join(ROOT, "data", "processed", "yeni.txt")
V3   = os.path.join(ROOT, "data", "processed", "Final_Dataset_Optimized_v3.csv")
OUT  = os.path.join(ROOT, "data", "processed", "final_genisletilmis_dataset_optimized_v3.csv")
SUSP = os.path.join(ROOT, "data", "processed", "suspicious_reviews.csv")
REPORT = os.path.join(ROOT, "data", "processed", "merge_yeni_report.json")

# ---- 1) Parse ----
with open(YENI, "r", encoding="utf-8") as f:
    raw = f.read()

# Bos satirla ayrilmis bloklar; coklu bos satir da ayirici sayilir.
blocks = re.split(r"\n\s*\n+", raw)
blocks = [b.strip() for b in blocks if b.strip()]

# Cok satirli bloklarda satirlar arasinda "\n" olabilir; bunlari bosluga cevir.
def normalize_block(b: str) -> str:
    # Satir sonlari -> bosluk; coklu bosluklari sadelestir
    s = re.sub(r"\s*\n\s*", " ", b)
    s = re.sub(r"\s+", " ", s).strip()
    return s

reviews = [normalize_block(b) for b in blocks]

# ---- 2) Akilli kisa-yorum birlestirme ----
# Kural: bir blok cok kisa (wc<8) ise SONRAKI blokla birlestirmeye calisma yerine
# blogun KENDI ICINDE zaten cok satir varsa (yukarida birlestirildi). Yani cok satirli
# yorumlar (10-13 ornegi gibi) zaten tek blok haline geldi. Ekstra bir yapay birlestirme
# yapmiyoruz; bu hatali concat riski yaratir.

def wc(s: str) -> int:
    return len(s.split())

# ---- 3) Filtreler ----
total_blocks = len(reviews)

# 3a) wc<8 ele
short_dropped = [r for r in reviews if wc(r) < 8]
reviews_long = [r for r in reviews if wc(r) >= 8]

# 3b) kendi icinde birebir duplicate ele (whitespace + lowercase normalize)
def norm_key(s: str) -> str:
    return re.sub(r"\s+", " ", s.lower().strip())

seen = {}
dedup_inner = []
inner_dups = 0
for r in reviews_long:
    k = norm_key(r)
    if k in seen:
        inner_dups += 1
        continue
    seen[k] = True
    dedup_inner.append(r)

# 3c) v3 ile cakisma
v3 = pd.read_csv(V3)
v3_keys = set(v3["comment_text"].astype(str).map(norm_key))
v3_overlap = 0
final = []
for r in dedup_inner:
    if norm_key(r) in v3_keys:
        v3_overlap += 1
        continue
    final.append(r)

# ---- 4) Supheli pozitif/negatif tespiti ----
# Strateji: COK ACIK pozitif/negatif sinyali yakalamaya calis. Esik dusuk olsun ki
# kullanici kalan supheli listeyi gozden gecirip karar verebilsin.
POZ_KEYS = [
    "harika", "muhteşem", "muhtesem", "mükemmel", "mukemmel",
    "kesinlikle tavsiye", "çok beğendik", "cok begendik", "bayıldık", "bayildik",
    "süper bir", "super bir", "çok memnun kaldık", "cok memnun kaldik",
    "tavsiye ederim", "tekrar geleceğiz", "tekrar gelecegiz",
]
NEG_KEYS = [
    "berbat", "rezalet", "iğrenç", "igrenc", "kesinlikle gitmeyin",
    "asla tavsiye etmem", "asla gitmeyin", "para tuzağı", "para tuzagi",
    "çok kötü bir otel", "cok kotu bir otel", "kandırıldık", "kandirildik",
    "tam bir hayal kırıklığı", "tam bir hayal kirikligi",
    "bir daha gelmem", "bir daha asla",
]

def detect_suspicion(text: str):
    low = text.lower()
    # turkce karakter normalize basit
    norm = (low.replace("ı","i").replace("İ","i").replace("ş","s").replace("ç","c")
                .replace("ğ","g").replace("ö","o").replace("ü","u"))
    pos_hits = [k for k in POZ_KEYS if k in low or k in norm]
    neg_hits = [k for k in NEG_KEYS if k in low or k in norm]
    if pos_hits and not neg_hits:
        return "supheli_pozitif", pos_hits
    if neg_hits and not pos_hits:
        return "supheli_negatif", neg_hits
    if pos_hits and neg_hits:
        # ikisi de varsa muhtemelen gercek notr (mixed) - supheli degil
        return "mixed_ok", pos_hits + neg_hits
    return None, []

suspicious_rows = []
clean_rows = []
mixed_count = 0
for r in final:
    flag, hits = detect_suspicion(r)
    if flag in ("supheli_pozitif", "supheli_negatif"):
        suspicious_rows.append({
            "suphe_tipi": flag,
            "anahtar_kelimeler": ", ".join(hits),
            "wc": wc(r),
            "comment_text": r,
        })
    else:
        if flag == "mixed_ok":
            mixed_count += 1
        clean_rows.append(r)

# ---- 5) v3 format satirlari ----
new_rows = []
for r in clean_rows:
    new_rows.append({
        "hotel_name":     "friend_hotel",
        "comment_text":   r,
        "source_url":     "-",
        "source":         "manuel_toplama",
        "rating_auto":    3,
        "sentiment_label":"nötr",
    })
new_df = pd.DataFrame(new_rows, columns=v3.columns)

# ---- 6) Yaz ----
merged = pd.concat([v3, new_df], ignore_index=True)
merged.to_csv(OUT, index=False, encoding="utf-8")

if suspicious_rows:
    pd.DataFrame(suspicious_rows).to_csv(SUSP, index=False, encoding="utf-8")

# ---- 7) Rapor ----
wc_stats = pd.Series([wc(r) for r in clean_rows]).describe().to_dict() if clean_rows else {}

report = {
    "input_blocks_total": total_blocks,
    "dropped_short_wc_lt_8": len(short_dropped),
    "dropped_inner_duplicates": inner_dups,
    "dropped_v3_overlap": v3_overlap,
    "suspicious_separated": len(suspicious_rows),
    "mixed_kept_as_notr": mixed_count,
    "added_to_dataset": len(clean_rows),
    "v3_size_before": len(v3),
    "v3_size_after":  len(merged),
    "wc_added_stats": wc_stats,
    "label_dist_before": v3["sentiment_label"].value_counts().to_dict(),
    "label_dist_after":  merged["sentiment_label"].value_counts().to_dict(),
    "source_dist_after": merged["source"].value_counts().to_dict(),
    "out_path": OUT,
    "suspicious_path": SUSP if suspicious_rows else None,
}
with open(REPORT, "w", encoding="utf-8") as f:
    json.dump(report, f, ensure_ascii=False, indent=2)

# Konsola ozet
print("=" * 70)
print("YENI.TXT -> v3 GENISLETME RAPORU")
print("=" * 70)
print(f"Girdi blok sayisi              : {total_blocks}")
print(f"Elenen (wc<8)                  : {len(short_dropped)}")
print(f"Elenen (kendi icinde duplicate): {inner_dups}")
print(f"Elenen (v3 ile cakisma)        : {v3_overlap}")
print(f"Supheli (ayri dosyaya)         : {len(suspicious_rows)}")
print(f"Mixed (notr olarak kaldi)      : {mixed_count}")
print(f"EKLENEN                        : {len(clean_rows)}")
print("-" * 70)
print(f"v3 onceki boyut : {len(v3)}")
print(f"v3 yeni boyut   : {len(merged)}")
print(f"v3 onceki dist  : {v3['sentiment_label'].value_counts().to_dict()}")
print(f"v3 yeni dist    : {merged['sentiment_label'].value_counts().to_dict()}")
print(f"Source dist     : {merged['source'].value_counts().to_dict()}")
if wc_stats:
    print(f"Eklenen wc      : median={wc_stats.get('50%',0):.0f} mean={wc_stats.get('mean',0):.1f} "
          f"min={wc_stats.get('min',0):.0f} max={wc_stats.get('max',0):.0f}")
print("-" * 70)
print(f"Cikti CSV       : {OUT}")
if suspicious_rows:
    print(f"Supheli CSV     : {SUSP}")
print(f"Rapor JSON      : {REPORT}")
print("=" * 70)
