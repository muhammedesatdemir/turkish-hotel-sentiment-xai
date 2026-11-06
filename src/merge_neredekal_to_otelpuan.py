# -*- coding: utf-8 -*-
"""
neredekal -> otelpuan birlestirme (hoca geri bildirimi A.1).

Hoca geri bildirimi:
    "neredekal 40 satir gereksiz; otelpuan ile birlestir, source kolonu
     otelyorum_otelpuan_birlesik gibi olsun."

Karar: source kolonunda mapping yap, satirlarda hicbir degisiklik yok.
    - 'otelpuan.com'  -> 'otelpuan_otelyorum'
    - 'neredekal'     -> 'otelpuan_otelyorum'
    - 'ai_generated'  : aynen kalir
    - 'manuel_toplama': aynen kalir

Girdi:
    data/processed/final_genisletilmis_dataset_optimized_v3.csv (14.337 satir)

Cikti:
    data/processed/final_genisletilmis_dataset_optimized_v3_v2.csv (14.337 satir)
    data/processed/merge_neredekal_report.json

Onemli not: source kolonu egitime DOGRUDAN girmez; train_optimized_v3.py
icindeki source_stratified_split sadece 'is_ai = (source == ai_generated)'
ikili etiketini kullanir. neredekal 'human' kategorisindeydi, otelpuan da
'human' kategorisinde - dolayisiyla mapping sonrasi stratifikasyon sonucu
DEGISMEZ. Yani daha onceden egitilmis modeller tekrar egitilmek zorunda
DEGILDIR.

Kosma:
    python src/merge_neredekal_to_otelpuan.py
"""
import os, json
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "data", "processed", "final_genisletilmis_dataset_optimized_v3.csv")
DST = os.path.join(ROOT, "data", "processed", "final_genisletilmis_dataset_optimized_v3_v2.csv")
REPORT = os.path.join(ROOT, "data", "processed", "merge_neredekal_report.json")

MERGE_MAP = {
    "otelpuan.com": "otelpuan_otelyorum",
    "neredekal":    "otelpuan_otelyorum",
}

# Beklenen son source dagilimi (rapor §3.10 Cizelge 1'den).
EXPECTED_AFTER = {
    "otelpuan_otelyorum": 11086,  # 11046 + 40
    "ai_generated":        2500,
    "manuel_toplama":       751,
}
EXPECTED_TOTAL = 14337

# ---- 1) Oku ----
df = pd.read_csv(SRC)
n0 = len(df)
src_before = df["source"].value_counts(dropna=False).to_dict()

# ---- 2) Mapping uygula ----
df["source"] = df["source"].astype(str).str.strip().replace(MERGE_MAP)
src_after = df["source"].value_counts(dropna=False).to_dict()

# ---- 3) Dogrulama ----
errors = []
if len(df) != n0:
    errors.append(f"satir sayisi degisti: {n0} -> {len(df)}")
if (df["source"] == "neredekal").any():
    errors.append("neredekal hala var (mapping basarisiz)")
if (df["source"] == "otelpuan.com").any():
    errors.append("otelpuan.com hala var (mapping basarisiz)")
for src, exp in EXPECTED_AFTER.items():
    got = int(src_after.get(src, 0))
    if got != exp:
        errors.append(f"{src}: beklenen={exp} gercek={got}")
if len(df) != EXPECTED_TOTAL:
    errors.append(f"toplam satir: beklenen={EXPECTED_TOTAL} gercek={len(df)}")

# ---- 4) Yaz ----
if not errors:
    df.to_csv(DST, index=False, encoding="utf-8")

# ---- 5) Rapor ----
report = {
    "input_csv":  SRC,
    "output_csv": DST if not errors else None,
    "rows_in":    n0,
    "rows_out":   len(df),
    "source_dist_before": src_before,
    "source_dist_after":  src_after,
    "merge_map": MERGE_MAP,
    "expected_after": EXPECTED_AFTER,
    "errors": errors,
}
with open(REPORT, "w", encoding="utf-8") as f:
    json.dump(report, f, ensure_ascii=False, indent=2)

# ---- 6) Konsol ozeti ----
print("=" * 70)
print("NEREDEKAL -> OTELPUAN_OTELYORUM BIRLESTIRME")
print("=" * 70)
print(f"Girdi CSV : {SRC}")
print(f"Satir     : {n0}")
print("-" * 70)
print("Source dist ONCE:")
for k, v in sorted(src_before.items(), key=lambda x: -x[1]):
    print(f"  {k:25s} : {v}")
print("Source dist SONRA:")
for k, v in sorted(src_after.items(), key=lambda x: -x[1]):
    print(f"  {k:25s} : {v}")
print("-" * 70)
if errors:
    print("HATALAR:")
    for e in errors:
        print(f"  - {e}")
    print("CIKTI YAZILMADI.")
else:
    print(f"Cikti CSV : {DST}")
    print("Tum dogrulamalar gecti.")
print(f"Rapor JSON: {REPORT}")
print("=" * 70)
