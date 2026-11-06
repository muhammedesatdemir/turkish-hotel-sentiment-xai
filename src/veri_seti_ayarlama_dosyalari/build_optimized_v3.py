"""
build_optimized_v3.py
=====================

Final_Dataset_PlusAI.csv -> Final_Dataset_Optimized_v3.csv

Amac:
  - Bozuk satirlari karantinaya al (label kolonuna metin kaymis vb.)
  - source kolonunu normalize et (otelpuan.com ailesi kucuk varyantlar)
  - Cok kisa yorumlari ele
  - Duplicate'leri kaldir
  - Rating/label celiskilerini ayri karantinaya yaz
  - AI-generated notrleri SILMEDEN kalite-bazli filtrele:
      * cok kisa (wc<=3) cikar
      * tam-metin duplicate cikar
      * cok yuksek frekansli n-gram kaliplarinda olanlardan ornekleme yap
      * tek-cumlelik mekanik kalip ifadeleri (idare ederdi / fena degildi /
        yeterliydi vb.) tamamen elemek yerine SAYI olarak sinirla
      * geriye kalanlardan ~2500 ornek hedefle
  - Optimize CSV + 4 karantina CSV + ozet rapor JSON uret.
"""

from __future__ import annotations

import json
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
INPUT_CSV = ROOT / "data" / "processed" / "Final_Dataset_PlusAI.csv"
OUT_DIR = ROOT / "data" / "processed"

OUT_OPTIMIZED = OUT_DIR / "Final_Dataset_Optimized_v3.csv"
QUAR_BROKEN = OUT_DIR / "v3_quarantine_broken_rows.csv"
QUAR_SHORT = OUT_DIR / "v3_quarantine_short.csv"
QUAR_DUP = OUT_DIR / "v3_quarantine_duplicates.csv"
QUAR_MISMATCH = OUT_DIR / "v3_quarantine_rating_label_mismatch.csv"
QUAR_AI_DROP = OUT_DIR / "v3_quarantine_ai_dropped.csv"
REPORT_JSON = OUT_DIR / "Final_Dataset_Optimized_v3.report.json"

LABELS = ["negatif", "nötr", "pozitif"]
LABEL_ALIAS = {
    "notr": "nötr", "ntr": "nötr", "neutral": "nötr",
    "neg": "negatif", "negative": "negatif",
    "pos": "pozitif", "positive": "pozitif",
}

# AI nötrlerden kalite filtresinden sonra kac tane kalsin?
AI_KEEP_TARGET = 2500
# AI nötrlerin minimum kelime sayisi (3 cok kalipa kapali, 4'e cekiyoruz)
AI_MIN_WORDS = 4
# Genel min word count (gercek + AI sonrasi)
GLOBAL_MIN_WORDS = 3
# Bir n-gram kalibinin AI icinde maksimum kac kez gorunmesine izin veriyoruz
AI_MAX_PATTERN_FREQ = 4
# Source kolonu otelpuan.com varyantlari icin regex
OTELPUAN_VARIANT_RE = re.compile(r"^otel ?puan", re.IGNORECASE)
# Kabul edilen kanonik source isimleri
CANONICAL_SOURCES = {"otelpuan.com", "neredekal", "ai_generated"}

SEED = 42


def normalize_label(x):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return None
    s = str(x).strip().lower()
    s = LABEL_ALIAS.get(s, s)
    return s if s in {"negatif", "nötr", "pozitif"} else None


def normalize_source(x):
    """otelpuan.* varyantlarini 'otelpuan.com' altinda topla, diger bilinmeyenleri 'unknown'."""
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "unknown"
    s = str(x).strip()
    s_low = s.lower()
    if s_low in CANONICAL_SOURCES:
        return s_low
    if OTELPUAN_VARIANT_RE.match(s):
        return "otelpuan.com"
    if "neredekal" in s_low:
        return "neredekal"
    if "ai" in s_low and "generated" in s_low:
        return "ai_generated"
    return "unknown"


def word_count(text):
    if pd.isna(text):
        return 0
    return len(str(text).split())


def pattern_signature(text):
    """
    Bir AI yorumu icin 'kalip imzasi':
      - kucuk harf
      - noktalama temizligi
      - ilk 4 token alinir (cogu AI yorumu zaten kisa)
    Ayni imzaya sahip cok sayida ornek varsa kalip uretimi var demektir.
    """
    if pd.isna(text):
        return ""
    s = str(text).lower()
    s = re.sub(r"[^\wçğıöşü\s]", " ", s, flags=re.UNICODE)
    s = re.sub(r"\s+", " ", s).strip()
    toks = s.split()
    return " ".join(toks[:4])


def lexical_diversity(text):
    """type-token ratio (TTR). 1.0 -> her kelime essiz, 0.0 -> ayni kelime."""
    toks = str(text).split()
    if not toks:
        return 0.0
    return len(set(t.lower() for t in toks)) / len(toks)


def main():
    rng = np.random.RandomState(SEED)
    print(f"[OKU] {INPUT_CSV}")
    df = pd.read_csv(INPUT_CSV, encoding="utf-8")
    rows_raw = len(df)
    print(f"[OKU] satir = {rows_raw}, kolonlar = {list(df.columns)}")

    report = {"input": str(INPUT_CSV), "rows_raw": rows_raw}

    # ==========================================================
    # 1) Bozuk satirlar: label kolonu kanonik degil VE
    #    rating_auto sayisal degil  -> kayma indikatoru.
    # ==========================================================
    df["_lbl_norm"] = df["sentiment_label"].map(normalize_label)
    rating_num = pd.to_numeric(df["rating_auto"], errors="coerce")

    # Tamamen bozuk: hem label gecersiz hem rating sayisal degil
    fully_broken_mask = df["_lbl_norm"].isna() & rating_num.isna()
    # Label gecersiz ama rating sayisal -> sadece label problemi
    label_only_bad_mask = df["_lbl_norm"].isna() & ~rating_num.isna()

    broken_total = int(fully_broken_mask.sum() + label_only_bad_mask.sum())
    print(f"[BOZUK] toplam gecersiz-label satir = {broken_total}")
    if broken_total:
        df.loc[fully_broken_mask | label_only_bad_mask].to_csv(
            QUAR_BROKEN, index=False, encoding="utf-8"
        )
    report["rows_broken_quarantined"] = broken_total
    df = df.loc[~(fully_broken_mask | label_only_bad_mask)].copy()
    df["sentiment_label"] = df["_lbl_norm"]
    df = df.drop(columns=["_lbl_norm"])
    rating_num = pd.to_numeric(df["rating_auto"], errors="coerce")

    # ==========================================================
    # 2) Source kolonunu normalize et + 'unknown' kaynaklari karantinaya at
    # ==========================================================
    df["source"] = df["source"].map(normalize_source)
    src_dist = df["source"].value_counts().to_dict()
    print(f"[SOURCE] normalize sonrasi = {src_dist}")
    report["source_distribution_after_normalize"] = {str(k): int(v) for k, v in src_dist.items()}

    unknown_mask = df["source"] == "unknown"
    unk_count = int(unknown_mask.sum())
    if unk_count:
        df.loc[unknown_mask].to_csv(QUAR_BROKEN, index=False, encoding="utf-8",
                                    mode="a", header=not QUAR_BROKEN.exists())
        print(f"[SOURCE] {unk_count} 'unknown' kaynak satir karantinaya atildi (CSV kayma kalintisi).")
    report["rows_unknown_source_quarantined"] = unk_count
    df = df.loc[~unknown_mask].copy()

    # ==========================================================
    # 3) Comment text temizligi (HTML, zero-width, multispace)
    # ==========================================================
    def clean_text(s):
        if pd.isna(s):
            return ""
        s = str(s)
        s = re.sub(r"<[^>]+>", " ", s)
        s = s.replace("\r", " ").replace("\n", " ").replace("\t", " ")
        s = re.sub(r"[​‌‍﻿]", " ", s)
        s = re.sub(r"\s+", " ", s).strip()
        return s

    df["comment_text"] = df["comment_text"].map(clean_text)
    df = df[df["comment_text"].str.len() > 0].copy()

    # ==========================================================
    # 4) Cok kisa yorumlari karantinaya al (genel)
    # ==========================================================
    wc_all = df["comment_text"].map(word_count)
    short_mask = wc_all < GLOBAL_MIN_WORDS
    short_count = int(short_mask.sum())
    print(f"[KISA] {short_count} satir < {GLOBAL_MIN_WORDS} kelime")
    if short_count:
        df.loc[short_mask].to_csv(QUAR_SHORT, index=False, encoding="utf-8")
    report["rows_short_quarantined"] = short_count
    df = df.loc[~short_mask].copy()

    # ==========================================================
    # 5) Duplicate temizligi (comment_text bazli, lower+strip)
    # ==========================================================
    norm_t = df["comment_text"].str.lower().str.strip()
    dup_mask = norm_t.duplicated(keep="first")
    dup_count = int(dup_mask.sum())
    print(f"[DUP] {dup_count} duplicate satir")
    if dup_count:
        df.loc[dup_mask].to_csv(QUAR_DUP, index=False, encoding="utf-8")
    report["rows_duplicates_quarantined"] = dup_count
    df = df.loc[~dup_mask].copy()

    # ==========================================================
    # 6) Rating/label celiskisi (sadece otelpuan.com'da var, AI degil)
    # ==========================================================
    rating_num = pd.to_numeric(df["rating_auto"], errors="coerce")
    mm_low = (rating_num <= 2) & (df["sentiment_label"].isin(["nötr", "pozitif"]))
    mm_high = (rating_num >= 4) & (df["sentiment_label"] == "negatif")
    mismatch_mask = (mm_low | mm_high).fillna(False)
    mm_count = int(mismatch_mask.sum())
    print(f"[CELISKI] rating-label = {mm_count} satir karantinaya")
    if mm_count:
        df.loc[mismatch_mask].to_csv(QUAR_MISMATCH, index=False, encoding="utf-8")
    report["rows_rating_label_mismatch_quarantined"] = mm_count
    df = df.loc[~mismatch_mask].copy()

    # ==========================================================
    # 7) AI-generated notrlerde KALITE FILTRESI
    # ==========================================================
    ai_mask = df["source"] == "ai_generated"
    ai = df.loc[ai_mask].copy()
    real = df.loc[~ai_mask].copy()

    print(f"[AI] filtre oncesi AI satir = {len(ai)}")
    report["ai_rows_before_filter"] = int(len(ai))

    ai["_wc"] = ai["comment_text"].map(word_count)
    ai["_sig"] = ai["comment_text"].map(pattern_signature)
    ai["_ttr"] = ai["comment_text"].map(lexical_diversity)
    ai["_norm"] = ai["comment_text"].str.lower().str.strip()

    drop_reasons = pd.Series([""] * len(ai), index=ai.index, dtype=object)

    # 7a) Cok kisa (<=3 kelime): mekanik kalip riski yuksek
    too_short = ai["_wc"] < AI_MIN_WORDS
    drop_reasons[too_short & (drop_reasons == "")] = "too_short"

    # 7b) Tam-metin duplicate (AI icinde)
    dup_in_ai = ai["_norm"].duplicated(keep="first")
    drop_reasons[dup_in_ai & (drop_reasons == "")] = "exact_duplicate_in_ai"

    # 7c) Yuksek frekansli kalip imzasi: bir _sig 4'ten fazla goruluyorsa
    #     ilk 4 hari (random) tut, gerisini sil.
    sig_counts = Counter(ai["_sig"])
    over_freq_sigs = {s for s, c in sig_counts.items() if c > AI_MAX_PATTERN_FREQ}
    print(f"[AI] yuksek frekansli kalip imzasi sayisi = {len(over_freq_sigs)}")
    # her over_freq sig icin ilk AI_MAX_PATTERN_FREQ orneji tut, digerlerini drop
    keep_idx = set()
    for sig in over_freq_sigs:
        idxs = ai.index[(ai["_sig"] == sig) & (drop_reasons == "")].tolist()
        if not idxs:
            continue
        # rastgele AI_MAX_PATTERN_FREQ tane tut, gerisini drop_reasons'a yaz
        chosen = list(rng.choice(idxs, size=min(AI_MAX_PATTERN_FREQ, len(idxs)), replace=False))
        keep_idx.update(chosen)
        for i in idxs:
            if i not in chosen:
                drop_reasons.at[i] = "pattern_oversampled"

    # 7d) Cok dusuk lexical diversity (TTR < 0.6) ve <= 5 kelime: tekrar/kalip
    low_div = (ai["_ttr"] < 0.6) & (ai["_wc"] <= 5)
    drop_reasons[low_div & (drop_reasons == "")] = "low_lexical_diversity_short"

    surviving = ai[drop_reasons == ""].copy()
    print(f"[AI] heuristik filtre sonrasi hayatta kalan = {len(surviving)}")

    # 7e) Hedef sayiyi yakala: stratify-by-signature ile orneklem
    if len(surviving) > AI_KEEP_TARGET:
        # her imza icin orantili pay
        sig_groups = surviving.groupby("_sig")
        group_sizes = {sig: len(g) for sig, g in sig_groups}
        n_groups = len(group_sizes)
        # Her gruptan en az 1 al, kalan kotayi orantili dagit
        base = {sig: 1 for sig in group_sizes}
        remaining = AI_KEEP_TARGET - n_groups
        if remaining < 0:
            # cok fazla grup var; tum gruplardan 1 alsak bile asar -> rastgele alt-orneklem
            sampled = surviving.sample(n=AI_KEEP_TARGET, random_state=SEED)
        else:
            total = sum(group_sizes.values())
            extras = {
                sig: int(round((group_sizes[sig] / total) * remaining))
                for sig in group_sizes
            }
            quotas = {sig: min(group_sizes[sig], base[sig] + extras[sig]) for sig in group_sizes}
            picked = []
            for sig, g in sig_groups:
                k = quotas[sig]
                if k <= 0:
                    continue
                if len(g) <= k:
                    picked.append(g)
                else:
                    picked.append(g.sample(n=k, random_state=SEED))
            sampled = pd.concat(picked, axis=0)
            # Yuvarlama hatalarini kapatmak icin gerekirse trim/topla
            if len(sampled) > AI_KEEP_TARGET:
                sampled = sampled.sample(n=AI_KEEP_TARGET, random_state=SEED)
            elif len(sampled) < AI_KEEP_TARGET:
                # eksigi kalan havuzdan rastgele tamamla
                pool = surviving.drop(index=sampled.index)
                need = AI_KEEP_TARGET - len(sampled)
                if len(pool) >= need:
                    sampled = pd.concat([sampled, pool.sample(n=need, random_state=SEED)], axis=0)
        # secilmeyenleri drop nedeniyle isaretle
        kept_idx = set(sampled.index)
        for i in surviving.index:
            if i not in kept_idx:
                drop_reasons.at[i] = "sampled_out_to_target"
        ai_kept = sampled
    else:
        ai_kept = surviving

    print(f"[AI] son tutulan AI satir = {len(ai_kept)} (hedef ~{AI_KEEP_TARGET})")
    report["ai_rows_kept"] = int(len(ai_kept))
    report["ai_drop_reason_counts"] = {
        r: int((drop_reasons == r).sum()) for r in sorted(set(drop_reasons.unique()) - {""})
    }

    ai_dropped = ai.loc[drop_reasons != ""].copy()
    ai_dropped["_drop_reason"] = drop_reasons.loc[drop_reasons != ""]
    if len(ai_dropped):
        ai_dropped.drop(columns=["_wc", "_sig", "_ttr", "_norm"], errors="ignore").to_csv(
            QUAR_AI_DROP, index=False, encoding="utf-8"
        )

    ai_kept = ai_kept.drop(columns=["_wc", "_sig", "_ttr", "_norm"], errors="ignore")

    # ==========================================================
    # 8) Birlestir
    # ==========================================================
    final = pd.concat([real, ai_kept], axis=0, ignore_index=True)
    # Son sefer comment_text bazli duplicate turamasin (real-AI cakismasi)
    norm_final = final["comment_text"].str.lower().str.strip()
    final = final.loc[~norm_final.duplicated(keep="first")].copy()

    # Karistir
    final = final.sample(frac=1.0, random_state=SEED).reset_index(drop=True)

    # ==========================================================
    # 9) Yaz + rapor
    # ==========================================================
    final.to_csv(OUT_OPTIMIZED, index=False, encoding="utf-8")
    print(f"[YAZ] {OUT_OPTIMIZED} -> {len(final)} satir")

    class_dist = final["sentiment_label"].value_counts().to_dict()
    src_dist_final = final["source"].value_counts().to_dict()
    report["rows_optimized"] = int(len(final))
    report["class_distribution_optimized"] = {str(k): int(v) for k, v in class_dist.items()}
    report["source_distribution_optimized"] = {str(k): int(v) for k, v in src_dist_final.items()}
    report["per_class_word_count"] = {}
    for lbl in LABELS:
        sub = final[final["sentiment_label"] == lbl]["comment_text"].map(word_count)
        report["per_class_word_count"][lbl] = {
            "n": int(len(sub)),
            "mean": float(sub.mean()) if len(sub) else 0.0,
            "median": float(sub.median()) if len(sub) else 0.0,
            "q05": float(sub.quantile(0.05)) if len(sub) else 0.0,
            "q95": float(sub.quantile(0.95)) if len(sub) else 0.0,
        }

    REPORT_JSON.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[RAPOR] {REPORT_JSON}")
    print("\n=== OZET ===")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
