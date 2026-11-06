"""
src/error_analysis.py
======================

v3_genisletilmis seed43 (best) icin yanlis siniflanan ornekleri toplar,
modelin softmax confidence'larini hesaplar ve heuristik kategorilere ayirir:

  - 'belirsiz_dengesiz'  : zit kutuplu kelimeler (ikinci en yuksek prob > 0.30)
  - 'guvenli_yanlis'     : model emin oldugu halde yanilmis (max prob > 0.80)
  - 'siyah_beyaz'        : metin acik tek kutuplu ama yanlis (heuristik kelime sayisi)
  - 'orta_seviye'        : digerleri

Ciktilar:
  reports/xai_final/error_analysis.csv      : tum hatalar + kategori
  reports/xai_final/error_analysis_summary.json : sayisal ozet
"""
from __future__ import annotations
import argparse, json, os, re
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from transformers import AutoModelForSequenceClassification, AutoTokenizer

LABELS = ["negatif", "nötr", "pozitif"]

POZ_KEYS = [
    "harika", "muhteşem", "muhtesem", "mükemmel", "mukemmel",
    "tavsiye ederim", "çok memnun", "cok memnun", "bayıldık", "bayildik",
    "süper", "super", "ilgili", "güler yüz", "guler yuz", "tertemiz",
    "lezzetli", "kaliteli", "memnun kaldık", "memnun kaldik",
]
NEG_KEYS = [
    "berbat", "rezalet", "iğrenç", "igrenc", "kesinlikle gitmeyin",
    "kötü", "kotu", "pis", "kirli", "asla", "para tuzağı", "para tuzagi",
    "kandırıldık", "kandirildik", "hayal kırıklığı", "hayal kirikligi",
    "bayat", "soğuk", "soguk", "şikayet", "sikayet", "rezil",
]


def count_polarity(text: str) -> tuple[int, int]:
    low = text.lower()
    norm = (low.replace("ı","i").replace("İ","i").replace("ş","s").replace("ç","c")
                .replace("ğ","g").replace("ö","o").replace("ü","u"))
    pos = sum(1 for k in POZ_KEYS if k in low or k in norm)
    neg = sum(1 for k in NEG_KEYS if k in low or k in norm)
    return pos, neg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", type=Path,
                    default=Path("models/optimized_v3/v3_genisletilmis/"
                                 "bert-base-turkish-cased_len256_ep5_lr2e-05_losswce_seed43/best_model"))
    ap.add_argument("--pred_csv",  type=Path,
                    default=Path("models/optimized_v3/v3_genisletilmis/"
                                 "bert-base-turkish-cased_len256_ep5_lr2e-05_losswce_seed43/test_predictions.csv"))
    ap.add_argument("--out_dir",   type=Path, default=Path("reports/xai_final"))
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] tahminler okunuyor: {args.pred_csv}")
    df = pd.read_csv(args.pred_csv)
    df["wc"] = df["comment_text"].astype(str).str.split().str.len()

    errs = df[df.correct == 0].copy().reset_index(drop=True)
    print(f"[INFO] yanlis siniflanan: {len(errs)} / {len(df)}  ({100*len(errs)/len(df):.1f}%)")

    # Confidence hesapla
    print(f"[INFO] model yukleniyor: {args.model_dir}")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(str(args.model_dir))
    mdl = AutoModelForSequenceClassification.from_pretrained(str(args.model_dir)).to(device).eval()

    all_probs = []
    BS = 16
    with torch.no_grad():
        for i in range(0, len(errs), BS):
            batch = errs["comment_text"].astype(str).iloc[i:i+BS].tolist()
            enc = tok(batch, padding=True, truncation=True, max_length=256,
                      return_tensors="pt").to(device)
            logits = mdl(**enc).logits
            probs = F.softmax(logits, dim=-1).cpu().numpy()
            all_probs.append(probs)
    P = np.vstack(all_probs)  # (n, 3)

    errs["p_negatif"] = P[:,0]
    errs["p_notr"]    = P[:,1]
    errs["p_pozitif"] = P[:,2]
    errs["max_prob"]  = P.max(axis=1)
    errs["second_prob"] = np.sort(P, axis=1)[:, -2]

    # Polarity counts
    pos_cnt, neg_cnt = [], []
    for t in errs["comment_text"].astype(str):
        p, n = count_polarity(t)
        pos_cnt.append(p); neg_cnt.append(n)
    errs["pos_keys"] = pos_cnt
    errs["neg_keys"] = neg_cnt

    # Kategorize et
    def categorize(row):
        # 1. ikinci olasilik yuksek -> model belirsiz
        if row.second_prob > 0.30:
            return "belirsiz_dengesiz"
        # 2. model emin ama yanlis (etiket gurultusu olabilir)
        if row.max_prob > 0.80:
            # eger metnin kutuplari etiketle celisiyorsa -> etiket gurultusu sinyali
            if row.sentiment_label == "negatif" and row.pos_keys >= 2 and row.neg_keys == 0:
                return "etiket_gurultusu_supheli"
            if row.sentiment_label == "pozitif" and row.neg_keys >= 2 and row.pos_keys == 0:
                return "etiket_gurultusu_supheli"
            return "guvenli_yanlis"
        # 3. orta seviye guvensizlik
        return "orta_seviye"

    errs["kategori"] = errs.apply(categorize, axis=1)

    # Hata yonu
    errs["yanlis_yon"] = errs["sentiment_label"] + "_to_" + errs["pred_label"]

    # CSV yaz
    out_csv = args.out_dir / "error_analysis.csv"
    cols_keep = ["comment_text", "sentiment_label", "pred_label", "yanlis_yon",
                 "wc", "p_negatif", "p_notr", "p_pozitif", "max_prob", "second_prob",
                 "pos_keys", "neg_keys", "kategori", "source"]
    errs[cols_keep].sort_values(["yanlis_yon","kategori"]).to_csv(out_csv, index=False, encoding="utf-8")
    print(f"[YAZILDI] {out_csv}")

    # Ozet
    summary = {
        "total_test": len(df),
        "total_errors": len(errs),
        "error_rate": round(len(errs)/len(df), 4),
        "errors_by_direction": errs["yanlis_yon"].value_counts().to_dict(),
        "errors_by_category": errs["kategori"].value_counts().to_dict(),
        "errors_by_dir_x_cat": errs.groupby(["yanlis_yon","kategori"]).size().to_dict(),
        "confidence_stats": {
            "mean_max_prob": round(float(errs.max_prob.mean()), 4),
            "median_max_prob": round(float(errs.max_prob.median()), 4),
            "high_conf_errors_count": int((errs.max_prob > 0.80).sum()),
            "low_conf_errors_count":  int((errs.max_prob < 0.50).sum()),
        },
        "wc_stats": {
            "mean": round(float(errs.wc.mean()), 1),
            "median": int(errs.wc.median()),
            "min": int(errs.wc.min()),
            "max": int(errs.wc.max()),
        },
        "by_source": errs["source"].value_counts().to_dict(),
    }
    # tuple keylerini stringle
    summary["errors_by_dir_x_cat"] = {f"{k[0]}__{k[1]}": v for k, v in summary["errors_by_dir_x_cat"].items()}

    out_json = args.out_dir / "error_analysis_summary.json"
    out_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[YAZILDI] {out_json}")

    # Konsola yazdir
    print()
    print("=" * 70)
    print(f"Toplam test                : {summary['total_test']}")
    print(f"Toplam hata                : {summary['total_errors']}  ({100*summary['error_rate']:.2f}%)")
    print(f"Ortalama max prob (hatada) : {summary['confidence_stats']['mean_max_prob']}")
    print(f"  >0.80 'guvenli yanlis'   : {summary['confidence_stats']['high_conf_errors_count']}")
    print(f"  <0.50 'belirsiz'         : {summary['confidence_stats']['low_conf_errors_count']}")
    print()
    print("Yon dagilimi:")
    for k, v in summary["errors_by_direction"].items():
        print(f"  {k:35s}: {v}")
    print()
    print("Kategori dagilimi:")
    for k, v in summary["errors_by_category"].items():
        print(f"  {k:30s}: {v}")
    print("=" * 70)


if __name__ == "__main__":
    main()
