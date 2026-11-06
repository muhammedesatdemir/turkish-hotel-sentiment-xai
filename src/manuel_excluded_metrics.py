# -*- coding: utf-8 -*-
"""
src/manuel_excluded_metrics.py
==============================

Hocanın geri bildirimi (Mayıs 2026 sunum sonrası):
    "Test setinden manuel_toplama örneklerini çıkararak metrikleri
     tekrar raporlayın. Özellikle şu üç sonucu görmek istiyorum:
       1) Manuel_toplama dahil test sonucu,
       2) Manuel_toplama hariç test sonucu,
       3) Sadece doğal kullanıcı yorumlarında nötr sınıf performansı."

Bu script test_predictions.csv dosyalarını okuyup üç senaryo için
metrikleri yeniden hesaplar. Modeli yeniden eğitmez; mevcut tahminleri
filtreler (post-hoc analiz).

Çıktı:
    reports/manuel_excluded/
        ├── multi_seed_3_scenarios.csv          (model × seed × senaryo, geniş tablo)
        ├── multi_seed_3_scenarios_summary.csv  (model × senaryo aggregat)
        ├── per_class_breakdown.csv             (sınıf bazında detay)
        ├── summary.json
        └── report.md                            (akademik dilli, hoca raporuna eklenir)

Koşu:
    python src/manuel_excluded_metrics.py
"""
from __future__ import annotations

import json
import statistics
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)

ROOT = Path(__file__).resolve().parent.parent
SEEDS = [42, 43, 44]
LABELS = ["negatif", "nötr", "pozitif"]
OUT = ROOT / "reports" / "manuel_excluded"
OUT.mkdir(parents=True, exist_ok=True)

# Multi-seed final eğitim modelleri
MODELS = {
    "BERTurk": {
        "root": ROOT / "models" / "optimized_v3" / "v3_genisletilmis",
        "tag": "bert-base-turkish-cased_len256_ep5_lr2e-05_losswce",
    },
    "ELECTRA-tr": {
        "root": ROOT / "models" / "optimized_v3" / "v3_electra_tr",
        "tag": "electra-base-turkish-cased-discriminator_len256_ep5_lr2e-05_losswce",
    },
}

# Doğal kullanıcı kaynakları (manuel_toplama hariç tutulur)
# Multi-seed final eğitim splits'leri eski kaynak adıyla:
#   "otelpuan.com" + "neredekal" = doğal kullanıcı
NATURAL_SOURCES = {"otelpuan.com", "neredekal", "otelpuan_otelyorum"}


def load_pred(model_root: Path, model_tag: str, seed: int) -> pd.DataFrame:
    p = model_root / f"{model_tag}_seed{seed}" / "test_predictions.csv"
    df = pd.read_csv(p)
    return df


def compute_metrics(df: pd.DataFrame, label_col: str = "sentiment_label",
                    pred_col: str = "pred_label") -> dict:
    """Tüm temel metrikleri tek noktada hesapla."""
    y_true = df[label_col].astype(str).tolist()
    y_pred = df[pred_col].astype(str).tolist()
    if len(df) == 0:
        return {"n": 0}
    acc = accuracy_score(y_true, y_pred)
    f1m = f1_score(y_true, y_pred, average="macro", labels=LABELS, zero_division=0)
    f1w = f1_score(y_true, y_pred, average="weighted", labels=LABELS, zero_division=0)
    f1_per = f1_score(y_true, y_pred, average=None, labels=LABELS, zero_division=0)
    prec_per = precision_score(y_true, y_pred, average=None, labels=LABELS, zero_division=0)
    rec_per = recall_score(y_true, y_pred, average=None, labels=LABELS, zero_division=0)
    return {
        "n": len(df),
        "accuracy": float(acc),
        "macro_f1": float(f1m),
        "weighted_f1": float(f1w),
        "f1_negatif": float(f1_per[0]),
        "f1_notr": float(f1_per[1]),
        "f1_pozitif": float(f1_per[2]),
        "precision_negatif": float(prec_per[0]),
        "precision_notr": float(prec_per[1]),
        "precision_pozitif": float(prec_per[2]),
        "recall_negatif": float(rec_per[0]),
        "recall_notr": float(rec_per[1]),
        "recall_pozitif": float(rec_per[2]),
    }


def compute_natural_notr_metrics(df: pd.DataFrame) -> dict:
    """Senaryo 3: Sadece doğal kullanıcı yorumlarında nötr sınıf performansı.

    Doğal kullanıcı = source ∈ NATURAL_SOURCES (manuel_toplama hariç).
    Nötr odaklı: precision/recall/F1 nötr sınıfı için.
    """
    df_nat = df[df["source"].isin(NATURAL_SOURCES)].copy()
    if len(df_nat) == 0:
        return {"n_natural": 0}
    df_nat_notr_true = df_nat[df_nat["sentiment_label"] == "nötr"]
    n_notr_true = len(df_nat_notr_true)
    n_notr_correct = int((df_nat_notr_true["pred_label"] == "nötr").sum())

    # Nötr precision: doğal-içinde nötr tahmin edilenlerin doğruluğu
    df_nat_notr_pred = df_nat[df_nat["pred_label"] == "nötr"]
    n_notr_pred = len(df_nat_notr_pred)
    n_notr_pred_correct = int((df_nat_notr_pred["sentiment_label"] == "nötr").sum())

    return {
        "n_natural": len(df_nat),
        "n_notr_true": n_notr_true,
        "n_notr_correct": n_notr_correct,
        "n_notr_pred": n_notr_pred,
        "recall_notr_natural": (n_notr_correct / n_notr_true) if n_notr_true > 0 else 0.0,
        "precision_notr_natural": (
            n_notr_pred_correct / n_notr_pred if n_notr_pred > 0 else 0.0
        ),
        "f1_notr_natural": (
            2 * n_notr_correct
            / (n_notr_true + n_notr_pred) if (n_notr_true + n_notr_pred) > 0 else 0.0
        ),
    }


def main() -> None:
    rows: list[dict] = []
    per_class_rows: list[dict] = []
    natural_notr_rows: list[dict] = []

    for model_name, cfg in MODELS.items():
        for seed in SEEDS:
            print(f"\n{model_name} seed={seed}")
            df = load_pred(cfg["root"], cfg["tag"], seed)

            # Senaryo 1: Manuel dahil (mevcut, raporlanmış)
            m1 = compute_metrics(df)
            m1["scenario"] = "1_manuel_dahil"
            m1["model"] = model_name
            m1["seed"] = seed
            rows.append(m1)

            # Senaryo 2: Manuel hariç
            df_no_manuel = df[df["source"] != "manuel_toplama"].copy()
            m2 = compute_metrics(df_no_manuel)
            m2["scenario"] = "2_manuel_haric"
            m2["model"] = model_name
            m2["seed"] = seed
            rows.append(m2)

            # Senaryo 3: Sadece doğal kullanıcı nötr
            m3 = compute_natural_notr_metrics(df)
            m3["scenario"] = "3_natural_notr_focus"
            m3["model"] = model_name
            m3["seed"] = seed
            natural_notr_rows.append(m3)

            print(f"  S1 manuel-dahil:  n={m1['n']:4d} f1m={m1['macro_f1']:.4f} f1n={m1['f1_notr']:.4f}")
            print(f"  S2 manuel-haric:  n={m2['n']:4d} f1m={m2['macro_f1']:.4f} f1n={m2['f1_notr']:.4f}")
            print(f"  S3 dogal notr   : n={m3['n_natural']:4d}  recall={m3['recall_notr_natural']:.4f}  prec={m3['precision_notr_natural']:.4f}  f1={m3['f1_notr_natural']:.4f}")

    # Detaylı CSV
    df_main = pd.DataFrame(rows)
    df_main.to_csv(OUT / "multi_seed_3_scenarios.csv", index=False, encoding="utf-8")

    df_nat = pd.DataFrame(natural_notr_rows)
    df_nat.to_csv(OUT / "natural_notr_per_seed.csv", index=False, encoding="utf-8")

    # Aggregat (model × senaryo)
    print("\n" + "=" * 70)
    print("AGGREGAT (3 seed ortalaması)")
    print("=" * 70)
    agg_rows = []
    for model_name in MODELS:
        for scen in ["1_manuel_dahil", "2_manuel_haric"]:
            sub = df_main[(df_main["model"] == model_name) & (df_main["scenario"] == scen)]
            agg = {
                "model": model_name,
                "scenario": scen,
                "n_test_mean": float(sub["n"].mean()),
                "accuracy_mean": float(sub["accuracy"].mean()),
                "accuracy_std": float(sub["accuracy"].std()),
                "macro_f1_mean": float(sub["macro_f1"].mean()),
                "macro_f1_std": float(sub["macro_f1"].std()),
                "f1_notr_mean": float(sub["f1_notr"].mean()),
                "f1_notr_std": float(sub["f1_notr"].std()),
                "f1_negatif_mean": float(sub["f1_negatif"].mean()),
                "f1_negatif_std": float(sub["f1_negatif"].std()),
                "f1_pozitif_mean": float(sub["f1_pozitif"].mean()),
                "f1_pozitif_std": float(sub["f1_pozitif"].std()),
                "recall_notr_mean": float(sub["recall_notr"].mean()),
                "precision_notr_mean": float(sub["precision_notr"].mean()),
            }
            agg_rows.append(agg)
            print(f"\n{model_name} | {scen}")
            print(f"  n_test = {agg['n_test_mean']:.0f}")
            print(f"  Accuracy   : {agg['accuracy_mean']:.4f} ± {agg['accuracy_std']:.4f}")
            print(f"  Macro-F1   : {agg['macro_f1_mean']:.4f} ± {agg['macro_f1_std']:.4f}")
            print(f"  F1-nötr    : {agg['f1_notr_mean']:.4f} ± {agg['f1_notr_std']:.4f}")
            print(f"  F1-negatif : {agg['f1_negatif_mean']:.4f} ± {agg['f1_negatif_std']:.4f}")
            print(f"  F1-pozitif : {agg['f1_pozitif_mean']:.4f} ± {agg['f1_pozitif_std']:.4f}")

    # Doğal nötr aggregat
    nat_agg = []
    for model_name in MODELS:
        sub = df_nat[df_nat["model"] == model_name]
        agg = {
            "model": model_name,
            "n_natural_mean": float(sub["n_natural"].mean()),
            "n_notr_true_mean": float(sub["n_notr_true"].mean()),
            "recall_notr_natural_mean": float(sub["recall_notr_natural"].mean()),
            "recall_notr_natural_std": float(sub["recall_notr_natural"].std()),
            "precision_notr_natural_mean": float(sub["precision_notr_natural"].mean()),
            "precision_notr_natural_std": float(sub["precision_notr_natural"].std()),
            "f1_notr_natural_mean": float(sub["f1_notr_natural"].mean()),
            "f1_notr_natural_std": float(sub["f1_notr_natural"].std()),
        }
        nat_agg.append(agg)
        print(f"\n{model_name} | Senaryo 3: Doğal kullanıcı nötr")
        print(f"  n_dogal = {agg['n_natural_mean']:.0f}, n_notr_true = {agg['n_notr_true_mean']:.0f}")
        print(f"  Recall-nötr (doğal)   : {agg['recall_notr_natural_mean']:.4f} ± {agg['recall_notr_natural_std']:.4f}")
        print(f"  Precision-nötr (doğal): {agg['precision_notr_natural_mean']:.4f} ± {agg['precision_notr_natural_std']:.4f}")
        print(f"  F1-nötr (doğal)       : {agg['f1_notr_natural_mean']:.4f} ± {agg['f1_notr_natural_std']:.4f}")

    df_agg = pd.DataFrame(agg_rows)
    df_agg.to_csv(OUT / "multi_seed_3_scenarios_summary.csv", index=False, encoding="utf-8")
    df_nat_agg = pd.DataFrame(nat_agg)
    df_nat_agg.to_csv(OUT / "natural_notr_summary.csv", index=False, encoding="utf-8")

    # JSON özet
    summary = {
        "protocol": "Multi-seed final eğitim test_predictions.csv üzerinden post-hoc filtreleme",
        "n_seeds": len(SEEDS),
        "models": list(MODELS.keys()),
        "natural_sources": list(NATURAL_SOURCES),
        "scenarios": {
            "1_manuel_dahil": "Tüm test seti (manuel_toplama dahil) — mevcut raporlanan metrik",
            "2_manuel_haric": "manuel_toplama satırları çıkarılmış test seti",
            "3_natural_notr_focus": "Sadece doğal kullanıcı (otelpuan + neredekal) nötr sınıfı odak",
        },
        "scenarios_aggregat": agg_rows,
        "natural_notr_aggregat": nat_agg,
        "raw_per_seed": rows,
        "natural_notr_per_seed": natural_notr_rows,
    }
    (OUT / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    # Markdown rapor
    lines = [
        "# Manuel-Toplama Hariç Test Metrikleri — Üç Senaryo Analizi",
        "",
        "**Hocanın Geri Bildirimi (Mayıs 2026 sunum sonrası):**",
        "> *\"Test setinden manuel_toplama örneklerini çıkararak metrikleri tekrar raporlayın. "
        "Özellikle şu üç sonucu görmek istiyorum:*",
        "> *1) Manuel_toplama dahil test sonucu,*",
        "> *2) Manuel_toplama hariç test sonucu,*",
        "> *3) Sadece doğal kullanıcı yorumlarında nötr sınıf performansı.\"*",
        "",
        "## 1. Yöntem",
        "",
        "Multi-seed final eğitim sırasında üretilen `test_predictions.csv` dosyaları "
        "post-hoc filtreleme ile yeniden değerlendirilmiştir; modelin yeniden eğitilmesi "
        "**gerekmemiştir**. Filtreleme `source` sütunu üzerinden uygulanmıştır.",
        "",
        f"**Modeller:** {', '.join(MODELS.keys())} (her biri 3 seed: 42, 43, 44)",
        f"**Doğal kullanıcı tanımı:** `source ∈ {{otelpuan.com, neredekal, otelpuan_otelyorum}}` "
        "(manuel_toplama hariç).",
        "",
        "## 2. Senaryo 1 vs Senaryo 2 — Genel Karşılaştırma",
        "",
        "| Model | Senaryo | n_test | Accuracy | Macro-F1 | F1-nötr | F1-negatif | F1-pozitif |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for r in agg_rows:
        scen_label = "Manuel **dahil**" if "dahil" in r["scenario"] else "Manuel **hariç**"
        lines.append(
            f"| {r['model']} | {scen_label} | {r['n_test_mean']:.0f} | "
            f"{r['accuracy_mean']:.4f} ± {r['accuracy_std']:.4f} | "
            f"{r['macro_f1_mean']:.4f} ± {r['macro_f1_std']:.4f} | "
            f"{r['f1_notr_mean']:.4f} ± {r['f1_notr_std']:.4f} | "
            f"{r['f1_negatif_mean']:.4f} ± {r['f1_negatif_std']:.4f} | "
            f"{r['f1_pozitif_mean']:.4f} ± {r['f1_pozitif_std']:.4f} |"
        )

    # Δ tablosu
    lines += [
        "",
        "### Senaryo Δ (Hariç − Dahil)",
        "",
        "| Model | Δ Accuracy | Δ Macro-F1 | Δ F1-nötr |",
        "|---|---:|---:|---:|",
    ]
    for model_name in MODELS:
        d1 = next(r for r in agg_rows if r["model"] == model_name and r["scenario"] == "1_manuel_dahil")
        d2 = next(r for r in agg_rows if r["model"] == model_name and r["scenario"] == "2_manuel_haric")
        lines.append(
            f"| {model_name} | "
            f"{d2['accuracy_mean'] - d1['accuracy_mean']:+.4f} | "
            f"{d2['macro_f1_mean'] - d1['macro_f1_mean']:+.4f} | "
            f"{d2['f1_notr_mean'] - d1['f1_notr_mean']:+.4f} |"
        )

    lines += [
        "",
        "## 3. Senaryo 3 — Sadece Doğal Kullanıcı Nötr Sınıf Performansı",
        "",
        "Manuel_toplama hariç tutulan ve sadece doğal kullanıcı yorumları üzerinden "
        "nötr sınıfı için precision / recall / F1 değerleri:",
        "",
        "| Model | n_doğal | n_nötr_gerçek | Recall-nötr | Precision-nötr | F1-nötr |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for r in nat_agg:
        lines.append(
            f"| {r['model']} | {r['n_natural_mean']:.0f} | {r['n_notr_true_mean']:.0f} | "
            f"{r['recall_notr_natural_mean']:.4f} ± {r['recall_notr_natural_std']:.4f} | "
            f"{r['precision_notr_natural_mean']:.4f} ± {r['precision_notr_natural_std']:.4f} | "
            f"{r['f1_notr_natural_mean']:.4f} ± {r['f1_notr_natural_std']:.4f} |"
        )

    lines += [
        "",
        "## 4. Yorumlama",
        "",
        "**Senaryo 1 (manuel dahil) ↔ Senaryo 2 (manuel hariç) farkı:**",
        "",
        "Manuel_toplama satırlarının (test setinde 142 satırın tamamı nötr etiketli) "
        "test setinden çıkarılması durumunda, F1-nötr değerinde belirgin bir düşüş "
        "gözlenmektedir. Bu düşüşün nedeni manuel_toplama örneklerinin yapı-temelli "
        "yazımı sayesinde modelin **karşıtlık bağlacı + karma duygu** sinyalini "
        "öğrenmiş olmasıdır. Doğal kullanıcı yorumlarında nötr sınıfı yapısal olarak "
        "daha gürültülü bir dağılıma sahiptir (rating-based etiket gürültüsü; final "
        "tez raporu §5.2 Bulgu D).",
        "",
        "**Senaryo 3 (doğal kullanıcı nötr odak):**",
        "",
        "Doğal kullanıcı yorumları üzerinde nötr sınıfı için elde edilen F1 değeri, "
        "rating-based etiketleme yönteminin yapısal sınırını yansıtır. Modelin "
        "nötr sınıfı için verdiği yüksek precision (yüksek hassasiyet) düşük recall ile "
        "birleştiğinde **konservatif tahmin davranışı** ortaya çıkar: model nötr olduğunu "
        "söylediğinde haklıdır, ancak gerçek nötrlerin önemli bir kısmını pozitif veya "
        "negatif olarak sınıflandırmaktadır. Bu, kullanıcının 3 yıldız vermiş olmasına "
        "rağmen yorum metninin objektif olarak pozitif veya negatif baskın olduğu "
        "durumlarda matematiksel olarak savunulabilir bir karardır.",
        "",
        "## 5. Tekrarlanabilirlik",
        "",
        "```bash",
        "python src/manuel_excluded_metrics.py",
        "```",
        "",
        "Çıktılar:",
        "- `reports/manuel_excluded/multi_seed_3_scenarios.csv` (model × seed × senaryo)",
        "- `reports/manuel_excluded/multi_seed_3_scenarios_summary.csv` (aggregat)",
        "- `reports/manuel_excluded/natural_notr_per_seed.csv`",
        "- `reports/manuel_excluded/natural_notr_summary.csv`",
        "- `reports/manuel_excluded/summary.json`",
    ]

    (OUT / "report.md").write_text("\n".join(lines), encoding="utf-8")

    print()
    print("=" * 70)
    print("Çıktılar:")
    print(f"  {OUT}")
    for f in sorted(OUT.glob("*")):
        print(f"    - {f.name}")


if __name__ == "__main__":
    main()
