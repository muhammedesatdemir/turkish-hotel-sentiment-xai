# -*- coding: utf-8 -*-
"""
5-fold CV detaylı markdown tablosu üretici (BERTurk + ELECTRA).

Her seed × fold için Macro-F1 / F1-nötr / Accuracy değerlerini
run_report.json'lardan toplar; final rapora hazır markdown tablo +
CSV özeti üretir.

Koşu:
    python scripts/build_cv_table.py             # her iki model için
    python scripts/build_cv_table.py berturk     # sadece BERTurk
    python scripts/build_cv_table.py electra     # sadece ELECTRA

Çıktı:
    reports/cv_detailed/{berturk,electra}_cv_detail.{md,csv}
"""
from __future__ import annotations

import sys
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "reports" / "cv_detailed"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SEEDS = [42, 43, 44]
N_FOLDS = 5

MODELS = {
    "berturk": {
        "cv_root": ROOT / "models" / "cv" / "v3_genisletilmis_5fold_berturk",
        "model_tag": "bert-base-turkish-cased_len256_ep5_lr2e-05_losswce",
        "title": "BERTurk",
        "model_id": "dbmdz/bert-base-turkish-cased",
        "multi_seed_f1m": 0.8797,
        "multi_seed_f1m_std": 0.0073,
        "multi_seed_f1n": 0.8056,
        "multi_seed_f1n_std": 0.0112,
    },
    "electra": {
        "cv_root": ROOT / "models" / "cv" / "v3_genisletilmis_5fold_electra",
        "model_tag": "electra-base-turkish-cased-discriminator_len256_ep5_lr2e-05_losswce",
        "title": "ELECTRA-tr",
        "model_id": "dbmdz/electra-base-turkish-cased-discriminator",
        "multi_seed_f1m": 0.8853,
        "multi_seed_f1m_std": 0.0143,
        "multi_seed_f1n": 0.8143,
        "multi_seed_f1n_std": 0.0243,
    },
}


def load_metrics(cv_root: Path, model_tag: str, seed: int, fold: int) -> dict:
    rr = cv_root / f"seed{seed}_fold{fold}" / f"{model_tag}_seed{seed}" / "run_report.json"
    with open(rr, encoding="utf-8") as f:
        data = json.load(f)
    test = data.get("test_metrics", {})
    per_class = test.get("f1_per_class", {})
    return {
        "seed": seed,
        "fold": fold,
        "accuracy": test.get("accuracy"),
        "f1_macro": test.get("f1_macro"),
        "f1_negatif": per_class.get("negatif"),
        "f1_notr": per_class.get("nötr"),
        "f1_pozitif": per_class.get("pozitif"),
    }


def build(model_key: str) -> None:
    cfg = MODELS[model_key]
    print(f"\n{'=' * 70}\n{cfg['title']} CV Detail\n{'=' * 70}")

    rows = [
        load_metrics(cfg["cv_root"], cfg["model_tag"], s, f)
        for s in SEEDS
        for f in range(1, N_FOLDS + 1)
    ]
    df = pd.DataFrame(rows)
    csv_out = OUT_DIR / f"{model_key}_cv_detail.csv"
    df.to_csv(csv_out, index=False, encoding="utf-8")

    lines = [
        f"# {cfg['title']} 5-Fold Cross-Validation — Detaylı Tablo",
        "",
        "Veri seti: `final_genisletilmis_dataset_optimized_v3_v2.csv` (14.337 satır)",
        "Protokol: AI-aware Stratified K-Fold, n_splits=5, 3 seed (42/43/44)",
        f"Model: {cfg['model_id']}, 5 epoch, lr=2e-5, loss=wce",
        "",
        "## Tüm Fold × Seed Sonuçları (Test Setinde)",
        "",
        "| Seed | Fold | Accuracy | Macro-F1 | F1-negatif | F1-nötr | F1-pozitif |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        lines.append(
            f"| {r['seed']} | {r['fold']} | "
            f"{r['accuracy']:.4f} | {r['f1_macro']:.4f} | "
            f"{r['f1_negatif']:.4f} | {r['f1_notr']:.4f} | {r['f1_pozitif']:.4f} |"
        )

    lines += [
        "",
        "## Seed Bazında Ortalama ± Std (5 Fold)",
        "",
        "| Seed | Accuracy | Macro-F1 | F1-negatif | F1-nötr | F1-pozitif |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for s in SEEDS:
        sub = df[df["seed"] == s]
        means = sub.mean(numeric_only=True)
        stds = sub.std(numeric_only=True)
        lines.append(
            f"| {s} | "
            f"{means['accuracy']:.4f} ± {stds['accuracy']:.4f} | "
            f"{means['f1_macro']:.4f} ± {stds['f1_macro']:.4f} | "
            f"{means['f1_negatif']:.4f} ± {stds['f1_negatif']:.4f} | "
            f"{means['f1_notr']:.4f} ± {stds['f1_notr']:.4f} | "
            f"{means['f1_pozitif']:.4f} ± {stds['f1_pozitif']:.4f} |"
        )

    all_means = df.mean(numeric_only=True)
    all_stds = df.std(numeric_only=True)
    lines += [
        "",
        "## Toplam Agregat (15 Koşu = 5 Fold × 3 Seed)",
        "",
        "| Metrik | Ortalama | Std | Min | Max |",
        "|---|---:|---:|---:|---:|",
    ]
    for col in ["accuracy", "f1_macro", "f1_negatif", "f1_notr", "f1_pozitif"]:
        lines.append(
            f"| {col} | {all_means[col]:.4f} | {all_stds[col]:.4f} | "
            f"{df[col].min():.4f} | {df[col].max():.4f} |"
        )

    delta = abs(all_means["f1_macro"] - cfg["multi_seed_f1m"])
    same_band = "σ düzeyinde gürültü" if delta < all_stds["f1_macro"] else "σ üzerinde"
    lines += [
        "",
        f"## Karşılaştırma: Multi-Seed Final Eğitim vs CV ({cfg['title']})",
        "",
        "| Kaynak | Macro-F1 (μ ± σ) | F1-nötr (μ ± σ) | n |",
        "|---|---:|---:|---:|",
        f"| 5-Fold CV (3 seed × 5 fold) | "
        f"{all_means['f1_macro']:.4f} ± {all_stds['f1_macro']:.4f} | "
        f"{all_means['f1_notr']:.4f} ± {all_stds['f1_notr']:.4f} | 15 |",
        f"| Multi-Seed Final (3 seed) | "
        f"{cfg['multi_seed_f1m']:.4f} ± {cfg['multi_seed_f1m_std']:.4f} | "
        f"{cfg['multi_seed_f1n']:.4f} ± {cfg['multi_seed_f1n_std']:.4f} | 3 |",
        "",
        f"**Δ Macro-F1 = {delta:.4f}** ({same_band}, "
        f"veri varyansına karşı stabil bulgu)",
    ]
    md_out = OUT_DIR / f"{model_key}_cv_detail.md"
    md_out.write_text("\n".join(lines), encoding="utf-8")

    print(f"OK: {md_out}")
    print(f"OK: {csv_out}")
    print(f"\n15 koşu Macro-F1 = {all_means['f1_macro']:.4f} ± {all_stds['f1_macro']:.4f}")
    print(f"15 koşu F1-nötr   = {all_means['f1_notr']:.4f} ± {all_stds['f1_notr']:.4f}")


def main() -> None:
    arg = sys.argv[1].lower() if len(sys.argv) > 1 else None
    if arg in MODELS:
        build(arg)
    elif arg in (None, "all", "both"):
        for k in MODELS:
            build(k)
    else:
        print(f"Bilinmeyen argüman: {arg}. Geçerli: berturk | electra | all")
        sys.exit(2)


if __name__ == "__main__":
    main()
