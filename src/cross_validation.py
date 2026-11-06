# -*- coding: utf-8 -*-
"""
src/cross_validation.py
=======================

AI-aware Stratified K-Fold Cross-Validation (hoca geri bildirimi A.3).

Hocanin itirazi:
    "CV yok. 5-fold/10-fold yapilmali. Stratified CV. Modelin guvenilirligini
     arttirir."

Mevcut multi-seed (3 seed) yalnizca INIT VARYANSI olcer; CV ise VERI VARYANSI'ni
olcer. Bu script 5-fold stratified CV ile veri varyansini olcup multi-seed
sonuclarini DOGRULAMAYI hedefler.

Kritik protokol (v3_baseline metric leakage'inin tekrarlanmamasi icin):
    1) AI ornekleri (source == ai_generated) HER ZAMAN train tarafinda kalir.
    2) StratifiedKFold yalnizca GERCEK kullanici yorumlarina (otelpuan_otelyorum
       + manuel_toplama, AI haric) uygulanir.
    3) Her fold'un test seti %100 gercek kullanici yorumudur.
    4) Her fold icin train'in icinden val (oran=val_size) ayrilir.
    5) AI'lar train'e eklenir.
    6) HER FOLD icin assert_no_leakage zorunlu.

Per-class oran (notr_test_boost): notr sinifi icin test orani %20, diger
siniflar icin %10. Mevcut nihai modelle ayni protokol.

Egitim, train_optimized_v3.run_one(cfg, external_splits=...) cagirisi ile
mevcut pipeline'a delege edilir. Boylece tokenizer, loss, metrics, raporlama,
hizdan tum altyapi yeniden kullanilir.

Kosma (BERTurk, 5 fold x 3 seed):
    python src/cross_validation.py \
        --csv data/processed/final_genisletilmis_dataset_optimized_v3_v2.csv \
        --run_name v3_genisletilmis_5fold_berturk \
        --model_name dbmdz/bert-base-turkish-cased \
        --seeds 42,43,44 \
        --n_splits 5

Dry-run (1 fold x 1 epoch sanity check):
    python src/cross_validation.py \
        --csv data/processed/final_genisletilmis_dataset_optimized_v3_v2.csv \
        --run_name v3_dryrun_5fold_berturk \
        --seeds 42 \
        --n_splits 5 \
        --dry_run \
        --epochs 1 \
        --max_folds 1
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

# Mevcut pipeline'i yeniden kullan
sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_optimized_v3 import (  # noqa: E402
    ExpConfig, LABELS, LABEL2ID, run_one,
)


# =============================================================================
# Fold split: AI-aware
# =============================================================================

def make_ai_aware_folds(
    df: pd.DataFrame,
    label_col: str,
    source_col: str,
    n_splits: int,
    seed: int,
    val_size: float,
    notr_test_boost: bool,
    notr_test_ratio: float,
    notr_val_ratio: float,
) -> list[tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]]:
    """
    Returns: list of (train_df, val_df, test_df) tuples, len == n_splits.

    Mantik:
        - AI'lari ayir (her fold'da train'e eklenecek).
        - Gercek df uzerinde StratifiedKFold(n_splits, shuffle=True, random_state=seed).
        - Her fold icin gercek_test_idx -> test_df. Geri kalan gercek -> train+val.
        - Train'den val ayrilir (label-stratified).
        - AI'lar train'e eklenir.
        - notr_test_boost: nötr siniflari icin test orani %20 hedefli; ancak
          n_splits=5 zaten 1/5 = %20 verir. notr_test_boost aktifken pozitif/
          negatif test orani da %20 olur (CV bunu degistirmez). Hocanin temel
          istegi 'kararlilik kanit' oldugu icin n_splits=5 ile %20-test orani
          dogrudan yeterlidir; bu nedenle notr_test_boost CV'de SADECE val
          oranini etkiler (notr_val_ratio).
    """
    is_ai = df[source_col].astype(str).str.strip().str.lower().eq("ai_generated")
    ai_df = df[is_ai].copy().reset_index(drop=True)
    real_df = df[~is_ai].copy().reset_index(drop=True)

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)

    rng = np.random.RandomState(seed)
    folds = []
    for fold_idx, (tv_idx, te_idx) in enumerate(skf.split(real_df, real_df[label_col])):
        test_df = real_df.iloc[te_idx].copy().reset_index(drop=True)
        tv_df = real_df.iloc[tv_idx].copy().reset_index(drop=True)

        # Train'den val ayir (label-stratified, sinif basina rastgele ornekleme).
        train_parts, val_parts = [], []
        for lbl in LABELS:
            sub = tv_df[tv_df[label_col] == lbl].copy()
            n = len(sub)
            if n == 0:
                continue
            if notr_test_boost and lbl == "nötr":
                v_ratio = notr_val_ratio
            else:
                v_ratio = val_size
            n_val = int(round(n * v_ratio))
            idx = np.arange(n)
            rng.shuffle(idx)
            sub_shuf = sub.iloc[idx].reset_index(drop=True)
            val_parts.append(sub_shuf.iloc[:n_val])
            train_parts.append(sub_shuf.iloc[n_val:])

        train_df = pd.concat(train_parts, ignore_index=True)
        val_df = pd.concat(val_parts, ignore_index=True)

        # AI'lari train'e ekle
        if len(ai_df) > 0:
            train_df = pd.concat([train_df, ai_df], ignore_index=True)

        # Karistir
        train_df = train_df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
        val_df = val_df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
        test_df = test_df.sample(frac=1.0, random_state=seed).reset_index(drop=True)

        folds.append((train_df, val_df, test_df))

    return folds


# =============================================================================
# Ana akis
# =============================================================================

def run_cv(args) -> int:
    csv_path = Path(args.csv).resolve()
    if not csv_path.exists():
        print(f"[HATA] CSV bulunamadi: {csv_path}", file=sys.stderr)
        return 1

    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
    n_splits = int(args.n_splits)
    max_folds = int(args.max_folds) if args.max_folds is not None else n_splits

    base_outdir = Path(args.base_outdir) / args.run_name
    base_outdir.mkdir(parents=True, exist_ok=True)

    # CSV'yi tek sefer oku (hizli sanity check icin); fold split icin kullanilir.
    df = pd.read_csv(csv_path)
    print(f"[OKU] {csv_path.name}  n={len(df)}")
    print(f"[OKU] source dist: {df['source'].value_counts().to_dict()}")
    print(f"[OKU] label  dist: {df['sentiment_label'].value_counts().to_dict()}")

    # Real sayisi ve AI sayisi loga
    is_ai = df["source"].astype(str).str.strip().str.lower().eq("ai_generated")
    n_ai = int(is_ai.sum())
    n_real = int((~is_ai).sum())
    print(f"[OKU] real(non-AI)={n_real}  AI={n_ai}")

    all_results = []
    t_overall_start = time.time()

    for seed in seeds:
        print(f"\n{'#' * 70}")
        print(f"# SEED = {seed}  (n_splits={n_splits}, max_folds={max_folds})")
        print(f"{'#' * 70}")

        # Bu seed icin fold listesi
        folds = make_ai_aware_folds(
            df, label_col="sentiment_label", source_col="source",
            n_splits=n_splits, seed=seed, val_size=args.val_size,
            notr_test_boost=args.notr_test_boost,
            notr_test_ratio=args.notr_test_ratio,
            notr_val_ratio=args.notr_val_ratio,
        )

        for fold_i, (train_df, val_df, test_df) in enumerate(folds):
            if fold_i >= max_folds:
                print(f"[CV] max_folds={max_folds} ulasildi, kalan fold'lar atlanir.")
                break

            run_name_fold = f"seed{seed}_fold{fold_i + 1}"

            # Sanity print
            tr_dist = train_df["sentiment_label"].value_counts().to_dict()
            va_dist = val_df["sentiment_label"].value_counts().to_dict()
            te_dist = test_df["sentiment_label"].value_counts().to_dict()
            ai_in_test = int((test_df["source"] == "ai_generated").sum())
            ai_in_val = int((val_df["source"] == "ai_generated").sum())
            print(f"\n--- FOLD {fold_i + 1}/{n_splits} (seed {seed}) ---")
            print(f"  train n={len(train_df)} dist={tr_dist}")
            print(f"  val   n={len(val_df)}  dist={va_dist}")
            print(f"  test  n={len(test_df)} dist={te_dist}")
            print(f"  AI in val={ai_in_val}  AI in test={ai_in_test}  (her ikisi 0 olmali)")

            if ai_in_test != 0 or ai_in_val != 0:
                print("[HATA] AI ornek val veya test'te! Protokol bozuldu.", file=sys.stderr)
                return 2

            # Cfg hazirla
            cfg = ExpConfig(
                csv_path=str(csv_path),  # bilgi amacli (run_one external_splits ile bypass)
                model_name=args.model_name,
                max_len=args.max_len,
                lr=args.lr,
                epochs=args.epochs,
                train_bs=args.train_bs,
                eval_bs=args.eval_bs,
                loss_type=args.loss_type,
                ai_only_in_train=True,  # savunma amacli; folds zaten AI-aware
                source_stratified=False,
                notr_test_boost=False,  # CV icin internal split kapali
                seed=seed,
                deterministic=True,
                base_outdir=str(base_outdir),
                run_name=run_name_fold,
                early_stop=(not args.no_early_stop),
                early_patience=args.early_patience,
            )

            t_fold_start = time.time()
            try:
                report = run_one(cfg, external_splits=(train_df, val_df, test_df))
            except Exception as e:
                print(f"[HATA] fold {fold_i + 1} seed {seed}: {e}", file=sys.stderr)
                raise
            elapsed_fold = time.time() - t_fold_start

            tm = report["test_metrics"]
            row = {
                "seed": seed,
                "fold": fold_i + 1,
                "accuracy": tm["accuracy"],
                "f1_macro": tm["f1_macro"],
                "f1_weighted": tm["f1_weighted"],
                "f1_negatif": tm["f1_per_class"]["negatif"],
                "f1_notr": tm["f1_per_class"]["nötr"],
                "f1_pozitif": tm["f1_per_class"]["pozitif"],
                "elapsed_sec": elapsed_fold,
                "out_dir": report["out_dir"],
                "train_n": len(train_df),
                "val_n": len(val_df),
                "test_n": len(test_df),
            }
            all_results.append(row)
            print(f"  [DONE] f1_macro={tm['f1_macro']:.4f}  acc={tm['accuracy']:.4f}  "
                  f"elapsed={elapsed_fold:.0f}s")

    # Ozet
    summary_df = pd.DataFrame(all_results)
    summary_path = base_outdir / "summary_cv.csv"
    summary_df.to_csv(summary_path, index=False, encoding="utf-8")

    # Aggregat istatistikler
    metric_cols = ["accuracy", "f1_macro", "f1_weighted", "f1_negatif", "f1_notr", "f1_pozitif"]
    agg = {
        "n_folds_total": int(len(summary_df)),
        "by_seed": {},
        "overall_mean": {c: float(summary_df[c].mean()) for c in metric_cols},
        "overall_std": {c: float(summary_df[c].std(ddof=1)) if len(summary_df) > 1 else 0.0 for c in metric_cols},
        "elapsed_sec_total": float(time.time() - t_overall_start),
    }
    for sd in summary_df["seed"].unique():
        sub = summary_df[summary_df["seed"] == sd]
        agg["by_seed"][int(sd)] = {
            "n_folds": int(len(sub)),
            "mean": {c: float(sub[c].mean()) for c in metric_cols},
            "std": {c: float(sub[c].std(ddof=1)) if len(sub) > 1 else 0.0 for c in metric_cols},
        }

    (base_outdir / "summary_cv.json").write_text(
        json.dumps(agg, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print("\n" + "=" * 70)
    print("CV SUMMARY")
    print("=" * 70)
    print(f"Toplam fold sayisi : {agg['n_folds_total']}")
    print(f"Toplam sure        : {agg['elapsed_sec_total']:.0f}s")
    print("Genel ortalama (mean +/- std):")
    for c in metric_cols:
        print(f"  {c:14s}: {agg['overall_mean'][c]:.4f} +/- {agg['overall_std'][c]:.4f}")
    print(f"\nCSV : {summary_path}")
    print(f"JSON: {base_outdir / 'summary_cv.json'}")
    return 0


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="AI-aware stratified k-fold CV")
    p.add_argument("--csv", type=str,
                   default="data/processed/final_genisletilmis_dataset_optimized_v3_v2.csv")
    p.add_argument("--run_name", type=str, default="v3_genisletilmis_5fold")
    p.add_argument("--model_name", type=str, default="dbmdz/bert-base-turkish-cased")
    p.add_argument("--seeds", type=str, default="42,43,44")
    p.add_argument("--n_splits", type=int, default=5)
    p.add_argument("--max_folds", type=int, default=None,
                   help="Sadece ilk K fold'u kos (debug/dry-run icin).")
    p.add_argument("--val_size", type=float, default=0.10)
    p.add_argument("--notr_test_boost", action="store_true",
                   help="CV'de notr_test_boost --val orani-- icin (test orani 1/n_splits zaten).")
    p.add_argument("--notr_test_ratio", type=float, default=0.20)
    p.add_argument("--notr_val_ratio", type=float, default=0.10)
    p.add_argument("--max_len", type=int, default=256)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--train_bs", type=int, default=16)
    p.add_argument("--eval_bs", type=int, default=32)
    p.add_argument("--loss_type", type=str, default="wce",
                   choices=["ce", "wce", "wce_smooth", "focal"])
    p.add_argument("--no_early_stop", action="store_true")
    p.add_argument("--early_patience", type=int, default=3)
    p.add_argument("--base_outdir", type=str, default="models/cv")
    p.add_argument("--dry_run", action="store_true",
                   help="Hizli kontrol: --epochs 1 --max_folds 1 esdegeri.")
    return p


def main():
    args = build_argparser().parse_args()
    if args.dry_run:
        print("[DRY-RUN] epochs=1, max_folds=1 zorlanir.")
        args.epochs = 1
        args.max_folds = 1
    return run_cv(args)


if __name__ == "__main__":
    sys.exit(main())
