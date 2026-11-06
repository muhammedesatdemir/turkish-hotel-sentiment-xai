# -*- coding: utf-8 -*-
"""Test seti dagilim farki check (multi-seed vs CV)."""
import pandas as pd

print('=' * 60)
print('MULTI-SEED v3_genisletilmis test seti dagilimi')
print('=' * 60)
for sd in [42, 43, 44]:
    path = f'models/optimized_v3/v3_genisletilmis/bert-base-turkish-cased_len256_ep5_lr2e-05_losswce_seed{sd}/test_predictions.csv'
    df = pd.read_csv(path)
    dist = df['sentiment_label'].value_counts().to_dict()
    pct = {k: f"{v/len(df)*100:.1f}%" for k, v in dist.items()}
    print(f'seed {sd}: total={len(df)}, dist={dist}')
    print(f'         pct={pct}')

print()
print('=' * 60)
print('CV v3_genisletilmis_5fold_berturk seed42 fold1-5 test dagilimi')
print('=' * 60)
for fold in range(1, 6):
    path = f'models/cv/v3_genisletilmis_5fold_berturk/seed42_fold{fold}/bert-base-turkish-cased_len256_ep5_lr2e-05_losswce_seed42/test_predictions.csv'
    df = pd.read_csv(path)
    dist = df['sentiment_label'].value_counts().to_dict()
    pct = {k: f"{v/len(df)*100:.1f}%" for k, v in dist.items()}
    print(f'fold {fold}: total={len(df)}, dist={dist}')
    print(f'         pct={pct}')
