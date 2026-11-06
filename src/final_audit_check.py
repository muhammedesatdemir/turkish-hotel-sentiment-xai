# -*- coding: utf-8 -*-
"""Son kontrol — EDA ve McNemar'da gozden kacirmis olabilecegimiz baska
hatalar/sorunlar var mi?"""
import pandas as pd
import numpy as np
from pathlib import Path

print('=' * 70)
print('KONTROL 1: EDA korelasyon matrisi YENIDEN OKU')
print('=' * 70)
corr = pd.read_csv('reports/eda/corr_matrix.csv', index_col=0)

# Her |r| >= 0.5 olani listele
print('\n|r| >= 0.5 olan tum ciftler:')
for i, fa in enumerate(corr.columns):
    for j, fb in enumerate(corr.columns):
        if i < j:
            r = corr.iloc[i, j]
            if abs(r) >= 0.5:
                print(f'  {fa:25s} <-> {fb:25s}: r = {r:+.4f}')

# Beklenmedik korelasyonlar
print('\nBEKLENMEDIK olabilecek korelasyonlar (|r| 0.3-0.5):')
for i, fa in enumerate(corr.columns):
    for j, fb in enumerate(corr.columns):
        if i < j:
            r = corr.iloc[i, j]
            if 0.3 <= abs(r) < 0.5:
                print(f'  {fa:25s} <-> {fb:25s}: r = {r:+.4f}')

print()
print('=' * 70)
print('KONTROL 2: McNemar testlerinde tutarlilik kontrolu')
print('=' * 70)
mcn = pd.read_csv('reports/mcnemar/multi_model/mcnemar_results.csv')

# Test sample size tutarli mi (n her seedde ayni olmali)?
print('\nTest set boyutu (n = a+b+c+d) seedlerde:')
for seed in [42, 43, 44]:
    sub = mcn[mcn['seed'] == seed]
    ns = sub['n'].unique()
    print(f'  seed {seed}: n values = {ns}')

# 3 seed'in tahminlerinde tutarlilik
print('\nHer model cifti icin 3 seed p-value dagilimi:')
for (a, b), grp in mcn.groupby(['model_a', 'model_b']):
    ps = sorted(grp['p_value'].dropna().tolist())
    bs = grp['b'].tolist()
    cs = grp['c'].tolist()
    print(f'  {a} vs {b}: p-values = {ps}')
    print(f'    b values: {bs}, c values: {cs}')

print()
print('=' * 70)
print('KONTROL 3: Multi-model ablasyon test seti boyutlari')
print('=' * 70)

# Her seed'in test set boyutunu kontrol et (4 model icin ayni olmali)
import json
for seed in [42, 43, 44]:
    print(f'\nSeed {seed}:')
    paths = {
        'BERTurk': f'models/optimized_v3/v3_genisletilmis/bert-base-turkish-cased_len256_ep5_lr2e-05_losswce_seed{seed}/run_report.json',
        'ELECTRA': f'models/optimized_v3/v3_electra_tr/electra-base-turkish-cased-discriminator_len256_ep5_lr2e-05_losswce_seed{seed}/run_report.json',
        'mBERT': f'models/optimized_v3/v3_mbert/bert-base-multilingual-cased_len256_ep5_lr2e-05_losswce_seed{seed}/run_report.json',
        'XLM-R': f'models/optimized_v3/v3_xlmr/xlm-roberta-base_len256_ep5_lr2e-05_losswce_seed{seed}/run_report.json',
    }
    for model, p in paths.items():
        if Path(p).exists():
            d = json.loads(Path(p).read_text(encoding='utf-8'))
            ts = d.get('split_sizes', {}).get('test', '?')
            tdist = d.get('split_class_dist', {}).get('test', {})
            test_acc = d.get('test_metrics', {}).get('accuracy', '?')
            test_f1m = d.get('test_metrics', {}).get('f1_macro', '?')
            print(f'  {model:10s}: test n={ts}, dist={tdist}, acc={test_acc:.4f}, f1m={test_f1m:.4f}')

print()
print('=' * 70)
print('KONTROL 4: Sentiment label tutarliligi (test setinde)')
print('=' * 70)
# Multi-model 4 modelin test seti AYNI mi (text bazinda)?
texts_per_model = {}
for seed in [42]:
    for model_name, base in [
        ('BERTurk', f'models/optimized_v3/v3_genisletilmis/bert-base-turkish-cased_len256_ep5_lr2e-05_losswce_seed{seed}/test_predictions.csv'),
        ('ELECTRA', f'models/optimized_v3/v3_electra_tr/electra-base-turkish-cased-discriminator_len256_ep5_lr2e-05_losswce_seed{seed}/test_predictions.csv'),
        ('mBERT', f'models/optimized_v3/v3_mbert/bert-base-multilingual-cased_len256_ep5_lr2e-05_losswce_seed{seed}/test_predictions.csv'),
        ('XLM-R', f'models/optimized_v3/v3_xlmr/xlm-roberta-base_len256_ep5_lr2e-05_losswce_seed{seed}/test_predictions.csv'),
    ]:
        df = pd.read_csv(base)
        texts_per_model[model_name] = set(df['comment_text'].astype(str).str.strip().str.lower())

    print(f'Seed {seed}:')
    for m1 in texts_per_model:
        for m2 in texts_per_model:
            if m1 < m2:
                inter = len(texts_per_model[m1] & texts_per_model[m2])
                u = len(texts_per_model[m1])
                v = len(texts_per_model[m2])
                print(f'  {m1} vs {m2}: |A|={u}, |B|={v}, |A∩B|={inter}, McNemar gecerli mi: {inter == u == v}')

print()
print('=' * 70)
print('KONTROL 5: CV verisi tutarlilik')
print('=' * 70)
cv = pd.read_csv('models/cv/v3_genisletilmis_5fold_berturk/summary_cv.csv')
print(f'Toplam fold sayisi: {len(cv)}')
print(f'Beklenen: 15 (3 seed x 5 fold)')
# Her seed'de 5 fold olmali
for seed in [42, 43, 44]:
    n = len(cv[cv['seed'] == seed])
    print(f'  seed {seed}: {n} fold')

print()
print('=' * 70)
print('KONTROL 6: CV test seti sablon-leakage var mi?')
print('=' * 70)
# Bir fold'un train ve test setinde manuel_toplama var mi?
sample_path = 'models/cv/v3_genisletilmis_5fold_berturk/seed42_fold1/bert-base-turkish-cased_len256_ep5_lr2e-05_losswce_seed42'
train = pd.read_csv(f'{sample_path}/splits/train.csv')
test = pd.read_csv(f'{sample_path}/splits/test.csv')

print(f'CV fold1 train: n={len(train)}')
print(f'  source dist: {train["source"].value_counts().to_dict()}')
print(f'CV fold1 test: n={len(test)}')
print(f'  source dist: {test["source"].value_counts().to_dict()}')

# Manuel_toplama notr correct rate (predictions varsa)
test_pred = pd.read_csv(f'{sample_path}/test_predictions.csv')
if 'source' in test_pred.columns:
    for src in test_pred['source'].unique():
        sub = test_pred[(test_pred['source'] == src) & (test_pred['sentiment_label'] == 'nötr')]
        if len(sub) > 0:
            rec = sub['correct'].mean()
            print(f'  Test {src} nötr recall (CV fold1): {rec:.4f} (n={len(sub)})')
