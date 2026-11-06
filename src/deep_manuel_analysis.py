# -*- coding: utf-8 -*-
"""Manuel_toplama derin analiz: ezberleme suphesi var mi?"""
import pandas as pd
from pathlib import Path

print('=' * 70)
print('MANUEL_TOPLAMA EZBERLEMESI ANALIZI')
print('=' * 70)

# Multi-seed v3_genisletilmis test predictions (3 seed)
for sd in [42, 43, 44]:
    print(f'\n--- Multi-seed seed {sd} ---')
    test = pd.read_csv(f'models/optimized_v3/v3_genisletilmis/bert-base-turkish-cased_len256_ep5_lr2e-05_losswce_seed{sd}/test_predictions.csv')
    train = pd.read_csv(f'models/optimized_v3/v3_genisletilmis/bert-base-turkish-cased_len256_ep5_lr2e-05_losswce_seed{sd}/splits/train.csv')

    # Test seti notr satirlari sources
    test_notr = test[test['sentiment_label'] == 'nötr']
    print(f'Test toplam: {len(test)}, test nötr: {len(test_notr)}')
    test_notr_src = test_notr['source'].value_counts().to_dict()
    print(f'Test nötr source dist: {test_notr_src}')

    # Per-source recall (notr siniftan)
    for src in test_notr['source'].unique():
        sub = test_notr[test_notr['source'] == src]
        rec = sub['correct'].mean()
        print(f'  {src} nötr recall: {rec:.4f} (n={len(sub)})')

    # Train'de manuel_toplama nasil dagilmis?
    train_manuel = train[train['source'] == 'manuel_toplama']
    print(f'Train manuel_toplama: {len(train_manuel)} satir')

    # Test'teki manuel_toplama yorumlarinin train'de yakini var mi?
    # Basit: ilk 5 kelime esitligi
    test_manuel = test[test['source'] == 'manuel_toplama']
    print(f'Test manuel_toplama: {len(test_manuel)} satir')

    if len(test_manuel) > 0 and len(train_manuel) > 0:
        train_first5 = train_manuel['comment_text'].astype(str).map(lambda s: ' '.join(s.lower().split()[:5])).tolist()
        train_first5_set = set(train_first5)

        match_count = 0
        for txt in test_manuel['comment_text']:
            f5 = ' '.join(str(txt).lower().split()[:5])
            if f5 in train_first5_set:
                match_count += 1
        print(f'Test manuel_toplama ilk-5-kelime match (train\'deki manuel_toplama ile): {match_count}/{len(test_manuel)} ({match_count/len(test_manuel)*100:.1f}%)')

# Manuel_toplama ic-ezberleme analizi
print()
print('=' * 70)
print('MANUEL_TOPLAMA IC-DUPLICATION SUPHESI')
print('=' * 70)
df = pd.read_csv('data/processed/final_genisletilmis_dataset_optimized_v3_v2.csv')
manuel = df[df['source'] == 'manuel_toplama'].copy()
manuel['first10'] = manuel['comment_text'].astype(str).map(lambda s: ' '.join(s.lower().split()[:10]))
print(f'Manuel_toplama toplam: {len(manuel)} satir')
print(f'Tekil ilk-10-kelime: {manuel["first10"].nunique()}')
print(f'Cogalan ilk-10-kelime gruplar (5+):')
top = manuel['first10'].value_counts().head(10)
for txt, cnt in top.items():
    if cnt >= 2:
        print(f'  ({cnt} kez): "{txt}..."')
