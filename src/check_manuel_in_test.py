# -*- coding: utf-8 -*-
"""Test setinde manuel_toplama'dan kac orneklem var?"""
import pandas as pd

# Multi-seed v3_genisletilmis test set - source bilgisi yok mu kontrol et
for sd in [42, 43, 44]:
    path = f'models/optimized_v3/v3_genisletilmis/bert-base-turkish-cased_len256_ep5_lr2e-05_losswce_seed{sd}/test_predictions.csv'
    df = pd.read_csv(path)
    print(f'=== Multi-seed seed {sd} test set ===')
    print(f'Columns: {list(df.columns)}')
    if 'source' in df.columns:
        src = df['source'].value_counts().to_dict()
        print(f'Source dist: {src}')
        # Manuel_toplama satirlari nasil dogru bilinir?
        manuel = df[df['source'] == 'manuel_toplama']
        if len(manuel) > 0:
            cor = manuel['correct'].sum() / len(manuel)
            print(f'manuel_toplama in test: n={len(manuel)}, correct rate={cor:.4f}')
            otelpuan = df[df['source'] == 'otelpuan_otelyorum']
            if len(otelpuan) > 0:
                cor_op = otelpuan[otelpuan['sentiment_label']=='nötr']['correct'].mean()
                print(f'otelpuan_otelyorum nötr in test: n={len(otelpuan[otelpuan["sentiment_label"]=="nötr"])}, correct rate={cor_op:.4f}')
        # Eski source isimleri olabilir (otelpuan.com)
        if 'otelpuan.com' in src or 'neredekal' in src:
            print('NOT: Eski source isimleri (otelpuan.com / neredekal) gozukuyor; v2 oncesi splitler.')
    print()
