# -*- coding: utf-8 -*-
"""
manuel_toplama recall %100 nereden geliyor? Decompose et:

1. Train'de yakın komşusu olan vs. olmayan test örneklerini ayır
2. Yakın komşusu olan: muhtemelen ezberleme
3. Yakın komşusu olmayan: stil/dagilim öğrenmesi

Bunu BERTurk seed42'de yapacağız.
"""
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

train = pd.read_csv('models/optimized_v3/v3_genisletilmis/bert-base-turkish-cased_len256_ep5_lr2e-05_losswce_seed42/splits/train.csv')
test_pred = pd.read_csv('models/optimized_v3/v3_genisletilmis/bert-base-turkish-cased_len256_ep5_lr2e-05_losswce_seed42/test_predictions.csv')

# Manuel_toplama'da train ve test
train_manuel = train[train['source'] == 'manuel_toplama'].reset_index(drop=True)
test_manuel = test_pred[test_pred['source'] == 'manuel_toplama'].reset_index(drop=True)
test_otelpuan_notr = test_pred[(test_pred['source'] == 'otelpuan.com') & (test_pred['sentiment_label'] == 'nötr')].reset_index(drop=True)

print(f'Train manuel: {len(train_manuel)}')
print(f'Test manuel: {len(test_manuel)}')
print(f'Test otelpuan nötr: {len(test_otelpuan_notr)}')

# TF-IDF: train_manuel vs test_manuel arası benzerlik
all_texts = list(train_manuel['comment_text']) + list(test_manuel['comment_text']) + list(test_otelpuan_notr['comment_text'])
vec = TfidfVectorizer(max_features=5000, ngram_range=(1, 2), lowercase=True, min_df=2)
X = vec.fit_transform(all_texts)

n_tm = len(train_manuel)
n_test_m = len(test_manuel)
n_test_o = len(test_otelpuan_notr)

# Test_manuel vs train_manuel
test_m_X = X[n_tm:n_tm + n_test_m]
train_m_X = X[:n_tm]
test_o_X = X[n_tm + n_test_m:]

sim_m_to_train = cosine_similarity(test_m_X, train_m_X)
sim_o_to_train = cosine_similarity(test_o_X, train_m_X)

# Her test_manuel örneği için en yüksek benzerlik
max_sim_m = sim_m_to_train.max(axis=1)
max_sim_o = sim_o_to_train.max(axis=1)

print()
print('Test manuel >> train_manuel EN YUKSEK cosine sim dagilimi:')
print(f'  median: {np.median(max_sim_m):.3f}')
print(f'  mean: {np.mean(max_sim_m):.3f}')
print(f'  q25/q75: {np.quantile(max_sim_m, 0.25):.3f} / {np.quantile(max_sim_m, 0.75):.3f}')
print(f'  >0.5: {(max_sim_m > 0.5).sum()} / {len(max_sim_m)}')
print(f'  >0.7: {(max_sim_m > 0.7).sum()} / {len(max_sim_m)}')
print(f'  >0.9: {(max_sim_m > 0.9).sum()} / {len(max_sim_m)}')

print()
print('Test otelpuan notr -> train_manuel EN YUKSEK sim:')
print(f'  median: {np.median(max_sim_o):.3f}')
print(f'  mean: {np.mean(max_sim_o):.3f}')
print(f'  >0.5: {(max_sim_o > 0.5).sum()} / {len(max_sim_o)}')

# Recall'ı sim eşiğine göre ayır
print()
print('=== RECALL DEKOMPOZISYON ===')
print()
print('Test_manuel:')
for thr in [0.0, 0.3, 0.5, 0.7]:
    near = max_sim_m >= thr
    if near.sum() > 0:
        rec = test_manuel.loc[near, 'correct'].mean()
        print(f'  sim >= {thr}: n={near.sum()}, recall={rec:.4f}')
    far = max_sim_m < thr
    if far.sum() > 0:
        rec = test_manuel.loc[far, 'correct'].mean()
        print(f'  sim <  {thr}: n={far.sum()}, recall={rec:.4f}')
    print()

# Eğer max_sim_m'in median'ı düşükse (örn. <0.3) o zaman ezberleme değil stil
# Eğer yüksekse (>0.7) o zaman direct ezberleme
print()
print('=== KARAR ===')
if np.median(max_sim_m) > 0.7:
    print('SONUÇ: Test manuel orneklerinin train\'de yakın komşusu var → ŞABLON EZBERLEME')
elif np.median(max_sim_m) > 0.4:
    print('SONUÇ: Test manuel orneklerinin train\'de orta benzerlikte komşusu var → KARMA (hem stil hem aug)')
else:
    print('SONUÇ: Test manuel orneklerinin train\'de yakın komşusu YOK → STİL/DAGILIM ezberleme')
print(f'  Median similarity: {np.median(max_sim_m):.3f}')
