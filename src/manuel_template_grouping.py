# -*- coding: utf-8 -*-
"""
manuel_toplama veri setinde 150 doğal yorumun şablon-augmentasyon ile 5x
çoğaltılmış grupları tespit etmek için clustering analizi.

Yaklaşım:
- Manuel_toplama 751 yorum
- Her yorumdan TF-IDF vektör çıkar
- Cosine similarity ile yakın komşuları bul
- Single-linkage clustering ile gruplara ayır (eşik = 0.5 cosine similarity)
- Her grubu inceleyip ortalama büyüklüğü kontrol et (~5 olması beklenir)
- Çıktı: her satıra group_id ata
"""
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict, Counter

# Yukle
df = pd.read_csv('data/processed/final_genisletilmis_dataset_optimized_v3_v2.csv')
manuel = df[df['source'] == 'manuel_toplama'].copy().reset_index(drop=True)
print(f'Manuel_toplama: {len(manuel)} satir')

# TF-IDF
texts = manuel['comment_text'].astype(str).tolist()
vec = TfidfVectorizer(
    max_features=5000,
    ngram_range=(1, 2),
    lowercase=True,
    min_df=2,
)
X = vec.fit_transform(texts)
print(f'TF-IDF shape: {X.shape}')

# Cosine similarity matrisi (sparse, simetrik)
print('Computing cosine similarity matrix...')
sim = cosine_similarity(X)
print(f'Sim matrix shape: {sim.shape}')

# Self-sim diagonali sıfırla
np.fill_diagonal(sim, 0)

# Esik hesapla: kaç komşu eşiği geçiyor (yüksek benzerlik = aynı grup)
THRESHOLDS = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
print('\nEşik tarama (her satırın yüksek-benzerlik komşu sayısı dağılımı):')
for thr in THRESHOLDS:
    neighbors = (sim >= thr).sum(axis=1)
    avg_neighbors = neighbors.mean()
    has_at_least_4 = (neighbors >= 4).sum()  # 5'li grup demek 4 komşu var
    print(f'  thr={thr}: ortalama komşu={avg_neighbors:.2f}, '
          f'≥4 komşusu olan satır={has_at_least_4} (5+li grup üyesi)')

# Single-linkage clustering: eşiği geçen kenarlar üzerinden Union-Find
def union_find_clustering(sim_matrix, threshold):
    n = sim_matrix.shape[0]
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x, y):
        px, py = find(x), find(y)
        if px != py:
            parent[px] = py

    # Esiği geçen tüm çiftleri birleştir
    for i in range(n):
        for j in range(i + 1, n):
            if sim_matrix[i, j] >= threshold:
                union(i, j)

    # Her satıra grup id'si ver
    groups = [find(i) for i in range(n)]
    return np.array(groups)

# 0.5 eşiğiyle dene
print('\n=== Clustering dene: threshold=0.5 ===')
groups_05 = union_find_clustering(sim, 0.5)
group_counts = Counter(groups_05)
print(f'Toplam grup: {len(group_counts)}')
print(f'Grup boyutu dağılımı: {Counter(group_counts.values())}')

# 0.4
print('\n=== Clustering dene: threshold=0.4 ===')
groups_04 = union_find_clustering(sim, 0.4)
gc04 = Counter(groups_04)
print(f'Toplam grup: {len(gc04)}')
print(f'Grup boyutu dağılımı: {Counter(gc04.values())}')

# 0.6
print('\n=== Clustering dene: threshold=0.6 ===')
groups_06 = union_find_clustering(sim, 0.6)
gc06 = Counter(groups_06)
print(f'Toplam grup: {len(gc06)}')
print(f'Grup boyutu dağılımı: {Counter(gc06.values())}')

# Hedef: ~150 grup, her biri ~5 örnek
# Hangi eşik en yakın?
target_groups = 150
target_avg_size = 5.0
print('\n=== Hedef analizi: ~150 grup, ~5 ortalama ===')
for thr_name, gc in [('0.4', gc04), ('0.5', group_counts), ('0.6', gc06)]:
    n_groups = len(gc)
    avg = sum(s for s in gc.values()) / n_groups
    n_5plus = sum(1 for s in gc.values() if s >= 4)
    print(f'  thr={thr_name}: {n_groups} grup, ortalama boyut={avg:.2f}, ≥4 üyeli grup={n_5plus}')
