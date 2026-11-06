# -*- coding: utf-8 -*-
"""Sinif-ici korelasyon dogrulama (Pearson r yorumumuzu sinama)."""
import pandas as pd

df = pd.read_csv('data/processed/final_genisletilmis_dataset_optimized_v3_v2.csv')

POZ = ['harika','muhteşem','muhtesem','mükemmel','mukemmel','süper','super',
       'tavsiye','memnun','kaliteli','guzel','güzel','temiz','rahat',
       'hizmet','ilgili','lezzetli','keyifli','begendi','beğendi']
NEG = ['berbat','rezalet','iğrenç','igrenc','kötü','kotu','pis','kirli',
       'sorun','şikayet','sikayet','kalitesiz','yavaş','yavas','uzak',
       'gürültü','gurultu','soğuk','soguk','boş','bos']

def kw(s, keys):
    if not isinstance(s, str):
        return 0
    low = s.lower()
    norm = (low.replace('ı','i').replace('ş','s').replace('ç','c')
                .replace('ğ','g').replace('ö','o').replace('ü','u'))
    return sum(1 for k in keys if k in low or k in norm)

df['pos_kw'] = df['comment_text'].map(lambda s: kw(s, POZ))
df['neg_kw'] = df['comment_text'].map(lambda s: kw(s, NEG))

print('=' * 70)
print('TUM VERI (n=14337):')
print(f'  pos_kw mean: {df["pos_kw"].mean():.3f}   neg_kw mean: {df["neg_kw"].mean():.3f}')
print(f'  Pearson r (pos_kw, neg_kw): {df[["pos_kw","neg_kw"]].corr().iloc[0,1]:.4f}')
print()

print('=' * 70)
print('SINIF ICI KORELASYONLAR:')
print('=' * 70)
for lbl in ['negatif', 'nötr', 'pozitif']:
    sub = df[df['sentiment_label'] == lbl]
    r = sub[['pos_kw', 'neg_kw']].corr().iloc[0, 1]
    pos_mean = sub['pos_kw'].mean()
    neg_mean = sub['neg_kw'].mean()
    print(f'{lbl} (n={len(sub)}):')
    print(f'  pos_kw mean: {pos_mean:.3f}   neg_kw mean: {neg_mean:.3f}')
    print(f'  Pearson r (sinif ici): {r:.4f}')
    print()

print('=' * 70)
print('NOTR ALT-KAYNAKLARI (AI haric, gercek):')
print('=' * 70)
notr_real = df[(df['sentiment_label'] == 'nötr') & (df['source'] != 'ai_generated')]
notr_ai = df[(df['sentiment_label'] == 'nötr') & (df['source'] == 'ai_generated')]
notr_manuel = df[df['source'] == 'manuel_toplama']
notr_otelpuan = df[(df['sentiment_label'] == 'nötr') & (df['source'] == 'otelpuan_otelyorum')]

for name, sub in [('notr (TUM AI haric)', notr_real),
                   ('notr (sadece AI)', notr_ai),
                   ('notr (sadece manuel_toplama)', notr_manuel),
                   ('notr (sadece otelpuan_otelyorum)', notr_otelpuan)]:
    if len(sub) > 1:
        r = sub[['pos_kw', 'neg_kw']].corr().iloc[0, 1]
        print(f'{name} (n={len(sub)}):')
        print(f'  pos_kw mean: {sub["pos_kw"].mean():.3f}   neg_kw mean: {sub["neg_kw"].mean():.3f}')
        print(f'  Pearson r: {r:.4f}')
        print()

print('=' * 70)
print('is_ai vs sentiment kategori cross-tab:')
print('=' * 70)
df['is_ai'] = (df['source'] == 'ai_generated').astype(int)
ct = pd.crosstab(df['is_ai'], df['sentiment_label'])
print(ct)
print()
print('Yorum: AI olan satirlarin %100 nötr; bu bilgi point-biserial veya')
print('kategorik korelasyon ile dogru yansir, Pearson(is_ai, sentiment_id={-1,0,+1})')
print('ile yansiyorsa NEAR-ZERO olur cunku ortalamalar denkler.')
print()
df['is_notr'] = (df['sentiment_label'] == 'nötr').astype(int)
print(f'Pearson r (is_ai, is_notr): {df[["is_ai","is_notr"]].corr().iloc[0,1]:.4f}')
print('  (bu deger AI<->notr deterministik iliskinin gercek olcusudur)')
