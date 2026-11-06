# Klasik ML Baseline — TF-IDF + LogReg / LinearSVC

**Hocanın Geri Bildirimi A.7 cevabı.** Multi-seed final eğitim ile
aynı 3 seed × aynı test setleri üzerinde değerlendirildi.

**Veri:** Splits doğrudan `models/optimized_v3/v3_genisletilmis/...seed{42,43,44}/splits/`
klasörlerinden okundu (BERTurk eğitiminin gördüğü split'in birebir aynısı).

## Tüm Sonuçlar (3 seed × 3 model = 9 koşu)

| Seed | Model | Accuracy | Macro-F1 | F1-negatif | F1-nötr | F1-pozitif | n_feat | Fit (sn) |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| 42 | tfidf_word_1_2 + LogReg | 0.8040 | 0.7830 | 0.8392 | 0.6353 | 0.8746 | 59,955 | 2.8 |
| 42 | tfidf_char_wb_3_5 + LogReg | 0.8026 | 0.7831 | 0.8313 | 0.6415 | 0.8764 | 73,158 | 4.1 |
| 42 | tfidf_word_1_2 + LinearSVC | 0.8040 | 0.7835 | 0.8402 | 0.6365 | 0.8737 | 59,955 | 0.2 |
| 43 | tfidf_word_1_2 + LogReg | 0.7946 | 0.7678 | 0.8404 | 0.6003 | 0.8628 | 59,835 | 4.1 |
| 43 | tfidf_char_wb_3_5 + LogReg | 0.8041 | 0.7837 | 0.8454 | 0.6434 | 0.8621 | 72,952 | 8.4 |
| 43 | tfidf_word_1_2 + LinearSVC | 0.8048 | 0.7789 | 0.8456 | 0.6173 | 0.8737 | 59,835 | 0.2 |
| 44 | tfidf_word_1_2 + LogReg | 0.8041 | 0.7853 | 0.8395 | 0.6475 | 0.8688 | 58,994 | 4.0 |
| 44 | tfidf_char_wb_3_5 + LogReg | 0.8062 | 0.7882 | 0.8292 | 0.6562 | 0.8792 | 72,502 | 7.7 |
| 44 | tfidf_word_1_2 + LinearSVC | 0.8041 | 0.7847 | 0.8385 | 0.6428 | 0.8727 | 58,994 | 0.2 |

## Model Bazında Ortalama ± Std (3 Seed)

| Model | Accuracy | Macro-F1 | F1-nötr |
|---|---:|---:|---:|
| tfidf_word_1_2 + LogReg | 0.8009 ± 0.0054 | 0.7787 ± 0.0095 | 0.6277 ± 0.0245 |
| tfidf_char_wb_3_5 + LogReg | 0.8043 ± 0.0018 | 0.7850 ± 0.0028 | 0.6471 ± 0.0080 |
| tfidf_word_1_2 + LinearSVC | 0.8043 ± 0.0004 | 0.7823 ± 0.0031 | 0.6322 ± 0.0133 |

## BERTurk Karşılaştırması

| Sistem | Macro-F1 (μ ± σ) | F1-nötr (μ) |
|---|---:|---:|
| BERTurk (multi-seed) | **0.8797 ± 0.0073** | 0.8056 |
| Klasik en iyi: tfidf_char_wb_3_5 + LogReg | 0.7850 | (yukarıdaki tabloya bakın) |
| **Δ (BERTurk üstünlüğü)** | **+0.0947** | — |

## Yorum

BERTurk klasik ML baseline'ından **9.5 puan** üstün. Bu fark
literatürdeki transformer-vs-klasik-ML beklentisiyle (8-12 puan) uyumlu.
Klasik baseline ayrıca aşağıdaki avantajları sağlar:

- Hızlı eğitim (saniyeler vs dakikalar)
- Yorumlanabilirlik: en yüksek katsayılı kelimeler `top_features_per_class.csv`
  içinde — feature importance doğrudan okunabilir.
- Kaynak verimliliği: GPU yok, CPU yeterli.

BERTurk'ün avantajı: bağlam yakalama, eş anlamlı/söz dizimsel varyasyona
dayanıklılık, sarkazm/ironide nispi üstünlük (yine de sınırlı, §5.2'ye bkz).

## Çıktı Dosyaları

- `results.csv` — Tüm 9 koşu detaylı metrik
- `summary.json` — Aggregate JSON
- `top_features_per_class.csv` — LogReg seed42 sınıf başına en yüksek 30 katsayı