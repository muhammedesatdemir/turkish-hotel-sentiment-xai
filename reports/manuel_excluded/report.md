# Manuel-Toplama Hariç Test Metrikleri — Üç Senaryo Analizi
## 1. Yöntem

Multi-seed final eğitim sırasında üretilen `test_predictions.csv` dosyaları post-hoc filtreleme ile yeniden değerlendirilmiştir; modelin yeniden eğitilmesi **gerekmemiştir**. Filtreleme `source` sütunu üzerinden uygulanmıştır.

**Modeller:** BERTurk, ELECTRA-tr (her biri 3 seed: 42, 43, 44)
**Doğal kullanıcı tanımı:** `source ∈ {otelpuan.com, neredekal, otelpuan_otelyorum}` (manuel_toplama hariç).

## 2. Senaryo 1 vs Senaryo 2 — Genel Karşılaştırma

| Model | Senaryo | n_test | Accuracy | Macro-F1 | F1-nötr | F1-negatif | F1-pozitif |
|---|---|---:|---:|---:|---:|---:|---:|
| BERTurk | Manuel **dahil** | 1393 | 0.8857 ± 0.0066 | 0.8797 ± 0.0073 | 0.8056 ± 0.0112 | 0.9047 ± 0.0111 | 0.9286 ± 0.0026 |
| BERTurk | Manuel **hariç** | 1249 | 0.8724 ± 0.0080 | 0.8428 ± 0.0118 | 0.6950 ± 0.0233 | 0.9047 ± 0.0111 | 0.9286 ± 0.0026 |
| ELECTRA-tr | Manuel **dahil** | 1393 | 0.8906 ± 0.0130 | 0.8853 ± 0.0143 | 0.8143 ± 0.0243 | 0.9094 ± 0.0186 | 0.9321 ± 0.0016 |
| ELECTRA-tr | Manuel **hariç** | 1249 | 0.8780 ± 0.0140 | 0.8510 ± 0.0175 | 0.7116 ± 0.0331 | 0.9094 ± 0.0186 | 0.9321 ± 0.0016 |

### Senaryo Δ (Hariç − Dahil)

| Model | Δ Accuracy | Δ Macro-F1 | Δ F1-nötr |
|---|---:|---:|---:|
| BERTurk | -0.0132 | -0.0369 | -0.1106 |
| ELECTRA-tr | -0.0126 | -0.0342 | -0.1027 |

## 3. Senaryo 3 — Sadece Doğal Kullanıcı Nötr Sınıf Performansı

Manuel_toplama hariç tutulan ve sadece doğal kullanıcı yorumları üzerinden nötr sınıfı için precision / recall / F1 değerleri:

| Model | n_doğal | n_nötr_gerçek | Recall-nötr | Precision-nötr | F1-nötr |
|---|---:|---:|---:|---:|---:|
| BERTurk | 1249 | 281 | 0.6302 ± 0.0462 | 0.7770 ± 0.0166 | 0.6950 ± 0.0233 |
| ELECTRA-tr | 1249 | 281 | 0.6573 ± 0.0346 | 0.7757 ± 0.0314 | 0.7116 ± 0.0331 |

## 4. Yorumlama

**Senaryo 1 (manuel dahil) ↔ Senaryo 2 (manuel hariç) farkı:**

Manuel_toplama satırlarının (test setinde 142 satırın tamamı nötr etiketli) test setinden çıkarılması durumunda, F1-nötr değerinde belirgin bir düşüş gözlenmektedir. Bu düşüşün nedeni manuel_toplama örneklerinin yapı-temelli yazımı sayesinde modelin **karşıtlık bağlacı + karma duygu** sinyalini öğrenmiş olmasıdır. Doğal kullanıcı yorumlarında nötr sınıfı yapısal olarak daha gürültülü bir dağılıma sahiptir.

**Senaryo 3 (doğal kullanıcı nötr odak):**

Doğal kullanıcı yorumları üzerinde nötr sınıfı için elde edilen F1 değeri, rating-based etiketleme yönteminin yapısal sınırını yansıtır. Modelin nötr sınıfı için verdiği yüksek precision (yüksek hassasiyet) düşük recall ile birleştiğinde **konservatif tahmin davranışı** ortaya çıkar: model nötr olduğunu söylediğinde haklıdır, ancak gerçek nötrlerin önemli bir kısmını pozitif veya negatif olarak sınıflandırmaktadır. Bu, kullanıcının 3 yıldız vermiş olmasına rağmen yorum metninin objektif olarak pozitif veya negatif baskın olduğu durumlarda matematiksel olarak savunulabilir bir karardır.

## 5. Tekrarlanabilirlik

```bash
python src/manuel_excluded_metrics.py
```

Çıktılar:
- `reports/manuel_excluded/multi_seed_3_scenarios.csv` (model × seed × senaryo)
- `reports/manuel_excluded/multi_seed_3_scenarios_summary.csv` (aggregat)
- `reports/manuel_excluded/natural_notr_per_seed.csv`
- `reports/manuel_excluded/natural_notr_summary.csv`
- `reports/manuel_excluded/summary.json`