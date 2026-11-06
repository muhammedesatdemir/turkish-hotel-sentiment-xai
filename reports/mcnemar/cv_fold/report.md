# CV-Fold McNemar Testi — BERTurk-CV vs ELECTRA-CV

**Karşılaştırma:** Her seed × fold için BERTurk ve ELECTRA aynı test setinde değerlendirildi.
**N test:** 15 (= 3 seed × 5 fold)
**Hipotez (H₀):** İki model aynı hata dağılımına sahip (kararlar ayırt edilemez).

## Tüm 15 Fold Sonuçları

| Seed | Fold | n | BERT doğru | ELECTRA doğru | b (sadece BERT) | c (sadece ELECTRA) | χ² | p-value | Yöntem |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| 42 | 1 | 2368 | 2135 | 2120 | 82 | 67 | 1.315 | 0.2514 | chi2_corrected |
| 42 | 2 | 2368 | 2164 | 2167 | 58 | 61 | 0.034 | 0.8545 | chi2_corrected |
| 42 | 3 | 2367 | 2143 | 2165 | 62 | 84 | 3.021 | 0.0822 | chi2_corrected |
| 42 | 4 | 2367 | 2135 | 2145 | 63 | 73 | 0.596 | 0.4403 | chi2_corrected |
| 42 | 5 | 2367 | 2141 | 2139 | 60 | 58 | 0.008 | 0.9267 | chi2_corrected |
| 43 | 1 | 2368 | 2153 | 2141 | 79 | 67 | 0.829 | 0.3626 | chi2_corrected |
| 43 | 2 | 2368 | 2132 | 2154 | 59 | 81 | 3.150 | 0.0759 | chi2_corrected |
| 43 | 3 | 2367 | 2145 | 2157 | 57 | 69 | 0.960 | 0.3271 | chi2_corrected |
| 43 | 4 | 2367 | 2153 | 2163 | 53 | 63 | 0.698 | 0.4034 | chi2_corrected |
| 43 | 5 | 2367 | 2120 | 2134 | 64 | 78 | 1.190 | 0.2753 | chi2_corrected |
| 44 | 1 | 2368 | 2118 | 2135 | 65 | 82 | 1.741 | 0.1869 | chi2_corrected |
| 44 | 2 | 2368 | 2126 | 2143 | 66 | 83 | 1.718 | 0.1899 | chi2_corrected |
| 44 | 3 | 2367 | 2152 | 2145 | 66 | 59 | 0.288 | 0.5915 | chi2_corrected |
| 44 | 4 | 2367 | 2134 | 2136 | 75 | 77 | 0.007 | 0.9354 | chi2_corrected |
| 44 | 5 | 2367 | 2133 | 2137 | 69 | 73 | 0.063 | 0.8012 | chi2_corrected |

## Özet İstatistikler

- **Min p-value:** 0.0759
- **Median p-value:** 0.3626
- **Max p-value:** 0.9354
- **Anlamlı (p<0,05, düzeltmesiz):** 0 / 15
- **Bonferroni eşik (0,05 / 15):** 0.00333
- **Bonferroni-corrected anlamlı:** 0 / 15

## Fisher's Combined Probability Test

15 bağımsız McNemar p-değerini tek bir meta-istatistik altında birleştirme:
- χ²_combined = −2 × Σ ln(p_i) = **31.984**
- df = 2 × n = **30**
- **p_combined = 0.3683**

> **Fisher meta-analizi sonucu:** p_combined = 0.3683 > 0,05 → BERTurk ve ELECTRA arasında **CV ortamında istatistiksel olarak anlamlı sistematik fark YOK**.

## Akademik Yorum

Bonferroni-corrected eşik 0.00333 altında 0 fold anlamlı çıkmıştır. Düzeltmesiz p<0,05 eşiğinde ise 0 fold anlamlı görünmektedir; ancak çoklu test düzeltmesi sonrası bu sayı düşer/sıfırlanır.

Multi-model ablasyondaki McNemar bulgusu (BERTurk-ELECTRA Bonferroni anlamsız, tek-split test seti üzerinde 3 seed) **CV ortamında 15 fold üzerinden bağımsız olarak doğrulanmıştır**: iki model arasındaki performans farkı seed/fold varyasyonu altında istatistiksel olarak ayırt edilemez. ELECTRA'nın nominal olarak ortalama 0,003 puanlık üstünlüğü σ_CV ≈ 0,007 ile aynı düzeydedir → **gürültü**.

## Reproducibility

```bash
python src/mcnemar_cv_test.py
```

**Çıktı:** `reports/mcnemar/cv_fold/{mcnemar_cv_results.csv, mcnemar_cv_summary.json, report.md}`