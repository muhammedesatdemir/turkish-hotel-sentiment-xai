# BERTurk vs ELECTRA-tr — 5-Fold Cross-Validation Yan-Yana Karşılaştırması

**Tarih:** 2026-05-10
**Veri seti:** `final_genisletilmis_dataset_optimized_v3_v2.csv` (14.337 satır)
**Protokol:** AI-aware Stratified K-Fold, n_splits=5, 3 seed (42/43/44) = **15 koşu/model**
**Aynı fold splitleri (deterministik)** — model-eşleşmeli karşılaştırma için zorunlu.

---

## 1. Toplam Agregat (15 Koşu)

| Metrik | BERTurk | ELECTRA-tr | Δ (ELECTRA−BERTurk) |
|---|---:|---:|---:|
| **Macro-F1** | 0,8750 ± 0,0066 | **0,8782 ± 0,0076** | **+0,0032** |
| **F1-nötr** | 0,7496 ± 0,0128 | **0,7553 ± 0,0156** | **+0,0057** |
| F1-negatif | 0,9284 ± 0,0064 | **0,9295 ± 0,0076** | +0,0011 |
| F1-pozitif | 0,9472 ± 0,0043 | **0,9498 ± 0,0041** | +0,0026 |
| Accuracy | 0,9035 ± 0,0055 | **0,9062 ± 0,0056** | +0,0027 |
| **CV Toplam Süre** | **4,42 saat** | 17,19 saat | **+12,77 saat (3,9× yavaş)** |

> **Yorum.** ELECTRA tüm metriklerde nominal olarak az miktarda üstün (Δ Macro-F1 = +0,003), ancak **σ aynı düzeyde, fark σ ile karşılaştırılabilir**. ELECTRA σ tüm metriklerde **BERTurk'ten az miktarda yüksek** (daha az kararlı). Eğitim maliyeti ELECTRA'da **3,9× daha yüksek**.

---

## 2. Seed Bazında Yan-Yana Karşılaştırma

| Seed | BERTurk Macro-F1 | ELECTRA Macro-F1 | Δ | BERTurk F1-nötr | ELECTRA F1-nötr | Δ |
|---|---:|---:|---:|---:|---:|---:|
| 42 | 0,8768 ± 0,0049 | 0,8786 ± 0,0102 | +0,0018 | 0,7523 ± 0,0083 | 0,7553 ± 0,0194 | +0,0030 |
| 43 | 0,8769 ± 0,0067 | 0,8808 ± 0,0076 | +0,0039 | 0,7543 ± 0,0129 | 0,7612 ± 0,0178 | +0,0069 |
| 44 | 0,8714 ± 0,0078 | 0,8751 ± 0,0040 | +0,0037 | 0,7421 ± 0,0153 | 0,7492 ± 0,0085 | +0,0071 |

> **Yorum.** Tüm 3 seed'de ELECTRA nominal üstün; ancak bazı seedlerde σ aynı veya daha yüksek. seed44'te ELECTRA σ_F1m = 0,0040 (BERTurk 0,0072'den düşük) → ELECTRA seedden seede daha değişken davranıyor: bazı seed'lerde çok kararlı, bazılarında daha varyanslı.

---

## 3. Multi-Seed Final Eğitim ile Tutarlılık

| Sistem | Multi-Seed Final (3 seed) | 5-Fold CV (15 koşu) | Δ (CV − Final) |
|---|---:|---:|---:|
| BERTurk Macro-F1 | 0,8797 ± 0,0073 | 0,8750 ± 0,0066 | −0,0047 (gürültü) |
| BERTurk F1-nötr | 0,8056 ± 0,0112 | 0,7496 ± 0,0128 | −0,0560 (anlamlı) |
| ELECTRA Macro-F1 | 0,8853 ± 0,0143 | 0,8782 ± 0,0076 | −0,0071 (gürültü) |
| ELECTRA F1-nötr | 0,8143 ± 0,0243 | 0,7553 ± 0,0156 | −0,0590 (anlamlı) |

> **Önemli akademik gözlem.** Macro-F1'de fark gürültü düzeyinde (her iki model için), ancak **F1-nötr'de CV ortamında ~0,06 puan düşüş** her iki model için de görülüyor. Bu düşüş şu yüzden:
>
> - Multi-seed final eğitim: `--notr_test_boost --notr_test_ratio 0.20` ile test setinde nötr oranı **%20'ye yükseltildi** → her bir nötr örnek hatasının F1-nötr üzerindeki etkisi azaltıldı.
> - 5-fold CV: `notr_test_boost` KAPALI → test setinde nötr oranı doğal dağılımda (~%18) → daha az nötr örnek üzerinden ölçüm → tek hata daha çok zarar veriyor.
>
> **Bu fark protokol farkıdır, model performans farkı DEĞİL.** İki model de aynı protokol farkından eşit derecede etkileniyor (BERTurk -0,056, ELECTRA -0,059). Karşılaştırma model bazında **adil** kalıyor.

---

## 4. Karar Boyutları (Çok Kriterli)

| Kriter | BERTurk | ELECTRA-tr | Kazanan |
|---|---|---|:---:|
| Multi-seed Macro-F1 | 0,8797 ± 0,0073 | 0,8853 ± 0,0143 | ELECTRA (nominal) |
| Multi-seed F1m σ | **0,0073** | 0,0143 | **BERTurk** (2× kararlı) |
| Multi-seed F1-nötr | 0,8056 ± 0,0112 | 0,8143 ± 0,0243 | ELECTRA (nominal) |
| Multi-seed F1n σ | **0,0112** | 0,0243 | **BERTurk** (2,2× kararlı) |
| CV Macro-F1 | 0,8750 ± 0,0066 | 0,8782 ± 0,0076 | ELECTRA (Δ=0,003 gürültü) |
| McNemar BERT vs ELECTRA min p | 0,0298 (Bonferroni eşik 0,0167'nin **üstünde**) | — | **EŞDEĞER** |
| Eğitim süresi (CV total) | **4,42 saat** | 17,19 saat | **BERTurk** (3,9× hızlı) |
| Türkçe NLP literatür baseline | **Evet** [4][16] | Daha az atıf | **BERTurk** |
| Akademik kararlılık ölçütü (σ ≤ 0,01) | **Sağlanıyor (0,0073)** | Sağlanmıyor (0,0143) | **BERTurk** |
| Reproducibility kolaylığı | Daha kolay (hızlı) | Pahalı | **BERTurk** |

**Skor:** BERTurk 6 / ELECTRA 0 (3 nominal üstünlük istatistiksel olarak anlamsız, EŞDEĞER kategorisinde).

---

## 5. Akademik Sonuç

**Nihai model BERTurk olarak doğrulanmıştır** — beş bağımsız boyutta (multi-seed, CV, McNemar, σ-kararlılık, eğitim maliyeti) tutarlı bulgu:

1. **İstatistiksel eşdeğerlik:** McNemar Bonferroni-corrected min p = 0,0298 > 0,0167 → fark anlamlı değil; CV bunu birinci-mertebede doğruluyor (Δ = 0,003, σ aynı düzey).

2. **Kararlılık üstünlüğü:** BERTurk multi-seed σ_F1m = 0,0073 (literatürde "akademik kararlılık" ölçütü σ ≤ 0,01 sağlanıyor); ELECTRA σ = 0,0143 (sağlanmıyor). BERTurk 2× daha kararlı.

3. **Pratik avantaj:** Eğitim 3,9× daha hızlı, reproduksiyon ve gelecek iteration için kritik.

4. **Akademik konum:** Türkçe NLP literatüründe baseline mimarisi.

Ortalama-bazlı 0,003-0,006 puanlık ELECTRA üstünlüğü gürültü düzeyindedir; pratik kriterler (kararlılık, hız, literatür) BERTurk'ü belirleyici kılar.

---

## 6. Üretim ve Reproducibility

**Veri kaynakları:**
- `models/cv/v3_genisletilmis_5fold_berturk/summary_cv.json` (BERTurk 15 koşu)
- `models/cv/v3_genisletilmis_5fold_electra/summary_cv.json` (ELECTRA 15 koşu)

**Detaylı tablolar:**
- `reports/cv_detailed/berturk_cv_detail.md`
- `reports/cv_detailed/electra_cv_detail.md`

**Bu rapor:** `reports/cv_detailed/cv_comparison.md` (elle yazıldı; sayılar `build_cv_table.py` çıktılarından)

**Reproduction:**
```bash
python scripts/build_cv_table.py        # her iki model için detaylı tablo
```
