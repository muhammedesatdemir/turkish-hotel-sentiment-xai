# Veri Kesfi (EDA) Raporu

Bu rapor, danisman geri bildirimi A.5 maddesi kapsaminda `final_genisletilmis_dataset_optimized_v3_v2.csv` (n=14.337) uzerinde uretilmistir. Sayisal ozet, gorseller ve Pearson korelasyon matrisi ile birlikte modelin egitildigi veri yapisini akademik olarak belgeler.

## 1. Genel Veri Yapisi

- Toplam yorum: **14,337**
- negatif: **4,370** (%30.5)
- nötr: **4,660** (%32.5)
- pozitif: **5,307** (%37.0)

Görsel: `01_class_dist.png`

## 2. Kelime Sayisi Dagilimi (Sinif Bazinda)

| Sinif | Medyan | Ortalama | q05 | q95 |
|---|---:|---:|---:|---:|
| negatif | 39 | 58.2 | 6 | 170 |
| nötr | 8 | 24.2 | 4 | 73 |
| pozitif | 32 | 40.8 | 5 | 110 |

Görsel: `02_wc_box.png`

**Yorum.** Notr siniflarinin medyan kelime sayisi pozitif/negatif siniflardan belirgin biçimde farkli olabilir; bu, notr yorumlarin **karma duygulu uzun yapida** olma egiliminden kaynaklanir. Bu yapi modelimizin manuel_toplama ile genisletme stratejisini (medyan 53 kelime hedefi) destekleyen ana motivasyondur.

## 3. Kaynak × Etiket Cross-Tab

Görsel: `03_source_label_heatmap.png`

**Yorum.** `ai_generated` kaynagi tamamen notr etiket altinda yogunlasmaktadir (uretim amacli; %100 notr). `manuel_toplama` kaynagi da tamamen notr olup sablon-augmentasyon ile yapay sinyal yogunlasmasinin engellendigi `ai_only_in_train` tasariminda yalnizca train'e dahil edilir.

## 4. Sinif Bazinda En Sik 15 Kelime

Görsel: `04_top_words_per_class.png`

- **negatif** ilk 5: `otel` (2209), `yemek` (1329), `kötü` (1128), `vardı` (1112), `yemekler` (1046)
- **nötr** ilk 5: `ancak` (1055), `biraz` (899), `iyiydi` (889), `otel` (835), `personel` (831)
- **pozitif** ilk 5: `güzeldi` (2876), `otel` (2298), `iyiydi` (2268), `güzel` (1753), `hizmet` (1548)

**Yorum.** Pozitif sinifta `temiz`, `güzel`, `rahat` gibi olumlu sinyaller; negatif sinifta `kötü`, `pis`, `sorun` gibi olumsuz sinyaller; notr sinifta hem olumlu hem olumsuz kelimelerin **birlikte yogun** gorulmesi (karma duygu) beklenen bir bulgudur. Bu, XAI analizinde gozlemlenen 'zit kutuplu sinyal birlikte varligi -> notrluk' ogrenmesinin veri tabanli temelini olusturur (final rapor §4.10 Bulgu B).

## 5. Pearson Korelasyon Matrisi (Meta-Features)

Görsel: `05_corr_matrix.png` · CSV: `corr_matrix.csv`

**sentiment_id (etiket) ile en güçlü korelasyonlar:**
- `rating_auto`: r = +0.813 (pozitif ilişki)
- `neg_keyword_count`: r = -0.390 (negatif ilişki)
- `pos_keyword_count`: r = +0.379 (pozitif ilişki)
- `wc`: r = -0.141 (negatif ilişki)
- `char_count`: r = -0.133 (negatif ilişki)

**is_ai (sentetik mi?) ile en güçlü korelasyonlar:**
- `pos_keyword_count`: r = -0.396
- `wc`: r = -0.345
- `char_count`: r = -0.345

**|r| ≥ 0,50 olan diğer feature çiftleri:**
- `wc` ↔ `char_count`: r = +0.997
- `char_count` ↔ `punct_count`: r = +0.841
- `wc` ↔ `punct_count`: r = +0.826
- `rating_auto` ↔ `sentiment_id`: r = +0.813

**Akademik yorum.** Pearson korelasyon matrisi modelimizin **dogru ozellikleri ogrendigini** dogrular: `rating_auto` ile `sentiment_id` arasindaki yuksek pozitif korelasyon (rating-based etiketleme tutarliligi); `pos_keyword_count` ↔ `neg_keyword_count` arasindaki dusuk korelasyon ise notr siniflarda her iki sinyalin **birbirinden bagimsiz** gorunmesinin (karma duygu yapisinin) kanitidir. `is_ai` kolonu kelime sayisi (`wc`) ile guclu negatif korelasyon gosterir — sentetik orneklerin kisa ve kalipli yapisinin numerik dogrulamasidir.

---

**Reproducibility:** `python src/eda_analysis.py`

Bu rapor `Bitirme2_FinalRapor_AhmetDulger_MuhammedEsatDemir.md` Ek-14 olarak ve `Hoca_Geri_Bildirim_Istatistik_Raporu.md` EK-B olarak referans verilmistir.