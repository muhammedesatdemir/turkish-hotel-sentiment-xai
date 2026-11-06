# XAI (Explainable AI) Analizi — Nihai Model `v3_genisletilmis`

**Üretim tarihi:** 1 Mayıs 2026
**Hedef model:** `dbmdz/bert-base-turkish-cased` fine-tune,
`models/optimized_v3/v3_genisletilmis/bert-base-turkish-cased_len256_ep5_lr2e-05_losswce_seed43/best_model`
**Seed:** 43 (3 seed arasında en yüksek Macro-F1 = 0.8871, F1-nötr = 0.8142)
**Test seti:** 1378 gerçek kullanıcı yorumu (AI augmented yok)
**Araçlar:** [LIME](https://github.com/marcotcr/lime) (`LimeTextExplainer`, `bow=False`, `num_samples=1500`)
ve [SHAP](https://github.com/shap/shap) (`shap.Explainer` + `maskers.Text`).
**Üretim scripti:** [`src/explain_xai.py`](../../src/explain_xai.py)

---

## 1. Klasör İçeriği

```
reports/xai_final/
├── README.md                                 # bu dosya
├── selected_examples.csv                     # 8 örnek metadata
├── summary.json                              # tüm LIME + SHAP top-features
├── error_analysis.csv                        # 148 yanlış sınıflanan örnek + confidence + kategori
├── error_analysis_summary.json               # hata analizi sayısal özeti
├── lime_case_01_dogru_negatif.html
├── lime_case_02_dogru_nötr.html
├── lime_case_03_dogru_pozitif.html
├── lime_case_04_hata_notr_negatif.html
├── lime_case_05_hata_notr_pozitif.html
├── lime_case_06_hata_pozitif_notr.html
├── lime_case_07_hata_negatif_notr.html
├── lime_case_08_dogru_notr_uzun.html
└── shap_case_01..08_<tür>.html               # SHAP karşılıkları
```

Her HTML dosyası bir tarayıcıda açılıp **interaktif kelime/token vurgularıyla** incelenebilir.

---

## 2. Neden XAI Yenilendi?

İlk XAI analizi (bkz. [`reports/xai/`](../xai/)) §19'da tamamlanmıştı, ancak `v3_bigtest` modeli üzerinde
(Macro-F1 = 0.8377, F1-nötr = 0.6852) yapılmıştı. §20'de **veri seti 14337 satıra genişletilince**
(751 doğal nötr eklendi) ve §21'deki multi-model ablasyon BERT-tr'yi nihai model olarak doğrulayınca,
XAI'nın **gerçek nihai model** üzerinde tekrarlanması gerekti. Bu klasör (`xai_final`) o yenileme çıktısıdır.

**Eski XAI ile fark:**
- F1-nötr 0.6852 → 0.8142 (+13 puan iyileşmiş model üzerinde analiz)
- Test seti boyutu büyüdü (782 → 1378 örnek, daha güvenilir hata örüntüsü)
- Hata analizi modülü eklendi (148 yanlış örneğin tamamı kategorize edildi)

---

## 3. Seçilen 8 Örneğin Yapısı

Hata yapısının **dört karakterini** yakalamak için `test_predictions.csv` üzerinden seçildi
(seçim deterministik, kategori bazlı ilk uygun örnek):

| # | Tür | True | Pred | wc | Amaç |
|:---:|---|:---:|:---:|:---:|---|
| 1 | dogru_negatif | negatif | negatif | 17 | temel negatif öğrenme kontrolü |
| 2 | dogru_nötr (kısa, karma) | nötr | nötr | 25 | "iyi+kötü" sinyalinin öğrenilip öğrenilmediği |
| 3 | dogru_pozitif | pozitif | pozitif | 17 | temel pozitif öğrenme kontrolü |
| 4 | hata_notr→negatif | nötr | negatif | 39 | en yaygın hata türü (41 vaka) |
| 5 | hata_notr→pozitif | nötr | pozitif | 22 | ikinci en yaygın (47 vaka) |
| 6 | hata_pozitif→nötr | pozitif | nötr | 21 | karma metinlerde nötre kayma (28 vaka) |
| 7 | hata_negatif→nötr | negatif | nötr | 39 | olumsuzu nötr sanma (31 vaka) |
| 8 | dogru_nötr_uzun | nötr | nötr | 36 | karmaşık uzun yorumda nötrü tutabilme |

---

## 4. Akademik Olarak En Değerli Bulgular

### Bulgu A — Model bağlamsal anlam öğrenmiş, lexicon-based değil

**Case 1 (doğru negatif):**
> *"Sadece casino ya önem verilmiş herşey dahil gidilmesine rağmen birçok şeyin ücreti var çok saçma bir düzen."*

LIME negatif sınıfa en büyük katkıyı veren kelimeler: **`saçma` (+0.50)**, `çok` (+0.10), `ücreti` (+0.10).
Tek başına nötr olabilecek "ücreti" kelimesi, "saçma" ile birlikte bağlam içinde
**negatif katkı** sağlıyor. Model BoW (Bag-of-Words) lexicon yerine **gerçek bağlamsal anlam**
öğrenmiş.

### Bulgu B — Nötr sınıfı zıt-kutuplu sinyalin *birlikte* varlığını öğrenmiş

**Case 2 (doğru nötr, kısa karma):**
> *"Etkinlik olmaması iyi değildi. Çok üzün süre kalınmaz. ... Olumlu yönü olarak temiz bir tesisti. Personel yaklaşımı iyi."*

LIME nötr katkıları: **`değildi` (+0.23)**, **`iyi` (+0.22)**, `olmaması` (+0.13), `Olumlu` (+0.07).

Model basit ortalama almıyor; **"iyi" ve "değildi" kelimelerinin birlikte varlığını**
nötrlük işareti olarak kodlamış. Bu, nötr sınıfının "tanımsal belirsizliği" sorununa
modelin verdiği **olgun bir yapısal cevaptır.**

### Bulgu C — Doğru-nötr-uzun: model "ama" yapısını yakalıyor

**Case 8 (doğru nötr, uzun karma):**
> *"Yol üstü bir duraktı, beklentim düşüktü. Oda temiz, yatak rahat. ... Personel az ama çalışkan. ... Genel anlamda fena değildi."*

LIME nötr katkıları: `temiz` (+0.15), `rahat` (+0.12), **`ama` (+0.10)**, `doyurucuydu` (+0.09).

`ama` bağlacının **doğrudan +0.10 katkı** vermesi çarpıcı: model, Türkçe karşıtlık
bağlaçlarını nötrlük göstergesi olarak öğrenmiş. Bu literatürde **discourse marker
recognition** olarak adlandırılan ileri seviye bir özellik.

### Bulgu D — Hatalar etiket gürültüsünden ve gerçek belirsizlikten kaynaklı

**Case 4 (nötr→negatif yanlış):**
> *"Otelde havuz ve gösteri imkanları gayet güzeldi lakin yemek lezzetleri çok kötüydü ... Otel kesinlikle aile çocuk oteli ... sevgililerin baş başa gitmesini önermem ..."*

LIME negatif katkıları: **`kötüydü` (+0.45)**, **`önermem` (+0.40)**, `kokuyordu` (+0.13).
LIME nötr katkıları (gerçek etiket için): `memnunduk` (+0.15), `önermem` (−0.15), `güzeldi` (+0.13).

Yorum **objektif olarak negatif kelime yoğun**: "çok kötüydü", "kokuyordu", "önermem".
Model'in negatif tahmin etmesi metin içeriğine **savunulabilir derecede tutarlı**.
Etiket "nötr" çünkü kullanıcı 3/5 puan vermiş — ama metin asıl tartışmalı.
Bu, **rating-based etiketlemenin yapısal sınırının** bir kanıtıdır.

### Bulgu E — Pozitif↔negatif karışması yok denecek kadar az

148 hatadan **yalnızca 1 tanesi pozitif↔negatif** (pozitif→negatif). Yani:
- Negatif↔nötr karışması: 72 örnek (%48.6)
- Nötr↔pozitif karışması: 75 örnek (%50.7)
- Pozitif↔negatif karışması: **1 örnek (%0.7)**

Model iki uç kategoriyi (pozitif/negatif) **neredeyse asla karıştırmıyor**.
Tüm hata orta sınıf (nötr) ekseninde toplanmış — bu, modelin
**duygu kutuplarını sağlam, kararsızlığı zor ayırt ettiğinin** kanıtı.

---

## 5. Hata Analizi — Tüm Yanlış Sınıflanan Örneklerin Toplu İncelemesi

`error_analysis.csv` dosyası 148 yanlış sınıflanan örneği içerir; her satır için aşağıdaki
bilgiler hesaplanmıştır:

| Sütun | Açıklama |
|---|---|
| `p_negatif`, `p_notr`, `p_pozitif` | Modelin softmax çıktısı (olasılıklar) |
| `max_prob` | En yüksek olasılık (model güveni) |
| `second_prob` | İkinci en yüksek olasılık (belirsizlik göstergesi) |
| `pos_keys`, `neg_keys` | Metinde geçen pozitif/negatif anahtar kelime sayıları |
| `kategori` | Heuristik hata kategorisi |
| `yanlis_yon` | true→pred yön etiketi (örn. `nötr_to_negatif`) |

### 5.1 Yön Dağılımı (148 hata)

| Yön | Sayı | Oran |
|---|:---:|:---:|
| nötr → pozitif | 47 | %31.8 |
| nötr → negatif | 41 | %27.7 |
| negatif → nötr | 31 | %20.9 |
| pozitif → nötr | 28 | %18.9 |
| pozitif → negatif | 1 | %0.7 |
| negatif → pozitif | 0 | %0.0 |

**Yorum:** Hatanın %59'u nötr'ün yanlış tahmin edilmesi (88/148). Nötr→negatif (41) ve nötr→pozitif (47)
sayıları yakın — model nötrü iki yöne **dengeli** kaybediyor, sistemik bias yok.

### 5.2 Kategori Dağılımı (heuristik)

| Kategori | Sayı | Oran | Açıklama |
|---|:---:|:---:|---|
| **guvenli_yanlis** | **95** | **%64.2** | max_prob > 0.80; model emin ama yanılmış |
| orta_seviye | 26 | %17.6 | 0.50 < max_prob ≤ 0.80; ortalama belirsizlik |
| **belirsiz_dengesiz** | **24** | **%16.2** | second_prob > 0.30; model iki sınıf arasında kararsız |
| etiket_gurultusu_supheli | 3 | %2.0 | yüksek güven + metnin kutbu etiketle ters |

### 5.3 Confidence İstatistikleri

- Hatadaki ortalama `max_prob`: **0.832** — model hatalarında bile genelde kendinden emin
- `max_prob > 0.80` olan 98 hata (%66.2) — bunlar daha çok **etiket gürültüsü** veya
  **gerçek metinsel belirsizlik** olabilir (Bulgu D ile örtüşür)
- `max_prob < 0.50` olan sadece 3 hata — modelin "hiç bilmediğim" durumlar nadir

### 5.4 Yön × Kategori Kesişimi

`error_analysis_summary.json` içinde `errors_by_dir_x_cat` anahtarı altında ayrıntılı.
En dikkat çekici bulgu: **nötr→pozitif/negatif hatalarının %70'i "guvenli_yanlis"** —
yani model nötrü yanlış sınıflandırırken kendinden emin oluyor; bu, nötr sınıfının
**kelime düzeyinde diğer iki sınıfla yapısal örtüşmesinin** doğal sonucudur (bkz. Bulgu D).

---

## 6. Final Raporda XAI'nin Yeri (Yazım Planı)

| Bölüm | İçerik | Kaynak |
|---|---|---|
| **Bulgular** | Case 1, 2, 8'in LIME görselleri Şekil olarak | `lime_case_01,02,08_*.html` |
| **Tartışma A — Bağlamsal öğrenme** | Bulgu A + B (Case 1, 2 LIME) | bu README §4 |
| **Tartışma B — Hata yapısı** | §5'in confusion-matrix sembiyozu | `error_analysis_summary.json` |
| **Sınırlamalar** | Bulgu D (rating-based etiket gürültüsü) | Case 4, 6 LIME |
| **Ek (appendix)** | Tüm 8 case için SHAP + LIME HTML'leri | `shap_*`, `lime_*` |

---

## 7. Tekrarlanabilirlik

```bash
# Sanal ortamı aktive et
venv\Scripts\activate

# XAI üretimi (~10-15 dk, 1 GPU)
python src/explain_xai.py \
  --model_dir "models/optimized_v3/v3_genisletilmis/bert-base-turkish-cased_len256_ep5_lr2e-05_losswce_seed43/best_model" \
  --pred_csv  "models/optimized_v3/v3_genisletilmis/bert-base-turkish-cased_len256_ep5_lr2e-05_losswce_seed43/test_predictions.csv" \
  --out_dir   "reports/xai_final" \
  --lime_samples 1500

# Hata analizi (~30 sn)
python src/error_analysis.py
```

LIME'ın `random_state=42` ile başlatılmasına rağmen, perturbation'lar üzerinden eğitilen
yerel sürrogat modelin sonuçları küçük gürültü içerebilir; ancak **top-3 öznitelikler
tutarlı şekilde aynıdır.**

---

## 8. Durum

XAI bileşeni **tamamlandı** ve nihai modele güncellendi. Proje şimdi:

| İş | Durum |
|---|:---:|
| Multi-seed BERT-tr (μ ± σ) | ✓ (§20) |
| 5 aşamalı ablasyon | ✓ (§20.8) |
| Multi-model karşılaştırması (4 model) | ✓ (§21) |
| **XAI yenileme (LIME + SHAP)** | **✓ (bu rapor)** |
| **Hata analizi (148 örnek)** | **✓ (`error_analysis.csv`)** |
| Final rapor `.md` revizyonu | ⏳ sıradaki adım |

---

## 9. Referanslar

- Ribeiro, M. T., Singh, S., & Guestrin, C. (2016). *"Why Should I Trust You?": Explaining the Predictions of Any Classifier.* KDD'16. (LIME)
- Lundberg, S. M., & Lee, S.-I. (2017). *A Unified Approach to Interpreting Model Predictions.* NIPS'17. (SHAP)
- Schweter, S. (2020). *BERTurk: BERT models for Turkish.* Zenodo. (`dbmdz/bert-base-turkish-cased`)
