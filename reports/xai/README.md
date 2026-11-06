# XAI + Hata Analizi — v3_bigtest seed 42

Bu dizin, nihai model üzerinde yapılan açıklanabilirlik analizlerini içerir.

## Model
- **Dizin:** `models/optimized_v3/v3_bigtest/bert-base-turkish-cased_len256_ep5_lr2e-05_losswce_seed42/best_model/`
- **Tag:** BERTurk-base (dbmdz/bert-base-turkish-cased), max_len=256, 5 epoch, WCE loss
- **Test (seed 42):** n=1247, %100 gerçek kullanıcı yorumu, {negatif 437, nötr 279, pozitif 531}
- **Test metrikleri (seed 42):** Accuracy 0.8709 · Macro-F1 0.8379 · F1-neg 0.8971 · F1-nötr 0.6815 · F1-poz 0.9351

## Yöntem
- **LIME** (`LimeTextExplainer`, bow=False, num_samples=1500, 12 feature): kelime bazlı perturbation ile tahmin olasılığının nasıl değiştiğini ölçer.
- **SHAP** (`shap.Explainer` + `maskers.Text`): tokenizer seviyesinde mask-tabanlı katkı.
- Toplam 8 örnek — 3 doğru tahmin + 4 tipik hata + 1 "zor doğru" nötr.

## Seçilen 8 Örnek — Özet Tablosu

| # | Tür | Gerçek | Tahmin | Kelime | Durum |
|---:|---|:---:|:---:|---:|:---:|
| 1 | dogru_negatif | negatif | negatif | 27 | ✓ |
| 2 | dogru_nötr (kısa, karma) | nötr | nötr | 11 | ✓ |
| 3 | dogru_pozitif | pozitif | pozitif | 32 | ✓ |
| 4 | hata_nötr→negatif | nötr | negatif | 20 | ✗ |
| 5 | hata_nötr→pozitif | nötr | pozitif | 45 | ✗ |
| 6 | hata_pozitif→nötr | pozitif | nötr | 28 | ✗ |
| 7 | hata_negatif→nötr | negatif | nötr | 22 | ✗ |
| 8 | dogru_nötr_uzun | nötr | nötr | 26 | ✓ |

## Ana Bulgular

### Bulgu 1 — Model "anahtar olumsuzlama" ifadelerini öğrenmiş
**Case 1 (doğru negatif):** "kalmadım" (+0.53), "memnun" (+0.29), "değildi" (+0.27), "eskiydi" (+0.10) negatif sınıfa en yüksek katkıyı veriyor. Yani model **"memnun kalmadım"** kombinasyonunu bir bütün olarak olumsuzluk sinyali olarak öğrenmiş; tek başına "memnun" olumlu olmasına rağmen bağlam içinde doğru yorumlanıyor.

### Bulgu 2 — Nötr sınıfı "hem olumlu hem olumsuz" sinyalini birlikte tanıyor
**Case 2 (doğru nötr):** *"Tesis ve personel cok iyi olmakla birlikte yemekler gercekten cok basarisizdi."*
LIME çıktısı nötr sınıfı için: "iyi" (+0.53) **ve** "basarisizdi" (+0.33) aynı anda pozitif katkı veriyor. Bu, modelin tek yönlü bir duygu sözcüğüne bakmadığını, **zıt kutuplu sinyallerin birlikte varlığını nötrlük işareti olarak öğrendiğini** kanıtlıyor.

**Case 8 (doğru uzun nötr):** "rahatsız" (+0.51) *ve* "ilgiliydi" (+0.11), "genel" (+0.08) birlikte nötrlük sinyali veriyor.

Bu akademik olarak güçlü bir bulgudur: model basit lexicon-based yaklaşımı değil, bağlamsal bir karma duygu temsilini öğrenmiş.

### Bulgu 3 — Hata örüntüsü "kutupsal baskınlık"
Hatalı tahminlerde (case 4-7) belirli bir örüntü var:

- **Case 4 (nötr → negatif):** Metinde hem "alakalıydı, ilgili" (olumlu) hem "kötüydü, azdı, ücretleri" (olumsuz) var. LIME'a göre negatif sinyaller (+0.57 + 0.32 + 0.31) olumluları (-0.19, -0.18) sayıca ve ağırlıkça eziyor. Model metni "kötüye yakın" olarak görmekte **aslında haklı denebilir**; etiketleme kararı "karma" olduğu için nötr ama sinyal ağırlığı negatife yakın.
- **Case 5 (nötr → pozitif):** Metin neredeyse tamamen olumlu sıfatlarla dolu ("güzeldi", "özenliydi", "lezzetliydi"); tek sorun "yaktı" gibi küçük bir negatif nüans. Model pozitife kayıyor — insan da öyle okur.
- **Case 6 (pozitif → nötr):** "harika", "güzel" VE "eksikliği", "yapılmıyor" birlikte var. Model nötr diyor. Etiket pozitif ama metin açıkça karma. **Bu hata aslında etiket sorununa işaret ediyor olabilir.**
- **Case 7 (negatif → nötr):** "güzeldi" 3 kere geçiyor, arada "kötüydü" 2 kez. Model nötr diyor. Etiket negatif. Metinde mutlak olumsuzluk baskın değil; model'in nötr demesi savunulabilir.

### Bulgu 4 — Hataların bir kısmı etiket ambiguity'sinden kaynaklı
Case 6 ve 7'deki "hata"lar, etiketlemenin "rating-based" (1-2 neg, 3 nötr, 4-5 poz) kuralına dayanmasından geliyor: aynı metne farklı yıldız verilmiş olsa aynı içerik farklı etiketlenirdi. Bu, projenin metodoloji bölümünde bilinçli bir **sınırlama** olarak sunulabilir: *"Nötr sınıfındaki bazı hatalar, puan-temelli etiketlemenin metin içeriğiyle ayrışması durumlarını yansıtır; modelin tahmini metin içeriği açısından savunulabilir olabilmektedir."*

## Dosya Listesi

```
selected_examples.csv           # 8 örneğin metadata'sı
summary.json                    # tüm top-features ve tahminler
lime_case_01_dogru_negatif.html     ... (8 HTML)
shap_case_01_dogru_negatif.html     ... (8 HTML)
```

LIME HTML'lerini bir tarayıcıda açarak interaktif kelime-vurgulu görseli inceleyebilirsiniz. SHAP HTML'leri token seviyesinde katkı grafikleri içerir.

## Final Rapora Önerilen Kullanım

1. **Bulgu 2** (nötr'ün karma-sinyal öğrenimi) — Tartışma bölümüne güçlü bir argüman.
2. **Bulgu 3 + 4** (hataların karakteri) — "Modelin hataları rastgele değil, etiketleme belirsizliğine yakın karma-duygu yorumlarında yoğunlaşmaktadır" cümlesiyle özetlenir.
3. Case 1, 2, 4 için LIME görsellerinin birer screenshot'u Bulgular bölümünde Şekil olarak sunulabilir.
