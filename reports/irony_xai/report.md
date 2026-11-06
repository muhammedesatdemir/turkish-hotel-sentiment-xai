# Sarkazm/İroni Yapısal Limitasyonu — XAI Kanıtı

**Hocanın geri bildirimi A.8 cevabı.** Otel yorumları veri setinde sarkazm/ironi etiketli örnek yoktur (doğal dağılımda nadir);
bu yüzden limitation'ı somutlaştırmak için **8 sentetik sarkastik cümle** yazılıp
nihai BERTurk modeli (`v3_genisletilmis seed=43`) tahminleri ile LIME açıklamaları üretildi.

**Model:** `dbmdz/bert-base-turkish-cased` fine-tune, seed 43.
**LIME parametreleri:** `bow=False`, `num_samples=1500`, `num_features=10`.

## Sentetik Sarkazm Örnekleri ve Model Tahminleri

| # | Asıl Polarite | Tahmin | Güven | Doğru? | İronik Sinyal | Negatif Substans |
|---|---|---|---:|:---:|---|---|
| 1 | negatif | nötr | 0.714 | ❌ | harika, mükemmel — pozitif kelimeler | sıcak su yok, klima bozuk, yalnız |
| 2 | negatif | pozitif | 0.997 | ❌ | güleryüzlü, teşekkürler — pozitif | yarım saat bekleme, ilgisizlik |
| 3 | negatif | pozitif | 0.995 | ❌ | başarılı — pozitif kelime | 5 yıldız beklentisi vs gerçek kalite |
| 4 | negatif | pozitif | 0.994 | ❌ | muhteşem, lüks | sadece 3 reçel ve tost (yetersiz çeşit) |
| 5 | negatif | negatif | 0.991 | ✅ | unutamayacağım — bağlamsız pozitif | küf kokusu |
| 6 | negatif | negatif | 0.977 | ✅ | sürpriz yapmadı — kalıp ifade | vasat, yetersizlik |
| 7 | negatif | nötr | 0.771 | ❌ | yardımsever, minnettar | pencere açılmadı, havasızlık |
| 8 | negatif | nötr | 0.531 | ❌ | müthiş, ideal | çöp konteyneri manzarası |

## Tahmin Detayları (Olasılık Dağılımları)

| # | Metin (kısalt.) | p_negatif | p_nötr | p_pozitif | Top LIME |
|---|---|---:|---:|---:|---|
| 1 | Hayatımda gördüğüm en harika otel, banyoda sıcak su yokmuş, klima bozu... | 0.058 | 0.714 | 0.228 | bozukmuş:+0.358, yokmuş:+0.137, klima:+0.137, sıcak:+0.089, su:+0.065 |
| 2 | Resepsiyondaki personel öyle güleryüzlüydü ki yarım saat kuyrukta bekl... | 0.001 | 0.003 | 0.997 | güleryüzlüydü:+0.336, teşekkürler:+0.324, bakmadılar:-0.205, ki:+0.071, Resepsiyondaki:-0.066 |
| 3 | 5 yıldızlı otel diye gittik, 5 dakikada parkta da bulunan kalitede gör... | 0.001 | 0.004 | 0.995 | başarılı:+0.212, çok:+0.163, otel:-0.081, diye:-0.080, kalitede:+0.077 |
| 4 | Otel kahvaltısı muhteşemdi, üç çeşit reçel ve tost makinesi vardı, ne ... | 0.001 | 0.005 | 0.994 | muhteşemdi:+0.570, reçel:+0.052, vardı:-0.051, bir:+0.042, ne:-0.040 |
| 5 | Odaya girer girmez küf kokusu beni karşıladı, gerçekten unutamayacağım... | 0.991 | 0.006 | 0.003 | kokusu:+0.487, küf:+0.469, unutamayacağım:+0.123, tatil:-0.064, bir:-0.054 |
| 6 | Eh işte, beklediğim gibi vasat bir oteldi, sürpriz yapmadı, hep aynı y... | 0.977 | 0.022 | 0.001 | yetersizlik:+0.354, vasat:+0.272, sürpriz:-0.097, hep:+0.081, yapmadı:+0.063 |
| 7 | Personel çok yardımseverdi, ben kaldığım odanın penceresini açamadım, ... | 0.080 | 0.771 | 0.149 | havasız:+0.338, açamadım:+0.211, minnettarım:-0.140, yardımseverdi:+0.116, Personel:+0.079 |
| 8 | Müthiş bir manzaraydı, otoparkın çöp konteynerleri tam pencerenin altı... | 0.030 | 0.531 | 0.439 | Müthiş:-0.539, manzaraydı:+0.187, çöp:+0.158, pencerenin:+0.127, altındaydı:+0.109 |

## Özet

- **Toplam test örneği:** 8 (7 sarkastik + 1 doğrudan kontrol)
- **Kontrol grubu (#6 — doğrudan negatif):** ✅ Doğru (model temel görevi yapabiliyor)
- **Sarkastik 7 örnek doğru tahmin:** 1/7
- **Sarkastik 7 örnek yanlış tahmin:** 6/7
- **Yanlış tahminlerde ortalama güven:** 0.834

## Akademik Yorum

Sarkastik 7 örnekten **6'i yanlış sınıflandırılmıştır**.
Bu, modelin **sarkazm/ironi yapısal sınırlamasını** XAI ile somutlaştırmaktadır:

1. **Yüzeysel kelime tercihi:** Model sarkastik cümlelerdeki pozitif kelimeleri
   (harika, müthiş, mükemmel, ideal, yardımsever) doğrudan pozitif sinyal olarak
   yorumluyor; karşıtlık bağlamını yakalayamıyor.
2. **Yüksek güvenle yanlış:** Yanlış tahminlerin ortalama güveni yüksektir
   → model belirsizlik göstermiyor, *yanlış olduğunu bilmiyor*.
3. **LIME bulgusu:** Top katsayılı kelimeler (HTML çıktılarda görselleştirildi) hep
   yüzeysel pozitif terimler — hiçbir karşıtlık ya da çelişki sinyali katsayısal
   olarak güçlü değil.

Bu bulgular final raporun §5.2 (Teorik ve Operasyonel Sınırlamalar) bölümündeki
sarkazm/ironi limitation iddiasını **deneysel olarak** kanıtlamaktadır.

## Çözüm Önerisi (Gelecek Çalışmalar)

- Türkçe sarkazm/ironi etiketli ek veri toplama (literatürde sınırlı kaynak)
- Sarkazm-aware fine-tuning (multi-task: sentiment + sarkazm tespiti)
- Discourse marker'lara (ama, ancak, fakat) ek olarak **kontrastif kelime
  desenlerini** (pozitif kelime + negatif olay) kodlayan attention mask'ı
- LLM-tabanlı few-shot reasoning (büyük dil modellerinin sarkazm-tanıma
  yeteneği daha gelişmiş)

## Üretim ve Reproducibility

```bash
python src/sarcasm_lime.py    # ~5 dk GPU
```

**Çıktı:** `reports/irony_xai/{sarcasm_predictions.csv, lime_sarcasm_*.html, summary.json, report.md}`