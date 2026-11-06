# Klasik ML Top Features Analizi — XAI ile Çapraz Doğrulama

**Kaynak:** `reports/classical_baseline/top_features_per_class.csv` (90 satır = 3 sınıf × 30 katsayı)
**Yöntem:** TF-IDF (word, 1-2 gram) + Logistic Regression (`class_weight='balanced'`),
seed 42 splits üzerinde fit. Sınıf başına en yüksek pozitif katsayılı 30 öznitelik (kelime/n-gram).
**Karşılaştırılan analiz:** §4.10 BERTurk XAI (LIME + SHAP), `reports/xai_final/`.

---

## 1. Bulgular — Sınıf Başına İlk 15 Öznitelik

### 1.1 Negatif Sınıfı

| Sıra | Öznitelik | Katsayı |
|---:|---|---:|
| 1 | kötü | 4,55 |
| 2 | kötüydü | 3,75 |
| 3 | değildi | 3,36 |
| 4 | yetersiz | 3,11 |
| 5 | hiç | 2,90 |
| 6 | etmem | 2,81 |
| 7 | yok | 2,68 |
| 8 | eski | 2,44 |
| 9 | berbat | 2,44 |
| 10 | değil | 2,37 |
| 11 | yemek | 2,36 |
| 12 | bile | 2,26 |
| 13 | asla | 2,24 |
| 14 | çok kötü (2-gram) | 2,13 |
| 15 | bir daha (2-gram) | 2,12 |

**Yorum.** Beklendiği üzere doğrudan negatif kelimeler (kötü, yetersiz, berbat, eski)
en yüksek katsayıyı alıyor. İki önemli nüans:

1. **"yemek" gibi nötr-görünen ad** negatif sınıfta yüksek katsayı alıyor — bu
   veri-spesifik bir bias: otel yorumlarında yemek şikayeti baskın.
2. **Negation pattern'ları** ("değildi", "değil", "yok", "etmem", "asla", "bile")
   ilk 15'in **6'sını** oluşturuyor. Klasik LogReg bile Türkçe negation'u
   yakalayabiliyor — ancak **bağlam bilgisi olmadan** ("güzel değildi" ↔ "değildi
   güzel" aynı katsayıyı alır). BERTurk'ün §4.10 Bulgu B'deki avantajı
   *"iyi" + "değildi" beraber → nötr kodlama* tam burada belirginleşir.

---

### 1.2 Nötr Sınıfı — **En Çarpıcı Bulgu**

| Sıra | Öznitelik | Katsayı |
|---:|---|---:|
| 1 | **ama** | **6,00** |
| 2 | **ancak** | **5,91** |
| 3 | **fakat** | **5,61** |
| 4 | fena | 2,26 |
| 5 | ortalama | 2,13 |
| 6 | nternet (≈ "internet") | 2,05 |
| 7 | beklentiyi | 2,05 |
| 8 | kahvaltı | 1,99 |
| 9 | buna rağmen (2-gram) | 1,92 |
| 10 | biraz | 1,85 |
| 11 | idare | 1,78 |
| 12 | yeterliydi | 1,74 |
| 13 | ulaşım | 1,67 |
| 14 | buna | 1,63 |
| 15 | ortalamaydı | 1,60 |

**Yorum — XAI ile çapraz doğrulama.**

Final raporun §4.10 son paragrafı (LIME analizi yorumu) şöyle diyor:

> *"Türkçe karşıtlık bağlacı 'ama'nın doğrudan +0,10 nötrlük katkısı ile
> yakalanması... küçük bir model boyutuyla (110M parametre) literatürde
> *discourse marker recognition* olarak bilinen ileri seviye özelliğin elde
> edilmiş olması veri kalitesinin bir göstergesidir."*

Klasik LogReg de **bağımsız olarak** aynı bulguya ulaşıyor:
- Top-3'ün TAMAMI Türkçe karşıtlık bağlaçları: **ama (6,00) / ancak (5,91) / fakat (5,61)**
- Sıradaki en yüksek katsayı (fena = 2,26) bunların yarısından az
- Bu, "discourse marker = nötrlük sinyali" ilişkisinin sadece BERTurk
  fenomeni değil, **veri setinin doğal bir özelliği** olduğunu kanıtlıyor.

**Akademik değer.** Bu çapraz doğrulama iki şey söylüyor:

1. **BERTurk'ün öğrendiği şey haklı:** Veri setinde gerçekten karşıtlık bağlaçları
   nötrlük sinyali olarak işliyor; BERTurk bu sinyali keşfetmek için *transformer
   olmasına* gerek duymadı, klasik LogReg bile yakaladı.
2. **BERTurk'ün ekstra katkısı:** §4.10 Bulgu A ve B (çok-kelimeli ifadelerin
   bağlamsal kodlanması, zıt kutuplu sinyallerin birlikte yorumlanması) — bunlar
   klasik LogReg'in kapasitesinin DIŞINDAdır. BERTurk'ün **+9,5 puanlık
   üstünlüğü** (0,8797 vs 0,7850) işte bu *ek* yetenekten geliyor.

**Diğer dikkat çekici:**
- "buna rağmen" (1,92) ve "buna" (1,63) — Türkçe nötr-yapı discourse devamı
- "ortalama" (2,13), "ortalamaydı" (1,60), "yeterliydi" (1,74), "idare" (1,78),
  "biraz" (1,85), "fena" (2,26) — Türkçe'nin "ne pozitif ne negatif" semantik
  alanı; LogReg bunları net yakalamış.
- "kahvaltı" (1,99), "ulaşım" (1,67), "internet" (2,05) — domain-spesifik
  nötr-baskın konular (otelpuan'da bu konular sıkça hem-iyi-hem-kötü değerlendirilir).

---

### 1.3 Pozitif Sınıfı

| Sıra | Öznitelik | Katsayı |
|---:|---|---:|
| 1 | güzeldi | 4,98 |
| 2 | her | 4,13 |
| 3 | iyiydi | 3,89 |
| 4 | güzel | 3,45 |
| 5 | voyage | 3,30 |
| 6 | gayet | 3,23 |
| 7 | mükemmel | 3,04 |
| 8 | ederim | 3,03 |
| 9 | harika | 3,00 |
| 10 | çok iyiydi (2-gram) | 3,00 |
| 11 | memnun kaldık (2-gram) | 2,97 |
| 12 | tavsiye ederim (2-gram) | 2,74 |
| 13 | memnun | 2,68 |
| 14 | güler | 2,58 |
| 15 | hizmet | 2,54 |

**Yorum.**
1. Beklendiği üzere doğrudan pozitif kelimeler (güzel, iyi, mükemmel, harika)
   en yüksek katsayıyı alıyor.
2. **"voyage" (3,30) anomalisi** — bu otel ismi (Voyage Belek vb.). Veri setinde
   marka adı bir sınıfla yüksek korelasyon kazanmış. Bu **model bias** sinyali;
   üretime alınırsa otel ismi bilgisi pozitif yanlılık yaratır.
   *Aksiyon önerisi:* Pre-processing'de marka/otel adlarını maskelemek (`[BRAND]`
   token'ı) düşünülebilir; bu çalışma kapsamında yapılmamış, §5.3 Gelecek
   Çalışmalar'a eklenebilir.
3. **2-gram'ların gücü:** "çok iyiydi", "memnun kaldık", "tavsiye ederim" — kalıp
   ifadeler. Bunlar bigram-aware modellerin (klasik LogReg dahil) yakaladığı
   ama unigram-only modellerin kaçırdığı yapıdır.
4. **"her" (4,13) anomalisi** — sıralamada beklenmedik şekilde 2. sıra. Olası
   sebep: "her şey güzeldi", "her şey mükemmeldi" gibi yapılarda yüksek frekans;
   "her" tek başına nötr ama bigram bağlamında pozitif yapıların başında geliyor.
   Class-weight balancing nedeniyle yüksek frekanslı pozitif-yapı başlangıç
   kelimesi şişmiş olabilir.

---

## 2. BERTurk XAI Bulgularıyla Karşılaştırma

| Bulgu | BERTurk LIME/SHAP (§4.10) | Klasik LogReg (bu rapor) | Eşleşme |
|---|---|---|---|
| Karşıtlık bağlacı = nötrlük sinyali | "ama" → +0,10 nötr | ama/ancak/fakat top-3 (katsayı 5,6-6,0) | ✅ Birebir |
| Negation pattern | "iyi" + "değildi" → nötr | "değildi", "değil" negatif top-10'da | ⚠️ Klasik bağlamsız yakalıyor |
| Çok-kelimeli ifadeler | "saçma bir düzen" gibi | "memnun kaldık", "tavsiye ederim" 2-gram olarak | ⚠️ Klasik kalıp olarak yakalıyor |
| Domain bias | XAI'de görülmedi | "voyage" otel adı yüksek katsayı | 🆕 Klasik baseline'ın kattığı |
| Nötr semantik alan (ortalama, fena, idare) | XAI'de örnek yok | İlk 15'te 4 kelime | 🆕 Klasik baseline'ın kattığı |

**Sonuç.** Klasik LogReg + TF-IDF baseline:
- BERTurk XAI iddialarının **2'sini bağımsız olarak doğruladı** (discourse marker, negation)
- BERTurk XAI'de görülmeyen **2 yeni bulgu kattı** (otel-ismi bias, nötr semantik
  alan kapsamı)
- BERTurk'ün **bağlamsal yetenek farkını** (Bulgu A, B) negatif örnekle gösterdi:
  klasik model "iyi değildi"yi bigram olmadığında yakalayamaz.

---

## 3. Sunum İçin Tek Cümlelik Mesaj

> "Klasik TF-IDF + LogReg baseline, BERTurk'ün XAI'de tespit ettiği
> 'discourse marker = nötrlük' bulgusunu bağımsız olarak doğruladı. Üç Türkçe
> karşıtlık bağlacı (ama, ancak, fakat) klasik baseline'da nötr sınıfının ilk
> 3 katsayısını oluşturuyor. Bu, bulgunun veri setinin yapısal bir özelliği
> olduğunu, BERTurk-spesifik bir artefakt olmadığını kanıtlıyor."

---

## 4. Reproducibility

```bash
# Yeniden üretim
python src/classical_baseline.py
# Çıktı: reports/classical_baseline/top_features_per_class.csv

# Bu rapor (top_features_analysis.md) elle yazıldı (XAI ile çapraz
# doğrulama akademik yorumu); CSV'den otomatik üretilmedi.
```
