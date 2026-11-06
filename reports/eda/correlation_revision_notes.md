# EDA — Korelasyon Yorumlarının Akademik Revizyonu

**Tarih:** 2026-05-09 (sunum öncesi öz-eleştiri)

Bu doküman, ilk EDA çıktısındaki Pearson korelasyon yorumlarımızın iki noktada
**istatistiksel olarak yanıltıcı** olduğunu tespit etmemiz üzerine yazılmıştır.
Hocaya sunulan rapora yanlış iddia girmesini engellemek için yorumları daha
dürüst ve nüanslı bir şekilde güncelliyoruz.

---

## Revizyon 1 — `pos_kw ↔ neg_kw = +0,021` yorumu

### İlk yorumumuz (HATA)
> "Pos ve neg anahtar kelime sayılarının near-zero korelasyonu (+0,021), nötr
> sınıfında karma duygu yapısının matematiksel ispatıdır."

### Tespit edilen sorun: Simpson Paradoksu

Tüm verinin Pearson r = +0,021 değeri sınıf-içi korelasyonların ortalaması
**değildir**. Sınıflar arası farklılık (pozitif sınıf çok pos_kw + az neg_kw;
negatif sınıf tam tersi) zıt yönde sinyaller üretir; bunlar bir araya geldiğinde
**ortak korelasyonu sıfıra yakın gösterir**, ancak **sınıf içinde** korelasyon
0 değildir.

### Sınıf-içi gerçek değerler

| Veri grubu | n | pos_kw mean | neg_kw mean | Pearson r (sınıf içi) |
|---|---:|---:|---:|---:|
| Tüm veri (karışık) | 14.337 | 2,031 | 0,631 | **+0,021** |
| Negatif sınıf | 4.370 | 1,523 | 1,343 | **+0,206** |
| **Nötr sınıf (tüm)** | **4.660** | **1,182** | **0,426** | **+0,195** |
| Pozitif sınıf | 5.307 | 3,194 | 0,225 | **+0,106** |
| Nötr (otelpuan_otelyorum) | 1.409 | 2,354 | 0,573 | **+0,207** |
| Nötr (manuel_toplama) | 751 | 1,563 | 0,871 | **−0,113** |
| Nötr (sadece AI) | 2.500 | 0,408 | 0,210 | −0,017 |

### Doğru yorumlama

Sınıf içinde her sınıfta **pozitif yönlü** korelasyon vardır (r ≈ +0,1 - +0,2);
bu, **uzun yorumların hem olumlu hem olumsuz anahtar kelime sayısının doğal
olarak yüksek olmasından** kaynaklanır (kelime havuzunun büyüklüğü artıyor).
Yani bu, semantik bir bulgu değil, **dilbilgisel uzunluk efektidir**.

**Önemli istisna:** `manuel_toplama` alt-grubu (n=751) içinde r = −0,113, yani
hafifçe negatif korelasyon. Bu **anlamlı** olabilir: manuel_toplama yorumlarımız
şablon-augmentasyon ile üretildiği için her bir yorum bilinçli olarak sınırlı
sayıda pos/neg sinyali içerir, birbirinin yerine geçer pozisyondalar. Bu,
şablon tasarımımızın **dengeli karma duygu yaratma hedefini** karşıladığının
**alt-grup bazlı kanıtıdır**.

### Final raporlardaki revizyon

- "Karma duygu yapısının matematiksel ispatı" iddiası **geri çekildi**
- Yerine: "Sınıf-içi korelasyonlar +0,1-0,2 bandında pozitif olup, bu durum
  uzun yorumlarda her iki sinyalin doğal birlikte artışını yansıtır
  (Simpson Paradoksu)"
- **manuel_toplama alt-grubunun r = −0,113** ayrı olarak vurgulanır;
  şablon tasarımımızın "dengeli karma duygu" hedefini karşıladığının kanıtı
  olarak sunulur

---

## Revizyon 2 — `is_ai ↔ sentiment_id = −0,037` (yanıltıcı near-zero)

### İlk yorumumuz (yanlış değil ama eksik)
Raporlarda doğrudan **is_ai ↔ sentiment_id ilişkisini sayısal olarak iddia
etmedik**, yalnızca `is_ai ↔ wc = −0,345` ve `is_ai ↔ pos_kw = −0,396`'yı
vurguladık. Ama Pearson matris görsellerinde **−0,037 değeri görünüyor**;
hocanın "AI-etiket arasında ilişki yokmuş" yanlış izlenimi edinmesi muhtemel.

### Tespit edilen sorun

`sentiment_id ∈ {−1, 0, +1}` üç-değerli kategorik değişken. AI satırların
**hepsi nötr (=0)**. AI olmayan satırların ortalaması da +0,079 (negatif/nötr/
pozitif görece dengeli, hafif pozitif eğilimli). İki grubun ortalamaları
neredeyse eşit → Pearson r ≈ 0. Bu, AI ↔ etiket ilişkisinin **olmadığını**
göstermez; **Pearson formülünün kategorik değişkenler için yanlış metrik
olduğunu** gösterir.

### Doğru metrik

| Yöntem | Değer | Anlam |
|---|---:|---|
| Pearson(is_ai, sentiment_id={−1,0,+1}) | **−0,037** | yanıltıcı near-zero |
| **Pearson(is_ai, is_notr={0,1})** | **+0,662** | doğru ölçü (point-biserial eşdeğeri) |

### Cross-tab (kategorik gerçek)

| | negatif | nötr | pozitif | Toplam |
|---|---:|---:|---:|---:|
| `is_ai = 0` (gerçek) | 4.370 | 2.160 | 5.307 | 11.837 |
| **`is_ai = 1` (sentetik)** | **0** | **2.500** | **0** | **2.500** |
| Toplam | 4.370 | 4.660 | 5.307 | 14.337 |

**AI satırların %100'ü nötr** — bu deterministik kategorik ilişki,
Pearson'da −0,037 değil **point-biserial r = +0,662** ile gerçek gücüyle
yansır.

### Final raporlardaki revizyon

- EDA Pearson matrisindeki −0,037 değeri **olduğu gibi tutulur** (sayısal
  olarak doğru)
- **Yanına bir not eklenir**: "Pearson kategorik değişkenler için optimal
  metrik değildir. is_ai ile kategorik nötr ilişkisi point-biserial
  r = +0,662 olup, AI örneklerinin **deterministik olarak nötr sınıfında
  yer almasının** doğru sayısal göstergesidir."
- **Cross-tab tablo eklenir** — kategorik gerçek görsel olarak da yansır

---

## Revizyon 3 — McNemar BERTurk vs XLM-RoBERTa nüansı

### İlk yorumumuz (iyimser)
> "Min p = 0,00453 < 0,0167 (Bonferroni). XLM-RoBERTa BERTurk'ten anlamlı
> düzeyde geride."

### Tespit edilen nüans

| Seed | p-value | α=0,05 anlamlı? |
|---:|---:|:---:|
| 42 | 0,6029 | ❌ |
| 43 | 0,0761 | ❌ |
| 44 | **0,0045** | ✅ |
| Min | 0,0045 | ✅ Bonferroni-corrected |
| Median | 0,0761 | — |

**Sadece 1 seedde anlamlı** (3'ten 1'i). Min-p approach
Bonferroni-corrected'da anlamlı görünüyor ama **median ≈ 0,076 anlamsız**.
Bu, **kararsız bir bulgu**.

Aynı durum **ELECTRA-tr vs XLM-RoBERTa** için de geçerli (3 seedden 1'i
anlamlı: 0,8521 / 0,0599 / <10⁻⁵).

### Doğru yorumlama

- BERTurk vs mBERT: 3/3 seed anlamlı → **çok güçlü** bulgu
- ELECTRA vs mBERT: 3/3 seed anlamlı → **çok güçlü** bulgu
- **BERTurk vs XLM-RoBERTa: 1/3 seed anlamlı → kararsız bulgu, seed 44 outlier**
- **ELECTRA-tr vs XLM-RoBERTa: 1/3 seed anlamlı → kararsız bulgu, seed 44 outlier**
- mBERT vs XLM-R: 2/3 seed anlamlı → orta güç

### Final raporlardaki revizyon

- "BERTurk vs XLM-R fark anlamlı" iddiası **kuvveti azaltılarak** raporlanır:
  "Seed 44'te güçlü farklılaşma; seed 42-43'te anlamsız. Bonferroni-corrected
  min-p eşik geçilse de **bulgu kararsızdır**"
- "Türkçe-spesifik üstün multilingual" iddiamız **mBERT için sağlam**
  (3/3 seed); **XLM-R için kısmen** (1/3 seed)
- Bu durum **akademik dürüstlüğü** artırır; hoca sayıları kontrol ederse
  tam cevabımız hazır olur

---

## Genel Akademik Konum

Bu üç revizyon ile:
1. **Hatalı bir "ispat" iddiamızı geri çektik** (pos↔neg karma duygu)
2. **Eksik bir gerçeği netleştirdik** (is_ai↔nötr point-biserial)
3. **İyimser bir bulguyu ölçüsünü doğru raporladık** (XLM-R kararsız)

**Sonuç:** Sunumda mahcup olma riskimizi minimuma indirdik. Her sayı şimdi
**doğru metrikle**, **doğru yorumla** raporlanmış durumda.
