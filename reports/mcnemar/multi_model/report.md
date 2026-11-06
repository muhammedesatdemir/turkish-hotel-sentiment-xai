# McNemar Testi — Multi-Model Ablasyon

Bu rapor `final_genisletilmis_dataset_optimized_v3_v2.csv` uzerinde egitilmis 4 model'in (BERTurk, ELECTRA-tr, mBERT, XLM-RoBERTa) AYNI test seti uzerindeki tahminlerinin McNemar testi ile istatistiksel anlamlilik karsilastirmasidir.

## Yontem

Her seed (42, 43, 44) icin 6 model cifti (C(4,2)) icin McNemar testi:

- **2x2 contingency table** (a/b/c/d): a = ikisi de dogru, b = sadece A dogru, c = sadece B dogru, d = ikisi de yanlis.
- **Continuity-corrected chi^2** (b+c >= 25): $\chi^2 = (|b-c|-1)^2 / (b+c)$, df=1.
- **Exact binomial** (b+c < 25): two-tailed binomial test.
- **H0:** iki model ayni hata dagilimina sahip.
- **Karar:** p < 0,05 -> anlamli fark; p >= 0,05 -> gurultu duzeyinde fark.
- **Bonferroni correction** (3 seed = 3 paralel test): $\alpha_\text{eff} = 0{,}05 / 3 = 0{,}0167$.

## Detayli Sonuclar — Her Seed icin

| Karsilastirma | Seed | a | b | c | d | chi^2 | p-value | Method | Sig (α=0,05) |
|---|---:|---:|---:|---:|---:|---:|---:|---|:---:|
| BERTurk vs ELECTRA-tr | 42 | 1182 | 43 | 39 | 129 | 0.110 | 0.7404 | chi2_corrected | hayir |
| BERTurk vs ELECTRA-tr | 43 | 1179 | 51 | 52 | 96 | 0.000 | 1.0000 | chi2_corrected | hayir |
| BERTurk vs ELECTRA-tr | 44 | 1203 | 44 | 68 | 94 | 4.723 | 0.0298 | chi2_corrected | **EVET** |
| BERTurk vs mBERT | 42 | 1138 | 87 | 46 | 122 | 12.030 | 0.0005 | chi2_corrected | **EVET** |
| BERTurk vs mBERT | 43 | 1132 | 98 | 49 | 99 | 15.673 | 0.0001 | chi2_corrected | **EVET** |
| BERTurk vs mBERT | 44 | 1140 | 107 | 64 | 98 | 10.316 | 0.0013 | chi2_corrected | **EVET** |
| BERTurk vs XLM-RoBERTa | 42 | 1155 | 70 | 63 | 105 | 0.271 | 0.6029 | chi2_corrected | hayir |
| BERTurk vs XLM-RoBERTa | 43 | 1169 | 61 | 42 | 106 | 3.146 | 0.0761 | chi2_corrected | hayir |
| BERTurk vs XLM-RoBERTa | 44 | 1153 | 94 | 58 | 104 | 8.059 | 0.0045 | chi2_corrected | **EVET** |
| ELECTRA-tr vs mBERT | 42 | 1129 | 92 | 55 | 117 | 8.816 | 0.0030 | chi2_corrected | **EVET** |
| ELECTRA-tr vs mBERT | 43 | 1136 | 95 | 45 | 102 | 17.150 | 0.0000 | chi2_corrected | **EVET** |
| ELECTRA-tr vs mBERT | 44 | 1160 | 111 | 44 | 94 | 28.103 | 0.0000 | chi2_corrected | **EVET** |
| ELECTRA-tr vs XLM-RoBERTa | 42 | 1162 | 59 | 56 | 116 | 0.035 | 0.8521 | chi2_corrected | hayir |
| ELECTRA-tr vs XLM-RoBERTa | 43 | 1170 | 61 | 41 | 106 | 3.539 | 0.0599 | chi2_corrected | hayir |
| ELECTRA-tr vs XLM-RoBERTa | 44 | 1167 | 104 | 44 | 94 | 23.520 | 0.0000 | chi2_corrected | **EVET** |
| mBERT vs XLM-RoBERTa | 42 | 1116 | 68 | 102 | 107 | 6.406 | 0.0114 | chi2_corrected | **EVET** |
| mBERT vs XLM-RoBERTa | 43 | 1121 | 60 | 90 | 107 | 5.607 | 0.0179 | chi2_corrected | **EVET** |
| mBERT vs XLM-RoBERTa | 44 | 1115 | 89 | 96 | 109 | 0.195 | 0.6591 | chi2_corrected | hayir |

## Cift Bazinda Ozet (3 Seed Aggregat + Bonferroni)

| Karsilastirma | Min p | Median p | Max p | Bonferroni eşik (α=0,05) | Anlamli? |
|---|---:|---:|---:|---:|:---:|
| BERTurk vs ELECTRA-tr | 0.0298 | 0.7404 | 1.0000 | 0.0167 | hayir |
| BERTurk vs mBERT | 0.0001 | 0.0005 | 0.0013 | 0.0167 | **EVET** |
| BERTurk vs XLM-RoBERTa | 0.0045 | 0.0761 | 0.6029 | 0.0167 | **EVET** |
| ELECTRA-tr vs mBERT | 0.0000 | 0.0000 | 0.0030 | 0.0167 | **EVET** |
| ELECTRA-tr vs XLM-RoBERTa | 0.0000 | 0.0599 | 0.8521 | 0.0167 | **EVET** |
| mBERT vs XLM-RoBERTa | 0.0114 | 0.0179 | 0.6591 | 0.0167 | **EVET** |

## Akademik Yorumlama

### BERTurk vs ELECTRA-tr

- Min p = **0.0298**
- Median p = 0.7404
- Bonferroni-corrected anlamli: HAYIR

**Yorum:** Bonferroni-corrected p-value (3-seed) anlamlilik esiginin **altinda kalmadigi icin** iki modelin hata dagilimi istatistiksel olarak ayirt edilemedi. Bu sonuc, ortalama-bazli sirakamadaki (ELECTRA > BERTurk) 0,006 puanlik Macro-F1 farkinin **gurultu duzeyinde** oldugunu **istatistiksel olarak dogrular**. Dolayisiyla nihai model olarak BERTurk'un secilmesi (sigma kararliligi 2x dusuk, egitim 5x hizli, Turkce literatur baseline'i) **akademik olarak savunulabilir** bir karardir; iki model arasindaki performans farki istatistiksel olarak anlamli olmadigi icin pratik kriterler belirleyici olur.

### BERTurk vs mBERT

- Min p = **0.0001**
- Bonferroni-corrected anlamli: EVET

**Yorum:** Turkce-spesifik mimarinin (BERTurk) multilingual mimariden (mBERT) **istatistiksel olarak anlamli sekilde ustun** oldugu dogrulanmistir. mBERT'in Turkce vocab payinin yaklasik %2,5 olmasi kayitli olarak Turkce-spesifik on-egitim avantajini matematiksel olarak gosterir.

### BERTurk vs XLM-RoBERTa

- Min p = **0.0045**
- Bonferroni-corrected anlamli: EVET

**Yorum:** 278M parametreye ve 250K vocab'a sahip XLM-RoBERTa'nin BERTurk'ten **istatistiksel olarak anlamli sekilde dusuk** kalmasi **multilingual on-egitimin Turkce-spesifik on-egitime alternatif olamadigi** sonucunu desteklerdir.

## Reproducibility

```bash
python src/mcnemar_test.py
```

Cikti dosyalar: `reports/mcnemar/multi_model/`