# Bagimsiz Sizinti Audit Raporu

Hoca geri bildirimi A.2 kapsaminda, mevcut `assert_no_leakage()` fonksiyonundan **bagimsiz** uc ayri yontemle sizinti dogrulamasi yapilmistir:

1. **Exact match** — `text.strip().lower()` ile birebir esitlik (mevcut yontem ile karsilastirma).
2. **Normalized match** — Turkce karakter + Unicode NFKC + lowercase + noktalama silme + multispace normalize.
3. **Near-duplicate (Jaccard)** — kelime kumesi tabanli benzerlik, J ≥ 0,80 ve J ≥ 0,90 esikleri.

## Sonuclar

| Konfigurasyon | n(train) | n(val) | n(test) | Exact tv | Exact tt | Exact vt | Norm tv | Norm tt | Norm vt | J≥0,80 tt | J≥0,90 tt | J≥0,80 tv | J≥0,90 tv |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| seed42_fold1 | 11022 | 947 | 2368 | 0 | 0 | 0 | 3 | 8 | 1 | 20 | 13 | 15 | 4 |
| seed42_fold2 | 11022 | 947 | 2368 | 0 | 0 | 0 | 1 | 11 | 0 | 30 | 13 | 12 | 4 |
| seed42_fold3 | 11022 | 948 | 2367 | 0 | 0 | 0 | 1 | 4 | 1 | 18 | 10 | 5 | 2 |
| seed42_fold4 | 11022 | 948 | 2367 | 0 | 0 | 0 | 3 | 4 | 1 | 23 | 10 | 17 | 5 |
| seed42_fold5 | 11022 | 948 | 2367 | 0 | 0 | 0 | 4 | 8 | 0 | 21 | 10 | 12 | 3 |
| seed43_fold1 | 11022 | 947 | 2368 | 0 | 0 | 0 | 2 | 8 | 1 | 19 | 8 | 13 | 5 |
| seed43_fold2 | 11022 | 947 | 2368 | 0 | 0 | 0 | 2 | 5 | 0 | 15 | 8 | 5 | 1 |
| seed43_fold3 | 11022 | 948 | 2367 | 0 | 0 | 0 | 6 | 3 | 0 | 26 | 13 | 6 | 1 |
| seed43_fold4 | 11022 | 948 | 2367 | 0 | 0 | 0 | 3 | 9 | 4 | 27 | 9 | 11 | 6 |
| seed43_fold5 | 11022 | 948 | 2367 | 0 | 0 | 0 | 1 | 10 | 1 | 28 | 9 | 8 | 1 |
| seed44_fold1 | 11022 | 947 | 2368 | 0 | 0 | 0 | 0 | 6 | 1 | 24 | 10 | 9 | 3 |
| seed44_fold2 | 11022 | 947 | 2368 | 0 | 0 | 0 | 2 | 7 | 0 | 30 | 14 | 12 | 4 |
| seed44_fold3 | 11022 | 948 | 2367 | 0 | 0 | 0 | 4 | 7 | 0 | 29 | 14 | 9 | 5 |
| seed44_fold4 | 11022 | 948 | 2367 | 0 | 0 | 0 | 5 | 8 | 0 | 23 | 9 | 9 | 4 |
| seed44_fold5 | 11022 | 948 | 2367 | 0 | 0 | 0 | 4 | 8 | 0 | 17 | 11 | 12 | 5 |

**Kisaltmalar:** tv = train↔val, tt = train↔test, vt = val↔test.

## Yorumlama

Bazi konfigurasyonlarda sifir-olmayan sizinti tespit edilmistir. Lutfen detayli `leakage_pairs_jaccard_*.csv` dosyalarini inceleyin.

Jaccard ≥ 0,80 esik degerinde tespit edilen ciftler **birebir kopya degildir**; kelime kumesi olcekli yakinligi gosterir. Bu olcek, ozellikle kisa metinlerde (bag of words 5-10 kelime) ayni kelime havuzundan gelen FARKLI yorumlari yanlis yakalayabilir; bu nedenle 0,80 listesi tum-veri-tabani benzerlik incelemesi icin kullanilmis, kesin sizinti kanit olarak J ≥ 0,90 esiki alinmistir.
