# Bagimsiz Sizinti Audit Raporu

Hoca geri bildirimi A.2 kapsaminda, mevcut `assert_no_leakage()` fonksiyonundan **bagimsiz** uc ayri yontemle sizinti dogrulamasi yapilmistir:

1. **Exact match** — `text.strip().lower()` ile birebir esitlik (mevcut yontem ile karsilastirma).
2. **Normalized match** — Turkce karakter + Unicode NFKC + lowercase + noktalama silme + multispace normalize.
3. **Near-duplicate (Jaccard)** — kelime kumesi tabanli benzerlik, J ≥ 0,80 ve J ≥ 0,90 esikleri.

## Sonuclar

| Konfigurasyon | n(train) | n(val) | n(test) | Exact tv | Exact tt | Exact vt | Norm tv | Norm tt | Norm vt | J≥0,80 tt | J≥0,90 tt | J≥0,80 tv | J≥0,90 tv |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| single | 11781 | 1178 | 1378 | 0 | 0 | 0 | 4 | 8 | 0 | 16 | 6 | 9 | 8 |

**Kisaltmalar:** tv = train↔val, tt = train↔test, vt = val↔test.

## Yorumlama

Bazi konfigurasyonlarda sifir-olmayan sizinti tespit edilmistir. Lutfen detayli `leakage_pairs_jaccard_*.csv` dosyalarini inceleyin.

Jaccard ≥ 0,80 esik degerinde tespit edilen ciftler **birebir kopya degildir**; kelime kumesi olcekli yakinligi gosterir. Bu olcek, ozellikle kisa metinlerde (bag of words 5-10 kelime) ayni kelime havuzundan gelen FARKLI yorumlari yanlis yakalayabilir; bu nedenle 0,80 listesi tum-veri-tabani benzerlik incelemesi icin kullanilmis, kesin sizinti kanit olarak J ≥ 0,90 esiki alinmistir.
