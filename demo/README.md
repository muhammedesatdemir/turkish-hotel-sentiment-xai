# Türkçe Otel Yorumu Duygu Analizi — Web Demo

Bitirme projesinin nihai modeli (v3_genisletilmis, BERTurk seed 43) üzerine inşa edilmiş tam yığın web uygulaması.

- **Backend:** FastAPI + Uvicorn — modeli RAM'e bir kez yükler, `/api/predict` (~50-200 ms), `/api/explain` (LIME, ~1-6 sn), `/api/stats/*`, `/api/errors`, `/api/predict/batch`
- **Frontend:** Vite + React 18 + TypeScript + Tailwind CSS — 5 sayfa (Tahmin, Dashboard, Toplu, Hata Analizi, Hakkında)

## Hızlı başlangıç

İki terminal açın.

### Terminal 1 — Backend

```powershell
cd demo/backend
..\..\venv\Scripts\activate
pip install -r requirements.txt   # ilk kullanımda
uvicorn main:app --reload --port 8000
```

İlk açılışta model 5-10 saniye içinde RAM'e (varsa GPU'ya) yüklenir.
Swagger UI: <http://localhost:8000/docs>

### Terminal 2 — Frontend

```powershell
cd demo/frontend
npm install     # ilk kullanımda
npm run dev
```

Tarayıcıda <http://localhost:5173/> açılır.

## Mimari

```
┌──────────────────────┐         HTTP/JSON         ┌──────────────────────┐
│  React Frontend      │ ◄────────────────────►   │  FastAPI Backend     │
│  TypeScript+Tailwind │                          │  Python              │
│  localhost:5173      │                          │  localhost:8000      │
└──────────────────────┘                          └──────────┬───────────┘
                                                             │ in-process
                                                             ▼
                                                  ┌──────────────────────┐
                                                  │  BERTurk fine-tuned  │
                                                  │  models/optimized_v3/│
                                                  │  ...seed43/best_model│
                                                  └──────────────────────┘
```

Backend'in MODEL_DIR yolu `demo/backend/config.py` içinde proje köküne göreceli olarak çözülür; modele kopyasız erişilir.

## Sayfalar

| Yol | İçerik |
|---|---|
| `/` | Tek yorum tahmini + LIME XAI |
| `/dashboard` | Genel metrikler, confusion matrix, multi-model karşılaştırma, ablation, veri seti dağılımı |
| `/batch` | CSV/TXT toplu analiz veya manuel satır girişi (max 50) |
| `/errors` | Test setindeki 148 hata, filtreli/sayfalı tablo, tıklayınca XAI modal |
| `/about` | Yöntem, veri seti, ekip |

## Endpoint Özeti

| Method | Path | Görev |
|---|---|---|
| GET    | `/api/health`                       | Sağlık kontrolü, model durumu |
| POST   | `/api/predict`                      | Tek yorum tahmini |
| POST   | `/api/explain`                      | LIME açıklaması (num_samples ayarlanabilir) |
| POST   | `/api/predict/batch`                | JSON list ile toplu tahmin |
| POST   | `/api/predict/batch/upload`         | Multipart CSV/TXT yükleme |
| GET    | `/api/stats/overview`               | Macro-F1, accuracy, F1-nötr |
| GET    | `/api/stats/confusion-matrix`       | 3-seed birleşik 4180 örneklik CM |
| GET    | `/api/stats/per-class`              | Sınıf bazlı precision/recall/F1 |
| GET    | `/api/stats/multi-model`            | 4 modelin karşılaştırması |
| GET    | `/api/stats/ablation`               | 5 aşamalı veri seti evrimi |
| GET    | `/api/stats/dataset`                | Sınıf/kaynak/split dağılımı |
| GET    | `/api/stats/error-summary`          | Hata özeti (yön/kategori/güven) |
| GET    | `/api/errors?direction=&category=`  | Filtreli/sayfalı 148 hata |
| GET    | `/api/errors/{id}`                  | Tek hata kaydı |
| GET    | `/api/examples`                     | 8 hazır demo örneği |

## Performans (CUDA 12.1 + RTX 3060)

| Endpoint | Süre |
|---|---|
| `/api/predict` (tek yorum) | 50-100 ms |
| `/api/explain` (kısa yorum, num_samples=300) | ~1 sn |
| `/api/explain` (uzun yorum, 90+ kelime) | ~4 sn |
| `/api/predict/batch` (5 yorum) | ~500 ms |
| `/api/stats/*` | <10 ms (RAM) |

CPU'da inference 3-5×, LIME 6-10× yavaşlar.

## Yapı

```
demo/
├── README.md
├── .gitignore
├── backend/
│   ├── config.py              # MODEL_DIR validation, CORS
│   ├── data.py                # Statik metrikler (overview, CM, multi-model, ablation, examples)
│   ├── error_store.py         # reports/xai_final/error_analysis.csv RAM'de
│   ├── inference.py           # ModelWrapper (predict + LIME explain)
│   ├── main.py                # FastAPI app, lifespan, 14 endpoint
│   ├── schemas.py             # Pydantic v2 modelleri
│   ├── requirements.txt
│   └── README.md
└── frontend/
    ├── package.json           # React 18 + Vite + Tailwind + TanStack Query + Recharts
    ├── tsconfig.json
    ├── vite.config.ts         # @-alias + manualChunks
    ├── tailwind.config.ts     # Sentiment palette
    ├── public/favicon.svg
    └── src/
        ├── api/               # client.ts (14 endpoint), types.ts (TypeScript)
        ├── components/        # ui/, layout/, prediction/, dashboard/, batch/, errors/, examples/
        ├── hooks/             # usePredict, useExplain, useBatch
        ├── lib/               # utils.ts, constants.ts, export.ts (CSV)
        ├── pages/             # 5 sayfa + 404
        ├── App.tsx            # Router (lazy-loaded routes)
        └── main.tsx           # QueryClient, BrowserRouter, Toaster
```

## Notlar

- Model dosyaları (`models/optimized_v3/.../best_model/`) ve raporlar (`reports/xai_final/`) demo tarafından **yalnızca okunur**; kopyalanmaz.
- Backend ve frontend ayrı portlarda; CORS yalnızca `localhost:5173` için açık.
- Demo Türkçe karakterleri (UTF-8: `ö`, `ş`, `ç`, `ğ`, `ı`, `ü`) tutarlı kullanır. Konsolda `cp1252` kaynaklı görsel bozulma olabilir; tarayıcı ve JSON yanıtı düzgündür.
- Test/auth/.env yok (kapsam dışı).

## Ekip

- **Geliştiriciler:** Ahmet Dülger · Muhammed Esat Demir
- **Danışman:** Dr. Öğr. Üyesi İlkay Yelmen
- 2026 · İstinye Üniversitesi
