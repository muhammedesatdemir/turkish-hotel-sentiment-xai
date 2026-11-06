# Backend — FastAPI + BERTurk

## Çalıştır

```powershell
cd demo/backend
..\..\venv\Scripts\activate
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```

Swagger UI: <http://localhost:8000/docs>

## Yapı

| Dosya | Görev |
|---|---|
| `config.py`         | `MODEL_DIR`, `REPORTS_DIR`, `LABELS`, `CORS_ORIGINS`, `validate_model_dir()` |
| `schemas.py`        | Pydantic v2 modelleri (Predict / Explain / Batch / Health) |
| `inference.py`      | `ModelWrapper.predict()`, `predict_proba_batch()`, `explain()` (LIME) |
| `data.py`           | Statik dict'ler — overview, confusion matrix, per-class, multi-model, ablation, dataset, error summary, 8 örnek |
| `error_store.py`    | `reports/xai_final/error_analysis.csv` RAM'de, in-memory filtreleme |
| `main.py`           | FastAPI app, lifespan, CORS, 14 endpoint |

## Önemli noktalar

- Model startup'ta tek sefer yüklenir (`lifespan` async context manager). CUDA varsa GPU'ya alınır.
- `id2label` sırası `["negatif", "nötr", "pozitif"]` olarak doğrulanır; uyumsuzlukta `RuntimeError`.
- LIME varsayılan `num_samples=300` (planda öngörülenden 5× hızlı, ~1 sn). Query param ile 50-2000 arası ayarlanabilir.
- `/api/predict/batch` JSON listesi (max 50) ve `/api/predict/batch/upload` multipart CSV (header otomatik atlanır) iki ayrı endpoint.
- Tüm hata yanıtları HTTP standartlarına uygun: 400 (boş), 413 (büyük), 422 (Pydantic), 503 (model yüklenmemiş), 500 (inference).

## Bağımlılıklar (`requirements.txt`)

- `fastapi >= 0.110`
- `uvicorn[standard] >= 0.27`
- `pydantic >= 2.0`
- `python-multipart >= 0.0.9`

`transformers`, `torch`, `lime`, `pandas`, `numpy` zaten proje köküne ait `venv`'de mevcut.
