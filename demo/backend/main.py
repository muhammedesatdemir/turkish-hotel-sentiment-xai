import csv
import io
import logging
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, HTTPException, Query, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from config import (
    CORS_ORIGINS,
    EDA_DIR,
    MAX_BATCH_ROWS,
    MAX_INPUT_WORDS,
    MODEL_DIR,
    REPORTS_DIR,
    validate_model_dir,
)
from data import (
    ABLATION_RESULTS,
    CLASSICAL_BASELINE,
    CONFUSION_MATRIX_3SEED,
    CV_RESULTS,
    DATASET_DISTRIBUTION,
    EDA_SUMMARY,
    ERROR_SUMMARY,
    EXAMPLES,
    LEAKAGE_AUDIT,
    MCNEMAR_SUMMARY,
    MULTI_MODEL_RESULTS,
    OVERVIEW_METRICS,
    PER_CLASS_METRICS_3SEED,
    SARCASM_ANALYSIS,
)
from error_store import ErrorStore
from inference import ModelWrapper
from schemas import (
    BatchItem,
    BatchRequest,
    BatchResponse,
    ExplainRequest,
    ExplainResponse,
    HealthResponse,
    PredictRequest,
    PredictResponse,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("demo.backend")


@asynccontextmanager
async def lifespan(app: FastAPI):
    validate_model_dir()
    logger.info("Validated MODEL_DIR=%s", MODEL_DIR)
    app.state.model = ModelWrapper(MODEL_DIR)

    error_csv = REPORTS_DIR / "error_analysis.csv"
    try:
        app.state.error_store = ErrorStore(error_csv)
    except Exception:
        logger.exception("Failed to load ErrorStore from %s", error_csv)
        app.state.error_store = None

    logger.info("Startup complete. Ready to serve requests.")
    yield
    logger.info("Shutting down.")


app = FastAPI(
    title="Türkçe Otel Yorumu Duygu Analizi",
    description="BERTurk fine-tuned model demo backend",
    version="0.2.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

# Statik EDA görsellerini servis et (reports/eda/*.png)
if EDA_DIR.exists():
    app.mount("/static/eda", StaticFiles(directory=EDA_DIR), name="eda")
    logger.info("Mounted EDA static files from %s", EDA_DIR)
else:
    logger.warning("EDA_DIR not found: %s — /static/eda mount skipped", EDA_DIR)


def _get_model(request: Request) -> ModelWrapper:
    model: ModelWrapper | None = getattr(request.app.state, "model", None)
    if model is None:
        raise HTTPException(status_code=503, detail="Model yüklenmemiş")
    return model


def _get_error_store(request: Request) -> ErrorStore:
    store: ErrorStore | None = getattr(request.app.state, "error_store", None)
    if store is None:
        raise HTTPException(status_code=503, detail="Hata analizi verisi yüklenmemiş")
    return store


@app.get("/")
def root():
    return {"name": "tr-hotel-sentiment-demo", "status": "ok"}


@app.get("/api/health", response_model=HealthResponse)
def health(request: Request):
    model: ModelWrapper | None = getattr(request.app.state, "model", None)
    store: ErrorStore | None = getattr(request.app.state, "error_store", None)
    if model is None:
        return HealthResponse(
            status="loading",
            model_loaded=False,
            model_dir=str(MODEL_DIR),
            device="unknown",
            labels=[],
            error_rows=0,
        )
    return HealthResponse(
        status="ok",
        model_loaded=True,
        model_dir=str(MODEL_DIR),
        device=str(model.device),
        labels=model.labels,
        error_rows=len(store.df) if store else 0,
    )


@app.post("/api/predict", response_model=PredictResponse)
def predict(req: PredictRequest, request: Request):
    text = req.text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="Metin boş olamaz")
    if len(text.split()) > MAX_INPUT_WORDS:
        raise HTTPException(
            status_code=413,
            detail=f"Maksimum {MAX_INPUT_WORDS} kelime",
        )
    model = _get_model(request)
    try:
        result = model.predict(text)
    except Exception:
        logger.exception("Inference failed")
        raise HTTPException(status_code=500, detail="Inference hatası")
    return PredictResponse(**result)


@app.post("/api/explain", response_model=ExplainResponse)
def explain(req: ExplainRequest, request: Request):
    text = req.text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="Metin boş olamaz")
    if len(text.split()) > MAX_INPUT_WORDS:
        raise HTTPException(
            status_code=413,
            detail=f"Maksimum {MAX_INPUT_WORDS} kelime",
        )
    model = _get_model(request)
    try:
        result = model.explain(text, num_samples=req.num_samples)
    except Exception:
        logger.exception("LIME failed")
        raise HTTPException(status_code=500, detail="LIME açıklaması üretilemedi")
    return ExplainResponse(**result)


@app.post("/api/predict/batch", response_model=BatchResponse)
def predict_batch(req: BatchRequest, request: Request):
    if len(req.texts) > MAX_BATCH_ROWS:
        raise HTTPException(
            status_code=413,
            detail=f"Maksimum {MAX_BATCH_ROWS} yorum",
        )
    model = _get_model(request)
    t0 = time.perf_counter()
    items: list[BatchItem] = []
    distribution = {label: 0 for label in model.labels}
    try:
        for i, text in enumerate(req.texts):
            text = (text or "").strip()
            if not text:
                continue
            r = model.predict(text)
            distribution[r["sentiment"]] += 1
            items.append(
                BatchItem(
                    index=i,
                    text=text,
                    sentiment=r["sentiment"],
                    confidence=r["confidence"],
                    probabilities=r["probabilities"],
                    wc=r["wc"],
                )
            )
    except Exception:
        logger.exception("Batch inference failed")
        raise HTTPException(status_code=500, detail="Batch inference hatası")
    elapsed_ms = int((time.perf_counter() - t0) * 1000)
    return BatchResponse(
        items=items,
        total=len(items),
        inference_time_ms=elapsed_ms,
        distribution=distribution,
    )


@app.post("/api/predict/batch/upload", response_model=BatchResponse)
async def predict_batch_upload(request: Request, file: UploadFile = File(...)):
    if not file.filename:
        raise HTTPException(status_code=400, detail="Dosya seçilmedi")
    raw = await file.read()
    try:
        text_str = raw.decode("utf-8")
    except UnicodeDecodeError:
        try:
            text_str = raw.decode("cp1254")
        except Exception:
            raise HTTPException(status_code=400, detail="Dosya UTF-8 olmalı")

    texts = _extract_texts_from_csv(text_str)
    if not texts:
        raise HTTPException(
            status_code=400,
            detail="Geçerli metin bulunamadı (tek sütunlu CSV bekleniyor)",
        )
    if len(texts) > MAX_BATCH_ROWS:
        raise HTTPException(
            status_code=413,
            detail=f"Maksimum {MAX_BATCH_ROWS} satır (yüklenen: {len(texts)})",
        )
    return predict_batch(BatchRequest(texts=texts), request)


def _extract_texts_from_csv(content: str) -> list[str]:
    """Read first non-empty cell of each row; skip a 'comment_text'/'text' header."""
    texts: list[str] = []
    reader = csv.reader(io.StringIO(content))
    for i, row in enumerate(reader):
        if not row:
            continue
        cell = (row[0] or "").strip()
        if not cell:
            continue
        if i == 0 and cell.lower() in {"text", "comment", "comment_text", "yorum"}:
            continue
        texts.append(cell)
    return texts


@app.get("/api/stats/overview")
def stats_overview():
    return OVERVIEW_METRICS


@app.get("/api/stats/confusion-matrix")
def stats_confusion_matrix():
    return CONFUSION_MATRIX_3SEED


@app.get("/api/stats/per-class")
def stats_per_class():
    return PER_CLASS_METRICS_3SEED


@app.get("/api/stats/multi-model")
def stats_multi_model():
    return MULTI_MODEL_RESULTS


@app.get("/api/stats/ablation")
def stats_ablation():
    return ABLATION_RESULTS


@app.get("/api/stats/dataset")
def stats_dataset():
    return DATASET_DISTRIBUTION


@app.get("/api/stats/cv")
def stats_cv():
    return CV_RESULTS


@app.get("/api/stats/mcnemar")
def stats_mcnemar():
    return MCNEMAR_SUMMARY


@app.get("/api/stats/classical-baseline")
def stats_classical_baseline():
    return CLASSICAL_BASELINE


@app.get("/api/stats/sarcasm")
def stats_sarcasm():
    return SARCASM_ANALYSIS


@app.get("/api/stats/leakage")
def stats_leakage():
    return LEAKAGE_AUDIT


@app.get("/api/stats/eda")
def stats_eda():
    images: list[dict] = []
    if EDA_DIR.exists():
        meta = {
            "01_class_dist.png": {
                "title": "Sınıf dağılımı",
                "description": "14.337 yorumun negatif/nötr/pozitif sınıflara göre sayı ve oran dağılımı.",
            },
            "02_wc_box.png": {
                "title": "Sınıf bazında kelime sayısı (tüm veri seti, AI dahil)",
                "description": (
                    "Tüm 14.337 yorumun kelime uzunluğu dağılımı. Nötr medyanı 8 kelime "
                    "GÖRÜNÜR; bu AI-üretimli 2.500 nötr örneğin (medyan 6 kelime, %53'ü) etkisidir. "
                    "AI hariç tutulduğunda — yani test setinin popülasyonunda — nötr medyanı "
                    "gerçekte 38 kelime'dir (otelpuan: 31, manuel_toplama: 53). AI yorumları "
                    "test setine sızdırılmadığı için bu medyan farkı yalnızca train tarafındadır."
                ),
            },
            "03_source_label_heatmap.png": {
                "title": "Kaynak × etiket heatmap",
                "description": "4 kaynak (otelpuan, ai_generated, manuel_toplama, neredekal) × 3 sınıf çapraz tablo.",
            },
            "04_top_words_per_class.png": {
                "title": "Sınıf bazında en sık 20 kelime",
                "description": "TF-IDF olmadan ham frekans; modelin gördüğü leksikal sinyal.",
            },
            "05_corr_matrix.png": {
                "title": "Pearson korelasyon matrisi (10 meta-özellik)",
                "description": "rating_auto ↔ sentiment_id = +0,8127 (etiketleme tutarlılığı). Simpson Paradoksu yorumu §4.5'te.",
            },
        }
        for fname, m in meta.items():
            if (EDA_DIR / fname).exists():
                images.append({
                    "url": f"/static/eda/{fname}",
                    "title": m["title"],
                    "description": m["description"],
                })

    return {**EDA_SUMMARY, "images": images}


@app.get("/api/stats/error-summary")
def stats_error_summary():
    return ERROR_SUMMARY


@app.get("/api/examples")
def examples():
    return {"items": EXAMPLES}


@app.get("/api/errors")
def list_errors(
    request: Request,
    direction: str | None = Query(None, description="yanlis_yon filter, e.g. 'nötr_to_pozitif' or 'all'"),
    category: str | None = Query(None, description="kategori filter, e.g. 'guvenli_yanlis' or 'all'"),
    min_conf: float | None = Query(None, ge=0.0, le=1.0),
    max_conf: float | None = Query(None, ge=0.0, le=1.0),
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
):
    store = _get_error_store(request)
    return store.list(
        direction=direction,
        category=category,
        min_conf=min_conf,
        max_conf=max_conf,
        limit=limit,
        offset=offset,
    )


@app.get("/api/errors/{error_id}")
def get_error(error_id: int, request: Request):
    store = _get_error_store(request)
    item = store.get(error_id)
    if item is None:
        raise HTTPException(status_code=404, detail="Hata kaydı bulunamadı")
    return item
