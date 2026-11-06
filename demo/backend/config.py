from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

MODEL_DIR = (
    PROJECT_ROOT
    / "models"
    / "optimized_v3"
    / "v3_genisletilmis"
    / "bert-base-turkish-cased_len256_ep5_lr2e-05_losswce_seed43"
    / "best_model"
)

REPORTS_DIR = PROJECT_ROOT / "reports" / "xai_final"
EDA_DIR = PROJECT_ROOT / "reports" / "eda"

LABELS = ["negatif", "nötr", "pozitif"]

MAX_SEQ_LENGTH = 256
MAX_INPUT_WORDS = 1000
MAX_BATCH_ROWS = 50

CORS_ORIGINS = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
]


def validate_model_dir() -> None:
    if not MODEL_DIR.exists():
        raise FileNotFoundError(f"MODEL_DIR not found: {MODEL_DIR}")
    required = ["config.json", "tokenizer.json", "tokenizer_config.json"]
    weights = ["model.safetensors", "pytorch_model.bin"]
    missing = [f for f in required if not (MODEL_DIR / f).exists()]
    if missing:
        raise FileNotFoundError(f"MODEL_DIR missing required files: {missing}")
    if not any((MODEL_DIR / w).exists() for w in weights):
        raise FileNotFoundError(
            f"MODEL_DIR has no weights file (expected one of {weights})"
        )
