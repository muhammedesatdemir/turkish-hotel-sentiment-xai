from typing import Any, Literal

from pydantic import BaseModel, Field

Sentiment = Literal["negatif", "nötr", "pozitif"]


class PredictRequest(BaseModel):
    text: str = Field(..., min_length=1, description="Türkçe otel yorumu")


class PredictResponse(BaseModel):
    sentiment: Sentiment
    confidence: float
    probabilities: dict[str, float]
    wc: int
    inference_time_ms: int


class ExplainRequest(BaseModel):
    text: str = Field(..., min_length=1)
    num_samples: int = Field(300, ge=50, le=2000)


class ExplainResponse(BaseModel):
    sentiment: Sentiment
    confidence: float
    probabilities: dict[str, float]
    lime_features: dict[str, list]
    tokens: list[str]
    num_samples: int
    explain_time_ms: int


class BatchRequest(BaseModel):
    texts: list[str] = Field(..., min_length=1, max_length=50)


class BatchItem(BaseModel):
    index: int
    text: str
    sentiment: Sentiment
    confidence: float
    probabilities: dict[str, float]
    wc: int


class BatchResponse(BaseModel):
    items: list[BatchItem]
    total: int
    inference_time_ms: int
    distribution: dict[str, int]


class HealthResponse(BaseModel):
    status: Literal["ok", "loading", "error"]
    model_loaded: bool
    model_dir: str
    device: str
    labels: list[str]
    error_rows: int


class StaticPayload(BaseModel):
    """Generic wrapper for hardcoded dict payloads (no strict shape enforcement)."""
    data: dict[str, Any]
