"""Hata analizi CSV'sini başlangıçta yükler, in-memory filtreleme sağlar."""
from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)


class ErrorStore:
    REQUIRED_COLS = [
        "comment_text", "sentiment_label", "pred_label", "yanlis_yon", "wc",
        "p_negatif", "p_notr", "p_pozitif", "max_prob", "second_prob",
        "kategori", "source",
    ]

    def __init__(self, csv_path: Path):
        self.csv_path = csv_path
        if not csv_path.exists():
            raise FileNotFoundError(f"Error analysis CSV not found: {csv_path}")
        df = pd.read_csv(csv_path, encoding="utf-8")
        missing = [c for c in self.REQUIRED_COLS if c not in df.columns]
        if missing:
            raise ValueError(
                f"error_analysis.csv missing columns: {missing}. Found: {df.columns.tolist()}"
            )
        df = df.reset_index(drop=True)
        df["id"] = df.index
        self.df = df
        logger.info("Loaded %d error rows from %s", len(df), csv_path)

    def list(
        self,
        direction: str | None = None,
        category: str | None = None,
        min_conf: float | None = None,
        max_conf: float | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> dict:
        df = self.df
        if direction and direction != "all":
            df = df[df["yanlis_yon"] == direction]
        if category and category != "all":
            df = df[df["kategori"] == category]
        if min_conf is not None:
            df = df[df["max_prob"] >= min_conf]
        if max_conf is not None:
            df = df[df["max_prob"] <= max_conf]

        total = len(df)
        page = df.iloc[offset : offset + limit]
        rows = []
        for _, r in page.iterrows():
            rows.append(
                {
                    "id": int(r["id"]),
                    "text": str(r["comment_text"]),
                    "true": str(r["sentiment_label"]),
                    "pred": str(r["pred_label"]),
                    "direction": str(r["yanlis_yon"]),
                    "wc": int(r["wc"]),
                    "probabilities": {
                        "negatif": float(r["p_negatif"]),
                        "nötr":    float(r["p_notr"]),
                        "pozitif": float(r["p_pozitif"]),
                    },
                    "max_prob": float(r["max_prob"]),
                    "second_prob": float(r["second_prob"]),
                    "category": str(r["kategori"]),
                    "source": str(r["source"]),
                }
            )
        return {
            "total": total,
            "limit": limit,
            "offset": offset,
            "rows": rows,
        }

    def get(self, error_id: int) -> dict | None:
        if error_id < 0 or error_id >= len(self.df):
            return None
        r = self.df.iloc[error_id]
        return {
            "id": int(r["id"]),
            "text": str(r["comment_text"]),
            "true": str(r["sentiment_label"]),
            "pred": str(r["pred_label"]),
            "direction": str(r["yanlis_yon"]),
            "wc": int(r["wc"]),
            "probabilities": {
                "negatif": float(r["p_negatif"]),
                "nötr":    float(r["p_notr"]),
                "pozitif": float(r["p_pozitif"]),
            },
            "max_prob": float(r["max_prob"]),
            "second_prob": float(r["second_prob"]),
            "category": str(r["kategori"]),
            "source": str(r["source"]),
        }
