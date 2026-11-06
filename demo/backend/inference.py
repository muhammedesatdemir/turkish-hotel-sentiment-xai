import logging
import time
from pathlib import Path

import numpy as np
import torch
from lime.lime_text import LimeTextExplainer
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from config import LABELS, MAX_SEQ_LENGTH

logger = logging.getLogger(__name__)

LIME_BATCH_SIZE = 32


class ModelWrapper:
    def __init__(self, model_dir: Path):
        self.model_dir = model_dir
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info("Loading tokenizer from %s", model_dir)
        self.tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
        logger.info("Loading model from %s onto %s", model_dir, self.device)
        self.model = AutoModelForSequenceClassification.from_pretrained(str(model_dir))
        self.model.to(self.device)
        self.model.eval()

        id2label = self.model.config.id2label
        ordered = [id2label[i] for i in sorted(id2label.keys(), key=int)] \
            if all(isinstance(k, str) for k in id2label.keys()) \
            else [id2label[i] for i in sorted(id2label.keys())]
        if ordered != LABELS:
            raise RuntimeError(
                f"Model id2label order {ordered} does not match expected {LABELS}"
            )
        self.labels = LABELS
        logger.info("Model loaded. Labels: %s", self.labels)

    @torch.inference_mode()
    def predict(self, text: str) -> dict:
        t0 = time.perf_counter()
        enc = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=MAX_SEQ_LENGTH,
            padding=False,
        ).to(self.device)
        logits = self.model(**enc).logits[0]
        probs = torch.softmax(logits, dim=-1).cpu().numpy()
        pred_idx = int(np.argmax(probs))
        elapsed_ms = int((time.perf_counter() - t0) * 1000)
        return {
            "sentiment": self.labels[pred_idx],
            "confidence": float(probs[pred_idx]),
            "probabilities": {
                self.labels[i]: float(probs[i]) for i in range(len(self.labels))
            },
            "wc": len(text.split()),
            "inference_time_ms": elapsed_ms,
        }

    @torch.inference_mode()
    def predict_proba_batch(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, len(self.labels)), dtype=np.float32)
        out = []
        for i in range(0, len(texts), LIME_BATCH_SIZE):
            chunk = texts[i : i + LIME_BATCH_SIZE]
            enc = self.tokenizer(
                chunk,
                return_tensors="pt",
                truncation=True,
                max_length=MAX_SEQ_LENGTH,
                padding=True,
            ).to(self.device)
            logits = self.model(**enc).logits
            probs = torch.softmax(logits, dim=-1).cpu().numpy()
            out.append(probs)
        return np.vstack(out)

    def explain(
        self,
        text: str,
        num_samples: int = 300,
        num_features: int = 12,
        top_k_per_class: int = 8,
    ) -> dict:
        t0 = time.perf_counter()
        explainer = LimeTextExplainer(
            class_names=self.labels,
            bow=False,
            random_state=42,
        )
        exp = explainer.explain_instance(
            text,
            self.predict_proba_batch,
            num_features=num_features,
            num_samples=num_samples,
            labels=tuple(range(len(self.labels))),
        )
        probs = self.predict_proba_batch([text])[0]
        pred_idx = int(np.argmax(probs))
        lime_features: dict[str, list[list]] = {}
        for cid, cname in enumerate(self.labels):
            feats = exp.as_list(label=cid)[:top_k_per_class]
            lime_features[cname] = [[w, float(s)] for w, s in feats]
        elapsed_ms = int((time.perf_counter() - t0) * 1000)
        return {
            "sentiment": self.labels[pred_idx],
            "confidence": float(probs[pred_idx]),
            "probabilities": {
                self.labels[i]: float(probs[i]) for i in range(len(self.labels))
            },
            "lime_features": lime_features,
            "tokens": text.split(),
            "num_samples": num_samples,
            "explain_time_ms": elapsed_ms,
        }
