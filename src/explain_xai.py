"""
src/explain_xai.py
==================

v3_genisletilmis/seed43/best_model uzerinde LIME + SHAP ile aciklama uretir.

Uretilen ciktilar (reports/xai/ altinda):
  - lime_case_<N>.html           : her ornek icin LIME kelime onemleri
  - shap_case_<N>.html            : her ornek icin SHAP token atifleri
  - summary.json                  : ornek listesi + tahmin + top-5 onemli token
  - README.md                     : hizli oku-anla ozeti

Kullanim:
  python src/explain_xai.py --n_examples 8

Varsayilan cikis: reports/xai/
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from transformers import AutoModelForSequenceClassification, AutoTokenizer

import shap
from lime.lime_text import LimeTextExplainer


# ============================================================================
# Sabitler
# ============================================================================

LABELS = ["negatif", "nötr", "pozitif"]
LABEL2ID = {l: i for i, l in enumerate(LABELS)}
ID2LABEL = {i: l for l, i in LABEL2ID.items()}

DEFAULT_MODEL_DIR = Path(
    "models/optimized_v3/v3_genisletilmis/"
    "bert-base-turkish-cased_len256_ep5_lr2e-05_losswce_seed43/best_model"
)
DEFAULT_PRED_CSV = Path(
    "models/optimized_v3/v3_genisletilmis/"
    "bert-base-turkish-cased_len256_ep5_lr2e-05_losswce_seed43/test_predictions.csv"
)
DEFAULT_OUT = Path("reports/xai")


# ============================================================================
# Model yukleme ve tahmin fonksiyonu (LIME ve SHAP icin ortak)
# ============================================================================

class ModelWrapper:
    def __init__(self, model_dir: Path, device: str = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
        self.model = AutoModelForSequenceClassification.from_pretrained(str(model_dir))
        self.model.to(self.device)
        self.model.eval()

    @torch.no_grad()
    def predict_proba(self, texts: List[str]) -> np.ndarray:
        """LIME/SHAP pipeline'i icin: metin listesi -> (n, 3) olasilik matrisi."""
        if isinstance(texts, str):
            texts = [texts]
        # Batch'le (LIME 5000 sample gonderebilir)
        out_probs = []
        BS = 32
        for i in range(0, len(texts), BS):
            batch = texts[i : i + BS]
            enc = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=256,
                return_tensors="pt",
            ).to(self.device)
            logits = self.model(**enc).logits
            probs = F.softmax(logits, dim=-1).cpu().numpy()
            out_probs.append(probs)
        return np.vstack(out_probs)


# ============================================================================
# Ornek secimi
# ============================================================================

def select_examples(pred_csv: Path) -> pd.DataFrame:
    """Hata analizi icin farkli kategorilerden ornekler sec."""
    df = pd.read_csv(pred_csv)
    df["wc"] = df["comment_text"].astype(str).str.split().str.len()

    picks = []

    # 1-3: her sinif icin en güvenli DOGRU tahmin (kisa ve net)
    for lbl in LABELS:
        sub = df[
            (df.correct == 1)
            & (df.sentiment_label == lbl)
            & (df.wc.between(10, 35))
        ]
        if len(sub):
            # Ilk birini al (deterministic, sort by index)
            picks.append(
                (f"dogru_{lbl}", sub.sort_index().iloc[0])
            )

    # 4: en yaygin hata — notr -> negatif (69 vaka)
    sub = df[(df.sentiment_label == "nötr") & (df.pred_label == "negatif") & (df.wc.between(15, 45))]
    if len(sub):
        picks.append(("hata_notr_negatif", sub.sort_index().iloc[0]))

    # 5: notr -> pozitif (42 vaka)
    sub = df[(df.sentiment_label == "nötr") & (df.pred_label == "pozitif") & (df.wc.between(15, 45))]
    if len(sub):
        picks.append(("hata_notr_pozitif", sub.sort_index().iloc[0]))

    # 6: pozitif -> notr (24 vaka; pozitif yorumu nötr sanıyor)
    sub = df[(df.sentiment_label == "pozitif") & (df.pred_label == "nötr") & (df.wc.between(15, 45))]
    if len(sub):
        picks.append(("hata_pozitif_notr", sub.sort_index().iloc[0]))

    # 7: negatif -> notr (22 vaka; olumsuzu nötr sanıyor)
    sub = df[(df.sentiment_label == "negatif") & (df.pred_label == "nötr") & (df.wc.between(15, 45))]
    if len(sub):
        picks.append(("hata_negatif_notr", sub.sort_index().iloc[0]))

    # 8: notr dogru (zor, cunku notr en zorlu sinif)
    sub = df[
        (df.correct == 1)
        & (df.sentiment_label == "nötr")
        & (df.wc.between(20, 50))
    ]
    if len(sub):
        picks.append(("dogru_notr_uzun", sub.sort_index().iloc[1 if len(sub) > 1 else 0]))

    # DataFrame'e cevir
    rows = []
    for kind, row in picks:
        rows.append(
            {
                "kind": kind,
                "text": row.comment_text,
                "wc": int(row.wc),
                "true": row.sentiment_label,
                "pred": row.pred_label,
                "correct": int(row.correct),
            }
        )
    return pd.DataFrame(rows)


# ============================================================================
# LIME aciklamalari
# ============================================================================

def run_lime(
    wrapper: ModelWrapper,
    examples_df: pd.DataFrame,
    out_dir: Path,
    num_samples: int = 1000,
    num_features: int = 12,
) -> List[dict]:
    """Her ornek icin LIME HTML ciktisi kaydet, top-features listesini dondur."""
    explainer = LimeTextExplainer(
        class_names=LABELS,
        bow=False,  # BoW kapali -> pozisyonel aciklama
        random_state=42,
    )

    lime_summary = []
    for i, row in examples_df.iterrows():
        text = str(row.text)
        true_label = row["true"]
        pred_label = row["pred"]
        kind = row["kind"]

        print(f"[LIME] case {i+1}/{len(examples_df)} ({kind}) ...", flush=True)

        # predict_proba(texts) -> (n, 3)
        exp = explainer.explain_instance(
            text,
            wrapper.predict_proba,
            num_features=num_features,
            num_samples=num_samples,
            labels=(0, 1, 2),
        )

        # HTML kaydet
        html_path = out_dir / f"lime_case_{i+1:02d}_{kind}.html"
        exp.save_to_file(str(html_path))

        # Tahmin edilen label icin top-features
        pred_id = LABEL2ID[pred_label]
        top_feats_pred = exp.as_list(label=pred_id)[:8]
        # True label icin de top-features
        true_id = LABEL2ID[true_label]
        top_feats_true = exp.as_list(label=true_id)[:8]

        lime_summary.append(
            {
                "idx": int(i) + 1,
                "kind": kind,
                "text": text,
                "true": true_label,
                "pred": pred_label,
                "correct": int(row.correct),
                "html": html_path.name,
                "top_features_for_predicted": top_feats_pred,
                "top_features_for_true": top_feats_true,
            }
        )

    return lime_summary


# ============================================================================
# SHAP aciklamalari (transformers pipeline ile)
# ============================================================================

def run_shap(
    wrapper: ModelWrapper,
    examples_df: pd.DataFrame,
    out_dir: Path,
) -> list:
    """SHAP text explainer — token bazli attribution.

    transformers.pipeline yerine dogrudan predict_proba wrap'lenir;
    maskeli text explainer kullanilir.
    """
    # SHAP Text explainer icin wrapper fonksiyon
    def f(texts):
        if isinstance(texts, np.ndarray):
            texts = texts.tolist()
        return wrapper.predict_proba(texts)

    masker = shap.maskers.Text(wrapper.tokenizer)
    explainer = shap.Explainer(f, masker, output_names=LABELS)

    shap_summary = []
    for i, row in examples_df.iterrows():
        text = str(row.text)
        kind = row["kind"]
        print(f"[SHAP] case {i+1}/{len(examples_df)} ({kind}) ...", flush=True)

        try:
            shap_values = explainer([text])
        except Exception as e:
            print(f"  [SHAP] HATA: {e}; atlaniyor")
            shap_summary.append(
                {
                    "idx": int(i) + 1,
                    "kind": kind,
                    "error": str(e),
                }
            )
            continue

        # HTML kaydet (text plot) - tum siniflar icin
        try:
            html_plot = shap.plots.text(shap_values, display=False)
            html_path = out_dir / f"shap_case_{i+1:02d}_{kind}.html"
            html_path.write_text(html_plot, encoding="utf-8")
        except Exception as e:
            print(f"  [SHAP] HTML render hatasi: {e}")
            html_path = None

        # Her sinif icin top-5 pozitif ve top-5 negatif katkili tokenlar
        # shap_values.values shape: (1, n_tokens, n_classes)
        try:
            tokens = shap_values.data[0]  # token listesi
            vals = shap_values.values[0]  # (n_tokens, 3)
            per_class_top = {}
            for cid, cname in enumerate(LABELS):
                pairs = [(str(t).strip(), float(vals[ti, cid])) for ti, t in enumerate(tokens)]
                pairs = [p for p in pairs if p[0]]  # bos olanlari at
                pos = sorted(pairs, key=lambda x: -x[1])[:5]
                neg = sorted(pairs, key=lambda x: x[1])[:5]
                per_class_top[cname] = {"pozitif_katki": pos, "negatif_katki": neg}
        except Exception as e:
            per_class_top = {"error": str(e)}

        shap_summary.append(
            {
                "idx": int(i) + 1,
                "kind": kind,
                "text": text,
                "true": row["true"],
                "pred": row["pred"],
                "html": html_path.name if html_path else None,
                "per_class_top_tokens": per_class_top,
            }
        )

    return shap_summary


# ============================================================================
# Ana akis
# ============================================================================

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model_dir", type=Path, default=DEFAULT_MODEL_DIR)
    p.add_argument("--pred_csv", type=Path, default=DEFAULT_PRED_CSV)
    p.add_argument("--out_dir", type=Path, default=DEFAULT_OUT)
    p.add_argument("--lime_samples", type=int, default=1000,
                   help="LIME perturbation sayisi (varsayilan 1000; hizli icin 500)")
    p.add_argument("--skip_shap", action="store_true",
                   help="SHAP'i atla (sadece LIME icin hizli deneme)")
    p.add_argument("--skip_lime", action="store_true")
    args = p.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Model yukle
    print(f"[INFO] model yukleniyor: {args.model_dir}")
    wrapper = ModelWrapper(args.model_dir)
    print(f"[INFO] device: {wrapper.device}")

    # 2. Ornek sec
    print(f"[INFO] ornekler: {args.pred_csv}")
    ex = select_examples(args.pred_csv)
    print(f"[INFO] secilen ornek sayisi: {len(ex)}")
    print(ex[["kind", "true", "pred", "wc"]].to_string())
    ex.to_csv(args.out_dir / "selected_examples.csv", index=False, encoding="utf-8")

    summary = {"model_dir": str(args.model_dir), "pred_csv": str(args.pred_csv)}
    summary["examples"] = ex.to_dict(orient="records")

    # 3. LIME
    if not args.skip_lime:
        print("\n[STAGE] LIME aciklamalari uretiliyor...")
        lime_summary = run_lime(
            wrapper, ex, args.out_dir, num_samples=args.lime_samples
        )
        summary["lime"] = lime_summary
    else:
        print("\n[STAGE] LIME atlandi.")

    # 4. SHAP
    if not args.skip_shap:
        print("\n[STAGE] SHAP aciklamalari uretiliyor...")
        shap_summary = run_shap(wrapper, ex, args.out_dir)
        summary["shap"] = shap_summary
    else:
        print("\n[STAGE] SHAP atlandi.")

    # 5. Summary JSON
    sum_path = args.out_dir / "summary.json"
    sum_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    print(f"\n[DONE] summary: {sum_path}")
    print(f"[DONE] cikti dizini: {args.out_dir.resolve()}")


if __name__ == "__main__":
    main()
