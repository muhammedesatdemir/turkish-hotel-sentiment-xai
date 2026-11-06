# -*- coding: utf-8 -*-
"""
Hocanin yeni talimatina gore: ELECTRA-tr modelinden 4 RENKLI grafik uretir.
  Fig.2  ROC curve (one-vs-rest)            <- ELECTRA gercek olasiliklari
  Fig.3  Confusion matrix                    <- ELECTRA gercek tahminleri
  Fig.4  SHAP summary plot                   <- ELECTRA gercek SHAP
  Fig.5  SHAP feature importance             <- ELECTRA gercek SHAP
Cikti: reports/ieee_electra_figurleri/
Cift dogrulama: canli tahmin vs kayitli test_predictions.csv
"""
import glob, collections
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn.preprocessing import label_binarize
from transformers import AutoTokenizer, AutoModelForSequenceClassification

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "reports" / "ieee_electra_figurleri"
OUT.mkdir(parents=True, exist_ok=True)
MODEL_DIR = glob.glob(str(ROOT / "models/optimized_v3/v3_electra_tr/*seed43/best_model"))[0]
TEST_CSV = glob.glob(str(ROOT / "models/optimized_v3/v3_electra_tr/*seed43/splits/test.csv"))[0]
PRED_CSV = glob.glob(str(ROOT / "models/optimized_v3/v3_electra_tr/*seed43/test_predictions.csv"))[0]

plt.rcParams["font.family"] = "DejaVu Sans"
plt.rcParams["font.size"] = 8
CLASSES = ["negative", "neutral", "positive"]
COL_W = 3.4
device = "cuda" if torch.cuda.is_available() else "cpu"
# Renkli palet (IEEE uyumlu, canli ama abartisiz)
C_ROC = ["#D62728", "#FF7F0E", "#2CA02C"]  # neg/notr/poz
CMAP = "viridis"
BAR = "#1F77B4"

print("ELECTRA model yukleniyor:", MODEL_DIR)
tok = AutoTokenizer.from_pretrained(MODEL_DIR)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR).to(device).eval()

df = pd.read_csv(TEST_CSV)
text_col = "comment_text" if "comment_text" in df.columns else df.columns[0]
if "labels" in df.columns:
    y_true = df["labels"].astype(int).values
else:
    m = {"negatif": 0, "nötr": 1, "pozitif": 2}
    y_true = df["sentiment_label"].map(m).astype(int).values
texts = df[text_col].astype(str).tolist()
print("Test ornegi:", len(texts))


@torch.no_grad()
def predict_proba(text_list, bs=32):
    out = []
    for i in range(0, len(text_list), bs):
        enc = tok(text_list[i:i+bs], return_tensors="pt", truncation=True,
                  padding=True, max_length=256).to(device)
        out.append(torch.softmax(model(**enc).logits, dim=-1).cpu().numpy())
    return np.vstack(out)


print("Olasiliklar (ELECTRA gercek cikti)...")
proba = predict_proba(texts)
y_pred = proba.argmax(axis=1)

pred_df = pd.read_csv(PRED_CSV)
if "pred_id" in pred_df.columns and len(pred_df) == len(y_pred):
    agree = (pred_df["pred_id"].astype(int).values == y_pred).mean()
    print(f"DOGRULAMA: canli vs kayitli test_predictions.csv = {agree:.4f}")
print(f"ELECTRA test accuracy (canli) = {(y_pred==y_true).mean():.4f}")


def fig_cm():
    cm = confusion_matrix(y_true, y_pred)
    cmn = cm / cm.sum(axis=1, keepdims=True)
    fig, ax = plt.subplots(figsize=(COL_W, 2.9), dpi=300)
    im = ax.imshow(cmn, cmap=CMAP, vmin=0, vmax=1)
    ax.set_xticks(range(3)); ax.set_yticks(range(3))
    ax.set_xticklabels(CLASSES, fontsize=8); ax.set_yticklabels(CLASSES, fontsize=8)
    ax.set_xlabel("Predicted", fontsize=8.5); ax.set_ylabel("True", fontsize=8.5)
    for i in range(3):
        for j in range(3):
            ax.text(j, i, f"{cm[i,j]}\n({cmn[i,j]*100:.1f}%)", ha="center", va="center",
                    fontsize=7, color="white" if cmn[i, j] < 0.6 else "black")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout(pad=0.3)
    p = OUT / "fig3_confusion_matrix_electra.png"
    fig.savefig(p, bbox_inches="tight", facecolor="white"); plt.close(fig)
    print("yazildi:", p.name)


def fig_roc():
    yb = label_binarize(y_true, classes=[0, 1, 2])
    fig, ax = plt.subplots(figsize=(COL_W, 2.9), dpi=300)
    for i, cname in enumerate(CLASSES):
        fpr, tpr, _ = roc_curve(yb[:, i], proba[:, i])
        ax.plot(fpr, tpr, color=C_ROC[i], lw=1.4, label=f"{cname} (AUC={auc(fpr,tpr):.3f})")
    ax.plot([0, 1], [0, 1], ":", color="gray", lw=0.8)
    ax.set_xlabel("False Positive Rate", fontsize=8.5)
    ax.set_ylabel("True Positive Rate", fontsize=8.5)
    ax.legend(fontsize=6.5, loc="lower right")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.02)
    fig.tight_layout(pad=0.3)
    p = OUT / "fig2_roc_curve_electra.png"
    fig.savefig(p, bbox_inches="tight", facecolor="white"); plt.close(fig)
    print("yazildi:", p.name)
    for i, c in enumerate(CLASSES):
        fpr, tpr, _ = roc_curve(yb[:, i], proba[:, i])
        print(f"  AUC[{c}] = {auc(fpr,tpr):.4f}")


fig_cm()
fig_roc()


def fig_shap():
    import shap
    rng = np.random.RandomState(42)
    idx = []
    for c in [0, 1, 2]:
        ci = np.where(y_true == c)[0]
        idx.extend(rng.choice(ci, min(25, len(ci)), replace=False).tolist())
    sub = [texts[i] for i in idx][:60]
    print(f"SHAP altkume: {len(sub)} ornek")

    def f(x):
        return predict_proba(list(x), bs=16)

    masker = shap.maskers.Text(tok)
    explainer = shap.Explainer(f, masker, output_names=CLASSES)
    sv = explainer(sub)

    abs_c = collections.defaultdict(float); cnt = collections.defaultdict(int)
    for i in range(len(sv)):
        vals = np.abs(sv[i].values).sum(axis=-1) if sv[i].values.ndim > 1 else np.abs(sv[i].values)
        for t, v in zip(sv[i].data, vals):
            t = str(t).strip()
            if len(t) < 2:
                continue
            abs_c[t] += float(v); cnt[t] += 1
    items = sorted([(t, abs_c[t]/cnt[t]) for t in abs_c if cnt[t] >= 2], key=lambda x: -x[1])[:12][::-1]

    fig, ax = plt.subplots(figsize=(COL_W, 3.0), dpi=300)
    ax.barh(range(len(items)), [v for _, v in items], color=BAR, edgecolor="black", lw=0.4, height=0.7)
    ax.set_yticks(range(len(items))); ax.set_yticklabels([t for t, _ in items], fontsize=7)
    ax.set_xlabel("Mean |SHAP value|", fontsize=8.5)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(pad=0.3)
    p5 = OUT / "fig5_shap_importance_electra.png"
    fig.savefig(p5, bbox_inches="tight", facecolor="white"); plt.close(fig)
    print("yazildi:", p5.name)

    try:
        fig = plt.figure(figsize=(COL_W, 3.0), dpi=300)
        shap.plots.bar(sv[:, :, 2].mean(0), max_display=12, show=False)
        plt.tight_layout()
        p4 = OUT / "fig4_shap_summary_electra.png"
        plt.savefig(p4, bbox_inches="tight", facecolor="white"); plt.close()
        print("yazildi:", p4.name)
    except Exception as e:
        print("Fig4 hata:", e)


try:
    fig_shap()
except Exception as e:
    print("SHAP hata:", repr(e))
print("OUT:", OUT)
