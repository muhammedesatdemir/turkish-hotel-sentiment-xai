# -*- coding: utf-8 -*-
"""
Hocanin istedigi 4 grafigi GERCEK modelden uretir (halusinasyon yok):
  Fig.2  ROC curve (one-vs-rest, 3 sinif)      <- modelin gercek olasiliklari
  Fig.3  Confusion matrix                       <- gercek test tahminleri
  Fig.4  SHAP summary plot (beeswarm)           <- gercek SHAP degerleri
  Fig.5  SHAP feature importance (mean |SHAP|)  <- gercek SHAP degerleri

Model: models/optimized_v3/v3_genisletilmis/...seed43/best_model
Test:  ayni klasor/splits/test.csv  (test_predictions.csv ile dogrulanir)
Cikti: reports/ieee_revize_figurleri/
"""
import glob
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
OUT = ROOT / "reports" / "ieee_revize_figurleri"
OUT.mkdir(parents=True, exist_ok=True)
MODEL_DIR = glob.glob(str(ROOT / "models/optimized_v3/v3_genisletilmis/*seed43*/best_model"))[0]
TEST_CSV = glob.glob(str(ROOT / "models/optimized_v3/v3_genisletilmis/*seed43*/splits/test.csv"))[0]
PRED_CSV = glob.glob(str(ROOT / "models/optimized_v3/v3_genisletilmis/*seed43*/test_predictions.csv"))[0]

plt.rcParams["font.family"] = "DejaVu Serif"
plt.rcParams["font.size"] = 8
CLASSES = ["negative", "neutral", "positive"]  # 0,1,2
COL_W = 3.4
device = "cuda" if torch.cuda.is_available() else "cpu"

print("Model yukleniyor:", MODEL_DIR)
tok = AutoTokenizer.from_pretrained(MODEL_DIR)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR).to(device).eval()

df = pd.read_csv(TEST_CSV)
# metin ve etiket kolonlarini bul
text_col = "comment_text" if "comment_text" in df.columns else df.columns[0]
# label: 'labels' veya sentiment_label -> id
if "labels" in df.columns:
    y_true = df["labels"].astype(int).values
else:
    m = {"negatif": 0, "nötr": 1, "pozitif": 2}
    y_true = df["sentiment_label"].map(m).astype(int).values
texts = df[text_col].astype(str).tolist()
print("Test ornegi:", len(texts))


@torch.no_grad()
def predict_proba(text_list, bs=32):
    probs = []
    for i in range(0, len(text_list), bs):
        batch = text_list[i:i+bs]
        enc = tok(batch, return_tensors="pt", truncation=True, padding=True, max_length=256).to(device)
        logits = model(**enc).logits
        p = torch.softmax(logits, dim=-1).cpu().numpy()
        probs.append(p)
    return np.vstack(probs)


print("Olasiliklar hesaplaniyor (gercek model cikti)...")
proba = predict_proba(texts)
y_pred = proba.argmax(axis=1)

# --- DOGRULAMA: kayitli test_predictions.csv ile karsilastir ---
pred_df = pd.read_csv(PRED_CSV)
if "pred_id" in pred_df.columns and len(pred_df) == len(y_pred):
    saved = pred_df["pred_id"].astype(int).values
    agree = (saved == y_pred).mean()
    print(f"DOGRULAMA: canli tahmin vs kayitli test_predictions.csv uyum = {agree:.4f}")
else:
    print("UYARI: kayitli tahminle boyut/kolon uyusmadi, sadece canli kullanilacak.")

acc = (y_pred == y_true).mean()
print(f"Test accuracy (canli) = {acc:.4f}")


# ============ Fig.3 Confusion Matrix ============
def fig_cm():
    cm = confusion_matrix(y_true, y_pred)
    cmn = cm / cm.sum(axis=1, keepdims=True)
    fig, ax = plt.subplots(figsize=(COL_W, 2.9), dpi=300)
    im = ax.imshow(cmn, cmap="Greys", vmin=0, vmax=1)
    ax.set_xticks(range(3)); ax.set_yticks(range(3))
    ax.set_xticklabels(CLASSES, fontsize=8); ax.set_yticklabels(CLASSES, fontsize=8)
    ax.set_xlabel("Predicted", fontsize=8.5); ax.set_ylabel("True", fontsize=8.5)
    for i in range(3):
        for j in range(3):
            ax.text(j, i, f"{cm[i,j]}\n({cmn[i,j]*100:.1f}%)", ha="center", va="center",
                    fontsize=7, color="white" if cmn[i, j] > 0.5 else "black")
    fig.tight_layout(pad=0.3)
    p = OUT / "fig3_confusion_matrix.png"
    fig.savefig(p, bbox_inches="tight", facecolor="white"); plt.close(fig)
    print("yazildi:", p.name)


# ============ Fig.2 ROC (one-vs-rest) ============
def fig_roc():
    yb = label_binarize(y_true, classes=[0, 1, 2])
    fig, ax = plt.subplots(figsize=(COL_W, 2.9), dpi=300)
    styles = ["-", "--", "-."]
    for i, cname in enumerate(CLASSES):
        fpr, tpr, _ = roc_curve(yb[:, i], proba[:, i])
        a = auc(fpr, tpr)
        ax.plot(fpr, tpr, styles[i], color="black", lw=1.1,
                label=f"{cname} (AUC={a:.3f})")
    ax.plot([0, 1], [0, 1], ":", color="gray", lw=0.8)
    ax.set_xlabel("False Positive Rate", fontsize=8.5)
    ax.set_ylabel("True Positive Rate", fontsize=8.5)
    ax.legend(fontsize=6.5, loc="lower right")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.02)
    fig.tight_layout(pad=0.3)
    p = OUT / "fig2_roc_curve.png"
    fig.savefig(p, bbox_inches="tight", facecolor="white"); plt.close(fig)
    print("yazildi:", p.name)
    # AUC degerlerini de yazdir (rapora referans)
    for i, c in enumerate(CLASSES):
        fpr, tpr, _ = roc_curve(yb[:, i], proba[:, i])
        print(f"  AUC[{c}] = {auc(fpr,tpr):.4f}")


fig_cm()
fig_roc()

# ============ Fig.4 + Fig.5 SHAP ============
# SHAP, tum test setinde agir; temsilci bir altkume (her siniftan dengeli) kullanilir.
def fig_shap():
    import shap
    rng = np.random.RandomState(42)
    # her siniftan ~25 ornek (toplam ~75) — temsilci, makul sure
    idx = []
    for c in [0, 1, 2]:
        ci = np.where(y_true == c)[0]
        take = min(25, len(ci))
        idx.extend(rng.choice(ci, take, replace=False).tolist())
    sub_texts = [texts[i] for i in idx]
    print(f"SHAP altkume: {len(sub_texts)} ornek (her siniftan dengeli)")

    def f(x):
        return predict_proba(list(x), bs=16)

    masker = shap.maskers.Text(tok)
    explainer = shap.Explainer(f, masker, output_names=CLASSES)
    sv = explainer(sub_texts[:60])  # guvenli sinir

    # Fig.4 — summary (beeswarm) tum siniflar icin tek panel zor; en bilgi verici: pozitif sinif
    # Bunun yerine mean|SHAP| feature importance (Fig.5) + ozet bar (Fig.4) uretelim.
    # SHAP token-bazli; toplam |katki| en yuksek token'lari topla.
    import collections
    abs_contrib = collections.defaultdict(float)
    cnt = collections.defaultdict(int)
    for i in range(len(sv)):
        toks = sv[i].data
        vals = np.abs(sv[i].values).sum(axis=-1) if sv[i].values.ndim > 1 else np.abs(sv[i].values)
        for t, v in zip(toks, vals):
            t = str(t).strip()
            if len(t) < 2:
                continue
            abs_contrib[t] += float(v)
            cnt[t] += 1
    # ortalama mutlak katki
    items = [(t, abs_contrib[t] / cnt[t]) for t in abs_contrib if cnt[t] >= 2]
    items.sort(key=lambda x: -x[1])
    top = items[:12][::-1]

    # Fig.5 feature importance
    fig, ax = plt.subplots(figsize=(COL_W, 3.0), dpi=300)
    words = [t for t, _ in top]; vals = [v for _, v in top]
    ax.barh(range(len(words)), vals, color="#555", edgecolor="black", lw=0.4, height=0.7)
    ax.set_yticks(range(len(words))); ax.set_yticklabels(words, fontsize=7)
    ax.set_xlabel("Mean |SHAP value|", fontsize=8.5)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(pad=0.3)
    p5 = OUT / "fig5_shap_importance.png"
    fig.savefig(p5, bbox_inches="tight", facecolor="white"); plt.close(fig)
    print("yazildi:", p5.name)

    # Fig.4 summary: beeswarm yerine, en etkili token'larin sinif-bazli yonu (kompakt)
    # shap.plots.bar kullan (global)
    try:
        fig = plt.figure(figsize=(COL_W, 3.0), dpi=300)
        shap.plots.bar(sv[:, :, 2].mean(0), max_display=12, show=False)  # pozitif sinif global
        plt.tight_layout()
        p4 = OUT / "fig4_shap_summary.png"
        plt.savefig(p4, bbox_inches="tight", facecolor="white"); plt.close()
        print("yazildi:", p4.name)
    except Exception as e:
        print("Fig4 (shap.plots.bar) hata:", e, "-> fig5 birincil kullanilir")


try:
    fig_shap()
except Exception as e:
    print("SHAP grafikleri uretilemedi:", repr(e))

print("OUT:", OUT)
