# -*- coding: utf-8 -*-
"""Final rapor icin akademik (sade) gorseller uretir. reports/rapor_figurleri/ altina."""
import csv
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "reports" / "rapor_figurleri"
OUT.mkdir(parents=True, exist_ok=True)
plt.rcParams["font.family"] = "DejaVu Sans"
plt.rcParams["axes.unicode_minus"] = False


def read_ms(path):
    rows = list(csv.DictReader(open(path, encoding="utf-8")))
    fm = np.array([float(r["f1_macro"]) for r in rows])
    return fm.mean(), fm.std()


# --- Figur A: Nihai model confusion matrix (seed43, akademik) ---
def cm_fig():
    cm = np.array([[406, 31, 0], [41, 322, 47], [1, 28, 502]])
    labels = ["negatif", "nötr", "pozitif"]
    cmn = cm / cm.sum(axis=1, keepdims=True)
    fig, ax = plt.subplots(figsize=(6.4, 5.6), dpi=200)
    im = ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(3)); ax.set_yticks(range(3))
    ax.set_xticklabels(labels, fontsize=12); ax.set_yticklabels(labels, fontsize=12)
    ax.set_xlabel("Tahmin Edilen Sınıf", fontsize=12, fontweight="bold")
    ax.set_ylabel("Gerçek Sınıf", fontsize=12, fontweight="bold")
    ax.set_title("Nihai Model (BERTurk) — Confusion Matrix", fontsize=13, fontweight="bold", pad=12)
    for i in range(3):
        for j in range(3):
            ax.text(j, i, f"{cm[i,j]}\n(%{cmn[i,j]*100:.1f})", ha="center", va="center",
                    fontsize=12, color="white" if cmn[i, j] > 0.5 else "#222", fontweight="bold")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Satır-normalize oran", fontsize=10)
    fig.tight_layout()
    p = OUT / "fig_confusion_matrix.png"
    fig.savefig(p, bbox_inches="tight", facecolor="white"); plt.close(fig)
    print("yazildi:", p)


# --- Figur B: 4-model Macro-F1 (akademik bar) ---
def models_fig():
    models = {
        "BERTurk": "models/optimized_v3/v3_genisletilmis/summary_multiseed.csv",
        "ELECTRA-tr": "models/optimized_v3/v3_electra_tr/summary_multiseed.csv",
        "XLM-RoBERTa": "models/optimized_v3/v3_xlmr/summary_multiseed.csv",
        "mBERT": "models/optimized_v3/v3_mbert/summary_multiseed.csv",
    }
    names, means, stds = [], [], []
    for n, p in models.items():
        m, s = read_ms(ROOT / p); names.append(n); means.append(m); stds.append(s)
    # ELECTRA en yuksek nominal -> vurgu; secilen BERTurk -> yildiz
    order = sorted(range(len(names)), key=lambda i: -means[i])
    names = [names[i] for i in order]; means = [means[i] for i in order]; stds = [stds[i] for i in order]
    colors = ["#4E79A7" if n != "BERTurk" else "#59A14F" for n in names]
    fig, ax = plt.subplots(figsize=(8, 5.2), dpi=200)
    x = np.arange(len(names))
    bars = ax.bar(x, means, yerr=stds, capsize=6, color=colors, width=0.6,
                  edgecolor="white", linewidth=1.2, error_kw=dict(ecolor="#444", lw=1.4))
    ax.set_xticks(x); ax.set_xticklabels(names, fontsize=11)
    ax.set_ylim(0.80, 0.91)
    ax.set_ylabel("Macro-F1 (3 seed ort. ± std)", fontsize=12, fontweight="bold")
    ax.set_title("Çok-Modelli Karşılaştırma — Aynı Protokol", fontsize=13, fontweight="bold", pad=12)
    for b, m, s, n in zip(bars, means, stds, names):
        star = " ★" if n == "BERTurk" else ""
        ax.text(b.get_x()+b.get_width()/2, m+s+0.003, f"{m:.4f}{star}".replace(".", ","),
                ha="center", va="bottom", fontsize=10.5, fontweight="bold")
    ax.spines[["top", "right"]].set_visible(False)
    ax.text(0, 0.803, "★ = nihai seçilen model (en düşük varyans + en hızlı eğitim)", fontsize=8.6, color="#666")
    fig.tight_layout()
    p = OUT / "fig_model_karsilastirma.png"
    fig.savefig(p, bbox_inches="tight", facecolor="white"); plt.close(fig)
    print("yazildi:", p)


# --- Figur C: Ablasyon ilerleme (baseline -> genisletilmis, F1-notr) ---
def ablation_fig():
    stages = ["baseline\n(AI her split)", "real_test\n(AI test'ten çıktı)", "fixed\n(eval düzeltme)",
              "bigtest\n(per-class split)", "genisletilmis\n(+751 nötr)"]
    macro = [0.78, 0.8195, 0.8303, 0.8377, 0.8797]
    notr = [0.55, 0.5856, 0.6156, 0.6852, 0.8056]
    fig, ax = plt.subplots(figsize=(9, 5.2), dpi=200)
    x = np.arange(len(stages))
    ax.plot(x, macro, "o-", color="#4E79A7", lw=2.2, ms=8, label="Macro-F1")
    ax.plot(x, notr, "s--", color="#E15759", lw=2.2, ms=8, label="F1-nötr")
    ax.set_xticks(x); ax.set_xticklabels(stages, fontsize=9.5)
    ax.set_ylabel("F1", fontsize=12, fontweight="bold")
    ax.set_title("Aşamalı İyileştirme: Nötr Sınıfın Güçlenmesi", fontsize=13, fontweight="bold", pad=12)
    ax.legend(fontsize=11, loc="lower right")
    ax.grid(axis="y", alpha=0.3)
    for xi, yi in zip(x, notr):
        ax.text(xi, yi-0.025, f"{yi:.3f}".replace(".", ","), ha="center", fontsize=8.6, color="#E15759")
    ax.set_ylim(0.50, 0.92)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    p = OUT / "fig_ablasyon_ilerleme.png"
    fig.savefig(p, bbox_inches="tight", facecolor="white"); plt.close(fig)
    print("yazildi:", p)


if __name__ == "__main__":
    cm_fig(); models_fig(); ablation_fig()
    print("OUT:", OUT)
