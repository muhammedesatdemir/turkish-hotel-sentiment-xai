# -*- coding: utf-8 -*-
"""IEEE bildirisi icin sade, tek-sutun (column-width) sekiller. Gri tonlu, akademik."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "reports" / "ieee_figurleri"
OUT.mkdir(parents=True, exist_ok=True)
plt.rcParams["font.family"] = "DejaVu Serif"  # IEEE Times benzeri
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["font.size"] = 8

COL_W = 3.4  # IEEE tek sutun inch


# --- Sekil 1: Pipeline (dikey, kompakt, tek sutun) ---
def fig_pipeline():
    steps = [
        ("Data Collection & Consolidation", "11,086 real + 751 curated + 2,500 synthetic"),
        ("Preprocessing & Labeling", "score-based 3-class, r=0.81"),
        ("Leakage-Free Split", "ai_only_in_train, test 100% real"),
        ("Model Training", "BERTurk + WCE, 3 seeds"),
        ("Statistical Validation", "McNemar + 5-fold CV"),
        ("Explainability", "LIME + SHAP"),
    ]
    fig, ax = plt.subplots(figsize=(COL_W, 4.3), dpi=300)
    ax.set_xlim(0, 10); ax.set_ylim(0, 100); ax.axis("off")
    n = len(steps)
    h = 12; gap = (100 - n*h) / (n+1)
    for i, (title, sub) in enumerate(steps):
        y = 100 - gap*(i+1) - h*(i+1) + h/2
        y0 = y - h/2
        box = FancyBboxPatch((1, y0), 8, h, boxstyle="round,pad=0.2,rounding_size=1.0",
                             linewidth=0.8, edgecolor="#333", facecolor="#EEEEEE")
        ax.add_patch(box)
        ax.text(5, y+1.8, title, ha="center", va="center", fontsize=7.2, fontweight="bold")
        ax.text(5, y-2.8, sub, ha="center", va="center", fontsize=5.8, color="#444", style="italic")
        if i < n-1:
            ax.add_patch(FancyArrowPatch((5, y0), (5, y0-gap),
                         arrowstyle="-|>", mutation_scale=10, color="#333", lw=1.0))
    fig.tight_layout(pad=0.2)
    p = OUT / "fig1_pipeline.png"
    fig.savefig(p, bbox_inches="tight", facecolor="white"); plt.close(fig)
    print("yazildi:", p)


# --- Sekil 2: LIME kelime katki (tek sutun, gri/koyu) ---
def fig_lime():
    d = json.load(open(ROOT / "reports/xai_final/summary.json", encoding="utf-8"))
    feats = d["lime"][0]["top_features_for_predicted"][:6]
    words = [w for w, _ in feats][::-1]
    vals = [v for _, v in feats][::-1]
    fig, ax = plt.subplots(figsize=(COL_W, 2.2), dpi=300)
    ax.barh(range(len(words)), vals, color="#555555", edgecolor="black", linewidth=0.5, height=0.65)
    ax.set_yticks(range(len(words))); ax.set_yticklabels(words, fontsize=8)
    ax.set_xlabel("LIME contribution to 'negative' class", fontsize=7.5)
    for i, v in enumerate(vals):
        ax.text(v+0.008, i, f"{v:+.2f}", va="center", fontsize=6.5)
    ax.set_xlim(0, max(vals)*1.2)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(labelsize=7)
    fig.tight_layout(pad=0.2)
    p = OUT / "fig2_lime.png"
    fig.savefig(p, bbox_inches="tight", facecolor="white"); plt.close(fig)
    print("yazildi:", p)


if __name__ == "__main__":
    fig_pipeline(); fig_lime()
    print("OUT:", OUT)
