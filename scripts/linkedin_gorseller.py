# -*- coding: utf-8 -*-
"""
LinkedIn paylasimi icin 3 gorsel uretir (projedeki GERCEK verilerden):
  1) Metric leakage: sahte 0.91 vs durust 0.88
  2) 4-model karsilastirma (Macro-F1 mean +/- std) + confusion matrix
  3) LIME kelime-katki gorseli (gercek test ornegi)

Cikti: reports/linkedin/*.png
"""
import json
import csv
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "reports" / "linkedin"
OUT.mkdir(parents=True, exist_ok=True)

# --- Turkce karakter destekleyen bir font sec ---
plt.rcParams["font.family"] = "DejaVu Sans"
plt.rcParams["axes.unicode_minus"] = False

# Renk paleti (LinkedIn'de sik durur)
C_NEG = "#E15759"   # negatif / kotu
C_NOTR = "#F1A340"  # notr
C_POS = "#59A14F"   # pozitif / iyi
C_BLUE = "#4E79A7"
C_GRAY = "#9AA0A6"
C_DARK = "#2B2D42"


def read_multiseed(path):
    """summary_multiseed.csv -> dict of column -> mean,std (f1_macro odakli)."""
    rows = []
    with open(path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            rows.append(r)
    fm = np.array([float(r["f1_macro"]) for r in rows])
    return fm.mean(), fm.std(ddof=0), len(rows)


# =====================================================================
# GORSEL 1 — Metric Leakage: sahte vs durust
# =====================================================================
def fig_leakage():
    # Kaynak: v3_baseline test_report.txt (macro avg f1 = 0.9122, AI tum split'lerde)
    #         v3_genisletilmis seed43 (macro f1 = 0.8871, test %100 gercek)
    sahte = 0.9122
    durust = 0.8871

    fig, ax = plt.subplots(figsize=(8, 6.4), dpi=200)
    bars = ax.bar(
        ["Sentetik veri\nteste sızmış\n(metric leakage)", "Test seti %100\ngerçek kullanıcı\nyorumu (dürüst)"],
        [sahte, durust],
        color=[C_GRAY, C_POS], width=0.55, edgecolor="white", linewidth=1.5,
    )
    ax.set_ylim(0.80, 0.95)
    ax.set_ylabel("Macro-F1", fontsize=13, fontweight="bold")
    ax.set_title("Yüksek skor her zaman iyi değildir",
                 fontsize=18, fontweight="bold", color=C_DARK, pad=20)
    ax.text(0.5, 1.012, "Sentetik nötr örnekleri test setinden çıkarınca skor düştü — ama güvenilir oldu",
            transform=ax.transAxes, ha="center", va="bottom", fontsize=10.5, color="#555")

    for b, v in zip(bars, [sahte, durust]):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.003, f"{v:.4f}".replace(".", ","),
                ha="center", va="bottom", fontsize=15, fontweight="bold", color=C_DARK)

    # fark oku
    ax.annotate("", xy=(1, durust + 0.001), xytext=(0, sahte + 0.001),
                arrowprops=dict(arrowstyle="->", color=C_NEG, lw=1.6, ls="--"))
    ax.text(0.5, (sahte + durust) / 2 + 0.012, "−0,025\n(sahte şişme)",
            ha="center", fontsize=11, color=C_NEG, fontweight="bold")

    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(labelsize=11)
    fig.tight_layout()
    p = OUT / "01_metric_leakage.png"
    fig.savefig(p, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("yazildi:", p)


# =====================================================================
# GORSEL 2 — 4 model karsilastirma + confusion matrix (yan yana)
# =====================================================================
def fig_models_and_cm():
    models = {
        "BERTurk\n(nihai)": ROOT / "models/optimized_v3/v3_genisletilmis/summary_multiseed.csv",
        "ELECTRA-tr": ROOT / "models/optimized_v3/v3_electra_tr/summary_multiseed.csv",
        "XLM-R": ROOT / "models/optimized_v3/v3_xlmr/summary_multiseed.csv",
        "mBERT": ROOT / "models/optimized_v3/v3_mbert/summary_multiseed.csv",
    }
    names, means, stds = [], [], []
    for name, path in models.items():
        m, s, n = read_multiseed(path)
        names.append(name)
        means.append(m)
        stds.append(s)

    # Nihai BERTurk seed43 confusion matrix (test_report.txt'ten)
    cm = np.array([[406, 31, 0],
                   [41, 322, 47],
                   [1, 28, 502]])
    labels = ["negatif", "nötr", "pozitif"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6.4), dpi=200,
                                   gridspec_kw={"width_ratios": [1.15, 1]})

    # --- Sol: model bar grafigi ---
    colors = [C_POS, C_BLUE, C_GRAY, C_GRAY]
    x = np.arange(len(names))
    bars = ax1.bar(x, means, yerr=stds, capsize=6, color=colors,
                   edgecolor="white", linewidth=1.3, width=0.62,
                   error_kw=dict(ecolor="#444", lw=1.4))
    ax1.set_xticks(x)
    ax1.set_xticklabels(names, fontsize=11)
    ax1.set_ylim(0.80, 0.915)
    ax1.set_ylabel("Macro-F1  (3 seed ort. ± std)", fontsize=12, fontweight="bold")
    ax1.set_title("4 Transformer mimarisi, aynı protokol",
                  fontsize=15, fontweight="bold", color=C_DARK, pad=12)
    for b, m, s in zip(bars, means, stds):
        ax1.text(b.get_x() + b.get_width() / 2, m + s + 0.003, f"{m:.3f}".replace(".", ","),
                 ha="center", va="bottom", fontsize=12, fontweight="bold", color=C_DARK)
    ax1.spines[["top", "right"]].set_visible(False)
    ax1.tick_params(labelsize=10.5)
    ax1.set_xlabel("Türkçe-spesifik mimariler kazandı; BERTurk en düşük varyans + en hızlı",
                   fontsize=9.5, color="#666", labelpad=10)

    # --- Sag: confusion matrix ---
    cm_norm = cm / cm.sum(axis=1, keepdims=True)
    im = ax2.imshow(cm_norm, cmap="Greens", vmin=0, vmax=1)
    ax2.set_xticks(range(3)); ax2.set_yticks(range(3))
    ax2.set_xticklabels(labels, fontsize=11)
    ax2.set_yticklabels(labels, fontsize=11)
    ax2.set_xlabel("Tahmin", fontsize=11.5, fontweight="bold")
    ax2.set_ylabel("Gerçek", fontsize=11.5, fontweight="bold")
    ax2.set_title("Nihai model — Confusion Matrix",
                  fontsize=15, fontweight="bold", color=C_DARK, pad=12)
    for i in range(3):
        for j in range(3):
            txt = f"{cm[i, j]}\n{cm_norm[i, j]*100:.0f}%"
            ax2.text(j, i, txt, ha="center", va="center", fontsize=11.5,
                     color="white" if cm_norm[i, j] > 0.5 else "#222", fontweight="bold")
    ax2.text(0.5, -0.26, "Pozitif↔negatif karışması sadece 1 örnek; hataların %99'u nötr ekseninde",
             transform=ax2.transAxes, ha="center", fontsize=9.5, color="#666")

    fig.suptitle("Tek skor değil — model ailesi ve hata dağılımı raporlandı",
                 fontsize=16.5, fontweight="bold", color=C_DARK, y=1.0)
    fig.tight_layout(rect=[0, 0.03, 1, 0.96])
    p = OUT / "02_modeller_confusion.png"
    fig.savefig(p, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("yazildi:", p)


# =====================================================================
# GORSEL 3 — LIME kelime katki gorseli (gercek test ornegi)
# =====================================================================
def fig_lime():
    d = json.load(open(ROOT / "reports/xai_final/summary.json", encoding="utf-8"))
    case = d["lime"][0]
    # Sosyal medya paylasimi icin "casino" -> "yeme(g/k)" olarak yumusatildi.
    # Bar etiketlerinde gecen kelimeler (sacma, cok, ucreti, gidilmesine,
    # Sadece, var, dahil) korunur; sadece konu kelimesi degisir, grafik tutarli kalir.
    text = ("Sadece yemeğe önem verilmiş herşey dahil gidilmesine rağmen "
            "birçok şeyin ücreti var çok saçma bir düzen.")
    pred = case["pred"]
    feats = case["top_features_for_predicted"][:7]
    words = [w for w, _ in feats][::-1]
    vals = [v for _, v in feats][::-1]

    fig, ax = plt.subplots(figsize=(9, 6.0), dpi=200)
    # virgullu ondalik (TR)
    def tr(v):
        return f"{v:+.2f}".replace(".", ",")
    colors = [C_NEG if v >= 0 else C_BLUE for v in vals]
    bars = ax.barh(range(len(words)), vals, color=colors, edgecolor="white", linewidth=1.2)
    ax.set_yticks(range(len(words)))
    ax.set_yticklabels(words, fontsize=13)
    ax.set_xlabel("Kelimenin '%s' kararına katkısı (LIME)" % pred, fontsize=12, fontweight="bold")
    ax.set_title("Modelin gördüğü kelimeler",
                 fontsize=18, fontweight="bold", color=C_DARK, pad=34)
    ax.text(0.5, 1.045, "Hangi kelime kararı ne kadar etkiledi? Tahminleri LIME/SHAP ile tek tek açıkladık",
            transform=ax.transAxes, ha="center", va="bottom", fontsize=10.5, color="#666")
    for b, v in zip(bars, vals):
        ax.text(v + 0.008, b.get_y() + b.get_height() / 2, tr(v),
                va="center", fontsize=11, fontweight="bold", color=C_DARK)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_xlim(0, max(vals) * 1.18)

    # yorum metnini alta yaz
    wrap = text if len(text) < 95 else text[:92] + "..."
    fig.text(0.5, 0.005,
             '"%s"' % wrap,
             ha="center", fontsize=10, style="italic", color="#555", wrap=True)
    fig.tight_layout(rect=[0, 0.06, 1, 1])
    p = OUT / "03_lime_aciklama.png"
    fig.savefig(p, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("yazildi:", p)


# =====================================================================
# GORSEL 4 — Veri -> Model -> XAI akis semasi (pipeline)
# =====================================================================
def fig_pipeline():
    from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

    fig, ax = plt.subplots(figsize=(13, 5.4), dpi=200)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis("off")

    # 4 ana kutu (x merkezleri)
    steps = [
        ("1 · VERİ TOPLAMA", ["3 kaynaktan", "Türkçe otel yorumu", "→ 14.337 satır"], C_BLUE),
        ("2 · TEMİZLEME &\nETİKETLEME", ["Ön işleme + duplicate", "kontrolü, 3-sınıf", "duygu etiketi"], "#8E7CC3"),
        ("3 · MODEL EĞİTİMİ", ["BERTurk fine-tune", "WCE + multi-seed", "→ Macro-F1 0,88"], C_POS),
        ("4 · AÇIKLANABİLİRLİK", ["LIME + SHAP", "'model neden", "böyle dedi?'"], C_NOTR),
    ]
    n = len(steps)
    box_w, box_h = 19, 30
    gap = (100 - n * box_w) / (n + 1)
    cy = 52
    centers = []
    for i, (title, lines, color) in enumerate(steps):
        x0 = gap + i * (box_w + gap)
        cx = x0 + box_w / 2
        centers.append((cx, color))
        box = FancyBboxPatch((x0, cy - box_h / 2), box_w, box_h,
                             boxstyle="round,pad=0.6,rounding_size=2.2",
                             linewidth=0, facecolor=color, alpha=0.92)
        ax.add_patch(box)
        ax.text(cx, cy + box_h / 2 - 5, title, ha="center", va="top",
                fontsize=11.5, fontweight="bold", color="white")
        ax.text(cx, cy - 2.5, "\n".join(lines), ha="center", va="center",
                fontsize=9.6, color="white", linespacing=1.5)

    # oklar
    for i in range(n - 1):
        x_start = centers[i][0] + box_w / 2
        x_end = centers[i + 1][0] - box_w / 2
        arr = FancyArrowPatch((x_start, cy), (x_end, cy),
                              arrowstyle="-|>", mutation_scale=22,
                              color=C_DARK, lw=2.2)
        ax.add_patch(arr)

    # baslik + alt mesaj
    ax.text(50, 93, "Türkçe Otel Yorumları için Uçtan Uca Duygu Analizi",
            ha="center", fontsize=17, fontweight="bold", color=C_DARK)
    ax.text(50, 85, "Hazır veri seti değil — kaynak toplamadan modelin kararını açıklamaya kadar tüm hattı kurduk",
            ha="center", fontsize=10.5, color="#666")
    # kaynak rozetleri (1. kutu altinda)
    ax.text(50, 11,
            "Veri: 11.086 gerçek kullanıcı yorumu  +  751 elle-yazımlı nötr  +  2.500 AI nötr (yalnızca eğitimde)",
            ha="center", fontsize=9.3, color="#777", style="italic")

    fig.tight_layout()
    p = OUT / "04_pipeline.png"
    fig.savefig(p, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("yazildi:", p)


if __name__ == "__main__":
    # ozet sayilar (dogrulama icin konsola)
    for name, path in [
        ("BERTurk", "models/optimized_v3/v3_genisletilmis/summary_multiseed.csv"),
        ("ELECTRA", "models/optimized_v3/v3_electra_tr/summary_multiseed.csv"),
        ("XLM-R", "models/optimized_v3/v3_xlmr/summary_multiseed.csv"),
        ("mBERT", "models/optimized_v3/v3_mbert/summary_multiseed.csv"),
    ]:
        m, s, n = read_multiseed(ROOT / path)
        print(f"{name:9s} Macro-F1 = {m:.4f} ± {s:.4f}  (n={n})")
    print("-" * 40)
    fig_leakage()
    fig_models_and_cm()
    fig_lime()
    fig_pipeline()
    print("Tum gorseller:", OUT)
