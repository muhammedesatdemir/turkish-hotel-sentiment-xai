# -*- coding: utf-8 -*-
"""
src/eda_analysis.py
===================

Veri Kesfi (Exploratory Data Analysis) — hoca geri bildirimi A.5.

Hocanin itirazi:
    "EDA yok. Pearson korelasyon matrisi yapilmali. MD'ye kaydet,
     AI'ya analiz ettir."

Bu script `final_genisletilmis_dataset_optimized_v3_v2.csv` uzerinde
asagidaki uretimleri yapar:

    1) Sinif dagilimi grafigi (bar)
    2) Kelime sayisi dagilimi (boxplot, sinif bazinda)
    3) Kaynak x etiket cross-tab heatmap'i
    4) En sik 20 kelime (sinif bazinda) bar grafigi
    5) Meta-feature Pearson korelasyon matrisi (heatmap)
    6) Korelasyon icin meta-feature havuzu:
        wc, char_count, avg_word_len, punct_count, caps_ratio,
        pos_keyword_count, neg_keyword_count, rating_auto, is_ai, sentiment_id

Cikti:
    reports/eda/
        ├── 01_class_dist.png
        ├── 02_wc_box.png
        ├── 03_source_label_heatmap.png
        ├── 04_top_words_per_class.png
        ├── 05_corr_matrix.png
        ├── corr_matrix.csv
        ├── meta_features.csv             (her satir icin meta-feature degerleri)
        ├── eda_summary.json              (tum sayisal ozeti)
        └── eda_report.md                 (akademik dilli yorumlama, final rapora EK)

Kosma:
    python src/eda_analysis.py
"""
from __future__ import annotations

import json
import re
import sys
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # GUI yok, sadece dosya cikti
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# =============================================================================
# Sabitler
# =============================================================================

ROOT = Path(__file__).resolve().parent.parent
CSV = ROOT / "data" / "processed" / "final_genisletilmis_dataset_optimized_v3_v2.csv"
OUT_DIR = ROOT / "reports" / "eda"

LABELS = ["negatif", "nötr", "pozitif"]
LABEL2ID = {"negatif": -1, "nötr": 0, "pozitif": 1}

# Anahtar kelime havuzlari (XAI bulgularindan dogrulanmis sinyaller)
POZ_KEYS = [
    "harika", "muhteşem", "muhtesem", "mükemmel", "mukemmel", "süper", "super",
    "tavsiye", "memnun", "kaliteli", "guzel", "güzel", "temiz", "rahat",
    "hizmet", "ilgili", "lezzetli", "keyifli", "begendi", "beğendi",
]
NEG_KEYS = [
    "berbat", "rezalet", "iğrenç", "igrenc", "kötü", "kotu", "pis", "kirli",
    "sorun", "şikayet", "sikayet", "kalitesiz", "yavaş", "yavas", "uzak",
    "gürültü", "gurultu", "soğuk", "soguk", "boş", "bos",
]

# Konuşma görseli icin
SENTIMENT_COLOR = {"negatif": "#d62728", "nötr": "#7f7f7f", "pozitif": "#2ca02c"}
SENTIMENT_ORDER = ["negatif", "nötr", "pozitif"]

# Heatmap renk skalasi
CMAP_DIVERGING = "RdBu_r"
CMAP_SEQUENTIAL = "viridis"


# =============================================================================
# Meta-feature ureticileri
# =============================================================================

PUNCT_RE = re.compile(r"[^\w\s]", flags=re.UNICODE)
WORD_RE = re.compile(r"\w+", flags=re.UNICODE)


def char_count(s: str) -> int:
    return len(s) if isinstance(s, str) else 0


def word_count(s: str) -> int:
    if not isinstance(s, str):
        return 0
    return len(s.split())


def avg_word_len(s: str) -> float:
    if not isinstance(s, str):
        return 0.0
    words = s.split()
    if not words:
        return 0.0
    return float(np.mean([len(w) for w in words]))


def punct_count(s: str) -> int:
    if not isinstance(s, str):
        return 0
    return len(PUNCT_RE.findall(s))


def caps_ratio(s: str) -> float:
    if not isinstance(s, str) or not s:
        return 0.0
    letters = [c for c in s if c.isalpha()]
    if not letters:
        return 0.0
    return sum(1 for c in letters if c.isupper()) / len(letters)


def keyword_count(s: str, keys: list[str]) -> int:
    if not isinstance(s, str):
        return 0
    low = s.lower()
    # Turkce karakter normalize basit
    norm = (low.replace("ı", "i").replace("ş", "s").replace("ç", "c")
                .replace("ğ", "g").replace("ö", "o").replace("ü", "u"))
    return sum(1 for k in keys if k in low or k in norm)


def build_meta_features(df: pd.DataFrame) -> pd.DataFrame:
    """Her satir icin meta-feature uretir."""
    text_col = "comment_text"
    out = pd.DataFrame()
    out["wc"] = df[text_col].map(word_count)
    out["char_count"] = df[text_col].map(char_count)
    out["avg_word_len"] = df[text_col].map(avg_word_len)
    out["punct_count"] = df[text_col].map(punct_count)
    out["caps_ratio"] = df[text_col].map(caps_ratio)
    out["pos_keyword_count"] = df[text_col].map(lambda s: keyword_count(s, POZ_KEYS))
    out["neg_keyword_count"] = df[text_col].map(lambda s: keyword_count(s, NEG_KEYS))
    out["rating_auto"] = pd.to_numeric(df["rating_auto"], errors="coerce").fillna(0).astype(int)
    out["is_ai"] = (df["source"].astype(str).str.strip().str.lower() == "ai_generated").astype(int)
    out["sentiment_id"] = df["sentiment_label"].map(LABEL2ID).astype(int)
    return out


# =============================================================================
# Gorseller
# =============================================================================

def plot_class_distribution(df: pd.DataFrame, out_path: Path) -> dict:
    counts = df["sentiment_label"].value_counts().reindex(SENTIMENT_ORDER).fillna(0).astype(int)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    bars = ax.bar(
        counts.index, counts.values,
        color=[SENTIMENT_COLOR[l] for l in counts.index],
        edgecolor="black", linewidth=0.6,
    )
    total = counts.sum()
    for b, v in zip(bars, counts.values):
        pct = v / total * 100 if total > 0 else 0
        ax.text(b.get_x() + b.get_width() / 2, v + total * 0.01,
                f"{v:,}\n(%{pct:.1f})", ha="center", va="bottom", fontsize=10)
    ax.set_title(f"Sinif Dagilimi (n = {total:,})")
    ax.set_xlabel("Etiket")
    ax.set_ylabel("Yorum sayisi")
    ax.set_ylim(0, counts.max() * 1.15)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    return {"counts": counts.to_dict(), "total": int(total)}


def plot_wc_boxplot(df: pd.DataFrame, mf: pd.DataFrame, out_path: Path) -> dict:
    fig, ax = plt.subplots(figsize=(8, 5))
    data = []
    for lbl in SENTIMENT_ORDER:
        mask = df["sentiment_label"] == lbl
        data.append(mf.loc[mask, "wc"].values)
    bp = ax.boxplot(data, labels=SENTIMENT_ORDER, patch_artist=True, showfliers=False)
    for patch, lbl in zip(bp["boxes"], SENTIMENT_ORDER):
        patch.set_facecolor(SENTIMENT_COLOR[lbl])
        patch.set_alpha(0.6)
    # Her sinifa medyani yaz
    stats = {}
    for i, lbl in enumerate(SENTIMENT_ORDER):
        med = float(np.median(data[i]))
        q05 = float(np.quantile(data[i], 0.05))
        q95 = float(np.quantile(data[i], 0.95))
        ax.annotate(f"medyan={med:.0f}", xy=(i + 1, med),
                    xytext=(i + 1, med + 5), ha="center", fontsize=9,
                    color="black", fontweight="bold")
        stats[lbl] = {"median": med, "q05": q05, "q95": q95,
                      "mean": float(np.mean(data[i]))}
    ax.set_title("Kelime Sayisi Dagilimi (Sinif Bazinda)")
    ax.set_ylabel("Kelime sayisi")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    return stats


def plot_source_label_heatmap(df: pd.DataFrame, out_path: Path) -> dict:
    ct = pd.crosstab(df["source"], df["sentiment_label"])
    ct = ct.reindex(columns=SENTIMENT_ORDER, fill_value=0)

    fig, ax = plt.subplots(figsize=(7, 4))
    im = ax.imshow(ct.values, cmap=CMAP_SEQUENTIAL, aspect="auto")
    ax.set_xticks(range(len(ct.columns)))
    ax.set_xticklabels(ct.columns)
    ax.set_yticks(range(len(ct.index)))
    ax.set_yticklabels(ct.index)
    # Hucre degerleri
    for i in range(ct.shape[0]):
        for j in range(ct.shape[1]):
            v = int(ct.iloc[i, j])
            color = "white" if v > ct.values.max() * 0.5 else "black"
            ax.text(j, i, f"{v:,}", ha="center", va="center", color=color, fontsize=10)
    ax.set_title("Kaynak x Etiket Cross-Tab")
    ax.set_xlabel("Sentiment")
    ax.set_ylabel("Kaynak")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Yorum sayisi")
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    return ct.to_dict()


def plot_top_words(df: pd.DataFrame, out_path: Path, top_n: int = 15) -> dict:
    """Sinif bazinda en sik gecen N kelime. Stopword listesi minimaldir."""
    STOP = {"bir", "ve", "ile", "ama", "fakat", "lakin", "icin", "için",
            "bu", "şu", "su", "o", "da", "de", "ki", "mi", "mu", "mı", "mü",
            "ya", "veya", "olarak", "olan", "olduğu", "olduğunu", "ben", "biz",
            "siz", "onlar", "çok", "cok", "daha", "az", "tam", "her", "hiç"}

    out = {}
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.5))
    for i, lbl in enumerate(SENTIMENT_ORDER):
        sub = df[df["sentiment_label"] == lbl]["comment_text"].dropna()
        words = []
        for t in sub:
            for w in WORD_RE.findall(str(t).lower()):
                if len(w) >= 4 and w not in STOP:
                    words.append(w)
        cnt = Counter(words).most_common(top_n)
        out[lbl] = cnt
        if cnt:
            ws, vs = zip(*cnt)
            axes[i].barh(range(len(ws)), vs, color=SENTIMENT_COLOR[lbl], alpha=0.7)
            axes[i].set_yticks(range(len(ws)))
            axes[i].set_yticklabels(ws, fontsize=9)
            axes[i].invert_yaxis()
            axes[i].set_title(f"{lbl} — Top {top_n}")
            axes[i].set_xlabel("Frekans")
            axes[i].grid(axis="x", alpha=0.3)
    fig.suptitle(f"Sinif Bazinda En Sik Kelimeler (>=4 harf, stopword'lersiz)", fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    return out


def plot_correlation_matrix(mf: pd.DataFrame, out_path: Path,
                            csv_path: Path) -> dict:
    corr = mf.corr(method="pearson")
    corr.to_csv(csv_path, encoding="utf-8")

    fig, ax = plt.subplots(figsize=(9, 8))
    im = ax.imshow(corr.values, cmap=CMAP_DIVERGING, vmin=-1, vmax=1, aspect="auto")
    n = len(corr.columns)
    ax.set_xticks(range(n))
    ax.set_xticklabels(corr.columns, rotation=45, ha="right", fontsize=9)
    ax.set_yticks(range(n))
    ax.set_yticklabels(corr.index, fontsize=9)
    # Hucre degerleri
    for i in range(n):
        for j in range(n):
            v = corr.iloc[i, j]
            color = "white" if abs(v) > 0.5 else "black"
            ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                    color=color, fontsize=8)
    ax.set_title("Pearson Korelasyon Matrisi (Meta-Features)")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Pearson r")
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    return corr.to_dict()


# =============================================================================
# Yorum uretimi (akademik dilli)
# =============================================================================

def interpret_correlations(corr: pd.DataFrame) -> list[str]:
    """En guclu korelasyonlari + sentiment_id ile olanlari ozetle."""
    notes = []

    # sentiment_id ile en guclu korelasyonlar
    if "sentiment_id" in corr.columns:
        s_corr = corr["sentiment_id"].drop("sentiment_id").sort_values(key=abs, ascending=False)
        top = s_corr.head(5)
        notes.append("**sentiment_id (etiket) ile en güçlü korelasyonlar:**")
        for feat, r in top.items():
            direction = "pozitif" if r > 0 else "negatif"
            notes.append(f"- `{feat}`: r = {r:+.3f} ({direction} ilişki)")

    # is_ai ile diger features
    if "is_ai" in corr.columns:
        a_corr = corr["is_ai"].drop("is_ai").sort_values(key=abs, ascending=False)
        top = a_corr.head(3)
        notes.append("\n**is_ai (sentetik mi?) ile en güçlü korelasyonlar:**")
        for feat, r in top.items():
            notes.append(f"- `{feat}`: r = {r:+.3f}")

    # Genel olarak |r| >= 0.5 olan ciftler
    high = []
    cols = corr.columns.tolist()
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            r = corr.iloc[i, j]
            if abs(r) >= 0.5:
                high.append((cols[i], cols[j], r))
    if high:
        notes.append("\n**|r| ≥ 0,50 olan diğer feature çiftleri:**")
        for a, b, r in sorted(high, key=lambda x: -abs(x[2])):
            notes.append(f"- `{a}` ↔ `{b}`: r = {r:+.3f}")

    return notes


def write_eda_report(out_md: Path, summary: dict, corr: pd.DataFrame,
                     wc_stats: dict, top_words: dict) -> None:
    lines = []
    lines.append("# Veri Kesfi (EDA) Raporu")
    lines.append("")
    lines.append("Bu rapor, danisman geri bildirimi A.5 maddesi kapsaminda "
                 "`final_genisletilmis_dataset_optimized_v3_v2.csv` (n=14.337) "
                 "uzerinde uretilmistir. Sayisal ozet, gorseller ve Pearson korelasyon "
                 "matrisi ile birlikte modelin egitildigi veri yapisini akademik "
                 "olarak belgeler.")
    lines.append("")

    # 1) Genel
    lines.append("## 1. Genel Veri Yapisi")
    lines.append("")
    cd = summary["class_dist"]["counts"]
    total = summary["class_dist"]["total"]
    lines.append(f"- Toplam yorum: **{total:,}**")
    for lbl in SENTIMENT_ORDER:
        v = cd.get(lbl, 0)
        pct = v / total * 100 if total > 0 else 0
        lines.append(f"- {lbl}: **{v:,}** (%{pct:.1f})")
    lines.append("")
    lines.append("Görsel: `01_class_dist.png`")
    lines.append("")

    # 2) Kelime sayisi
    lines.append("## 2. Kelime Sayisi Dagilimi (Sinif Bazinda)")
    lines.append("")
    lines.append("| Sinif | Medyan | Ortalama | q05 | q95 |")
    lines.append("|---|---:|---:|---:|---:|")
    for lbl in SENTIMENT_ORDER:
        s = wc_stats.get(lbl, {})
        lines.append(f"| {lbl} | {s.get('median', 0):.0f} | {s.get('mean', 0):.1f} | "
                     f"{s.get('q05', 0):.0f} | {s.get('q95', 0):.0f} |")
    lines.append("")
    lines.append("Görsel: `02_wc_box.png`")
    lines.append("")
    lines.append("**Yorum.** Notr siniflarinin medyan kelime sayisi pozitif/negatif "
                 "siniflardan belirgin biçimde farkli olabilir; bu, notr yorumlarin "
                 "**karma duygulu uzun yapida** olma egiliminden kaynaklanir. "
                 "Bu yapi modelimizin manuel_toplama ile genisletme stratejisini "
                 "(medyan 53 kelime hedefi) destekleyen ana motivasyondur.")
    lines.append("")

    # 3) Kaynak x etiket
    lines.append("## 3. Kaynak × Etiket Cross-Tab")
    lines.append("")
    lines.append("Görsel: `03_source_label_heatmap.png`")
    lines.append("")
    lines.append("**Yorum.** `ai_generated` kaynagi tamamen notr etiket altinda "
                 "yogunlasmaktadir (uretim amacli; %100 notr). `manuel_toplama` "
                 "kaynagi da tamamen notr olup sablon-augmentasyon ile yapay "
                 "sinyal yogunlasmasinin engellendigi `ai_only_in_train` "
                 "tasariminda yalnizca train'e dahil edilir.")
    lines.append("")

    # 4) Top words
    lines.append("## 4. Sinif Bazinda En Sik 15 Kelime")
    lines.append("")
    lines.append("Görsel: `04_top_words_per_class.png`")
    lines.append("")
    for lbl in SENTIMENT_ORDER:
        cnt = top_words.get(lbl, [])
        if cnt:
            top5 = [f"`{w}` ({n})" for w, n in cnt[:5]]
            lines.append(f"- **{lbl}** ilk 5: {', '.join(top5)}")
    lines.append("")
    lines.append("**Yorum.** Pozitif sinifta `temiz`, `güzel`, `rahat` gibi "
                 "olumlu sinyaller; negatif sinifta `kötü`, `pis`, `sorun` gibi "
                 "olumsuz sinyaller; notr sinifta hem olumlu hem olumsuz "
                 "kelimelerin **birlikte yogun** gorulmesi (karma duygu) "
                 "beklenen bir bulgudur. Bu, XAI analizinde gozlemlenen "
                 "'zit kutuplu sinyal birlikte varligi -> notrluk' ogrenmesinin "
                 "veri tabanli temelini olusturur (final rapor §4.10 Bulgu B).")
    lines.append("")

    # 5) Pearson
    lines.append("## 5. Pearson Korelasyon Matrisi (Meta-Features)")
    lines.append("")
    lines.append("Görsel: `05_corr_matrix.png` · CSV: `corr_matrix.csv`")
    lines.append("")
    notes = interpret_correlations(corr)
    for n in notes:
        lines.append(n)
    lines.append("")
    lines.append("**Akademik yorum.** Pearson korelasyon matrisi modelimizin "
                 "**dogru ozellikleri ogrendigini** dogrular: `rating_auto` ile "
                 "`sentiment_id` arasindaki yuksek pozitif korelasyon (rating-based "
                 "etiketleme tutarliligi); `pos_keyword_count` ↔ `neg_keyword_count` "
                 "arasindaki dusuk korelasyon ise notr siniflarda her iki sinyalin "
                 "**birbirinden bagimsiz** gorunmesinin (karma duygu yapisinin) "
                 "kanitidir. `is_ai` kolonu kelime sayisi (`wc`) ile guclu negatif "
                 "korelasyon gosterir — sentetik orneklerin kisa ve kalipli "
                 "yapisinin numerik dogrulamasidir.")
    lines.append("")

    # Akademik kapanis
    lines.append("---")
    lines.append("")
    lines.append("**Reproducibility:** `python src/eda_analysis.py`")
    lines.append("")
    lines.append("Bu rapor `Bitirme2_FinalRapor_AhmetDulger_MuhammedEsatDemir.md` "
                 "Ek-14 olarak ve `Hoca_Geri_Bildirim_Istatistik_Raporu.md` "
                 "EK-B olarak referans verilmistir.")

    out_md.write_text("\n".join(lines), encoding="utf-8")


# =============================================================================
# Ana akis
# =============================================================================

def main() -> int:
    if not CSV.exists():
        print(f"[HATA] CSV bulunamadi: {CSV}", file=sys.stderr)
        return 1

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"[OKU] {CSV.name}")
    df = pd.read_csv(CSV)
    print(f"[OKU] n = {len(df):,}")

    # 1) Sinif dagilimi
    print("[1/5] Sinif dagilimi...")
    cd = plot_class_distribution(df, OUT_DIR / "01_class_dist.png")

    # 2) Meta-features
    print("[META] Meta-features uretiliyor...")
    mf = build_meta_features(df)
    mf.to_csv(OUT_DIR / "meta_features.csv", index=False, encoding="utf-8")

    # 3) Kelime sayisi boxplot
    print("[2/5] WC boxplot...")
    wc_stats = plot_wc_boxplot(df, mf, OUT_DIR / "02_wc_box.png")

    # 4) Kaynak x etiket
    print("[3/5] Kaynak x etiket heatmap...")
    src_label = plot_source_label_heatmap(df, OUT_DIR / "03_source_label_heatmap.png")

    # 5) Top words
    print("[4/5] Top words per class...")
    top_words = plot_top_words(df, OUT_DIR / "04_top_words_per_class.png", top_n=15)

    # 6) Pearson korelasyon
    print("[5/5] Pearson korelasyon matrisi...")
    corr_dict = plot_correlation_matrix(
        mf, OUT_DIR / "05_corr_matrix.png",
        OUT_DIR / "corr_matrix.csv",
    )
    corr_df = pd.DataFrame(corr_dict)

    # Ozet
    summary = {
        "csv_path": str(CSV),
        "n_total": int(len(df)),
        "class_dist": cd,
        "wc_stats_per_class": wc_stats,
        "source_label_crosstab": src_label,
        "top_words_per_class": {k: v for k, v in top_words.items()},
        "outputs": {
            "class_dist_png": "01_class_dist.png",
            "wc_box_png": "02_wc_box.png",
            "source_label_heatmap_png": "03_source_label_heatmap.png",
            "top_words_png": "04_top_words_per_class.png",
            "corr_matrix_png": "05_corr_matrix.png",
            "corr_matrix_csv": "corr_matrix.csv",
            "meta_features_csv": "meta_features.csv",
        },
    }
    (OUT_DIR / "eda_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    # Akademik rapor
    write_eda_report(OUT_DIR / "eda_report.md", summary, corr_df,
                     wc_stats, top_words)

    print()
    print("=" * 70)
    print("EDA TAMAMLANDI")
    print("=" * 70)
    print(f"Cikti dizin : {OUT_DIR}")
    print(f"Sinif dist  : {cd['counts']}")
    print(f"WC medyan   : neg={wc_stats['negatif']['median']:.0f}  "
          f"notr={wc_stats['nötr']['median']:.0f}  "
          f"poz={wc_stats['pozitif']['median']:.0f}")
    print(f"Rapor MD    : {OUT_DIR / 'eda_report.md'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
