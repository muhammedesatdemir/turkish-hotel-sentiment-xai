# -*- coding: utf-8 -*-
"""
src/leakage_audit.py
====================

Bagimsiz sizinti audit scripti (hoca geri bildirimi A.2).

Hocanin itirazi:
    "Leakage kacmamasi icin AI'ya analiz ettir, kod ver, AI 'kesinlikle yok' desin."

Bizim cevabimiz: AI'ya dogrudan sorma akademik acidan zayif (LLM hallusinasyonu
riski). Yerine, mevcut assert_no_leakage'tan FARKLI uc ayri yontemle bagimsiz
dogrulama yapiyoruz:

    1) Exact match (lower+strip)               - mevcut yontem (kontrol icin)
    2) Normalized match (TR lowercase + accent + punct strip)
    3) Near-duplicate (Jaccard kelime kumesi)  - esik 0.80 ve 0.90

Cikti:
    reports/leakage_audit/<run_name>/
        report.md                   - akademik dilli ozet (final raporda EK olur)
        leakage_summary.json        - tum sayilar + esikler + hash
        leakage_pairs_jaccard_080.csv  - varsa, J>=0.80 yakin esleme listesi
        leakage_pairs_jaccard_090.csv  - varsa, J>=0.90 yakin esleme listesi

Kosma (mevcut nihai modelin splitleri):
    python src/leakage_audit.py \
        --train models/optimized_v3/v3_genisletilmis/bert-base-turkish-cased_len256_ep5_lr2e-05_losswce_seed42/splits/train.csv \
        --val   models/optimized_v3/v3_genisletilmis/bert-base-turkish-cased_len256_ep5_lr2e-05_losswce_seed42/splits/val.csv \
        --test  models/optimized_v3/v3_genisletilmis/bert-base-turkish-cased_len256_ep5_lr2e-05_losswce_seed42/splits/test.csv \
        --run_name v3_genisletilmis_seed42

Veya CV fold'lari icin (CV bittikten sonra):
    python src/leakage_audit.py \
        --cv_root models/cv/v3_genisletilmis_5fold_berturk \
        --run_name v3_genisletilmis_5fold_berturk
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import unicodedata
from pathlib import Path

import pandas as pd


# =============================================================================
# Normalize fonksiyonlari
# =============================================================================

# Turkce karakter -> ASCII benzeri (sadece accent + dotted-i icin; "ç" -> "c" vb.)
TR_ACCENT_MAP = str.maketrans({
    "ı": "i", "İ": "i", "ş": "s", "Ş": "s",
    "ç": "c", "Ç": "c", "ğ": "g", "Ğ": "g",
    "ö": "o", "Ö": "o", "ü": "u", "Ü": "u",
})
PUNCT_RE = re.compile(r"[^\w\s]", flags=re.UNICODE)
MULTISPACE_RE = re.compile(r"\s+")


def norm_exact(s: str) -> str:
    """Mevcut assert_no_leakage ile ayni: lower + strip."""
    if s is None:
        return ""
    return str(s).strip().lower()


def norm_full(s: str) -> str:
    """Daha sert normalize: TR accent, lower, NFKC, punct strip, multispace."""
    if s is None:
        return ""
    s = str(s)
    s = unicodedata.normalize("NFKC", s)
    s = s.translate(TR_ACCENT_MAP)
    s = s.lower()
    s = PUNCT_RE.sub(" ", s)
    s = MULTISPACE_RE.sub(" ", s).strip()
    return s


def to_word_set(s: str) -> set:
    """Jaccard icin kelime kumesi (norm_full sonrasi)."""
    n = norm_full(s)
    if not n:
        return set()
    return set(n.split())


# =============================================================================
# Sizinti tespit
# =============================================================================

def exact_overlap(a: list[str], b: list[str]) -> set[str]:
    sa = set(norm_exact(x) for x in a)
    sb = set(norm_exact(x) for x in b)
    sa.discard(""); sb.discard("")
    return sa & sb


def normalized_overlap(a: list[str], b: list[str]) -> set[str]:
    sa = set(norm_full(x) for x in a)
    sb = set(norm_full(x) for x in b)
    sa.discard(""); sb.discard("")
    return sa & sb


def jaccard_pairs(a_texts: list[str], b_texts: list[str], threshold: float,
                  min_words: int = 5) -> list[tuple]:
    """
    a_texts'teki her metin icin b_texts icinde Jaccard >= threshold olanlari bul.
    BUYUK TEST: O(n*m), 11K x 1.4K = 15M cift; cok yavas. Optimize:
    - Kelime kumesi olustur, kisa (<min_words) olanlari atla
    - Esle icin: a_word_sets, b_word_sets dictlerinden ortaklik tarama
    - Daha da iyi: inverted index (kelime -> b_idx), ortak kelimesi olan
      ciftleri filtrele.

    Inverted index yaklasimi:
        - b'deki her metin icin word_set
        - inv_idx: word -> [b_idx, ...]
        - a'daki her metin icin: kelimelerinden b_idx_set topla, sadece bunlarda
          jaccard hesapla.
    """
    a_sets = [to_word_set(t) for t in a_texts]
    b_sets = [to_word_set(t) for t in b_texts]

    # Inverted index: kelime -> b indeksleri
    inv = {}
    for j, ws in enumerate(b_sets):
        if len(ws) < min_words:
            continue
        for w in ws:
            inv.setdefault(w, []).append(j)

    pairs = []
    for i, a_ws in enumerate(a_sets):
        if len(a_ws) < min_words:
            continue
        # Aday b indeksleri: a'nin kelimelerinden gelen b'ler
        cand = set()
        for w in a_ws:
            cand.update(inv.get(w, ()))
        for j in cand:
            b_ws = b_sets[j]
            inter = a_ws & b_ws
            union = a_ws | b_ws
            if not union:
                continue
            jac = len(inter) / len(union)
            if jac >= threshold:
                pairs.append({
                    "a_idx": i,
                    "b_idx": j,
                    "jaccard": jac,
                    "a_text": a_texts[i],
                    "b_text": b_texts[j],
                })
    return pairs


# =============================================================================
# Audit ana akis
# =============================================================================

def audit_three_splits(train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame,
                       text_col: str, out_dir: Path, label: str = "") -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)

    a_train = train[text_col].astype(str).tolist()
    a_val = val[text_col].astype(str).tolist()
    a_test = test[text_col].astype(str).tolist()

    # 1) Exact
    e_tv = exact_overlap(a_train, a_val)
    e_tt = exact_overlap(a_train, a_test)
    e_vt = exact_overlap(a_val, a_test)

    # 2) Normalized
    n_tv = normalized_overlap(a_train, a_val)
    n_tt = normalized_overlap(a_train, a_test)
    n_vt = normalized_overlap(a_val, a_test)

    # 3) Jaccard near-dup, sadece train-test ve train-val (test-val ufak)
    j080_tt = jaccard_pairs(a_train, a_test, threshold=0.80)
    j090_tt = jaccard_pairs(a_train, a_test, threshold=0.90)
    j080_tv = jaccard_pairs(a_train, a_val, threshold=0.80)
    j090_tv = jaccard_pairs(a_train, a_val, threshold=0.90)

    # 0.80 ve 0.90 listelerinden yuksek jaccard'a gore sirali kayit
    if j080_tt:
        pd.DataFrame(j080_tt).sort_values("jaccard", ascending=False)\
            .to_csv(out_dir / f"leakage_pairs_jaccard_080_{label}_train_test.csv",
                    index=False, encoding="utf-8")
    if j090_tt:
        pd.DataFrame(j090_tt).sort_values("jaccard", ascending=False)\
            .to_csv(out_dir / f"leakage_pairs_jaccard_090_{label}_train_test.csv",
                    index=False, encoding="utf-8")
    if j080_tv:
        pd.DataFrame(j080_tv).sort_values("jaccard", ascending=False)\
            .to_csv(out_dir / f"leakage_pairs_jaccard_080_{label}_train_val.csv",
                    index=False, encoding="utf-8")
    if j090_tv:
        pd.DataFrame(j090_tv).sort_values("jaccard", ascending=False)\
            .to_csv(out_dir / f"leakage_pairs_jaccard_090_{label}_train_val.csv",
                    index=False, encoding="utf-8")

    summary = {
        "label": label,
        "n_train": len(train),
        "n_val": len(val),
        "n_test": len(test),
        "exact": {
            "train_val": len(e_tv),
            "train_test": len(e_tt),
            "val_test": len(e_vt),
        },
        "normalized_tr_accent_punct": {
            "train_val": len(n_tv),
            "train_test": len(n_tt),
            "val_test": len(n_vt),
        },
        "jaccard_near_dup": {
            "train_test_ge_080": len(j080_tt),
            "train_test_ge_090": len(j090_tt),
            "train_val_ge_080": len(j080_tv),
            "train_val_ge_090": len(j090_tv),
        },
    }
    return summary


def write_report_md(summaries: list[dict], out_path: Path, csv_hash: str = "") -> None:
    """Akademik dilli markdown ozet."""
    lines = []
    lines.append("# Bagimsiz Sizinti Audit Raporu")
    lines.append("")
    lines.append("Hoca geri bildirimi A.2 kapsaminda, mevcut `assert_no_leakage()` "
                 "fonksiyonundan **bagimsiz** uc ayri yontemle sizinti dogrulamasi "
                 "yapilmistir:")
    lines.append("")
    lines.append("1. **Exact match** — `text.strip().lower()` ile birebir esitlik (mevcut yontem ile karsilastirma).")
    lines.append("2. **Normalized match** — Turkce karakter + Unicode NFKC + lowercase + noktalama silme + multispace normalize.")
    lines.append("3. **Near-duplicate (Jaccard)** — kelime kumesi tabanli benzerlik, J ≥ 0,80 ve J ≥ 0,90 esikleri.")
    lines.append("")
    if csv_hash:
        lines.append(f"**Veri seti SHA-256 (ilk 16 hex):** `{csv_hash}`")
        lines.append("")
    lines.append("## Sonuclar")
    lines.append("")
    lines.append("| Konfigurasyon | n(train) | n(val) | n(test) | Exact tv | Exact tt | Exact vt | Norm tv | Norm tt | Norm vt | J≥0,80 tt | J≥0,90 tt | J≥0,80 tv | J≥0,90 tv |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for s in summaries:
        e = s["exact"]; n = s["normalized_tr_accent_punct"]; j = s["jaccard_near_dup"]
        lines.append(
            f"| {s['label']} | {s['n_train']} | {s['n_val']} | {s['n_test']} | "
            f"{e['train_val']} | {e['train_test']} | {e['val_test']} | "
            f"{n['train_val']} | {n['train_test']} | {n['val_test']} | "
            f"{j['train_test_ge_080']} | {j['train_test_ge_090']} | "
            f"{j['train_val_ge_080']} | {j['train_val_ge_090']} |"
        )
    lines.append("")
    lines.append("**Kisaltmalar:** tv = train↔val, tt = train↔test, vt = val↔test.")
    lines.append("")

    # Otomatik yorumlama
    has_any_leakage = False
    for s in summaries:
        e = s["exact"]; n = s["normalized_tr_accent_punct"]; j = s["jaccard_near_dup"]
        if (e["train_val"] + e["train_test"] + e["val_test"]
            + n["train_val"] + n["train_test"] + n["val_test"]
            + j["train_test_ge_090"] + j["train_val_ge_090"]) > 0:
            has_any_leakage = True
            break

    lines.append("## Yorumlama")
    lines.append("")
    if has_any_leakage:
        lines.append("Bazi konfigurasyonlarda sifir-olmayan sizinti tespit edilmistir. "
                     "Lutfen detayli `leakage_pairs_jaccard_*.csv` dosyalarini inceleyin.")
    else:
        lines.append("**Tum konfigurasyonlarda exact ve normalized eslestirme uzerinden "
                     "sizinti sifirdir.** Jaccard ≥ 0,90 esik degerinde de yakin-duplicate "
                     "tespit edilmemistir; veri setinde train/val/test arasinda hem birebir "
                     "hem de yakin-icerik sizmasi yoktur. Bu sonuc, kodda kullanilan "
                     "`assert_no_leakage()` mekanizmasinin uretim sirasinda yeterli "
                     "korumayi sagladigini bagimsiz olarak dogrular.")
    lines.append("")
    lines.append("Jaccard ≥ 0,80 esik degerinde tespit edilen ciftler **birebir kopya degildir**; "
                 "kelime kumesi olcekli yakinligi gosterir. Bu olcek, ozellikle kisa metinlerde "
                 "(bag of words 5-10 kelime) ayni kelime havuzundan gelen FARKLI yorumlari "
                 "yanlis yakalayabilir; bu nedenle 0,80 listesi tum-veri-tabani benzerlik "
                 "incelemesi icin kullanilmis, kesin sizinti kanit olarak J ≥ 0,90 esiki "
                 "alinmistir.")
    lines.append("")

    out_path.write_text("\n".join(lines), encoding="utf-8")


# =============================================================================
# CLI
# =============================================================================

def main():
    p = argparse.ArgumentParser(description="Bagimsiz sizinti audit (3 yontem).")
    p.add_argument("--train", type=str, default=None, help="Tek konfigurasyon: train.csv")
    p.add_argument("--val", type=str, default=None)
    p.add_argument("--test", type=str, default=None)
    p.add_argument("--cv_root", type=str, default=None,
                   help="CV koklu dizin (icindeki tum seed*_fold*/.../splits taranir).")
    p.add_argument("--run_name", type=str, required=True)
    p.add_argument("--text_col", type=str, default="comment_text")
    p.add_argument("--out_root", type=str, default="reports/leakage_audit")
    args = p.parse_args()

    out_dir = Path(args.out_root) / args.run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    summaries = []

    if args.cv_root:
        cv_root = Path(args.cv_root)
        # Her fold'un splits/ dizini
        fold_dirs = sorted([d for d in cv_root.glob("seed*_fold*")
                            if d.is_dir()])
        if not fold_dirs:
            print(f"[HATA] CV fold dizini bulunamadi: {cv_root}", file=sys.stderr)
            return 1
        for fd in fold_dirs:
            # icindeki bert-base-turkish-cased_*/splits/
            cand = list(fd.glob("*/splits"))
            if not cand:
                print(f"[UYARI] {fd}: splits/ dizini yok, atlaniyor.")
                continue
            sp = cand[0]
            try:
                tr = pd.read_csv(sp / "train.csv")
                va = pd.read_csv(sp / "val.csv")
                te = pd.read_csv(sp / "test.csv")
            except Exception as e:
                print(f"[UYARI] {fd}: csv okunamadi ({e}), atlaniyor.")
                continue
            label = fd.name
            print(f"[AUDIT] {label}: train={len(tr)} val={len(va)} test={len(te)}")
            s = audit_three_splits(tr, va, te, args.text_col, out_dir, label=label)
            summaries.append(s)
    else:
        if not (args.train and args.val and args.test):
            print("[HATA] Tek konfigurasyon icin --train --val --test gerekli.", file=sys.stderr)
            return 2
        tr = pd.read_csv(args.train)
        va = pd.read_csv(args.val)
        te = pd.read_csv(args.test)
        label = "single"
        print(f"[AUDIT] {label}: train={len(tr)} val={len(va)} test={len(te)}")
        s = audit_three_splits(tr, va, te, args.text_col, out_dir, label=label)
        summaries.append(s)

    # Ozet JSON
    summary_path = out_dir / "leakage_summary.json"
    summary_path.write_text(
        json.dumps({"run_name": args.run_name, "n_configs": len(summaries),
                    "results": summaries}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    # Markdown rapor
    report_path = out_dir / "report.md"
    write_report_md(summaries, report_path)

    # Konsol ozeti
    print()
    print("=" * 70)
    print(f"LEAKAGE AUDIT: {args.run_name}")
    print("=" * 70)
    for s in summaries:
        e = s["exact"]; n = s["normalized_tr_accent_punct"]; j = s["jaccard_near_dup"]
        print(f"[{s['label']}] n={s['n_train']}/{s['n_val']}/{s['n_test']}")
        print(f"  exact tv/tt/vt = {e['train_val']}/{e['train_test']}/{e['val_test']}")
        print(f"  norm  tv/tt/vt = {n['train_val']}/{n['train_test']}/{n['val_test']}")
        print(f"  jacc  tt(.80/.90) = {j['train_test_ge_080']}/{j['train_test_ge_090']}")
        print(f"  jacc  tv(.80/.90) = {j['train_val_ge_080']}/{j['train_val_ge_090']}")
    print("=" * 70)
    print(f"Markdown: {report_path}")
    print(f"JSON    : {summary_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
