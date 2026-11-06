# -*- coding: utf-8 -*-
"""
ELECTRA-tr'yi SIZINTILI veri (v3_baseline, test'te 250 AI) + seed 42 ile egitir.
Amac: BERTurk'un sahte 0.9122 skoruyla birebir karsilastirilabilir bir ELECTRA
sahte-skoru uretmek. Boylece leakage bolumu %100 ELECTRA olur.

Protokol BERTurk v3_baseline ile ayni: ayni split dosyalari, WCE, seed 42,
max_len 256, lr 2e-5, 5 epoch, FP16.
Cikti: reports/electra_leakage/result.json
"""
import glob, json, random
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_linear_schedule_with_warmup
from sklearn.metrics import f1_score, accuracy_score, classification_report
from sklearn.utils.class_weight import compute_class_weight

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "reports" / "electra_leakage"
OUT.mkdir(parents=True, exist_ok=True)
SEED = 42
MODEL = "dbmdz/electra-base-turkish-cased-discriminator"
MAXLEN, LR, EPOCHS, BS = 256, 2e-5, 5, 16
LABELS = {"negatif": 0, "nötr": 1, "pozitif": 2}

# ayni sizintili split (BERTurk v3_baseline seed42)
SPLIT = Path(glob.glob(str(ROOT / "models/optimized_v3/v3_baseline/*/splits"))[0])

random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
dev = "cuda" if torch.cuda.is_available() else "cpu"


def load(split):
    df = pd.read_csv(SPLIT / f"{split}.csv")
    txt = df["comment_text"].astype(str).tolist()
    if "labels" in df.columns:
        y = df["labels"].astype(int).tolist()
    else:
        y = df["sentiment_label"].map(LABELS).astype(int).tolist()
    return txt, y, df


tr_x, tr_y, tr_df = load("train")
te_x, te_y, te_df = load("test")
ai_in_test = (te_df["source"] == "ai_generated").sum() if "source" in te_df.columns else "?"
print(f"Train: {len(tr_x)} | Test: {len(te_x)} (test AI: {ai_in_test})  <- SIZINTILI dogrulama")

tok = AutoTokenizer.from_pretrained(MODEL)


class DS(Dataset):
    def __init__(self, x, y):
        self.x, self.y = x, y
    def __len__(self): return len(self.x)
    def __getitem__(self, i):
        e = tok(self.x[i], truncation=True, padding="max_length", max_length=MAXLEN, return_tensors="pt")
        return {k: v.squeeze(0) for k, v in e.items()}, torch.tensor(self.y[i], dtype=torch.long)


def collate(b):
    xs = {k: torch.stack([d[0][k] for d in b]) for k in b[0][0]}
    ys = torch.stack([d[1] for d in b])
    return xs, ys


tr_loader = DataLoader(DS(tr_x, tr_y), batch_size=BS, shuffle=True, collate_fn=collate)
te_loader = DataLoader(DS(te_x, te_y), batch_size=32, shuffle=False, collate_fn=collate)

model = AutoModelForSequenceClassification.from_pretrained(MODEL, num_labels=3).to(dev)

cw = compute_class_weight("balanced", classes=np.array([0, 1, 2]), y=np.array(tr_y))
crit = nn.CrossEntropyLoss(weight=torch.tensor(cw, dtype=torch.float).to(dev))
opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
total = len(tr_loader) * EPOCHS
sched = get_linear_schedule_with_warmup(opt, int(0.1 * total), total)
scaler = torch.cuda.amp.GradScaler()

print("Egitim basliyor (ELECTRA, sizintili, seed42)...")
for ep in range(EPOCHS):
    model.train()
    for xs, ys in tr_loader:
        xs = {k: v.to(dev) for k, v in xs.items()}; ys = ys.to(dev)
        opt.zero_grad()
        with torch.cuda.amp.autocast():
            out = model(**xs); loss = crit(out.logits, ys)
        scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); sched.step()
    print(f"  epoch {ep+1}/{EPOCHS} bitti")

# degerlendirme
model.eval(); preds = []
with torch.no_grad():
    for xs, ys in te_loader:
        xs = {k: v.to(dev) for k, v in xs.items()}
        with torch.cuda.amp.autocast():
            logits = model(**xs).logits
        preds.extend(logits.argmax(-1).cpu().tolist())

acc = accuracy_score(te_y, preds)
mf1 = f1_score(te_y, preds, average="macro")
rep = classification_report(te_y, preds, target_names=["negatif", "nötr", "pozitif"], digits=4)
print("\n=== ELECTRA SIZINTILI (LEAKY) SONUC ===")
print(f"Accuracy = {acc:.4f}")
print(f"Macro-F1 = {mf1:.4f}   <-- ELECTRA sahte-skor (BERTurk leaky = 0.9122)")
print(rep)

json.dump({
    "model": "ELECTRA-tr",
    "setting": "leaky (synthetic in all splits, v3_baseline split, seed 42)",
    "test_ai_count": int(ai_in_test) if ai_in_test != "?" else None,
    "accuracy": round(acc, 4),
    "macro_f1": round(mf1, 4),
    "berturk_leaky_macro_f1_ref": 0.9122,
}, open(OUT / "result.json", "w"), indent=2)
print("\nyazildi:", OUT / "result.json")
