# src/compute_class_weights_3class_splits3.py

from pathlib import Path
import numpy as np
import pandas as pd
import torch
from sklearn.utils.class_weight import compute_class_weight

TRAIN_CSV = Path("data/processed/splits3/train.csv")

# Etiket sırası SABİT olmalı (model + report için)
LABELS = ["negatif", "nötr", "pozitif"]

def main():
    if not TRAIN_CSV.exists():
        raise FileNotFoundError(f"Train dosyası bulunamadı: {TRAIN_CSV}")

    df = pd.read_csv(TRAIN_CSV)

    if "sentiment_label" not in df.columns:
        raise ValueError("train.csv içinde sentiment_label kolonu yok.")

    y = df["sentiment_label"].astype(str).str.strip().str.lower().values

    # sadece LABELS içindeki sınıflar kalsın
    mask = np.isin(y, LABELS)
    y = y[mask]

    counts = {lab: int((y == lab).sum()) for lab in LABELS}
    print(f"[INFO] Train size: {len(y)}")
    print(f"[INFO] Counts: {counts}")

    classes = np.array(LABELS)
    weights = compute_class_weight(class_weight="balanced", classes=classes, y=y)

    print(f"[OK] LABELS: {LABELS}")
    print(f"[OK] Weights: {weights.tolist()}")

    w_tensor = torch.tensor(weights, dtype=torch.float)
    print(f"[OK] Torch tensor: {w_tensor}")

if __name__ == "__main__":
    main()
