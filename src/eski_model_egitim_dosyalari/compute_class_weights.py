from pathlib import Path
import pandas as pd
import torch

INPUT_TRAIN = Path("data/processed/splits/train.csv")

# Etiket sırası sabit olmalı
LABELS = ["çok olumsuz", "olumsuz", "nötr", "olumlu", "çok olumlu"]

def main():
    df = pd.read_csv(INPUT_TRAIN)

    counts = df["sentiment_label"].value_counts().to_dict()
    N = len(df)
    K = len(LABELS)

    weights = []
    for lab in LABELS:
        n_c = counts.get(lab, 0)
        w = (N / (K * n_c)) if n_c > 0 else 0.0
        weights.append(w)

    w_tensor = torch.tensor(weights, dtype=torch.float)

    print("[INFO] Train size:", N)
    print("[INFO] Counts:", counts)
    print("[OK] LABELS:", LABELS)
    print("[OK] Weights:", weights)
    print("[OK] Torch tensor:", w_tensor)

if __name__ == "__main__":
    main()
