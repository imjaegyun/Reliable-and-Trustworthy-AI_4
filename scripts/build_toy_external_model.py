"""Create a small synthetic 2D dataset and a tiny PyTorch classifier.

The generated files are:
- `models/toy_binary_mlp.pt` (state_dict of a small 2-layer MLP)
- `data/toy_dataset.csv` (normalized toy dataset)
- `data/toy_reference_point.json` (one center sample and metadata)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset

from custom.toy_model_data import build_toy_model


ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "toy_binary_mlp.pt"
DATA_PATH = ROOT / "data" / "toy_dataset.csv"
REFERENCE_PATH = ROOT / "data" / "toy_reference_point.json"


def generate_dataset(seed: int, n_per_class: int = 1000) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    # Two Gaussian clouds in 2D (normalized afterwards).
    x0 = rng.normal(loc=-1.0, scale=0.4, size=(n_per_class, 2))
    x1 = rng.normal(loc=1.0, scale=0.4, size=(n_per_class, 2))

    # Make class0 and class1 separated more in each dimension.
    x = np.vstack([x0, x1])
    y = np.concatenate([np.zeros(n_per_class), np.ones(n_per_class)])

    # Normalize to [0,1] so that bound radii are easy to interpret.
    x_min = x.min(axis=0)
    x_max = x.max(axis=0)
    x_norm = (x - x_min) / (x_max - x_min + 1e-8)
    return x_norm.astype(np.float32), y.astype(np.int64)


def train_model(x: np.ndarray, y: np.ndarray, epochs: int = 80, seed: int = 1) -> nn.Module:
    torch.manual_seed(seed)
    tensor_x = torch.tensor(x, dtype=torch.float32)
    tensor_y = torch.tensor(y, dtype=torch.long)

    dataset = TensorDataset(tensor_x, tensor_y)
    loader = DataLoader(dataset, batch_size=64, shuffle=True)

    model = build_toy_model(in_channel=2, out_dim=2)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=2e-2)

    for epoch in range(1, epochs + 1):
        total = 0.0
        correct = 0
        total_count = 0
        for batch_x, batch_y in loader:
            optimizer.zero_grad()
            logits = model(batch_x)
            loss = loss_fn(logits, batch_y)
            loss.backward()
            optimizer.step()
            total += float(loss.item()) * len(batch_x)
            pred = logits.argmax(dim=-1)
            correct += int((pred == batch_y).sum().item())
            total_count += len(batch_x)
        if epoch % 20 == 0:
            acc = correct / max(1, total_count)
            print(f"epoch={epoch:03d} loss={total / max(1, total_count):.4f} acc={acc:.4f}")

    return model


def save_dataset_csv(x: np.ndarray, y: np.ndarray) -> None:
    DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    header = "x0,x1,label"
    arr = np.column_stack([x, y[:, None]])
    np.savetxt(DATA_PATH, arr, delimiter=",", fmt="%.6f", header=header, comments="")


def save_reference_point(x: np.ndarray, y: np.ndarray, model: nn.Module) -> None:
    with torch.no_grad():
        logits = model(torch.tensor(x, dtype=torch.float32))
        pred = logits.argmax(dim=-1).numpy()
    correct_idx = np.where(pred == y)[0]
    label0_mask = np.array([], dtype=np.int64)
    if len(correct_idx) > 0:
        label0_mask = correct_idx[y[correct_idx] == 0]
    if len(label0_mask) == 0:
        # fallback
        label0_mask = np.where(y == 0)[0]

    # Use the point with the largest margin for class 0 as center sample.
    probs = torch.softmax(logits, dim=1).numpy()
    margins = probs[:, 0] - probs[:, 1]
    cands = label0_mask[margins[label0_mask].argsort()[::-1]]
    center_idx = int(cands[0])
    center = x[center_idx]

    REFERENCE_PATH.parent.mkdir(parents=True, exist_ok=True)
    REFERENCE_PATH.write_text(
        json.dumps(
            {
                "index": center_idx,
                "label": int(y[center_idx]),
                "point": center.tolist(),
                "margin": float(margins[center_idx]),
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--n-per-class", type=int, default=800)
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    if MODEL_PATH.exists() and DATA_PATH.exists() and not args.force:
        raise SystemExit(
            "Existing artifacts already exist. Use --force to overwrite."
        )

    x, y = generate_dataset(seed=args.seed, n_per_class=args.n_per_class)
    print(f"Generated dataset: x={x.shape}, y={y.shape}")

    save_dataset_csv(x, y)

    model = train_model(x, y, epochs=args.epochs, seed=args.seed)
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict()}, MODEL_PATH)
    print(f"saved model: {MODEL_PATH}")

    save_reference_point(x, y, model)
    print(f"saved dataset: {DATA_PATH}")
    print(f"saved reference point: {REFERENCE_PATH}")


if __name__ == "__main__":
    main()
