"""Custom model + dataloader definitions for α,β-CROWN toy experiments.

This file is intentionally lightweight and self-contained so it can be imported by
`test.py` and α,β-CROWN custom loaders.
"""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np


def build_toy_model(in_channel: int = 2, out_dim: int = 2):
    """Return a small MLP used as external model in this assignment."""
    return ToyBinaryMLP(in_dim=in_channel, hidden=16, out_dim=out_dim)


class ToyBinaryMLP(nn.Module):
    def __init__(self, in_dim: int = 2, hidden: int = 16, out_dim: int = 2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def load_toy_box_data(dataset_csv: str | Path = "data/toy_dataset.csv", batch_size: int = 64):
    """Load the generated toy dataset for local robustness verification examples.

    Expected CSV format:
      - columns: x0, x1, label
      - values are normalized to [0, 1]
    """
    path = Path(dataset_csv)
    if not path.exists():
        raise FileNotFoundError(f"dataset not found: {path}")

    data = np.loadtxt(path, delimiter=",", skiprows=1)
    x = torch.as_tensor(data[:, :2], dtype=torch.float32)
    y = torch.as_tensor(data[:, 2], dtype=torch.long)

    dataset = TensorDataset(x, y)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False)


def select_reference_point(dataset_csv: str | Path = "data/toy_dataset.csv", target_label: int = 0):
    """Return one reference feature row for `target_label` with deterministic ordering."""
    path = Path(dataset_csv)
    if not path.exists():
        raise FileNotFoundError(f"dataset not found: {path}")

    data = np.loadtxt(path, delimiter=",", skiprows=1)
    mask = data[:, 2] == float(target_label)
    candidates = data[mask][:, :2]
    if len(candidates) == 0:
        raise ValueError(f"no row with label={target_label} in {path}")

    # A deterministic representative point: 75% quantile position.
    idx = max(0, int(len(candidates) * 0.75) - 1)
    return candidates[idx]
