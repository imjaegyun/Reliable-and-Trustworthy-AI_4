"""Custom model and data functions for the assignment alpha-beta-CROWN run."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "toy_binary_mlp.pt"
DATA_PATH = ROOT / "data" / "toy_dataset.csv"
REFERENCE_PATH = ROOT / "data" / "toy_reference_point.json"


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


def build_toy_model(in_channel: int = 2, out_dim: int = 2):
    """Return the small MLP and load the trained checkpoint when available."""
    model = ToyBinaryMLP(in_dim=in_channel, hidden=16, out_dim=out_dim)
    if MODEL_PATH.exists():
        checkpoint = torch.load(MODEL_PATH, map_location="cpu")
        state_dict = checkpoint.get("state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
        model.load_state_dict(state_dict)
    return model


def toy_box_data(spec):
    """Return one toy verification instance for alpha-beta-CROWN.

    alpha-beta-CROWN customized data loaders return:
      X, labels, data_max, data_min, eps
    """
    eps = spec.get("epsilon", 0.05)
    if REFERENCE_PATH.exists():
        payload = json.loads(REFERENCE_PATH.read_text(encoding="utf-8"))
        point = payload["point"]
        label = payload["label"]
    else:
        data = np.loadtxt(DATA_PATH, delimiter=",", skiprows=1)
        point = data[0, :2].tolist()
        label = int(data[0, 2])

    x = torch.tensor([point], dtype=torch.float32)
    labels = torch.tensor([label], dtype=torch.long)
    data_max = torch.ones_like(x)
    data_min = torch.zeros_like(x)
    eps_tensor = torch.tensor(eps, dtype=torch.float32).reshape(1, -1)
    return x, labels, data_max, data_min, eps_tensor


def load_toy_box_data(dataset_csv: str | Path = DATA_PATH, batch_size: int = 64):
    """Load the generated toy dataset for manual checks."""
    path = Path(dataset_csv)
    if not path.exists():
        raise FileNotFoundError(f"dataset not found: {path}")

    data = np.loadtxt(path, delimiter=",", skiprows=1)
    x = torch.as_tensor(data[:, :2], dtype=torch.float32)
    y = torch.as_tensor(data[:, 2], dtype=torch.long)
    return DataLoader(TensorDataset(x, y), batch_size=batch_size, shuffle=False)


def select_reference_point(dataset_csv: str | Path = DATA_PATH, target_label: int = 0):
    """Return one deterministic reference feature row for a target label."""
    path = Path(dataset_csv)
    if not path.exists():
        raise FileNotFoundError(f"dataset not found: {path}")

    data = np.loadtxt(path, delimiter=",", skiprows=1)
    mask = data[:, 2] == float(target_label)
    candidates = data[mask][:, :2]
    if len(candidates) == 0:
        raise ValueError(f"no row with label={target_label} in {path}")

    idx = max(0, int(len(candidates) * 0.75) - 1)
    return candidates[idx]
