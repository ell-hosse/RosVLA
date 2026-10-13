"""Train-only standardisation and early stopping (as in CoVLA/vla_trajecotry_head/utils.py)."""
from __future__ import annotations

import numpy as np


class Standardiser:
    """Zero-mean, unit-variance over the last axis, fitted on training rows."""

    def __init__(self, x: np.ndarray):
        flat = x.reshape(-1, x.shape[-1])
        self.mean = flat.mean(axis=0)
        self.std = np.maximum(flat.std(axis=0), 1e-6)

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return ((x - self.mean) / self.std).astype(np.float32)

    def inverse(self, x: np.ndarray) -> np.ndarray:
        return (x * self.std + self.mean).astype(np.float32)


class EarlyStopper:
    """Early stopping on a lower-is-better metric; keeps the best weights."""

    def __init__(self, patience: int, min_delta: float = 0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.best = float("inf")
        self.best_epoch = -1
        self.best_state = None
        self.n_bad = 0

    def step(self, metric: float, epoch: int, state_dict) -> bool:
        if metric < self.best - self.min_delta:
            self.best = metric
            self.best_epoch = epoch
            self.best_state = {k: v.detach().cpu().clone() for k, v in state_dict.items()}
            self.n_bad = 0
        else:
            self.n_bad += 1
        return self.n_bad >= self.patience
