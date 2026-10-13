"""Closed-form linear model: state window -> trajectory window.

    y = bias + W @ standardise(x)

x is (vEgo, aEgo, steeringAngleDeg) at 20 points over 3 s and y the (x, y)
positions over the same window. Fitted with ridge regression in closed form.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from config import DOWNSAMPLE


class LinearWorldModel:
    """Ridge regression with an unpenalised bias."""

    def __init__(self, alpha: float = 1.0):
        self.alpha = float(alpha)
        self.mu: np.ndarray | None = None
        self.sd: np.ndarray | None = None
        self.Wb: np.ndarray | None = None   # (n_features + 1, n_targets); last row = bias

    def fit(self, X: np.ndarray, Y: np.ndarray) -> "LinearWorldModel":
        X = np.asarray(X, dtype=np.float64)
        Y = np.asarray(Y, dtype=np.float64)
        if len(X) <= X.shape[1]:
            raise ValueError(
                f"only {len(X)} training rows for {X.shape[1]} features -- "
                "too few to fit safely, check the data split"
            )

        self.mu = X.mean(axis=0)
        self.sd = X.std(axis=0)
        self.sd[self.sd < 1e-8] = 1.0   # constant input columns
        Z = (X - self.mu) / self.sd
        Z1 = np.hstack([Z, np.ones((len(Z), 1))])

        D = Z1.shape[1]
        A = Z1.T @ Z1 + self.alpha * np.eye(D)
        A[-1, -1] -= self.alpha         # do not penalise the bias term
        B = Z1.T @ Y
        self.Wb = np.linalg.solve(A, B)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.Wb is None:
            raise RuntimeError("call fit() (or load()) before predict()")
        X = np.asarray(X, dtype=np.float64)
        Z = (X - self.mu) / self.sd
        Z1 = np.hstack([Z, np.ones((len(Z), 1))])
        return Z1 @ self.Wb

    @property
    def n_params(self) -> int:
        return int(self.Wb.size + self.mu.size + self.sd.size)

    def save(self, path: Path) -> None:
        payload = {
            "alpha": self.alpha,
            "mu": self.mu.tolist(),
            "sd": self.sd.tolist(),
            "Wb": self.Wb.tolist(),
            "n_features": int(self.mu.shape[0]),
            "n_targets": int(self.Wb.shape[1]),
        }
        Path(path).write_text(json.dumps(payload, indent=2))

    @classmethod
    def load(cls, path: Path) -> "LinearWorldModel":
        payload = json.loads(Path(path).read_text())
        m = cls(alpha=payload["alpha"])
        m.mu = np.array(payload["mu"], dtype=np.float64)
        m.sd = np.array(payload["sd"], dtype=np.float64)
        m.Wb = np.array(payload["Wb"], dtype=np.float64)
        return m


def ade_fde(pred: np.ndarray, true: np.ndarray, n_wp: int = DOWNSAMPLE):
    """ADE / FDE in metres. pred, true: (N, n_wp * 2) flattened waypoints."""
    p = pred.reshape(-1, n_wp, 2)
    t = true.reshape(-1, n_wp, 2)
    d = np.linalg.norm(p - t, axis=2)      # (N, n_wp) per-waypoint distance
    return float(d.mean()), float(d[:, -1].mean())


def rmse(pred: np.ndarray, true: np.ndarray) -> float:
    return float(np.sqrt(np.mean((pred - true) ** 2)))


def mae(pred: np.ndarray, true: np.ndarray) -> float:
    return float(np.mean(np.abs(pred - true)))
