"""Trajectory (ADE/FDE) and action metrics.

The model is deterministic, so there is no best-of-K metric here.
"""

from __future__ import annotations

import numpy as np


def ade_fde(pred: np.ndarray, true: np.ndarray) -> tuple:
    """ADE and FDE for (N, T, 2) trajectories in metres."""
    d = np.linalg.norm(pred - true, axis=-1)          # (N, T)
    return float(d.mean()), float(d[:, -1].mean())


def action_report(pred: np.ndarray, true: np.ndarray, classes: tuple) -> dict:
    acc = float((pred == true).mean())
    per_class = {}
    for i, name in enumerate(classes):
        mask = true == i
        if mask.any():
            per_class[name] = {
                "support": int(mask.sum()),
                "recall": round(float((pred[mask] == i).mean()), 4),
            }
    return {"accuracy": round(acc, 4), "per_class_recall": per_class}


def aggregate(per_fold: list) -> dict:
    """Mean and std of every scalar metric across folds."""
    out = {}
    for key in per_fold[0]:
        vals = [f[key] for f in per_fold if isinstance(f.get(key), (int, float))]
        if len(vals) == len(per_fold):
            out[key] = {
                "mean": round(float(np.mean(vals)), 4),
                "std": round(float(np.std(vals)), 4),
            }
    return out
