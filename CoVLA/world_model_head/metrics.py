"""Evaluation metrics.

Same ADE / FDE / min-ADE / min-FDE as ../vla_trajecotry_head/metrics.py, plus

  * per_step_displacement_m: error at each waypoint over the horizon;
  * action_report: ranges of the predicted actions, tanh saturation, and how
    much displacement the learned residual contributes.

Text-embedding fidelity (ablation b) matches ../vla_trajecotry_head.
"""
from __future__ import annotations

import numpy as np

from config import ACC_MAX, KAPPA_MAX


def ade_fde(pred: np.ndarray, true: np.ndarray) -> tuple[float, float]:
    d = np.linalg.norm(pred - true, axis=-1)
    return float(d.mean()), float(d[:, -1].mean())


def min_ade_fde(preds: np.ndarray, true: np.ndarray) -> tuple[float, float]:
    """Best-of-K ADE/FDE. preds: (K, N, T, 2)."""
    d = np.linalg.norm(preds - true[None], axis=-1)
    return float(d.mean(axis=2).min(axis=0).mean()), float(d[:, :, -1].min(axis=0).mean())


def per_step_displacement(pred: np.ndarray, true: np.ndarray) -> list[float]:
    d = np.linalg.norm(pred - true, axis=-1)
    return [round(float(x), 4) for x in d.mean(axis=0)]


def trajectory_report(pred: np.ndarray, true: np.ndarray) -> dict:
    ade, fde = ade_fde(pred, true)
    d = np.linalg.norm(pred - true, axis=-1)
    return {
        "n": int(len(true)),
        "ade_m": round(ade, 4),
        "fde_m": round(fde, 4),
        "ade_median_m": round(float(np.median(d.mean(axis=1))), 4),
        "fde_p90_m": round(float(np.percentile(d[:, -1], 90)), 4),
        "per_step_displacement_m": per_step_displacement(pred, true),
    }


def action_report(actions: np.ndarray, residual: np.ndarray | None = None) -> dict:
    """actions: (N, 19, 2) as emitted. residual: (N, 20, 2) metres, or None."""
    a, k = actions[..., 0], actions[..., 1]
    rep = {
        "accel_mps2": {
            "mean": round(float(a.mean()), 4),
            "std": round(float(a.std()), 4),
            "p01": round(float(np.percentile(a, 1)), 4),
            "p99": round(float(np.percentile(a, 99)), 4),
            "saturated_frac": round(float((np.abs(a) > 0.99 * ACC_MAX).mean()), 5),
        },
        "curvature_1pm": {
            "mean": round(float(k.mean()), 6),
            "std": round(float(k.std()), 6),
            "p01": round(float(np.percentile(k, 1)), 6),
            "p99": round(float(np.percentile(k, 99)), 6),
            "saturated_frac": round(float((np.abs(k) > 0.99 * KAPPA_MAX).mean()), 5),
        },
        "mean_abs_jerk_mps3": round(float(np.abs(np.diff(a, axis=1)).mean() / 0.15), 4),
    }
    if residual is not None:
        rmag = np.linalg.norm(residual, axis=-1)
        rep["residual"] = {
            "mean_abs_m": round(float(rmag.mean()), 4),
            "final_step_mean_abs_m": round(float(rmag[:, -1].mean()), 4),
            "note": "displacement added AFTER the kinematic rollout; if this "
                    "approaches the ADE, the physics is no longer doing the work",
        }
    return rep


def cosine(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = a / np.maximum(np.linalg.norm(a, axis=-1, keepdims=True), 1e-8)
    b = b / np.maximum(np.linalg.norm(b, axis=-1, keepdims=True), 1e-8)
    return (a * b).sum(-1)


def text_fidelity_report(pred_emb, true_emb, ceiling_emb, mean_emb) -> dict:
    cos_model = cosine(pred_emb, true_emb)
    cos_ceiling = cosine(ceiling_emb, true_emb)
    cos_mean = cosine(np.repeat(mean_emb[None], len(true_emb), axis=0), true_emb)
    span = float(cos_ceiling.mean() - cos_mean.mean())
    scaled = ((float(cos_model.mean()) - float(cos_mean.mean())) / span
              if abs(span) > 1e-6 else float("nan"))
    return {
        "n": int(len(true_emb)),
        "cosine_model": round(float(cos_model.mean()), 4),
        "cosine_pca_ceiling": round(float(cos_ceiling.mean()), 4),
        "cosine_mean_baseline": round(float(cos_mean.mean()), 4),
        "scaled_score": round(scaled, 4),
    }
