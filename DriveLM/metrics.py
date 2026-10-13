"""Evaluation metrics.

Trajectory
----------
* ADE / FDE / min-ADE / min-FDE / per-step error over the 6 future waypoints
  (the origin is not counted, unlike CoVLA).
* nuScenes planning L2 at 1 / 2 / 3 s in both conventions:
    - "stp3"  (ST-P3, VAD, OpenDriveVLA, BEV-Planner): mean L2 over all
      waypoints up to t;
    - "uniad" (UniAD, GPT-Driver): L2 at t.
  "Avg" is the mean over the three horizons. Collision rate is not computed.

Text
----
* cosine vs. PCA ceiling and train-mean baseline, scaled score (as in CoVLA)
* retrieval accuracy against the training caption bank
* DriveLM behavior accuracy (Acc / Speed / Steer) of the retrieved caption
"""
from __future__ import annotations

import numpy as np

from config import ACC_MAX, DT, KAPPA_MAX, L2_IDX, L2_TIMES_S


def ade_fde(pred: np.ndarray, true: np.ndarray) -> tuple[float, float]:
    d = np.linalg.norm(pred - true, axis=-1)
    return float(d.mean()), float(d[:, -1].mean())


def min_ade_fde(preds: np.ndarray, true: np.ndarray) -> tuple[float, float]:
    """Best-of-K ADE/FDE. preds: (K, N, T, 2)."""
    d = np.linalg.norm(preds - true[None], axis=-1)
    return float(d.mean(axis=2).min(axis=0).mean()), float(d[:, :, -1].min(axis=0).mean())


def nuscenes_l2(pred: np.ndarray, true: np.ndarray) -> dict:
    d = np.linalg.norm(pred - true, axis=-1)            # (N, 6)
    out = {}
    for conv in ("stp3", "uniad"):
        vals = {}
        for t, i in zip(L2_TIMES_S, L2_IDX):
            v = d[:, : i + 1].mean() if conv == "stp3" else d[:, i].mean()
            vals[f"{t:.0f}s"] = round(float(v), 4)
        vals["avg"] = round(float(np.mean(list(vals.values()))), 4)
        out[conv] = vals
    return out


def trajectory_report(pred: np.ndarray, true: np.ndarray, mask=None) -> dict:
    if mask is not None:
        pred, true = pred[mask], true[mask]
    ade, fde = ade_fde(pred, true)
    d = np.linalg.norm(pred - true, axis=-1)
    return {
        "n": int(len(true)),
        "ade_m": round(ade, 4),
        "fde_m": round(fde, 4),
        "ade_median_m": round(float(np.median(d.mean(axis=1))), 4),
        "fde_p90_m": round(float(np.percentile(d[:, -1], 90)), 4),
        "per_step_displacement_m": [round(float(x), 4) for x in d.mean(axis=0)],
        "per_step_time_s": [round(DT * (k + 1), 2) for k in range(d.shape[1])],
        "l2": nuscenes_l2(pred, true),
    }


def action_report(actions: np.ndarray, residual: np.ndarray | None = None) -> dict:
    a, k = actions[..., 0], actions[..., 1]
    rep = {
        "accel_mps2": {
            "mean": round(float(a.mean()), 4), "std": round(float(a.std()), 4),
            "p01": round(float(np.percentile(a, 1)), 4),
            "p99": round(float(np.percentile(a, 99)), 4),
            "saturated_frac": round(float((np.abs(a) > 0.99 * ACC_MAX).mean()), 5),
        },
        "curvature_1pm": {
            "mean": round(float(k.mean()), 6), "std": round(float(k.std()), 6),
            "p01": round(float(np.percentile(k, 1)), 6),
            "p99": round(float(np.percentile(k, 99)), 6),
            "saturated_frac": round(float((np.abs(k) > 0.99 * KAPPA_MAX).mean()), 5),
        },
        "mean_abs_jerk_mps3": round(float(np.abs(np.diff(a, axis=1)).mean() / DT), 4),
    }
    if residual is not None:
        rmag = np.linalg.norm(residual[:, 1:], axis=-1)
        rep["residual"] = {
            "mean_abs_m": round(float(rmag.mean()), 4),
            "final_step_mean_abs_m": round(float(rmag[:, -1].mean()), 4),
            "note": "displacement added AFTER the kinematic rollout; if this approaches "
                    "the ADE, the physics is no longer doing the work",
        }
    return rep


def cosine(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = a / np.maximum(np.linalg.norm(a, axis=-1, keepdims=True), 1e-8)
    b = b / np.maximum(np.linalg.norm(b, axis=-1, keepdims=True), 1e-8)
    return (a * b).sum(-1)


def text_fidelity_report(pred_emb, true_emb, ceiling_emb, mean_emb) -> dict:
    """Text cosine metrics. scaled_score: 1 = PCA ceiling, 0 = mean baseline."""
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


_STEER = (("slightly steering to the left", "slight_left"),
          ("slightly steering to the right", "slight_right"),
          ("steering to the left", "left"), ("steering to the right", "right"),
          ("going straight", "straight"))
_SPEED = (("not moving", "stopped"), ("very fast", "very_fast"),
          ("driving fast", "fast"), ("normal speed", "normal"),
          ("driving slowly", "slow"), ("very slowly", "very_slow"))


def parse_behavior(text: str) -> tuple[str | None, str | None]:
    t = text.lower()
    steer = next((lab for kw, lab in _STEER if kw in t), None)
    speed = next((lab for kw, lab in _SPEED if kw in t), None)
    return steer, speed


def retrieval_report(pred_emb, true_texts, true_behaviors, bank_texts, bank_emb,
                     bank_behaviors) -> dict:
    """Decode the generated embedding by nearest neighbour in the training caption bank.

    Also returns DriveLM behavior accuracy (Acc / Speed / Steer) of the retrieved
    caption.
    """
    bank_n = bank_emb / np.maximum(np.linalg.norm(bank_emb, axis=1, keepdims=True), 1e-8)
    pred_n = pred_emb / np.maximum(np.linalg.norm(pred_emb, axis=1, keepdims=True), 1e-8)
    nearest = (pred_n @ bank_n.T).argmax(axis=1)
    retrieved = [bank_texts[i] for i in nearest]
    bank_set = set(bank_texts)
    covered = np.array([t in bank_set for t in true_texts])
    correct = np.array([r == t for r, t in zip(retrieved, true_texts)])

    steer_ok, speed_ok, both_ok = [], [], []
    for i, gt in zip(nearest, true_behaviors):
        ps, pv = parse_behavior(bank_behaviors[i])
        gs, gv = parse_behavior(gt)
        steer_ok.append(ps == gs)
        speed_ok.append(pv == gv)
        both_ok.append(ps == gs and pv == gv)
    return {
        "bank_size": len(bank_texts),
        "retrieval_accuracy": round(float(correct.mean()), 4),
        "retrieval_accuracy_covered": (round(float(correct[covered].mean()), 4)
                                       if covered.any() else None),
        "bank_coverage": round(float(covered.mean()), 4),
        "behavior_accuracy": round(float(np.mean(both_ok)), 4),
        "behavior_speed_accuracy": round(float(np.mean(speed_ok)), 4),
        "behavior_steer_accuracy": round(float(np.mean(steer_ok)), 4),
    }


def behavior_majority_baseline(train_behaviors, test_behaviors) -> dict:
    """Baseline: always predict the most common training behavior."""
    from collections import Counter
    maj = Counter(train_behaviors).most_common(1)[0][0]
    ms, mv = parse_behavior(maj)
    parsed = [parse_behavior(b) for b in test_behaviors]
    return {
        "majority_behavior": maj,
        "behavior_accuracy": round(float(np.mean([p == (ms, mv) for p in parsed])), 4),
        "behavior_speed_accuracy": round(float(np.mean([p[1] == mv for p in parsed])), 4),
        "behavior_steer_accuracy": round(float(np.mean([p[0] == ms for p in parsed])), 4),
    }
