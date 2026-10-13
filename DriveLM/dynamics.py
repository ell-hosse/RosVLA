"""World model: differentiable kinematic rollout, its inverse, and analytic baselines.

Same integrator as CoVLA/world_model_head/dynamics.py with 6 steps of
dt = 0.5 s (the keyframe spacing). State (x, y, psi, v) in the ego frame of
the keyframe, x forward, y left. For k = 1..6:

    v_k     = clamp(v_{k-1} + a_k * dt, V_MIN, V_MAX)
    dpsi_k  = v_k * kappa_k * dt
    x_k     = x_{k-1} + v_k * dt * cos(psi_{k-1} + dpsi_k / 2)
    y_k     = y_{k-1} + v_k * dt * sin(psi_{k-1} + dpsi_k / 2)
    psi_k   = psi_{k-1} + dpsi_k

With the midpoint heading, inverse_dynamics recovers controls that reproduce
the ground-truth path exactly (checked by self_test).

Baselines / priors (no parameters):

    cv    hold speed, drive straight
    ca    hold the measured acceleration, drive straight
    ctrv  hold speed and curvature
    ctra  hold both

Curvature priors use the measured yaw rate, kappa0 = yaw_rate / v. The
steering-angle map is still fitted on train and reported.
"""
from __future__ import annotations

import numpy as np
import torch

from config import ACC_MAX, DT, KAPPA_MAX, N_STEPS, V_MAX, V_MIN

PRIOR_NAMES = ("cv", "ca", "ctrv", "ctra")


def rollout(v0: torch.Tensor, actions: torch.Tensor, dt: float = DT) -> torch.Tensor:
    """(B,) speed + (B, 6, 2) actions -> (B, 7, 2) metres; waypoint 0 = origin."""
    if actions.shape[-2] != N_STEPS:
        raise ValueError(f"expected {N_STEPS} action steps, got {actions.shape[-2]}")
    zero = torch.zeros_like(v0)
    x, y, psi, v = zero, zero, zero, v0
    xs, ys = [x], [y]
    for k in range(N_STEPS):
        a, kappa = actions[:, k, 0], actions[:, k, 1]
        v = torch.clamp(v + a * dt, min=V_MIN, max=V_MAX)
        dpsi = v * kappa * dt
        psi_mid = psi + 0.5 * dpsi
        x = x + v * dt * torch.cos(psi_mid)
        y = y + v * dt * torch.sin(psi_mid)
        psi = psi + dpsi
        xs.append(x)
        ys.append(y)
    return torch.stack([torch.stack(xs, dim=1), torch.stack(ys, dim=1)], dim=-1)


def rollout_np(v0: np.ndarray, actions: np.ndarray, dt: float = DT) -> np.ndarray:
    with torch.no_grad():
        out = rollout(torch.as_tensor(v0, dtype=torch.float64),
                      torch.as_tensor(actions, dtype=torch.float64), dt)
    return out.numpy()


def _wrap(x: np.ndarray) -> np.ndarray:
    return (x + np.pi) % (2 * np.pi) - np.pi


def inverse_dynamics(traj7: np.ndarray, v0: np.ndarray, dt: float = DT, eps: float = 1e-2):
    """(N, 7, 2) path incl. origin + (N,) v0 -> ((N, 6, 2) actions, (N,) moving flag).

    Actions are not clipped, so rollout(inverse(p)) == p.
    """
    traj7 = np.asarray(traj7, dtype=np.float64)
    v0 = np.asarray(v0, dtype=np.float64)
    n = len(traj7)
    d = traj7[:, 1:] - traj7[:, :-1]
    seg = np.linalg.norm(d, axis=-1)
    moving = seg > eps
    h = np.arctan2(d[..., 1], d[..., 0])

    actions = np.zeros((n, N_STEPS, 2), dtype=np.float64)
    psi = np.zeros(n)
    v_prev = v0.copy()
    for k in range(N_STEPS):
        v_k = seg[:, k] / dt
        a_k = (v_k - v_prev) / dt
        h_k = np.where(moving[:, k], h[:, k], psi)
        dpsi = 2.0 * _wrap(h_k - psi)
        kappa_k = np.where(moving[:, k], dpsi / np.maximum(seg[:, k], eps), 0.0)
        actions[:, k, 0] = a_k
        actions[:, k, 1] = kappa_k
        psi = psi + dpsi
        v_prev = v_k
    return actions, moving.all(axis=1)


def clip_actions(actions: np.ndarray) -> np.ndarray:
    out = actions.copy()
    out[..., 0] = np.clip(out[..., 0], -ACC_MAX, ACC_MAX)
    out[..., 1] = np.clip(out[..., 1], -KAPPA_MAX, KAPPA_MAX)
    return out


def measured_kappa(v0: np.ndarray, yaw_rate0: np.ndarray, v_min: float = 1.0) -> np.ndarray:
    """kappa = yaw_rate / v; 0 below v_min where it is dominated by noise."""
    v0 = np.asarray(v0, dtype=np.float64)
    k = np.where(v0 > v_min, np.asarray(yaw_rate0, dtype=np.float64) / np.maximum(v0, v_min), 0.0)
    return np.clip(k, -KAPPA_MAX, KAPPA_MAX)


def identify_steering_map(steering_deg, v0, kappa1, v_min: float = 3.0,
                          n_grid: int = 81, k_max: float = 0.02) -> dict:
    """Fit kappa = deg2rad(steer) / (L_eff * (1 + K_us v^2)) on training rows.

    Reported only (the priors use the yaw rate). The nuScenes steering signal is
    the steering-wheel angle, so L_eff includes the steering ratio.
    """
    delta = np.deg2rad(np.asarray(steering_deg, dtype=np.float64))
    v0 = np.asarray(v0, dtype=np.float64)
    kappa1 = np.asarray(kappa1, dtype=np.float64)
    m = (v0 > v_min) & np.isfinite(kappa1) & (np.abs(kappa1) < KAPPA_MAX)
    if m.sum() < 50:
        return {"fitted": False, "n_fit": int(m.sum())}
    best = None
    for K in np.linspace(0.0, k_max, n_grid):
        f = delta[m] / (1.0 + K * v0[m] ** 2)
        ff = float(f @ f)
        if ff < 1e-12:
            continue
        inv_L = float(f @ kappa1[m]) / ff
        resid = float(np.mean((kappa1[m] - inv_L * f) ** 2))
        if best is None or resid < best[0]:
            best = (resid, float(K), inv_L)
    resid, K_us, inv_L = best
    pred = inv_L * delta[m] / (1.0 + K_us * v0[m] ** 2)
    ss_res = float(np.sum((kappa1[m] - pred) ** 2))
    ss_tot = float(np.sum((kappa1[m] - kappa1[m].mean()) ** 2)) or 1.0
    return {"fitted": True,
            "equation": "kappa = deg2rad(steer_deg) / (L_eff * (1 + K_us * v^2))",
            "L_eff_m": round(1.0 / inv_L, 4) if abs(inv_L) > 1e-9 else None,
            "K_us": round(K_us, 5), "n_fit": int(m.sum()),
            "r2": round(1.0 - ss_res / ss_tot, 4)}


def yaw_rate_kappa_r2(v0, yaw_rate0, kappa1, v_min: float = 3.0) -> float:
    """How well the measured kappa0 predicts the first-step curvature (train)."""
    k0 = measured_kappa(v0, yaw_rate0)
    m = (np.asarray(v0) > v_min) & (np.abs(kappa1) < KAPPA_MAX)
    if m.sum() < 50:
        return float("nan")
    ss_res = float(np.sum((kappa1[m] - k0[m]) ** 2))
    ss_tot = float(np.sum((kappa1[m] - kappa1[m].mean()) ** 2)) or 1.0
    return round(1.0 - ss_res / ss_tot, 4)


def prior_actions(name: str, v0, a0, kappa0) -> np.ndarray:
    n = len(v0)
    z = np.zeros((n, N_STEPS), dtype=np.float64)
    A = np.repeat(np.asarray(a0, dtype=np.float64)[:, None], N_STEPS, axis=1)
    K = np.repeat(np.asarray(kappa0, dtype=np.float64)[:, None], N_STEPS, axis=1)
    acc = {"cv": z, "ca": A, "ctrv": z, "ctra": A, "none": z}[name]
    kap = {"cv": z, "ca": z, "ctrv": K, "ctra": K, "none": z}[name]
    return clip_actions(np.stack([acc, kap], axis=-1))


def baseline_report(traj6: np.ndarray, v0, a0, kappa0) -> dict:
    from metrics import trajectory_report
    out = {}
    for name in PRIOR_NAMES:
        pred = rollout_np(v0, prior_actions(name, v0, a0, kappa0))[:, 1:]
        r = trajectory_report(pred, traj6)
        out[name] = {"params": 0, "ade_m": r["ade_m"], "fde_m": r["fde_m"], "l2": r["l2"]}
    return out


def self_test(n: int = 512, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    v0 = rng.uniform(2.0, 15.0, size=n)
    actions = np.stack([rng.uniform(-1.5, 1.5, size=(n, N_STEPS)),
                        rng.uniform(-0.05, 0.05, size=(n, N_STEPS))], axis=-1)
    traj = rollout_np(v0, actions)
    rec, valid = inverse_dynamics(traj, v0)
    traj2 = rollout_np(v0, rec)
    pos_err = float(np.abs(traj2 - traj).max())
    act_err = float(np.abs(rec - actions).max())
    return {"n": n, "max_position_roundtrip_error_m": pos_err,
            "max_action_roundtrip_error": act_err, "all_rows_valid": bool(valid.all()),
            "passed": pos_err < 1e-4 and act_err < 1e-3}


if __name__ == "__main__":
    import json
    print(json.dumps(self_test(), indent=2))
