"""World model: differentiable kinematic rollout, its inverse, the steering->curvature
fit, and the analytic baselines.

Rollout
-------
State (x, y, psi, v) in the anchor frame (x forward, y lateral, psi = 0 at
t = 0). Actions are acceleration `a` and curvature `kappa`. For k = 1 .. 19,
dt = 0.15 s:

    v_k     = clamp(v_{k-1} + a_k * dt, V_MIN, V_MAX)
    dpsi_k  = v_k * kappa_k * dt
    x_k     = x_{k-1} + v_k * dt * cos(psi_{k-1} + dpsi_k / 2)
    y_k     = y_{k-1} + v_k * dt * sin(psi_{k-1} + dpsi_k / 2)
    psi_k   = psi_{k-1} + dpsi_k

This is the integrator of RosVLA/trajectory_predictor/trajectory_calculator.py
with curvature instead of yaw rate, which stays well-conditioned as v -> 0.

Inverse
-------
Recovers the (a, kappa) sequence that reproduces a path exactly. Segment
headings h_k = atan2(dy, dx) equal the midpoint heading, so

    dpsi_k  = 2 * wrap(h_k - psi_{k-1}),     psi_k = psi_{k-1} + dpsi_k
    v_k     = |p_k - p_{k-1}| / dt,          a_k = (v_k - v_{k-1}) / dt
    kappa_k = dpsi_k / (v_k * dt)

self_test() checks rollout(inverse_dynamics(p)) == p. The recovered actions
are used by --pretrain-epochs and by the oracle_actions report row.
"""
from __future__ import annotations

import numpy as np
import torch

from config import ACC_MAX, DT, KAPPA_MAX, N_STEPS, N_WAYPOINTS, V_MAX, V_MIN


def rollout(v0: torch.Tensor, actions: torch.Tensor, dt: float = DT) -> torch.Tensor:
    """(B,) initial speed + (B, N_STEPS, 2) actions (a, kappa) -> (B, N_WAYPOINTS, 2) metres.

    Waypoint 0 is the origin.
    """
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
    """NumPy version of rollout(), used by the baselines."""
    with torch.no_grad():
        out = rollout(
            torch.as_tensor(v0, dtype=torch.float64),
            torch.as_tensor(actions, dtype=torch.float64),
            dt,
        )
    return out.numpy()


def _wrap(x: np.ndarray) -> np.ndarray:
    return (x + np.pi) % (2 * np.pi) - np.pi


def inverse_dynamics(
    traj: np.ndarray, v0: np.ndarray, dt: float = DT, eps: float = 1e-3
) -> tuple[np.ndarray, np.ndarray]:
    """(N, 20, 2) path + (N,) initial speed -> ((N, 19, 2) actions, valid mask).

    valid is False for rows where the car is nearly stopped and heading /
    curvature are undefined. Actions are not clipped.
    """
    traj = np.asarray(traj, dtype=np.float64)
    v0 = np.asarray(v0, dtype=np.float64)
    n = len(traj)

    d = traj[:, 1:] - traj[:, :-1]                      # (N, 19, 2)
    seg = np.linalg.norm(d, axis=-1)                    # (N, 19)
    moving = seg > eps
    h = np.arctan2(d[..., 1], d[..., 0])                # (N, 19) segment headings

    actions = np.zeros((n, N_STEPS, 2), dtype=np.float64)
    psi = np.zeros(n)
    v_prev = v0.copy()
    for k in range(N_STEPS):
        v_k = seg[:, k] / dt
        a_k = (v_k - v_prev) / dt
        # no displacement: keep the previous heading
        h_k = np.where(moving[:, k], h[:, k], psi)
        dpsi = 2.0 * _wrap(h_k - psi)
        denom = np.maximum(seg[:, k], eps)
        kappa_k = np.where(moving[:, k], dpsi / denom, 0.0)
        actions[:, k, 0] = a_k
        actions[:, k, 1] = kappa_k
        psi = psi + dpsi
        v_prev = v_k

    valid = moving.all(axis=1)
    return actions, valid


def clip_actions(actions: np.ndarray) -> np.ndarray:
    out = actions.copy()
    out[..., 0] = np.clip(out[..., 0], -ACC_MAX, ACC_MAX)
    out[..., 1] = np.clip(out[..., 1], -KAPPA_MAX, KAPPA_MAX)
    return out


def identify_steering_map(
    steering_deg: np.ndarray, v0: np.ndarray, kappa1: np.ndarray,
    v_min: float = 3.0, n_grid: int = 81, k_max: float = 0.02,
) -> dict:
    """Fit the single-track understeer relation on training rows:

        kappa = delta_rad / (L_eff * (1 + K_us * v^2))

    L_eff combines wheelbase and steering ratio. Solved with a grid over K_us
    and closed-form least squares for 1/L_eff. Rows slower than `v_min` are
    excluded from the fit.
    """
    delta = np.deg2rad(np.asarray(steering_deg, dtype=np.float64))
    v0 = np.asarray(v0, dtype=np.float64)
    kappa1 = np.asarray(kappa1, dtype=np.float64)

    m = (v0 > v_min) & np.isfinite(kappa1) & (np.abs(kappa1) < KAPPA_MAX)
    if m.sum() < 100:
        return {"fitted": False, "inv_L_eff": 0.0, "K_us": 0.0, "n_fit": int(m.sum())}

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
    return {
        "fitted": True,
        "equation": "kappa = deg2rad(steeringAngleDeg) / (L_eff * (1 + K_us * vEgo^2))",
        "inv_L_eff": round(inv_L, 6),
        "L_eff_m": round(1.0 / inv_L, 4) if abs(inv_L) > 1e-9 else None,
        "K_us": round(K_us, 5),
        "n_fit": int(m.sum()),
        "fit_v_min_mps": v_min,
        "residual_mse": round(resid, 8),
        "r2": round(1.0 - ss_res / ss_tot, 4),
    }


def apply_steering_map(steering_deg: np.ndarray, v0: np.ndarray, smap: dict) -> np.ndarray:
    if not smap.get("fitted"):
        return np.zeros_like(np.asarray(v0, dtype=np.float64))
    delta = np.deg2rad(np.asarray(steering_deg, dtype=np.float64))
    v0 = np.asarray(v0, dtype=np.float64)
    kappa = smap["inv_L_eff"] * delta / (1.0 + smap["K_us"] * v0 ** 2)
    return np.clip(kappa, -KAPPA_MAX, KAPPA_MAX)


def prior_actions(
    name: str, v0: np.ndarray, a0: np.ndarray, kappa0: np.ndarray
) -> np.ndarray:
    """(N, 19, 2) constant action sequences for the analytic baselines / priors.

        cv    a = 0,  kappa = 0
        ca    a = a0, kappa = 0
        ctrv  a = 0,  kappa = kappa0
        ctra  a = a0, kappa = kappa0
        none  same as cv, named separately in reports
    """
    n = len(v0)
    z = np.zeros((n, N_STEPS), dtype=np.float64)
    A = {"cv": z, "ca": np.repeat(a0[:, None], N_STEPS, axis=1),
         "ctrv": z, "ctra": np.repeat(a0[:, None], N_STEPS, axis=1),
         "none": z}[name]
    K = {"cv": z, "ca": z, "ctrv": np.repeat(kappa0[:, None], N_STEPS, axis=1),
         "ctra": np.repeat(kappa0[:, None], N_STEPS, axis=1), "none": z}[name]
    return np.stack([A, K], axis=-1)


def baseline_report(
    traj_true: np.ndarray, v0: np.ndarray, a0: np.ndarray, kappa0: np.ndarray
) -> dict:
    """ADE/FDE of each analytic baseline through the same rollout."""
    out = {}
    for name in ("cv", "ca", "ctrv", "ctra"):
        pred = rollout_np(v0, prior_actions(name, v0, a0, kappa0))
        d = np.linalg.norm(pred - traj_true, axis=-1)
        out[name] = {
            "params": 0,
            "ade_m": round(float(d.mean()), 4),
            "fde_m": round(float(d[:, -1].mean()), 4),
        }
    return out


def self_test(n: int = 512, seed: int = 0) -> dict:
    """Check that rollout -> inverse_dynamics -> rollout is a round trip."""
    rng = np.random.default_rng(seed)
    v0 = rng.uniform(2.0, 28.0, size=n)
    actions = np.stack([
        rng.uniform(-2.0, 2.0, size=(n, N_STEPS)),
        rng.uniform(-0.02, 0.02, size=(n, N_STEPS)),
    ], axis=-1)
    traj = rollout_np(v0, actions)
    rec_actions, valid = inverse_dynamics(traj, v0)
    traj2 = rollout_np(v0, rec_actions)

    pos_err = float(np.abs(traj2 - traj).max())
    act_err = float(np.abs(rec_actions - actions).max())
    return {
        "n": n,
        "max_position_roundtrip_error_m": pos_err,
        "max_action_roundtrip_error": act_err,
        "all_rows_valid": bool(valid.all()),
        "passed": pos_err < 1e-4 and act_err < 1e-3,
    }


if __name__ == "__main__":
    import json
    print(json.dumps(self_test(), indent=2))
    print(f"\ndt = {DT} s, {N_STEPS} steps, horizon = {N_STEPS * DT:.2f} s, "
          f"{N_WAYPOINTS} waypoints (index 0 = origin)")
