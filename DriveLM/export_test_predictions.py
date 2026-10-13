"""Run the saved DriveLM models (seed 0, caption) on the test split and save
per-row predictions for the direct head (D1) and the world-model head (B).

    python export_test_predictions.py      # writes figures/drivelm_test_predictions.npz
"""
from __future__ import annotations

import json
import pathlib
import sys
from pathlib import Path

import numpy as np
import torch

if sys.platform != "win32":                  # checkpoints were saved on Windows
    pathlib.WindowsPath = pathlib.PureWindowsPath

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from config import N_FUTURE, Config                                         # noqa: E402
from data import load_splits                                                 # noqa: E402
from dynamics import measured_kappa, prior_actions                          # noqa: E402
from model import DirectVLA, WorldModelVLA                                   # noqa: E402
from train_wm import pre_tanh                                                # noqa: E402

t = lambda x: torch.as_tensor(np.asarray(x), dtype=torch.float32)


def run(family, ablation):
    cfg = Config(family=family, ablation=ablation, text_field="caption")
    cfg.seed = 0
    sp = load_splits(cfg, verbose=False)[0]["test"]
    rd = HERE / "runs" / family / ablation / "caption" / "seed_0"
    ck = torch.load(rd / "model.pt", map_location="cpu", weights_only=False)
    img = t((sp.img_feat - ck["img_mean"]) / ck["img_std"])
    hist = t((sp.hist - ck["hist_mean"]) / ck["hist_std"])
    ld = ck["pca_basis"].shape[0]
    gen = torch.Generator().manual_seed(cfg.seed)
    if family == "direct":
        m = DirectVLA(cfg, img.shape[1], sp.hist.shape[2], ld)
        m.load_state_dict(ck["model_state"]); m.eval()
        _, th = m.predict(img, hist, None, cfg.flow_steps, gen)
        pred = (th.numpy() * ck["wp_std"] + ck["wp_mean"]).reshape(-1, N_FUTURE, 2)
    else:
        m = WorldModelVLA(cfg, img.shape[1], sp.hist.shape[2], ld)
        m.load_state_dict(ck["model_state"]); m.eval()
        k0 = measured_kappa(sp.v0, sp.yaw_rate0)
        prior = t(pre_tanh(prior_actions(ck["prior"], sp.v0, sp.a0, k0)))
        _, tr7, _, _ = m.predict(img, hist, None, t(sp.v0), prior, cfg.flow_steps, gen)
        pred = tr7.numpy()[:, 1:]
    err = np.linalg.norm(pred - sp.traj, axis=-1)
    rep = json.loads((rd / "report.json").read_text())["test"]
    print(f"{family}/{ablation}: n={len(pred)} ADE {err.mean():.4f} (report {rep['ade_m']})")
    return sp, pred, err


def main():
    sp, pd, ed = run("direct", "d1_flow_generated")
    _, pw, ew = run("wm", "b_vla_diffusion")
    out = HERE / "figures"; out.mkdir(exist_ok=True)
    d = np.load(HERE / "cache" / "dataset.npz", allow_pickle=True)
    img_of = dict(zip(d["sample_token"], d["img"]))
    np.savez(out / "drivelm_test_predictions.npz",
             gt=sp.traj, pred_direct=pd, pred_wm=pw,
             ade_direct=ed.mean(1), ade_wm=ew.mean(1), fde_direct=ed[:, -1], fde_wm=ew[:, -1],
             sample_token=sp.sample_token, scene=sp.scene_name,
             img=np.array([img_of[s] for s in sp.sample_token]),
             caption=sp.cap_text.astype(str), behavior=sp.behavior.astype(str), v0=sp.v0)


if __name__ == "__main__":
    main()
