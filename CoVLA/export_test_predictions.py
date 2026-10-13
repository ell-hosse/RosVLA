"""Run the saved CoVLA models on the test split and save per-row predictions.

    python export_test_predictions.py direct   # vla_trajecotry_head/runs/model.pt
    python export_test_predictions.py wm       # world_model_head/runs_b_vla_diffusion/model.pt

Writes figures/covla_<mode>_test_predictions.npz and checks ADE against the
model's report.json.
"""
from __future__ import annotations

import json
import pathlib
import sys
from pathlib import Path, PureWindowsPath

import numpy as np
import torch

if sys.platform != "win32":                  # checkpoints were saved on Windows
    pathlib.WindowsPath = pathlib.PureWindowsPath

ROOT = Path(__file__).resolve().parent
MODE = sys.argv[1]
PKG = ROOT / ("vla_trajecotry_head" if MODE == "direct" else "world_model_head")
sys.path.insert(0, str(PKG))

from config import Config                    # noqa: E402

PRE = Path(sys.argv[2]) if len(sys.argv) > 2 else None


def test_rows(cfg):
    """Rebuild load_split('test') from the history and image-feature caches only."""
    pre = PRE or cfg.preprocessed_dir
    d = np.load(pre / "cache_2hz" / "test.npz", allow_pickle=True)
    paths = np.load(pre / "cache_2hz" / "paths_test.npz", allow_pickle=True)["paths"]
    n = len(d["frame_idx"])
    rng = np.random.default_rng(cfg.seed)
    idx = np.sort(rng.choice(n, size=cfg.max_test, replace=False)) if cfg.max_test < n else np.arange(n)

    h = np.load(PKG / "cache" / "history_test.npz")
    assert np.array_equal(h["frame_idx"], d["frame_idx"][idx])
    f = np.load(PKG / "cache" / "imgfeat_test.npz", allow_pickle=True)
    have = {PureWindowsPath(p).as_posix().split("covla_preprocessed/")[-1] for p in f["paths"]}
    rel = [PureWindowsPath(str(p)).as_posix() for p in paths[idx]]
    keep = h["valid"] & np.array([r in have for r in rel])
    assert keep.sum() == len(f["feat"])

    traj = d["traj"][idx][keep].astype(np.float32)
    traj = traj - traj[:, :1]
    return dict(
        img=f["feat"], hist=h["hist"][keep], traj=traj,
        state=d["state"][idx][keep][:, :3].astype(np.float32),
        rel=np.array(rel)[keep],
        video=d["video_names"][d["video_code"][idx][keep]],
        frame=d["frame_idx"][idx][keep],
    )


def main():
    cfg = Config(ablation="b_vla_diffusion") if MODE == "wm" else Config()
    S = test_rows(cfg)
    run = PKG / ("runs" if MODE == "direct" else "runs_b_vla_diffusion")
    ck = torch.load(run / "model.pt", map_location="cpu", weights_only=False)
    t = lambda x: torch.as_tensor(np.asarray(x), dtype=torch.float32)
    img = t((S["img"] - ck["img_mean"]) / ck["img_std"])
    hist = t((S["hist"] - ck["hist_mean"]) / ck["hist_std"])
    latent_dim = ck["pca_basis"].shape[0]
    gen = torch.Generator().manual_seed(cfg.seed)

    if MODE == "direct":
        from model import CoVLAFlowModel
        m = CoVLAFlowModel(img.shape[1], hist.shape[1], hist.shape[2], latent_dim, cfg)
        m.load_state_dict(ck["model_state"]); m.eval()
        _, tr = m.predict(img, hist, cfg.flow_steps, gen)
        pred = (tr.numpy() * ck["wp_std"] + ck["wp_mean"]).reshape(-1, 20, 2)
        stored = json.loads((run / "report.json").read_text())["test"]["ade"]
    else:
        from dynamics import apply_steering_map, prior_actions
        from train import pre_tanh
        from model import WorldModelVLA
        v0, a0, st = S["state"][:, 0], S["state"][:, 1], S["state"][:, 2]
        kappa0 = apply_steering_map(st, v0, ck["steering_map"])
        prior = t(pre_tanh(prior_actions(ck["prior"], v0, a0, kappa0)))
        m = WorldModelVLA("b_vla_diffusion", img.shape[1], hist.shape[2], latent_dim, cfg)
        m.load_state_dict(ck["model_state"]); m.eval()
        _, tr, _, _ = m.predict(img, hist, t(v0), prior, steps=cfg.flow_steps, generator=gen)
        pred = tr.numpy()
        stored = json.loads((run / "report.json").read_text())["test"]["ade_m"]

    err = np.linalg.norm(pred - S["traj"], axis=-1)
    print(f"{MODE}: n={len(pred)}  ADE {err.mean():.4f} (report {stored})  FDE {err[:, -1].mean():.4f}")
    out = ROOT / "figures"; out.mkdir(exist_ok=True)
    np.savez(out / f"covla_{MODE}_test_predictions.npz", pred=pred, gt=S["traj"],
             ade=err.mean(1), fde=err[:, -1], rel=S["rel"], video=S["video"],
             frame=S["frame"], v0=S["state"][:, 0])


if __name__ == "__main__":
    main()
