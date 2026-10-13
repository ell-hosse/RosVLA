"""Run the saved per-fold models on their held-out scenario and save predictions.

    python export_predictions.py   # writes runs/test_predictions.npz
"""
from __future__ import annotations

import json
import pathlib
import sys
from pathlib import Path

import numpy as np
import torch

if sys.platform != "win32":
    pathlib.WindowsPath = pathlib.PureWindowsPath

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from config import Config                          # noqa: E402
from data import load_samples, scenarios_of        # noqa: E402
from encoders import image_features                # noqa: E402
from model import VLAModel                         # noqa: E402


def main():
    cfg = Config()
    S = load_samples(cfg)
    img_all = image_features(S, cfg)
    tel_all = np.stack([s.telemetry for s in S])
    wp_all = np.stack([s.waypoints for s in S])
    scen = np.array([s.scenario for s in S])
    summ = json.loads((HERE / "runs" / "summary.json").read_text())["folds"]
    out = {"scenario": [], "frame": [], "pred": [], "gt": [], "ade": []}
    for fi, held in enumerate(scenarios_of(S)):
        te = np.where(scen == held)[0]
        ck = torch.load(HERE / "runs" / f"fold_{fi:02d}_{held}" / "model.pt",
                        map_location="cpu", weights_only=False)
        m = VLAModel(img_all.shape[1], tel_all.shape[1], ck["latent_dim"], cfg)
        m.load_state_dict(ck["model_state"]); m.eval()
        img = torch.tensor(((img_all[te] - ck["img_mean"]) / ck["img_std"]).astype(np.float32))
        tel = torch.tensor(((tel_all[te] - ck["tel_mean"]) / ck["tel_std"]).astype(np.float32))
        _, th, _ = m.predict(img, tel, cfg.flow_steps, torch.Generator().manual_seed(cfg.seed + fi))
        pred = (th.numpy() * ck["wp_std"] + ck["wp_mean"]).reshape(len(te), cfg.n_waypoints, 2)
        err = np.linalg.norm(pred - wp_all[te], axis=-1)
        print(f"fold {fi} {held:<20s} ADE {err.mean():.4f} (stored {summ[fi]['ade']})")
        for j, i in enumerate(te):
            out["scenario"].append(held); out["frame"].append(S[i].frame)
            out["pred"].append(pred[j]); out["gt"].append(wp_all[i]); out["ade"].append(err[j].mean())
    np.savez(HERE / "runs" / "test_predictions.npz", **{k: np.array(v) for k, v in out.items()})


if __name__ == "__main__":
    main()
