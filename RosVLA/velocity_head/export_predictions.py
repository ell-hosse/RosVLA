"""Run the saved per-fold models on their held-out windows and save predictions.

    python export_predictions.py    # writes paper_figures/test_predictions.npz

Uses the same evaluation path as run_cv.py and checks ADE/FDE against
runs/fold_*/metrics.json.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from config import Config                                   # noqa: E402
from data import load_samples, scenarios_of, tc             # noqa: E402
from encoders import image_features                         # noqa: E402
from model import VelocityVLA                               # noqa: E402

if sys.platform != "win32":       # checkpoints were saved on Windows
    import pathlib
    pathlib.WindowsPath = pathlib.PureWindowsPath

ACTIONS = ("FORWARD", "RIGHT_TURN", "LEFT_TURN", "STOP", "REVERSE_LEFT")


SUMMARY = json.loads((HERE / "runs" / "summary.json").read_text())


def main():
    cfg = Config(calculator_dir=HERE.parent / "trajectory_predictor" / "trajectory_predictors")
    samples = load_samples(cfg)
    img_all = image_features(samples, cfg)
    tel_all = np.stack([s.tel for s in samples])
    wp_all = np.stack([s.wp for s in samples])
    dt_all = np.stack([s.dt for s in samples])
    warm_all = np.stack([s.warm for s in samples])
    chunk_all = np.stack([s.chunk for s in samples]).reshape(len(samples), cfg.horizon, 2)
    scen_all = np.array([s.scenario for s in samples])

    out = {k: [] for k in ("scenario", "frame", "fold", "pred", "gt", "ade", "fde",
                           "pred_action", "true_action", "pred_cmd", "true_cmd")}
    for fold_i, held in enumerate(scenarios_of(samples)):
        te = np.where(scen_all == held)[0]
        ck = torch.load(HERE / "runs" / f"fold_{fold_i:02d}_{held}" / "model.pt",
                        weights_only=False)
        plant = tc.PlantModel.from_dict(ck["stage2"])
        model = VelocityVLA(img_all.shape[1], tel_all.shape[1], ck["latent_dim"], cfg)
        model.load_state_dict(ck["model_state"])
        model.eval()

        img = torch.tensor(((img_all[te] - ck["img_mean"]) / ck["img_std"]).astype(np.float32))
        tel = torch.tensor(((tel_all[te] - ck["tel_mean"]) / ck["tel_std"]).astype(np.float32))
        gen = torch.Generator().manual_seed(cfg.seed + fold_i)
        _, v, w, logits = model.predict(img, tel, cfg.flow_steps, gen)
        cmd = np.stack([v.numpy() * ck["v_std"] + ck["v_mean"],
                        w.numpy() * ck["w_std"] + ck["w_mean"]], axis=2)
        pred = np.stack([plant(c, wm, d) for c, wm, d in zip(cmd, warm_all[te], dt_all[te])])
        gt = wp_all[te]
        err = np.linalg.norm(pred - gt, axis=-1)

        ref = SUMMARY["folds"][fold_i]
        print(f"fold {fold_i} {held:<20s} ADE {err.mean():.4f} (stored {ref['ade']:.4f})  "
              f"FDE {err[:, -1].mean():.4f} (stored {ref['fde']:.4f})")

        for j, i in enumerate(te):
            out["scenario"].append(held); out["frame"].append(samples[i].frame)
            out["fold"].append(fold_i); out["pred"].append(pred[j]); out["gt"].append(gt[j])
            out["ade"].append(err[j].mean()); out["fde"].append(err[j, -1])
            out["pred_action"].append(ACTIONS[int(logits[j].argmax())])
            out["true_action"].append(ACTIONS[samples[i].action])
            out["pred_cmd"].append(cmd[j]); out["true_cmd"].append(chunk_all[i])

    dst = HERE / "paper_figures"
    dst.mkdir(exist_ok=True)
    np.savez(dst / "test_predictions.npz", **{k: np.array(v) for k, v in out.items()})
    a = np.array(out["ade"])
    print(f"\n{len(a)} test windows | pooled ADE {a.mean():.4f} | saved -> {dst/'test_predictions.npz'}")


if __name__ == "__main__":
    main()
