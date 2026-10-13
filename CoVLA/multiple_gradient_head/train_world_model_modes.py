#!/usr/bin/env python3
"""Gradient-routing modes for the CoVLA world-model VLA (ablation b_vla_diffusion).

    python train_world_model_modes.py                                   # baseline
    python train_world_model_modes.py --trunk split --grad-mode staged
    python train_world_model_modes.py --max-train 4000 --epochs 6       # quick check

World-model counterpart of train_modes.py. Data, inverse dynamics, steering
map, prior, rollout, action bounds and evaluate() are imported from
../world_model_head, so --trunk shared --grad-mode joint reproduces
train.py --ablation b_vla_diffusion (test ADE 0.9073 m).

Ablations a and c have no flow head and are not supported here.
Results go to runs_wm/<trunk>-<grad_mode>/report.json.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

HERE = Path(__file__).resolve().parent
WM = HERE.parent / "world_model_head"
sys.path.insert(0, str(HERE))
sys.path.insert(1, str(WM))

from config import ACC_MAX, DT, KAPPA_MAX, N_STEPS, N_WAYPOINTS, Config   # noqa: E402
from data import load_split                                               # noqa: E402
from dynamics import (apply_steering_map, baseline_report, clip_actions,  # noqa: E402
                      identify_steering_map, inverse_dynamics, prior_actions,
                      rollout_np, self_test)
from latent import TextLatent, pca_report                                 # noqa: E402
from model import parameter_report                                        # noqa: E402
from train import evaluate, pre_tanh, to_tensor                           # noqa: E402
from utils import Standardiser                                            # noqa: E402

from modes import add_mode_arguments, mode_from_args                      # noqa: E402
from wm_heads import build_model, shared_parameters                       # noqa: E402
from wm_trainer import train                                              # noqa: E402

ABLATION = "b_vla_diffusion"


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--ablation", default=ABLATION,
                    help="only b_vla_diffusion has a flow head; the others are refused")
    ap.add_argument("--preprocessed-dir", default=None)
    ap.add_argument("--meta-dir", default=None)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--patience", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--pca-dim", type=int, default=None)
    ap.add_argument("--max-train", type=int, default=None)
    ap.add_argument("--max-val", type=int, default=None)
    ap.add_argument("--max-test", type=int, default=None)
    ap.add_argument("--prior", default=None, choices=["none", "cv", "ca", "ctrv", "ctra"])
    ap.add_argument("--no-residual", action="store_true")
    ap.add_argument("--pretrain-epochs", type=int, default=None)
    ap.add_argument("--rebuild-history-cache", action="store_true")
    ap.add_argument("--rebuild-image-cache", action="store_true")
    ap.add_argument("--device", default=None, choices=["cuda", "cpu"])
    ap.add_argument("--out-dir", default=None, help="override runs_wm/<mode>/")
    add_mode_arguments(ap)
    return ap


def config_from_args(args, out_dir: Path) -> Config:
    cfg = Config(ablation=args.ablation, out_dir=out_dir)
    for attr, val in (
        ("epochs", args.epochs), ("early_stop_patience", args.patience),
        ("batch_size", args.batch_size), ("lr", args.lr), ("pca_dim", args.pca_dim),
        ("max_train", args.max_train), ("max_val", args.max_val),
        ("max_test", args.max_test), ("prior", args.prior),
        ("pretrain_epochs", args.pretrain_epochs), ("device", args.device),
    ):
        if val is not None:
            setattr(cfg, attr, val)
    if args.no_residual:
        cfg.use_residual = False
    if args.preprocessed_dir:
        cfg.preprocessed_dir = Path(args.preprocessed_dir)
    if args.meta_dir:
        cfg.meta_dir = Path(args.meta_dir)
    return cfg


def run(args, verbose: bool = True) -> dict:
    mode = mode_from_args(args)
    if args.ablation != ABLATION:
        raise SystemExit(
            f"ablation {args.ablation!r} has no flow head, so there is nothing to "
            f"route between two objectives. Only {ABLATION!r} is supported here; "
            f"run ../world_model_head/train.py for the others."
        )

    out_dir = Path(args.out_dir) if args.out_dir else HERE / "runs_wm" / mode.name
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg = config_from_args(args, out_dir)

    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    device = cfg.device if (cfg.device == "cpu" or torch.cuda.is_available()) else "cpu"

    print("#" * 72)
    print(f"# WORLD-MODEL VLA   ablation={cfg.ablation}   MODE={mode.name}")
    print(f"#   {mode.describe(cfg.p_sampled_z)}")
    print(f"#   prior={cfg.prior}  residual={cfg.use_residual}  device={device}")
    print("#" * 72)

    # data
    splits = {n: load_split(cfg, n, force_history=args.rebuild_history_cache,
                            force_images=args.rebuild_image_cache)
              for n in ("train", "val", "test")}
    train_s, val_s, test_s = splits["train"], splits["val"], splits["test"]
    print(f"\nsizes: train={len(train_s)}  val={len(val_s)}  test={len(test_s)}")
    if train_s.approximate_history:
        print("  [!] history windows are the 2 Hz fallback, not true 20 Hz.")

    # fit standardisers and PCA on train only
    hist_std = Standardiser(train_s.hist)
    img_std = Standardiser(train_s.img_feat)
    latent = TextLatent(train_s.cap_emb, cfg.pca_dim)
    mean_emb = train_s.cap_emb.mean(axis=0)
    mean_emb = mean_emb / max(np.linalg.norm(mean_emb), 1e-8)
    print(f"PCA text latent: dim={latent.dim}  explained={latent.explained:.4f}")

    gt_actions, _ = {}, {}
    for name, sp in splits.items():
        gt_actions[name], _v = inverse_dynamics(sp.traj, sp.v0)

    smap = identify_steering_map(train_s.steer0, train_s.v0, gt_actions["train"][:, 0, 1])
    if smap["fitted"]:
        print(f"identified steering map: L_eff={smap['L_eff_m']:.3f} m  "
              f"K_us={smap['K_us']:.4f}  R^2={smap['r2']:.4f}")

    baselines, priors_pre, oracle = {}, {}, {}
    for name, sp in splits.items():
        kappa0 = apply_steering_map(sp.steer0, sp.v0, smap)
        baselines[name] = baseline_report(sp.traj, sp.v0, sp.a0, kappa0)
        priors_pre[name] = pre_tanh(prior_actions(cfg.prior, sp.v0, sp.a0, kappa0))
        exact = rollout_np(sp.v0, gt_actions[name])
        clipped = rollout_np(sp.v0, clip_actions(gt_actions[name]))
        oracle[name] = {
            "exact": {"ade_m": round(float(np.linalg.norm(exact - sp.traj, axis=-1).mean()), 6)},
            "clipped_to_envelope": {
                "ade_m": round(float(np.linalg.norm(clipped - sp.traj, axis=-1).mean()), 4),
                "fde_m": round(float(np.linalg.norm(clipped - sp.traj, axis=-1)[:, -1].mean()), 4),
                "note": f"best reachable ADE for ANY model bounded by |a|<={ACC_MAX} "
                        f"m/s^2, |kappa|<={KAPPA_MAX} 1/m",
            },
        }
    print(f"test CA baseline ADE {baselines['test']['ca']['ade_m']:.4f} m   "
          f"action-envelope ceiling {oracle['test']['clipped_to_envelope']['ade_m']:.4f} m")

    # pack
    def pack(name):
        sp = splits[name]
        img = to_tensor(img_std(sp.img_feat), device)
        return {
            "img": img, "img_t": img,
            "hist": to_tensor(hist_std(sp.hist), device),
            "v0": to_tensor(sp.v0, device),
            "prior_pre": to_tensor(priors_pre[name], device),
            "traj": to_tensor(sp.traj, device),
            "gt_act": to_tensor(clip_actions(gt_actions[name]), device),
            "z": to_tensor(latent.encode(sp.cap_emb), device),
        }

    P = {name: pack(name) for name in splits}
    loader = DataLoader(
        TensorDataset(P["train"]["img_t"], P["train"]["hist"], P["train"]["v0"],
                      P["train"]["prior_pre"], P["train"]["traj"],
                      P["train"]["gt_act"], P["train"]["z"]),
        batch_size=cfg.batch_size, shuffle=True,
    )

    # model
    model = build_model(
        ablation=cfg.ablation, img_dim=train_s.img_feat.shape[1],
        history_feat_dim=train_s.hist.shape[2], latent_dim=latent.dim,
        cfg=cfg, mode=mode,
    ).to(device)
    size_info = parameter_report(model)
    print(f"model: {size_info['total_params']:,} trainable params; "
          f"by module: {size_info['by_module']}")

    act_scale = to_tensor(
        np.maximum(clip_actions(gt_actions["train"]).std(axis=(0, 1)), 1e-3), device
    )

    print("\n" + "=" * 72)
    print(f"TRAINING (max {cfg.epochs} epochs, patience={cfg.early_stop_patience}, "
          f"{cfg.pretrain_epochs} action-pretrain epochs)")
    print("=" * 72)
    train_log = train(cfg, model, loader, P["val"], val_s.traj, P["val"]["z"],
                      act_scale, device, mode, shared_parameters(model, mode),
                      verbose=verbose)

    # report
    print("\n" + "=" * 72); print("FINAL EVALUATION"); print("=" * 72)
    report = {
        "ablation": cfg.ablation,
        "mode": mode.to_dict(cfg.p_sampled_z),
        "model_size": size_info,
        "dynamics_self_test": self_test(),
        "identified_steering_map": smap,
        "prior": cfg.prior,
        "dt_s": DT, "n_steps": N_STEPS, "n_waypoints": N_WAYPOINTS,
        "horizon_s": round(N_STEPS * DT, 4),
        "approximate_history": bool(train_s.approximate_history),
        "n_rows": {k: len(v) for k, v in splits.items()},
        "baselines": baselines,
        "oracle_actions": oracle,
        "pca": pca_report(latent, {k: v.cap_emb for k, v in splits.items()}),
        **train_log,
    }

    for name in ("train", "val", "test"):
        r = evaluate(model, cfg, P[name], splits[name], latent, mean_emb, device)
        report[name] = r
        print(f"\n  [{name}]  n={r['n']}")
        print(f"    ADE {r['ade_m']:.4f} m   FDE {r['fde_m']:.4f} m")
        key = f"min_ade_k{cfg.n_samples_eval}"
        if key in r:
            print(f"    min-ADE(K={cfg.n_samples_eval}) {r[key]:.4f}")
        if "residual" in r["actions"]:
            print(f"    learned residual carries "
                  f"{r['actions']['residual']['mean_abs_m']:.4f} m of the "
                  f"{r['ade_m']:.4f} m error scale")
        if "text" in r:
            t = r["text"]
            print(f"    text cosine: model={t['cosine_model']:.4f}  "
                  f"ceiling={t['cosine_pca_ceiling']:.4f}  "
                  f"scaled={t['scaled_score']:.4f}")

    ckpt = {
        "model_state": model.state_dict(),
        "mode": mode.to_dict(cfg.p_sampled_z),
        "hist_mean": hist_std.mean, "hist_std": hist_std.std,
        "img_mean": img_std.mean, "img_std": img_std.std,
        "steering_map": smap, "prior": cfg.prior,
        "pca_mean": latent.mean, "pca_basis": latent.basis,
        "pca_scale": latent.scale, "mean_emb": mean_emb,
        "config": {k: str(v) for k, v in vars(cfg).items()},
    }
    torch.save(ckpt, out_dir / "model.pt")
    (out_dir / "report.json").write_text(json.dumps(report, indent=2, default=str))
    print(f"\nreport -> {out_dir / 'report.json'}")
    print(f"  test ADE {report['test']['ade_m']:.4f} m   "
          f"(published ablation-b baseline: 0.9073 m)")
    return report


def main():
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
