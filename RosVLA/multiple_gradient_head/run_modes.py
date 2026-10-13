#!/usr/bin/env python3
"""Leave-one-scenario-out CV for the velocity head under different gradient-routing modes.

    python run_modes.py                                     # baseline (as ../velocity_head)
    python run_modes.py --trunk split --grad-mode staged
    python run_modes.py --grad-mode pcgrad
    python run_modes.py --folds 2 --epochs 20               # quick smoke run

Data, encoders, PCA, the stage-2 plant model and metrics are imported from
../velocity_head; only the training loop changes. Results go to
runs/<trunk>-<grad_mode>/.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
SIBLING = HERE.parent / "velocity_head"
sys.path.insert(0, str(HERE))
sys.path.insert(1, str(SIBLING))

from config import Config, find_calculator_dir                       # noqa: E402
from data import command_prototypes, describe, load_samples, scenarios_of  # noqa: E402
from encoders import image_features, text_matrix                     # noqa: E402
from latent import TextLatent                                        # noqa: E402
from metrics import (                                                # noqa: E402
    action_report, ade_fde, aggregate, min_ade_fde, text_report, velocity_report,
)
from model import parameter_report                                   # noqa: E402
from run_cv import Standardiser, load_fold_calculator, rollout       # noqa: E402

from heads import build_model                                        # noqa: E402
from modes import add_mode_arguments, mode_from_args                 # noqa: E402
from trainer import train_fold                                       # noqa: E402


def resolve_calculator_dir(explicit: str | None) -> Path:
    """Directory with the exported stage-2 fold models (../trajectory_predictor first)."""
    if explicit:
        return Path(explicit).expanduser().resolve()
    local = find_calculator_dir(HERE.parent / "trajectory_predictor")
    if local.is_dir() and any(local.glob("fold_*.json")):
        return local
    return Config().calculator_dir


def build_config(args) -> Config:
    cfg = Config()
    cfg.calculator_dir = resolve_calculator_dir(args.calculator_dir)
    if args.epochs:
        cfg.epochs = args.epochs
    if args.text_field:
        cfg.text_field = args.text_field
    if args.hist:
        cfg.hist = args.hist
    return cfg


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--text-field", type=str, default=None)
    ap.add_argument("--hist", type=int, default=None)
    ap.add_argument("--calculator-dir", type=str, default=None)
    ap.add_argument("--folds", type=int, default=None, help="limit folds (debug)")
    ap.add_argument("--out-dir", type=str, default=None,
                    help="override runs/<trunk>-<grad_mode>/")
    ap.add_argument("--quiet", action="store_true")
    add_mode_arguments(ap)
    args = ap.parse_args()

    mode = mode_from_args(args)
    cfg = build_config(args)
    p_sampled = mode.resolved_p_sampled_z(cfg.p_sampled_z)

    out_dir = Path(args.out_dir) if args.out_dir else HERE / "runs" / mode.name
    out_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    print("=" * 72)
    print(f"MODE  {mode.name}")
    print("=" * 72)
    print(f"  {mode.describe(cfg.p_sampled_z)}")
    if mode.is_staged:
        print("  phase A: trunk + flow head on the flow loss, then both frozen")
        print("  phase B: control head only, always fed a SAMPLED latent"
              if p_sampled >= 1.0 else
              f"  phase B: control head only, sampled latent p={p_sampled}")
    if mode.grad_mode == "pcgrad":
        print("  conflicting flow/control gradients projected apart at the shared trunk")
    if not args.quiet:
        print("\n" + "=" * 72); print("DATA"); print("=" * 72)

    samples = load_samples(cfg)
    if not args.quiet:
        print(describe(samples, cfg))
        print("\nFROZEN ENCODERS")

    img_all = image_features(samples, cfg)
    txt_all, _ = text_matrix(samples, cfg)
    tel_all = np.stack([s.tel for s in samples])
    chunk_all = np.stack([s.chunk for s in samples]).reshape(len(samples), cfg.horizon, 2)
    wp_all = np.stack([s.wp for s in samples])
    dt_all = np.stack([s.dt for s in samples])
    warm_all = np.stack([s.warm for s in samples])
    act_all = np.array([s.action for s in samples])
    scen_all = np.array([s.scenario for s in samples])
    texts_all = [s.text for s in samples]

    folds = scenarios_of(samples)
    if args.folds:
        folds = folds[:args.folds]

    print(f"\n  stage-2 models <- {cfg.calculator_dir}")
    print(f"  {len(folds)} folds (leave-one-scenario-out)  |  results -> {out_dir}")

    per_fold, size_info = [], None

    for fold_i, held_out in enumerate(folds):
        te = np.where(scen_all == held_out)[0]
        tr = np.where(scen_all != held_out)[0]
        plant, plant_file = load_fold_calculator(cfg, held_out)

        # fit on the training scenarios only
        img_std = Standardiser(img_all[tr])
        tel_std = Standardiser(tel_all[tr])
        v_std = Standardiser(chunk_all[tr, :, 0])
        w_std = Standardiser(chunk_all[tr, :, 1])
        latent = TextLatent(txt_all[tr], cfg.pca_dim)
        protos = command_prototypes(samples, tr)

        bank_texts = sorted(set(texts_all[i] for i in tr))
        bank_emb = np.stack(
            [txt_all[[i for i in tr if texts_all[i] == t][0]] for t in bank_texts]
        )
        train_mean_emb = txt_all[tr].mean(axis=0)
        train_mean_emb /= max(np.linalg.norm(train_mean_emb), 1e-8)

        counts = np.bincount(act_all[tr], minlength=cfg.n_actions).astype(np.float32)
        weight = torch.tensor(
            np.where(counts > 0, 1.0 / np.maximum(counts, 1.0), 0.0)
            / max((1.0 / np.maximum(counts[counts > 0], 1.0)).mean(), 1e-8),
            dtype=torch.float32,
        )

        def pack(idx):
            return (
                torch.tensor(img_std(img_all[idx])),
                torch.tensor(tel_std(tel_all[idx])),
                torch.tensor(latent.encode(txt_all[idx]), dtype=torch.float32),
                torch.tensor(v_std(chunk_all[idx, :, 0])),
                torch.tensor(w_std(chunk_all[idx, :, 1])),
                torch.tensor(act_all[idx], dtype=torch.long),
            )

        model = build_model(img_all.shape[1], tel_all.shape[1], latent.dim, cfg, mode)
        if size_info is None:
            size_info = parameter_report(model)
            print(f"\nMODEL  {size_info['total_params']:,} trainable params "
                  f"({size_info['size_fp16_mb']} MB fp16)")
            for name, k in size_info["by_module"].items():
                print(f"    {name:<16s} {k:>10,}")

        t0 = time.time()
        train_log = train_fold(cfg, model, pack(tr), weight, cfg.seed + fold_i, mode)
        train_secs = time.time() - t0

        # evaluate
        model.eval()
        img_te, tel_te, _, _, _, _ = pack(te)
        gen = torch.Generator().manual_seed(cfg.seed + fold_i)
        z_hat, v_hat, w_hat, logits = model.predict(img_te, tel_te, cfg.flow_steps, gen)

        pred_chunk = np.stack(
            [v_std.invert(v_hat.numpy()), w_std.invert(w_hat.numpy())], axis=2
        )
        true_chunk = chunk_all[te]
        vel = velocity_report(pred_chunk, true_chunk, protos)

        traj_pred = rollout(plant, pred_chunk, warm_all[te], dt_all[te])
        ade, fde = ade_fde(traj_pred, wp_all[te])
        traj_oracle = rollout(plant, true_chunk, warm_all[te], dt_all[te])
        ade_o, fde_o = ade_fde(traj_oracle, wp_all[te])
        hold = np.repeat(warm_all[te][:, -1:, :], cfg.horizon, axis=1)
        ade_h, fde_h = ade_fde(rollout(plant, hold, warm_all[te], dt_all[te]), wp_all[te])

        multi = []
        for k in range(cfg.n_samples_eval):
            g = torch.Generator().manual_seed(1000 * (fold_i + 1) + k)
            _, v_k, w_k, _ = model.predict(img_te, tel_te, cfg.flow_steps, g)
            c_k = np.stack([v_std.invert(v_k.numpy()), w_std.invert(w_k.numpy())], axis=2)
            multi.append(rollout(plant, c_k, warm_all[te], dt_all[te]))
        m_ade, m_fde = min_ade_fde(np.stack(multi), wp_all[te])

        txt = text_report(
            pred_emb=latent.decode(z_hat.numpy()),
            true_emb=txt_all[te],
            ceiling_emb=latent.reconstruct(txt_all[te]),
            bank_emb=bank_emb, bank_texts=bank_texts,
            true_texts=[texts_all[i] for i in te],
            mean_emb=train_mean_emb,
        )

        preds = logits.argmax(1).numpy()
        result = {
            "fold": fold_i,
            "held_out": held_out,
            "mode": mode.to_dict(cfg.p_sampled_z),
            "stage2_model": plant_file,
            "n_train": int(len(tr)), "n_test": int(len(te)),
            "latent_dim": latent.dim,
            "pca_explained": round(latent.explained, 4),
            **train_log,
            "velocity": vel,
            "ade": round(ade, 4), "fde": round(fde, 4),
            "ade_oracle_commands": round(ade_o, 4),
            "fde_oracle_commands": round(fde_o, 4),
            "ade_hold_last_command": round(ade_h, 4),
            "fde_hold_last_command": round(fde_h, 4),
            f"min_ade_k{cfg.n_samples_eval}": round(m_ade, 4),
            f"min_fde_k{cfg.n_samples_eval}": round(m_fde, 4),
            "action": action_report(preds, act_all[te], cfg.action_classes),
            "accuracy": action_report(preds, act_all[te], cfg.action_classes)["accuracy"],
            **{f"text_{k}": v for k, v in txt.items()},
            "train_seconds": round(train_secs, 1),
        }
        per_fold.append(result)

        fold_dir = out_dir / f"fold_{fold_i:02d}_{held_out}"
        fold_dir.mkdir(parents=True, exist_ok=True)
        torch.save({
            "model_state": model.state_dict(),
            "mode": mode.to_dict(cfg.p_sampled_z),
            "latent_dim": latent.dim,
            "pca_mean": latent.mean, "pca_basis": latent.basis, "pca_scale": latent.scale,
            "img_mean": img_std.mean, "img_std": img_std.std,
            "tel_mean": tel_std.mean, "tel_std": tel_std.std,
            "v_mean": v_std.mean, "v_std": v_std.std,
            "w_mean": w_std.mean, "w_std": w_std.std,
            "prototypes": protos, "bank_texts": bank_texts,
            "stage2": plant.to_dict(), "stage2_file": plant_file,
            "config": vars(cfg),
        }, fold_dir / "model.pt")
        (fold_dir / "metrics.json").write_text(json.dumps(result, indent=2, default=str))

        extra = ""
        if "grad_cosine_mean_over_training" in result:
            extra = f"  cos={result['grad_cosine_mean_over_training']:+.3f}"
        print(f"  fold {fold_i:2d} {held_out:<20s} n={len(te):3d}  "
              f"v_mae={vel['v']['mae']:.4f}  w_mae={vel['w']['mae']:.4f}  "
              f"ADE={ade:.4f} (oracle {ade_o:.4f})  acc={result['accuracy']:.3f}"
              f"{extra}  [{train_secs:.0f}s]")

    summary = {
        "mode": mode.to_dict(cfg.p_sampled_z),
        "n_folds": len(per_fold),
        "model_size": size_info,
        "config": {k: str(v) for k, v in vars(cfg).items()},
        "average": aggregate(per_fold),
        "folds": per_fold,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str))

    avg = summary["average"]

    def row(label, key, unit=""):
        if key in avg:
            print(f"  {label:<40s} {avg[key]['mean']:8.4f} +/- {avg[key]['std']:.4f}{unit}")

    print("\n" + "=" * 72)
    print(f"{mode.name.upper()}   AVERAGE OVER {len(per_fold)} FOLDS   (mean +/- std)")
    print("=" * 72)
    row("v  MAE (m/s)", "velocity.v.mae")
    row("w  MAE (rad/s)", "velocity.w.mae")
    row("exact command match (snapped)", "velocity.snapped_exact_match")
    print()
    row("ADE (m)", "ade")
    row("FDE (m)", "fde")
    row(f"min-ADE over {cfg.n_samples_eval} samples (m)", f"min_ade_k{cfg.n_samples_eval}")
    row("ADE with TRUE commands (floor)", "ade_oracle_commands")
    row("ADE holding the last command", "ade_hold_last_command")
    print()
    row("action accuracy (current step)", "accuracy")
    row("text cosine, model", "text_cosine_model")
    row("text cosine, PCA ceiling", "text_cosine_pca_ceiling")
    row("text retrieval accuracy", "text_retrieval_accuracy")
    if "grad_cosine_mean_over_training" in avg:
        print()
        row("flow-vs-control grad cosine", "grad_cosine_mean_over_training")
        row("fraction of conflicting batches", "frac_conflicting_over_training")
    if "pcgrad_projection_rate" in avg:
        row("pcgrad projection rate", "pcgrad_projection_rate")
    print(f"\n  summary -> {out_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
