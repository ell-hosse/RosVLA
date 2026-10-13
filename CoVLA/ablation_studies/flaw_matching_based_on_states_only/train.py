#!/usr/bin/env python3
"""Ablation: flow-match the caption embedding from vehicle state only.

    past 3 s vehicle state  --flow matching-->  PCA caption embedding
    image + past 3 s vehicle state + generated caption embedding  -->  future trajectory

Control and ablation runs share this folder and write to separate runs/
subdirectories:

    python train.py --flow-uses-image     # control (image + state -> z)
    python train.py                       # ablation (state only -> z)
    python train.py --traj-state-only     # fully state-only pipeline

    python train.py --max-train 20000 --epochs 50
    python train.py --device cpu

The main metric is text.scaled_score (1 = PCA ceiling, 0 = mean caption).
If the ablation matches the control, the flow head is not using the image.
ADE/FDE is secondary since the trajectory head is unchanged.

Uses the video-level split of covla_preprocessed, up to 50 epochs with early
stopping on validation ADE (patience 5). Standardisers and PCA are fitted on
train only.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import Config                                        # noqa: E402
from data import load_split                                      # noqa: E402
from latent import TextLatent, pca_report                        # noqa: E402
from metrics import ade_fde, min_ade_fde, text_fidelity_report   # noqa: E402
from model import CoVLAFlowModel, parameter_report                # noqa: E402
from utils import EarlyStopper, Standardiser                     # noqa: E402


def to_tensor(x: np.ndarray, device: str) -> torch.Tensor:
    return torch.as_tensor(x, dtype=torch.float32, device=device)


@torch.no_grad()
def evaluate(model, cfg, img, hist, traj_true_m, cap_emb_true, latent, mean_emb,
             wp_std, device, steps=None, n_samples=None):
    """Full report for one split. traj_true_m is in metres (not standardised)."""
    model.eval()
    steps = steps or cfg.flow_steps
    n_samples = n_samples or cfg.n_samples_eval

    gen = torch.Generator(device=device).manual_seed(cfg.seed)
    z_hat, traj_hat = model.predict(img, hist, steps, gen)
    traj_pred = wp_std.inverse(traj_hat.cpu().numpy()).reshape(-1, 20, 2)
    ade, fde = ade_fde(traj_pred, traj_true_m)

    samples = []
    for k in range(n_samples):
        g = torch.Generator(device=device).manual_seed(1000 + k)
        _, t_k = model.predict(img, hist, steps, g)
        samples.append(wp_std.inverse(t_k.cpu().numpy()).reshape(-1, 20, 2))
    m_ade, m_fde = min_ade_fde(np.stack(samples), traj_true_m)

    pred_emb = latent.decode(z_hat.cpu().numpy())
    ceiling_emb = latent.reconstruct(cap_emb_true)
    text_report = text_fidelity_report(pred_emb, cap_emb_true, ceiling_emb, mean_emb)

    return {
        "ade": round(ade, 4), "fde": round(fde, 4),
        f"min_ade_k{n_samples}": round(m_ade, 4), f"min_fde_k{n_samples}": round(m_fde, 4),
        "text": text_report,
    }


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
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
    ap.add_argument("--rebuild-history-cache", action="store_true")
    ap.add_argument("--rebuild-image-cache", action="store_true")
    ap.add_argument("--device", default=None, choices=["cuda", "cpu"])
    ap.add_argument(
        "--flow-uses-image", action="store_true",
        help="CONTROL RUN: give the flow head the baseline's [image, vehicle state] "
             "conditioning instead of state only.",
    )
    ap.add_argument(
        "--traj-state-only", action="store_true",
        help="Also drop the image from the TRAJECTORY head (fully state-only pipeline). "
             "A separate experiment, not a substitute for the default run.",
    )
    args = ap.parse_args()

    cfg = Config()
    if args.epochs is not None:
        cfg.epochs = args.epochs
    if args.patience is not None:
        cfg.early_stop_patience = args.patience
    if args.batch_size is not None:
        cfg.batch_size = args.batch_size
    if args.lr is not None:
        cfg.lr = args.lr
    if args.pca_dim is not None:
        cfg.pca_dim = args.pca_dim
    if args.max_train is not None:
        cfg.max_train = args.max_train
    if args.max_val is not None:
        cfg.max_val = args.max_val
    if args.max_test is not None:
        cfg.max_test = args.max_test
    if args.device is not None:
        cfg.device = args.device
    if args.preprocessed_dir:
        cfg.preprocessed_dir = Path(args.preprocessed_dir)
    if args.meta_dir:
        cfg.meta_dir = Path(args.meta_dir)
    if args.flow_uses_image:
        cfg.flow_uses_image = True
    if args.traj_state_only:
        cfg.traj_uses_image = False

    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    device = cfg.device if (cfg.device == "cpu" or torch.cuda.is_available()) else "cpu"
    print(f"device: {device}")

    _c = lambda flag: "image + vehicle state" if flag else "VEHICLE STATE ONLY (no image)"
    print(f"ablation variant: {cfg.variant}"
          f"{'   <- this is the CONTROL (= baseline)' if cfg.flow_uses_image else ''}")
    print(f"  flow head conditioning: {_c(cfg.flow_uses_image)}")
    print(f"  traj head conditioning: {_c(cfg.traj_uses_image)}")

    print("=" * 72); print("LOADING DATA"); print("=" * 72)
    train = load_split(cfg, "train", force_history=args.rebuild_history_cache,
                        force_images=args.rebuild_image_cache)
    val = load_split(cfg, "val", force_history=args.rebuild_history_cache,
                      force_images=args.rebuild_image_cache)
    test = load_split(cfg, "test", force_history=args.rebuild_history_cache,
                       force_images=args.rebuild_image_cache)
    print(f"\nfinal sizes: train={len(train.traj)}  val={len(val.traj)}  test={len(test.traj)}")
    if train.approximate_history:
        print("  [!] history windows are the 2 Hz-fallback approximation, not true 20 Hz "
              "(CoVLA-metadata/states was not found under --meta-dir) -- see README.md.")

    # fit standardisers and PCA on train only
    img_std = Standardiser(train.img_feat)
    hist_std = Standardiser(train.hist)
    traj_flat_tr = train.traj.reshape(len(train.traj), -1)
    wp_std = Standardiser(traj_flat_tr)
    latent = TextLatent(train.cap_emb, cfg.pca_dim)
    mean_emb = train.cap_emb.mean(axis=0)
    mean_emb = mean_emb / max(np.linalg.norm(mean_emb), 1e-8)

    print(f"\nPCA text latent: dim={latent.dim}  "
          f"cumulative explained variance (train)={latent.explained:.4f}")

    def pack(split):
        img = to_tensor(img_std(split.img_feat), device)
        hist = to_tensor(hist_std(split.hist), device)
        z = to_tensor(latent.encode(split.cap_emb), device)
        traj = to_tensor(wp_std(split.traj.reshape(len(split.traj), -1)), device)
        return img, hist, z, traj

    img_tr, hist_tr, z_tr, traj_tr = pack(train)
    img_va, hist_va, z_va, traj_va = pack(val)
    img_te, hist_te, z_te, traj_te = pack(test)

    train_loader = DataLoader(
        TensorDataset(img_tr, hist_tr, z_tr, traj_tr),
        batch_size=cfg.batch_size, shuffle=True,
    )

    model = CoVLAFlowModel(
        img_dim=train.img_feat.shape[1],
        history_len=train.hist.shape[1],
        history_feat_dim=train.hist.shape[2],
        latent_dim=latent.dim,
        cfg=cfg,
    ).to(device)

    size_info = parameter_report(model)
    print(f"\nmodel: {size_info['total_params']:,} trainable params "
          f"({size_info['size_fp32_mb']} MB fp32); by module: {size_info['by_module']}")

    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.epochs)
    stopper = EarlyStopper(cfg.early_stop_patience, cfg.early_stop_min_delta)

    print("\n" + "=" * 72); print(f"TRAINING (max {cfg.epochs} epochs, "
                                    f"early stop patience={cfg.early_stop_patience})")
    print("=" * 72)
    history_log = []
    for epoch in range(cfg.epochs):
        model.train()
        t0 = time.time()
        flow_losses, traj_losses = [], []
        for bi, bh, bz, btr in train_loader:
            b = bi.shape[0]
            # separate conditioning for the flow and trajectory heads
            cond_flow, cond_traj = model.conditions(bi, bh)

            # rectified flow: straight line from noise to data
            s = torch.rand(b, 1, device=device)
            eps = torch.randn_like(bz)
            z_s = (1 - s) * eps + s * bz
            v_target = bz - eps
            v_pred = model.flow(z_s, s.squeeze(1), cond_flow)
            loss_flow = F.mse_loss(v_pred, v_target)

            # trajectory head, fed the true z or a sampled one
            if torch.rand(()) < cfg.p_sampled_z:
                with torch.no_grad():
                    z_in = model.flow.sample(cond_flow, cfg.flow_steps)
            else:
                z_in = bz
            traj_pred = model.traj_head(cond_traj, z_in)
            loss_traj = F.smooth_l1_loss(traj_pred, btr)

            loss = cfg.w_flow * loss_flow + cfg.w_traj * loss_traj
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            opt.step()

            flow_losses.append(loss_flow.item())
            traj_losses.append(loss_traj.item())
        sched.step()

        # early stopping on single-sample validation ADE
        model.eval()
        with torch.no_grad():
            gen = torch.Generator(device=device).manual_seed(cfg.seed + epoch)
            _, traj_hat = model.predict(img_va, hist_va, cfg.flow_steps, gen)
            traj_pred_m = wp_std.inverse(traj_hat.cpu().numpy()).reshape(-1, 20, 2)
            val_ade, val_fde = ade_fde(traj_pred_m, val.traj)

        dt = time.time() - t0
        print(f"  epoch {epoch:3d}  flow_loss={np.mean(flow_losses):.4f}  "
              f"traj_loss={np.mean(traj_losses):.4f}  val_ADE={val_ade:.4f}  "
              f"val_FDE={val_fde:.4f}  [{dt:.1f}s]")
        history_log.append({
            "epoch": epoch, "flow_loss": float(np.mean(flow_losses)),
            "traj_loss": float(np.mean(traj_losses)), "val_ade": val_ade, "val_fde": val_fde,
        })

        if stopper.step(val_ade, epoch, model.state_dict()):
            print(f"  early stopping at epoch {epoch} "
                  f"(best val_ADE={stopper.best:.4f} at epoch {stopper.best_epoch})")
            break

    model.load_state_dict(stopper.best_state)
    print(f"\nrestored best checkpoint from epoch {stopper.best_epoch} "
          f"(val_ADE={stopper.best:.4f})")

    print("\n" + "=" * 72); print("FINAL EVALUATION"); print("=" * 72)
    report = {
        "ablation": {
            "name": "flaw_matching_based_on_states_only",
            "variant": cfg.variant,
            "is_control_run": bool(cfg.flow_uses_image and cfg.traj_uses_image),
            "flow_head_conditioning": (
                "image + vehicle state" if cfg.flow_uses_image
                else "vehicle state ONLY (no image)"
            ),
            "trajectory_head_conditioning": (
                "image + vehicle state" if cfg.traj_uses_image
                else "vehicle state ONLY (no image)"
            ),
            "flow_cond_dim": int(model.flow_cond_dim),
            "traj_cond_dim": int(model.traj_cond_dim),
        },
        "model_size": size_info,
        "pca": pca_report(latent, {"train": train.cap_emb, "val": val.cap_emb,
                                    "test": test.cap_emb}),
        "best_epoch": stopper.best_epoch,
        "history_log": history_log,
        "approximate_history": bool(train.approximate_history),
        "n_rows": {"train": len(train.traj), "val": len(val.traj), "test": len(test.traj)},
    }

    for name, img_t, hist_t, traj_true, cap_true in (
        ("train", img_tr, hist_tr, train.traj, train.cap_emb),
        ("val", img_va, hist_va, val.traj, val.cap_emb),
        ("test", img_te, hist_te, test.traj, test.cap_emb),
    ):
        r = evaluate(model, cfg, img_t, hist_t, traj_true, cap_true, latent, mean_emb,
                     wp_std, device)
        report[name] = r
        print(f"\n  [{name}]  n={len(traj_true)}")
        print(f"    ADE {r['ade']:.4f} m   FDE {r['fde']:.4f} m   "
              f"min-ADE(K={cfg.n_samples_eval}) {r[f'min_ade_k{cfg.n_samples_eval}']:.4f}   "
              f"min-FDE(K={cfg.n_samples_eval}) {r[f'min_fde_k{cfg.n_samples_eval}']:.4f}")
        t = r["text"]
        print(f"    text cosine: model={t['cosine_model']:.4f}  "
              f"PCA-ceiling={t['cosine_pca_ceiling']:.4f}  "
              f"mean-baseline={t['cosine_mean_baseline']:.4f}  "
              f"scaled={t['scaled_score']:.4f}  (1.0=ceiling, 0.0=no better than mean)")

    print(f"\n  --- PCA performance (projecting {cfg.text_emb_dim}-d captions down to "
          f"{latent.dim}-d) ---")
    for name in ("train", "val", "test"):
        p = report["pca"][name]
        print(f"    {name:<5} n={p['n']:6d}  reconstruction cosine={p['reconstruction_cosine']:.4f}"
              f"  MSE={p['reconstruction_mse']:.6f}")

    out_dir = cfg.out_dir / cfg.variant
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model_state": model.state_dict(),
        "img_mean": img_std.mean, "img_std": img_std.std,
        "hist_mean": hist_std.mean, "hist_std": hist_std.std,
        "wp_mean": wp_std.mean, "wp_std": wp_std.std,
        "pca_mean": latent.mean, "pca_basis": latent.basis, "pca_scale": latent.scale,
        "mean_emb": mean_emb, "config": vars(cfg),
    }, out_dir / "model.pt")
    (out_dir / "report.json").write_text(json.dumps(report, indent=2, default=str))
    print(f"\nmodel       -> {out_dir / 'model.pt'}")
    print(f"full report -> {out_dir / 'report.json'}")


if __name__ == "__main__":
    main()
