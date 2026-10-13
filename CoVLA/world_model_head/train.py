#!/usr/bin/env python3
"""Train the CoVLA world-model trajectory head (three ablations).

    python train.py --ablation a_world_model_only     # world model alone
    python train.py --ablation b_vla_diffusion        # with the flow caption head
    python train.py --ablation c_vla_no_text          # no text
    python train.py --all                             # all three
    python train.py --self-test                       # check the integrator

    python train.py --all --max-train 100000 --max-val 10000 --max-test 10000
    python train.py --ablation a_world_model_only --prior cv --no-residual
    python train.py --device cpu

Each ablation writes to runs_<ablation>/. Standardisers, the PCA latent and
the steering->curvature fit use the training split only.

Reports also include:

  baselines                CV / CA / CTRV / CTRA through the same rollout
  oracle_actions           inverse-dynamics actions rolled out (ADE should be ~0)
  oracle_actions_clipped   the same actions clipped to the action bounds
                           (best ADE reachable in this action space)
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

from config import (ABLATIONS, ACC_MAX, DT, HERE, KAPPA_MAX,      # noqa: E402
                    N_STEPS, N_WAYPOINTS, Config)
from data import load_split                                        # noqa: E402
from dynamics import (apply_steering_map, baseline_report, clip_actions,   # noqa: E402
                      identify_steering_map, inverse_dynamics, prior_actions,
                      rollout, rollout_np, self_test)
from latent import TextLatent, pca_report                          # noqa: E402
from metrics import (action_report, ade_fde, min_ade_fde,          # noqa: E402
                     text_fidelity_report, trajectory_report)
from model import WorldModelVLA, parameter_report                   # noqa: E402
from utils import EarlyStopper, Standardiser                        # noqa: E402


def to_tensor(x, device):
    return torch.as_tensor(np.asarray(x), dtype=torch.float32, device=device)


def pre_tanh(actions: np.ndarray) -> np.ndarray:
    """Map prior actions to pre-tanh space so that raw == 0 reproduces the prior.

    Clamped to 0.999 to keep atanh finite.
    """
    out = np.empty_like(actions, dtype=np.float32)
    out[..., 0] = np.arctanh(np.clip(actions[..., 0] / ACC_MAX, -0.999, 0.999))
    out[..., 1] = np.arctanh(np.clip(actions[..., 1] / KAPPA_MAX, -0.999, 0.999))
    return out


@torch.no_grad()
def evaluate(model, cfg, pack, split, latent=None, mean_emb=None, device="cpu"):
    """Full report for one split, in metres."""
    model.eval()
    img, hist, v0, prior_pre = pack["img"], pack["hist"], pack["v0"], pack["prior_pre"]
    gen = torch.Generator(device=device).manual_seed(cfg.seed)

    z_hat, traj_hat, actions, residual = model.predict(
        img, hist, v0, prior_pre, steps=cfg.flow_steps, generator=gen
    )
    traj_pred = traj_hat.cpu().numpy()
    rep = trajectory_report(traj_pred, split.traj)
    rep["actions"] = action_report(
        actions.cpu().numpy(),
        residual.cpu().numpy() if residual is not None else None,
    )

    if model.uses_flow:
        samples = []
        for k in range(cfg.n_samples_eval):
            g = torch.Generator(device=device).manual_seed(1000 + k)
            _, t_k, _, _ = model.predict(img, hist, v0, prior_pre, cfg.flow_steps, g)
            samples.append(t_k.cpu().numpy())
        m_ade, m_fde = min_ade_fde(np.stack(samples), split.traj)
        rep[f"min_ade_k{cfg.n_samples_eval}"] = round(m_ade, 4)
        rep[f"min_fde_k{cfg.n_samples_eval}"] = round(m_fde, 4)
        pred_emb = latent.decode(z_hat.cpu().numpy())
        rep["text"] = text_fidelity_report(
            pred_emb, split.cap_emb, latent.reconstruct(split.cap_emb), mean_emb
        )
    return rep


def run_one(cfg: Config, args) -> dict:
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    device = cfg.device if (cfg.device == "cpu" or torch.cuda.is_available()) else "cpu"

    print("\n" + "#" * 72)
    print(f"# ABLATION: {cfg.ablation}")
    print(f"#   image: {cfg.uses_image}   flow/diffusion: {cfg.uses_flow}   "
          f"text reaching the action head: {cfg.uses_text}")
    print(f"#   prior: {cfg.prior}   residual: {cfg.use_residual}   device: {device}")
    print("#" * 72)

    # data
    splits = {}
    for name in ("train", "val", "test"):
        splits[name] = load_split(cfg, name, force_history=args.rebuild_history_cache,
                                  force_images=args.rebuild_image_cache)
    train, val, test = splits["train"], splits["val"], splits["test"]
    print(f"\nfinal sizes: train={len(train)}  val={len(val)}  test={len(test)}")
    if train.approximate_history:
        print("  [!] history windows are the 2 Hz fallback, not true 20 Hz "
              "(CoVLA-metadata/states not found) -- see README.md")

    # fit standardisers and PCA on train only
    hist_std = Standardiser(train.hist)
    img_std = Standardiser(train.img_feat) if cfg.uses_image else None

    latent = mean_emb = None
    if cfg.uses_text:
        latent = TextLatent(train.cap_emb, cfg.pca_dim)
        mean_emb = train.cap_emb.mean(axis=0)
        mean_emb = mean_emb / max(np.linalg.norm(mean_emb), 1e-8)
        print(f"\nPCA text latent: dim={latent.dim}  "
              f"cumulative explained variance (train)={latent.explained:.4f}")

    # inverse dynamics: actions that reproduce each ground-truth path
    gt_actions, gt_valid = {}, {}
    for name, sp in splits.items():
        a, v = inverse_dynamics(sp.traj, sp.v0)
        gt_actions[name], gt_valid[name] = a, v
        print(f"  [{name}] inverse dynamics: {v.mean() * 100:.1f}% of rows moving at "
              f"every step (the rest were stopped and are curvature-free by definition)")

    # steering -> curvature fit (train only)
    smap = identify_steering_map(train.steer0, train.v0, gt_actions["train"][:, 0, 1])
    if smap["fitted"]:
        print(f"\nidentified: kappa = deg2rad(steer) / (L_eff * (1 + K_us * v^2))   "
              f"L_eff={smap['L_eff_m']:.3f} m  K_us={smap['K_us']:.4f}  R^2={smap['r2']:.4f}")
        if smap["r2"] < 0.4:
            print("  [!] a weak fit: a single frame's steering angle is a noisy estimate "
                  "of the path curvature. This is why `ca` beats `ctra` as a prior.")

    # priors, baselines, oracles
    baselines = {}
    priors_pre, oracle = {}, {}
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

    print("\nzero-parameter baselines through the same rollout (test):")
    for k, v in baselines["test"].items():
        print(f"    {k.upper():<5} ADE {v['ade_m']:.4f} m   FDE {v['fde_m']:.4f} m")
    print(f"    oracle actions, exact:   ADE {oracle['test']['exact']['ade_m']:.2e} m "
          f"(integrator round-trip check -- should be ~0)")
    print(f"    oracle actions, clipped: ADE "
          f"{oracle['test']['clipped_to_envelope']['ade_m']:.4f} m  <- the action "
          f"envelope's own ceiling")

    # pack
    def pack(name):
        sp = splits[name]
        n = len(sp)
        img = (to_tensor(img_std(sp.img_feat), device) if cfg.uses_image
               else torch.zeros(n, 1, device=device))
        return {
            "img": (img if cfg.uses_image else None),
            "img_t": img,
            "hist": to_tensor(hist_std(sp.hist), device),
            "v0": to_tensor(sp.v0, device),
            "prior_pre": to_tensor(priors_pre[name], device),
            "traj": to_tensor(sp.traj, device),
            "gt_act": to_tensor(clip_actions(gt_actions[name]), device),
            "z": (to_tensor(latent.encode(sp.cap_emb), device) if cfg.uses_text
                  else torch.zeros(n, 1, device=device)),
        }

    P = {name: pack(name) for name in splits}
    loader = DataLoader(
        TensorDataset(P["train"]["img_t"], P["train"]["hist"], P["train"]["v0"],
                      P["train"]["prior_pre"], P["train"]["traj"],
                      P["train"]["gt_act"], P["train"]["z"]),
        batch_size=cfg.batch_size, shuffle=True,
    )

    # model
    model = WorldModelVLA(
        ablation=cfg.ablation,
        img_dim=(train.img_feat.shape[1] if cfg.uses_image else 0),
        history_feat_dim=train.hist.shape[2],
        latent_dim=(latent.dim if cfg.uses_text else 0),
        cfg=cfg,
    ).to(device)
    size_info = parameter_report(model)
    print(f"\nmodel: {size_info['total_params']:,} trainable params "
          f"({size_info['size_fp32_mb']} MB fp32); by module: {size_info['by_module']}")

    # stronger weight decay on the residual
    res_params = list(model.residual_head.parameters()) if model.residual_head else []
    res_ids = {id(p) for p in res_params}
    base_params = [p for p in model.parameters() if id(p) not in res_ids]
    opt = torch.optim.AdamW(
        [{"params": base_params, "weight_decay": cfg.weight_decay},
         {"params": res_params, "weight_decay": cfg.residual_weight_decay}],
        lr=cfg.lr,
    )
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.epochs)
    stopper = EarlyStopper(cfg.early_stop_patience, cfg.early_stop_min_delta)

    act_scale = to_tensor(
        np.maximum(clip_actions(gt_actions["train"]).std(axis=(0, 1)), 1e-3), device
    )

    # training
    print("\n" + "=" * 72)
    print(f"TRAINING (max {cfg.epochs} epochs, early stop patience="
          f"{cfg.early_stop_patience}, {cfg.pretrain_epochs} action-pretrain epochs)")
    print("=" * 72)
    history_log = []
    for epoch in range(cfg.epochs):
        model.train()
        t0 = time.time()
        flow_losses, traj_losses, act_losses = [], [], []
        pretrain = epoch < cfg.pretrain_epochs

        for bi, bh, bv, bp, btr, bga, bz in loader:
            img_in = bi if cfg.uses_image else None
            cond = model.condition(img_in, bh)

            loss = 0.0
            loss_flow = None
            z_in = None
            if model.uses_flow:
                b = bh.shape[0]
                s = torch.rand(b, 1, device=device)
                eps = torch.randn_like(bz)
                z_s = (1 - s) * eps + s * bz
                v_target = bz - eps
                loss_flow = F.mse_loss(model.flow(z_s, s.squeeze(1), cond), v_target)
                loss = loss + cfg.w_flow * loss_flow
                # use a sampled latent with probability p_sampled_z
                if torch.rand(()) < cfg.p_sampled_z:
                    with torch.no_grad():
                        z_in = model.flow.sample(cond, cfg.flow_steps)
                else:
                    z_in = bz

            actions, residual = model.act(cond, z_in, bp)

            if pretrain:
                # warm start: supervise on the inverse-dynamics actions
                loss_act = F.mse_loss(actions / act_scale, bga / act_scale)
                loss = loss + loss_act
                act_losses.append(loss_act.item())
                traj_losses.append(float("nan"))
            else:
                traj = rollout(bv, actions)
                if residual is not None:
                    traj = traj + residual
                loss_traj = F.smooth_l1_loss(traj, btr)
                # smoothness penalty on the actions
                da = actions[:, 1:, 0] - actions[:, :-1, 0]
                dk = actions[:, 1:, 1] - actions[:, :-1, 1]
                loss_smooth = cfg.w_jerk * (da ** 2).mean() + cfg.w_dkappa * (dk ** 2).mean()
                loss = loss + cfg.w_traj * loss_traj + loss_smooth
                traj_losses.append(loss_traj.item())
            if loss_flow is not None:
                flow_losses.append(loss_flow.item())

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            opt.step()
        sched.step()

        model.eval()
        with torch.no_grad():
            gen = torch.Generator(device=device).manual_seed(cfg.seed + epoch)
            _, traj_hat, _, _ = model.predict(
                P["val"]["img"], P["val"]["hist"], P["val"]["v0"],
                P["val"]["prior_pre"], cfg.flow_steps, gen,
            )
            val_ade, val_fde = ade_fde(traj_hat.cpu().numpy(), val.traj)

        rec = {"epoch": epoch, "val_ade": val_ade, "val_fde": val_fde,
               "pretrain": bool(pretrain)}
        msg = f"  epoch {epoch:3d}"
        if flow_losses:
            rec["flow_loss"] = float(np.mean(flow_losses))
            msg += f"  flow={rec['flow_loss']:.4f}"
        if pretrain:
            rec["action_loss"] = float(np.mean(act_losses))
            msg += f"  action={rec['action_loss']:.4f} (pretrain)"
        else:
            rec["traj_loss"] = float(np.mean(traj_losses))
            msg += f"  traj={rec['traj_loss']:.4f}"
        print(msg + f"  val_ADE={val_ade:.4f}  val_FDE={val_fde:.4f}  "
                    f"[{time.time() - t0:.1f}s]")
        history_log.append(rec)

        if not pretrain and stopper.step(val_ade, epoch, model.state_dict()):
            print(f"  early stopping at epoch {epoch} (best val_ADE={stopper.best:.4f} "
                  f"at epoch {stopper.best_epoch})")
            break

    if stopper.best_state is not None:
        model.load_state_dict(stopper.best_state)
        print(f"\nrestored best checkpoint from epoch {stopper.best_epoch} "
              f"(val_ADE={stopper.best:.4f})")

    # report
    print("\n" + "=" * 72); print("FINAL EVALUATION"); print("=" * 72)
    report = {
        "ablation": cfg.ablation,
        "what_it_sees": {
            "image": cfg.uses_image,
            "past_3s_state_window": True,
            "caption_embedding": ("generated by the rectified-flow head"
                                  if cfg.uses_flow else "none at all"),
        },
        "trajectory_is_produced_by": (
            "a differentiable kinematic rollout of predicted (a, kappa) controls"
            + (" plus a gated learned residual" if cfg.use_residual else "")
        ),
        "model_size": size_info,
        "dynamics_self_test": self_test(),
        "identified_steering_map": smap,
        "prior": cfg.prior,
        "dt_s": DT, "n_steps": N_STEPS, "n_waypoints": N_WAYPOINTS,
        "horizon_s": round(N_STEPS * DT, 4),
        "best_epoch": stopper.best_epoch,
        "history_log": history_log,
        "approximate_history": bool(train.approximate_history),
        "n_rows": {k: len(v) for k, v in splits.items()},
        "baselines": baselines,
        "oracle_actions": oracle,
    }
    if cfg.uses_text:
        report["pca"] = pca_report(
            latent, {k: v.cap_emb for k, v in splits.items()}
        )

    for name in ("train", "val", "test"):
        r = evaluate(model, cfg, P[name], splits[name], latent, mean_emb, device)
        report[name] = r
        print(f"\n  [{name}]  n={r['n']}")
        print(f"    ADE {r['ade_m']:.4f} m   FDE {r['fde_m']:.4f} m")
        if f"min_ade_k{cfg.n_samples_eval}" in r:
            print(f"    min-ADE(K={cfg.n_samples_eval}) "
                  f"{r[f'min_ade_k{cfg.n_samples_eval}']:.4f}   "
                  f"min-FDE(K={cfg.n_samples_eval}) "
                  f"{r[f'min_fde_k{cfg.n_samples_eval}']:.4f}")
        best_base = min(baselines[name].items(), key=lambda kv: kv[1]["ade_m"])
        base_ade = best_base[1]["ade_m"]
        delta = (f"{100 * (1 - r['ade_m'] / base_ade):+.1f}% on ADE" if base_ade > 1e-9
                 else "n/a (that baseline is exact on this data)")
        print(f"    best zero-param baseline on this split: {best_base[0].upper()} "
              f"ADE {base_ade:.4f} m  ->  world model is {delta}")
        if "residual" in r["actions"]:
            print(f"    learned residual carries {r['actions']['residual']['mean_abs_m']:.4f} m "
                  f"of the {r['ade_m']:.4f} m error scale")
        if "text" in r:
            t = r["text"]
            print(f"    text cosine: model={t['cosine_model']:.4f}  "
                  f"ceiling={t['cosine_pca_ceiling']:.4f}  "
                  f"mean-baseline={t['cosine_mean_baseline']:.4f}  "
                  f"scaled={t['scaled_score']:.4f}")

    out_dir = cfg.out_dir
    ckpt = {
        "model_state": model.state_dict(),
        "hist_mean": hist_std.mean, "hist_std": hist_std.std,
        "steering_map": smap, "prior": cfg.prior,
        "config": {k: str(v) for k, v in vars(cfg).items()},
    }
    if cfg.uses_image:
        ckpt |= {"img_mean": img_std.mean, "img_std": img_std.std}
    if cfg.uses_text:
        ckpt |= {"pca_mean": latent.mean, "pca_basis": latent.basis,
                 "pca_scale": latent.scale, "mean_emb": mean_emb}
    torch.save(ckpt, out_dir / "model.pt")
    (out_dir / "report.json").write_text(json.dumps(report, indent=2, default=str))
    print(f"\nmodel       -> {out_dir / 'model.pt'}")
    print(f"full report -> {out_dir / 'report.json'}")
    return report


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ablation", default="a_world_model_only", choices=list(ABLATIONS))
    ap.add_argument("--all", action="store_true", help="run all three ablations in order")
    ap.add_argument("--self-test", action="store_true",
                    help="round-trip the kinematic integrator and exit")
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
    args = ap.parse_args()

    if args.self_test:
        print(json.dumps(self_test(), indent=2))
        return

    todo = list(ABLATIONS) if args.all else [args.ablation]
    for ablation in todo:
        # pass out_dir to the constructor so only runs_<ablation>/ is created
        cfg = Config(ablation=ablation, out_dir=HERE / f"runs_{ablation}")
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
        run_one(cfg, args)

    if len(todo) > 1:
        print("\nall three ablations done -- run `python compare.py` for the table")


if __name__ == "__main__":
    main()
