#!/usr/bin/env python3
"""DriveLM-nuScenes VLA with the world-model trajectory head.
Counterpart of CoVLA/world_model_head.

The action head predicts 6 (acceleration, curvature) controls, one per 0.5 s
interval, as a bounded correction to an analytic prior. Waypoints come from
the kinematic rollout in dynamics.py plus a gated residual initialised at 0.

    python train_wm.py --self-test                     # integrator round trip
    python train_wm.py --ablation a_world_model_only   # CAN window only
    python train_wm.py --all --seeds 0 1 2             # all four ablations, 3 seeds
    python train_wm.py --all --prior ca --no-residual  # rollout only

Ablations:
    a_world_model_only   past CAN window -> actions -> rollout. No image, no text.
    b_vla_diffusion      image + CAN --flow--> caption latent;
                         [image, CAN, latent] -> actions -> rollout
    c_vla_no_text        image + CAN -> actions -> rollout
    d_vla_gt_caption     oracle: ground-truth caption latent to the action head

Reports include the CV / CA / CTRV / CTRA baselines, the inverse-dynamics
oracle (should be ~0) and the action-bound ceiling.
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

from common import (TextSide, Timer, add_common_args, apply_common_args,   # noqa: E402
                    baselines_for, pick_device, print_text, print_traj, run_dir,
                    set_seed, subgroup_report, to_tensor, write_summary)
from config import (ACC_MAX, DT, HORIZON_S, KAPPA_MAX, N_STEPS,            # noqa: E402
                    WM_ABLATIONS, Config)
from data import load_splits                                               # noqa: E402
from dynamics import (PRIOR_NAMES, clip_actions, identify_steering_map,    # noqa: E402
                      inverse_dynamics, measured_kappa, prior_actions, rollout_np,
                      self_test, yaw_rate_kappa_r2)
from latent import pca_report                                              # noqa: E402
from metrics import action_report, min_ade_fde, trajectory_report          # noqa: E402
from model import WorldModelVLA, parameter_report                          # noqa: E402
from utils import EarlyStopper, Standardiser                               # noqa: E402


def pre_tanh(actions: np.ndarray) -> np.ndarray:
    out = np.empty_like(actions, dtype=np.float32)
    out[..., 0] = np.arctanh(np.clip(actions[..., 0] / ACC_MAX, -0.999, 0.999))
    out[..., 1] = np.arctanh(np.clip(actions[..., 1] / KAPPA_MAX, -0.999, 0.999))
    return out


@torch.no_grad()
def evaluate(model, cfg, P, split, text, device, sweep=False) -> dict:
    model.eval()
    args = (P["img"], P["hist"], P["z"], P["v0"], P["prior_pre"])
    gen = torch.Generator(device=device).manual_seed(cfg.seed)
    z_hat, traj7, actions, residual = model.predict(*args, cfg.flow_steps, gen)
    pred = traj7.cpu().numpy()[:, 1:]
    rep = trajectory_report(pred, split.traj)
    rep["subgroups"] = subgroup_report(pred, split)
    rep["actions"] = action_report(actions.cpu().numpy(),
                                   residual.cpu().numpy() if residual is not None else None)
    if model.uses_flow and not model.gt_text_input:
        samples = []
        for k in range(cfg.n_samples_eval):
            g = torch.Generator(device=device).manual_seed(1000 + k)
            samples.append(model.predict(*args, cfg.flow_steps, g)[1].cpu().numpy()[:, 1:])
        m_ade, m_fde = min_ade_fde(np.stack(samples), split.traj)
        rep[f"min_ade_k{cfg.n_samples_eval}"] = round(m_ade, 4)
        rep[f"min_fde_k{cfg.n_samples_eval}"] = round(m_fde, 4)
    if model.uses_flow and split.has_text:
        rep["text"] = text.report(z_hat.cpu().numpy(), split)
    if sweep and model.uses_flow:
        rows = []
        for steps in cfg.step_sweep:
            g = torch.Generator(device=device).manual_seed(cfg.seed)
            with Timer() as tm:
                zs, t7, _, _ = model.predict(*args, steps, g)
            r = {"steps": steps,
                 "ade_m": trajectory_report(t7.cpu().numpy()[:, 1:], split.traj)["ade_m"],
                 "ms_per_sample": round(1000 * tm.dt / len(split), 4)}
            if split.has_text:
                r["cosine_model"] = text.report(zs.cpu().numpy(), split)["cosine_model"]
            rows.append(r)
        rep["step_sweep"] = rows
    return rep


def run_one(cfg: Config, args, seed: int) -> dict:
    cfg.seed = seed
    set_seed(seed)
    device = pick_device(cfg)
    out = run_dir(cfg, seed)
    out.mkdir(parents=True, exist_ok=True)

    print("\n" + "#" * 78)
    print(f"# WORLD MODEL / {cfg.ablation}   text_field="
          f"{cfg.text_field if cfg.uses_text else '-'}   seed={seed}   device={device}")
    print(f"#   image: {cfg.uses_image}   flow: {cfg.uses_flow}   GT caption into head: "
          f"{cfg.gt_text_input}   prior: {cfg.prior}   residual: {cfg.use_residual}")
    print("#" * 78)

    splits, info = load_splits(cfg, force_images=args.rebuild_image_cache)
    train = splits["train"]

    img_std = Standardiser(train.img_feat) if cfg.uses_image else None
    hist_std = Standardiser(train.hist)
    text = TextSide(train, cfg) if cfg.uses_text else None

    # inverse dynamics, measured curvature, baselines, oracles
    gt_act, kappa0, baselines, oracle = {}, {}, {}, {}
    for name, sp in splits.items():
        gt_act[name], _ = inverse_dynamics(sp.traj7, sp.v0)
        kappa0[name] = measured_kappa(sp.v0, sp.yaw_rate0)
        baselines[name] = baselines_for(sp)
        exact = rollout_np(sp.v0, gt_act[name])[:, 1:]
        clipped = rollout_np(sp.v0, clip_actions(gt_act[name]))[:, 1:]
        oracle[name] = {
            "exact": {"ade_m": round(float(np.linalg.norm(exact - sp.traj, axis=-1).mean()), 6)},
            "clipped_to_envelope": {
                "ade_m": round(float(np.linalg.norm(clipped - sp.traj, axis=-1).mean()), 4),
                "fde_m": round(float(np.linalg.norm(clipped - sp.traj, axis=-1)[:, -1].mean()), 4),
                "note": f"best ADE reachable by ANY model with |a|<={ACC_MAX}, |kappa|<={KAPPA_MAX}"},
        }
    smap = identify_steering_map(train.steer0, train.v0, gt_act["train"][:, 0, 1])
    yaw_r2 = yaw_rate_kappa_r2(train.v0, train.yaw_rate0, gt_act["train"][:, 0, 1])

    prior = cfg.prior
    if prior == "auto":   # chosen on train
        prior = min(PRIOR_NAMES, key=lambda k: baselines["train"][k]["ade_m"])
    print(f"  prior: {prior}"
          + (" (auto: lowest TRAIN ADE of cv/ca/ctrv/ctra)" if cfg.prior == "auto" else ""))
    print(f"  measured kappa0 = yaw_rate / v explains R^2={yaw_r2} of the first-step curvature "
          f"(steering-wheel map R^2={smap.get('r2')})")
    print("  zero-parameter baselines (test): " + "  ".join(
        f"{k.upper()} {v['ade_m']:.3f}" for k, v in baselines["test"].items()))
    print(f"  oracle actions: exact ADE {oracle['test']['exact']['ade_m']:.1e} m (must be ~0); "
          f"clipped {oracle['test']['clipped_to_envelope']['ade_m']:.4f} m")

    def pack(name):
        sp = splits[name]
        n = len(sp)
        img = to_tensor(img_std(sp.img_feat), device) if cfg.uses_image else None
        z = to_tensor(text.z(sp), device) if (text and sp.has_text) else None
        return {
            "img": img, "img_t": img if img is not None else torch.zeros(n, 1, device=device),
            "hist": to_tensor(hist_std(sp.hist), device),
            "z": z, "z_t": z if z is not None else torch.zeros(n, 1, device=device),
            "v0": to_tensor(sp.v0, device),
            "prior_pre": to_tensor(pre_tanh(prior_actions(prior, sp.v0, sp.a0, kappa0[name])), device),
            "traj7": to_tensor(sp.traj7, device),
            "gt_act": to_tensor(clip_actions(gt_act[name]), device),
        }

    P = {k: pack(k) for k in splits}
    T = P["train"]
    loader = DataLoader(TensorDataset(T["img_t"], T["hist"], T["z_t"], T["v0"], T["prior_pre"],
                                      T["traj7"], T["gt_act"]),
                        batch_size=cfg.batch_size, shuffle=True,
                        generator=torch.Generator().manual_seed(seed))

    model = WorldModelVLA(cfg, img_dim=(train.img_feat.shape[1] if cfg.uses_image else 0),
                          hist_feat_dim=train.hist.shape[2],
                          latent_dim=(text.latent.dim if text else 0)).to(device)
    size = parameter_report(model)
    print(f"  model: {size['total_params']:,} trainable params; {size['by_module']}")

    res_params = list(model.residual_head.parameters()) if model.residual_head else []
    res_ids = {id(p) for p in res_params}
    opt = torch.optim.AdamW(
        [{"params": [p for p in model.parameters() if id(p) not in res_ids],
          "weight_decay": cfg.weight_decay},
         {"params": res_params, "weight_decay": cfg.residual_weight_decay}], lr=cfg.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.epochs)
    stopper = EarlyStopper(cfg.early_stop_patience, cfg.early_stop_min_delta)
    act_scale = to_tensor(np.maximum(clip_actions(gt_act["train"]).std(axis=(0, 1)), 1e-3), device)

    # check: the untrained head reproduces the prior
    with torch.no_grad():
        model.eval()
        _, t7, _, _ = model.predict(P["val"]["img"], P["val"]["hist"], P["val"]["z"],
                                    P["val"]["v0"], P["val"]["prior_pre"], cfg.flow_steps)
        d0 = float(np.linalg.norm(t7.cpu().numpy()[:, 1:] - splits["val"].traj, axis=-1).mean())
    print(f"  untrained model val ADE {d0:.4f} == prior '{prior}' val ADE "
          f"{baselines['val'][prior]['ade_m']:.4f}")

    log = []
    for epoch in range(cfg.epochs):
        model.train()
        t0 = time.time()
        fl, tl, al = [], [], []
        pretrain = epoch < cfg.pretrain_epochs
        for bi, bh, bz, bv, bp, btr, bga in loader:
            cond, fcond = model.encode(bi if cfg.uses_image else None, bh)
            loss = 0.0
            if model.uses_flow:
                loss_flow = model.flow_loss(fcond, bz)
                loss = loss + cfg.w_flow * loss_flow
                fl.append(loss_flow.item())
            z_in = model.text_latent_for_head(fcond, bz, cfg.flow_steps, training=True,
                                              p_sampled=cfg.p_sampled_z)
            actions, residual = model.act(cond, z_in, bp)
            if pretrain:
                la = F.mse_loss(actions / act_scale, bga / act_scale)
                loss = loss + la
                al.append(la.item())
            else:
                traj = model.roll(bv, actions, residual)
                lt = F.smooth_l1_loss(traj[:, 1:], btr[:, 1:])
                da = actions[:, 1:, 0] - actions[:, :-1, 0]
                dk = actions[:, 1:, 1] - actions[:, :-1, 1]
                loss = loss + cfg.w_traj * lt + cfg.w_jerk * (da ** 2).mean() \
                    + cfg.w_dkappa * (dk ** 2).mean()
                tl.append(lt.item())
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            opt.step()
        sched.step()

        model.eval()
        with torch.no_grad():
            g = torch.Generator(device=device).manual_seed(seed + epoch)
            V = P["val"]
            _, t7, _, _ = model.predict(V["img"], V["hist"], V["z"], V["v0"], V["prior_pre"],
                                        cfg.flow_steps, g)
            d = np.linalg.norm(t7.cpu().numpy()[:, 1:] - splits["val"].traj, axis=-1)
            val_ade, val_fde = float(d.mean()), float(d[:, -1].mean())
        rec = {"epoch": epoch, "val_ade": val_ade, "val_fde": val_fde, "pretrain": pretrain}
        if fl:
            rec["flow_loss"] = float(np.mean(fl))
        if tl:
            rec["traj_loss"] = float(np.mean(tl))
        if al:
            rec["action_loss"] = float(np.mean(al))
        log.append(rec)
        if epoch % 5 == 0 or epoch == cfg.epochs - 1:
            print(f"  epoch {epoch:3d}  " + (f"flow={rec['flow_loss']:.4f}  " if fl else "")
                  + (f"traj={rec['traj_loss']:.4f}  " if tl else f"action={rec['action_loss']:.4f}  ")
                  + f"val_ADE={val_ade:.4f}  val_FDE={val_fde:.4f}  [{time.time() - t0:.1f}s]")
        if not pretrain and stopper.step(val_ade, epoch, model.state_dict()):
            print(f"  early stopping at epoch {epoch} (best val_ADE={stopper.best:.4f} "
                  f"at epoch {stopper.best_epoch})")
            break
    if stopper.best_state is not None:
        model.load_state_dict(stopper.best_state)

    report = {
        "family": "wm", "ablation": cfg.ablation, "seed": seed,
        "text_field": cfg.text_field if cfg.uses_text else None,
        "what_it_sees": {"image": cfg.uses_image, "past_can_window": True,
                         "caption_into_head": ("ground truth (oracle)" if cfg.gt_text_input else
                                               "generated by the flow head" if cfg.uses_flow
                                               else "none")},
        "trajectory_is_produced_by": "differentiable kinematic rollout of predicted (a, kappa)"
                                     + (" + gated learned residual" if cfg.use_residual else ""),
        "model_size": size, "dynamics_self_test": self_test(),
        "prior": prior, "prior_requested": cfg.prior,
        "measured_kappa_r2_first_step": yaw_r2, "identified_steering_map": smap,
        "dt_s": DT, "n_steps": N_STEPS, "horizon_s": HORIZON_S,
        "best_epoch": stopper.best_epoch, "history_log": log, "data": info,
        "baselines": baselines, "oracle_actions": oracle,
    }
    if text:
        report["pca"] = pca_report(text.latent, {k: v.cap_emb for k, v in splits.items()
                                                 if v.has_text})

    print("\n  FINAL EVALUATION (best checkpoint, epoch %d)" % stopper.best_epoch)
    for name, sp in splits.items():
        if cfg.gt_text_input and not sp.has_text:
            continue
        r = evaluate(model, cfg, P[name], sp, text, device, sweep=(name == "test"))
        report[name] = r
        print(f"  [{name}] n={r['n']}")
        print_traj(r)
        mk = f"min_ade_k{cfg.n_samples_eval}"
        if mk in r:
            print(f"    min-ADE(K={cfg.n_samples_eval}) {r[mk]:.4f}")
        if "text" in r:
            print_text(r["text"])
        best = min(baselines[name].items(), key=lambda kv: kv[1]["ade_m"])
        print(f"    best 0-param baseline: {best[0].upper()} ADE {best[1]['ade_m']:.4f}   "
              + (f"residual carries {r['actions']['residual']['mean_abs_m']:.4f} m"
                 if "residual" in r["actions"] else ""))

    ckpt = {"model_state": model.state_dict(), "config": {k: str(v) for k, v in vars(cfg).items()},
            "hist_mean": hist_std.mean, "hist_std": hist_std.std, "prior": prior}
    if img_std:
        ckpt |= {"img_mean": img_std.mean, "img_std": img_std.std}
    if text:
        ckpt |= {"pca_mean": text.latent.mean, "pca_basis": text.latent.basis,
                 "pca_scale": text.latent.scale, "mean_emb": text.mean_emb}
    torch.save(ckpt, out / "model.pt")
    (out / "report.json").write_text(json.dumps(report, indent=2, default=str))
    print(f"  -> {out / 'report.json'}")
    return report


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ablation", default="b_vla_diffusion", choices=list(WM_ABLATIONS))
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--prior", default=None, choices=["auto", "none", *PRIOR_NAMES])
    ap.add_argument("--no-residual", action="store_true")
    ap.add_argument("--pretrain-epochs", type=int, default=None)
    add_common_args(ap)
    args = ap.parse_args()

    if args.self_test:
        print(json.dumps(self_test(), indent=2))
        return

    todo = list(WM_ABLATIONS) if args.all else [args.ablation]
    for ab in todo:
        reports = []
        for seed in args.seeds:
            cfg = apply_common_args(Config(family="wm", ablation=ab), args)
            if args.prior:
                cfg.prior = args.prior
            if args.no_residual:
                cfg.use_residual = False
            if args.pretrain_epochs is not None:
                cfg.pretrain_epochs = args.pretrain_epochs
            reports.append(run_one(cfg, args, seed))
        split_names = [k for k in ("train", "val", "test", "nusc_val") if k in reports[0]]
        s = write_summary(run_dir(cfg, 0).parent, reports, split_names,
                          {"family": "wm", "ablation": ab,
                           "text_field": cfg.text_field if cfg.uses_text else None,
                           "seeds": args.seeds, "prior": reports[0]["prior"],
                           "residual": cfg.use_residual})
        t = s.get("test", {})
        if t:
            print(f"\n  == {ab}: test ADE {t['ade_m']['mean']:.4f} +- {t['ade_m']['std']:.4f} "
                  f"over {s['n_seeds']} seed(s)")
    print("\nrun `python compare.py` for the tables")


if __name__ == "__main__":
    main()
