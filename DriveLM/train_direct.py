#!/usr/bin/env python3
"""DriveLM-nuScenes flow-matching VLA with a direct trajectory head.
Counterpart of CoVLA/vla_trajecotry_head and its ablation studies.

    image + past CAN window --flow matching--> PCA caption embedding
    image + past CAN window + generated caption embedding --> 6 future waypoints (3 s)

    python train_direct.py                                   # d1_flow_generated, seed 0
    python train_direct.py --all --seeds 0 1 2               # all ablations, 3 seeds
    python train_direct.py --ablation d3_no_text
    python train_direct.py --all --text-field behavior
    python train_direct.py --all --text-field caption_full

Ablations (see model.py):
    d1_flow_generated   trajectory head gets the generated caption latent
    d2_gt_caption       oracle: trajectory head gets the ground-truth latent. The
                        DriveLM behavior sentence is derived from the future path,
                        so this is an upper bound only.
    d3_no_text          no flow head and no caption
    d4_flow_state_only  flow head conditioned on the CAN window only

Standardisers, PCA and the retrieval bank are fitted on train. Early stopping
uses validation ADE with a generated latent.
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
from config import DIRECT_ABLATIONS, N_FUTURE, Config                     # noqa: E402
from data import load_splits                                               # noqa: E402
from latent import pca_report                                              # noqa: E402
from metrics import min_ade_fde, trajectory_report                         # noqa: E402
from model import DirectVLA, parameter_report                              # noqa: E402
from utils import EarlyStopper, Standardiser                               # noqa: E402


@torch.no_grad()
def evaluate(model, cfg, P, split, wp_std, text, device, sweep=False) -> dict:
    model.eval()
    img, hist, z = P["img"], P["hist"], P["z"]
    gen = torch.Generator(device=device).manual_seed(cfg.seed)
    z_hat, t_hat = model.predict(img, hist, z, cfg.flow_steps, gen)
    pred = wp_std.inverse(t_hat.cpu().numpy()).reshape(-1, N_FUTURE, 2)
    rep = trajectory_report(pred, split.traj)
    rep["subgroups"] = subgroup_report(pred, split)

    if model.uses_flow and not model.gt_text_input:
        samples = []
        for k in range(cfg.n_samples_eval):
            g = torch.Generator(device=device).manual_seed(1000 + k)
            _, t_k = model.predict(img, hist, z, cfg.flow_steps, g)
            samples.append(wp_std.inverse(t_k.cpu().numpy()).reshape(-1, N_FUTURE, 2))
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
                zs, ts = model.predict(img, hist, z, steps, g)
            p = wp_std.inverse(ts.cpu().numpy()).reshape(-1, N_FUTURE, 2)
            r = {"steps": steps, "ade_m": trajectory_report(p, split.traj)["ade_m"],
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
    print(f"# DIRECT / {cfg.ablation}   text_field={cfg.text_field if cfg.uses_text else '-'}"
          f"   seed={seed}   device={device}")
    print(f"#   image: {cfg.uses_image}   flow: {cfg.uses_flow}   GT caption into head: "
          f"{cfg.gt_text_input}   flow sees image: {cfg.uses_flow and not cfg.flow_cond_state_only}")
    print("#" * 78)

    splits, info = load_splits(cfg, force_images=args.rebuild_image_cache)
    train = splits["train"]

    # fit standardisers and PCA on train only
    img_std = Standardiser(train.img_feat) if cfg.uses_image else None
    hist_std = Standardiser(train.hist)
    wp_std = Standardiser(train.traj.reshape(len(train), -1))
    text = TextSide(train, cfg) if cfg.uses_text else None
    if text:
        print(f"  PCA text latent: dim={text.latent.dim} (from {text.latent.n_unique_train} unique "
              f"train texts), explained variance {text.latent.explained:.4f}")

    def pack(sp):
        n = len(sp)
        return {
            "img": to_tensor(img_std(sp.img_feat), device) if cfg.uses_image else None,
            "img_t": (to_tensor(img_std(sp.img_feat), device) if cfg.uses_image
                      else torch.zeros(n, 1, device=device)),
            "hist": to_tensor(hist_std(sp.hist), device),
            "z": (to_tensor(text.z(sp), device) if (text and sp.has_text) else None),
            "z_t": (to_tensor(text.z(sp), device) if (text and sp.has_text)
                    else torch.zeros(n, 1, device=device)),
            "traj": to_tensor(wp_std(sp.traj.reshape(n, -1)), device),
        }

    P = {k: pack(v) for k, v in splits.items()}
    loader = DataLoader(TensorDataset(P["train"]["img_t"], P["train"]["hist"],
                                      P["train"]["z_t"], P["train"]["traj"]),
                        batch_size=cfg.batch_size, shuffle=True,
                        generator=torch.Generator().manual_seed(seed))

    model = DirectVLA(cfg, img_dim=(train.img_feat.shape[1] if cfg.uses_image else 0),
                      hist_feat_dim=train.hist.shape[2],
                      latent_dim=(text.latent.dim if text else 0)).to(device)
    size = parameter_report(model)
    print(f"  model: {size['total_params']:,} trainable params; {size['by_module']}")

    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.epochs)
    stopper = EarlyStopper(cfg.early_stop_patience, cfg.early_stop_min_delta)

    log = []
    for epoch in range(cfg.epochs):
        model.train()
        t0 = time.time()
        fl, tl = [], []
        for bi, bh, bz, btr in loader:
            cond, fcond = model.encode(bi if cfg.uses_image else None, bh)
            loss = 0.0
            if model.uses_flow:
                loss_flow = model.flow_loss(fcond, bz)
                loss = loss + cfg.w_flow * loss_flow
                fl.append(loss_flow.item())
            z_in = model.text_latent_for_head(fcond, bz, cfg.flow_steps, training=True,
                                              p_sampled=cfg.p_sampled_z)
            loss_traj = F.smooth_l1_loss(model.traj_head(cond, z_in), btr)
            loss = loss + cfg.w_traj * loss_traj
            tl.append(loss_traj.item())
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            opt.step()
        sched.step()

        model.eval()
        with torch.no_grad():
            g = torch.Generator(device=device).manual_seed(seed + epoch)
            _, t_hat = model.predict(P["val"]["img"], P["val"]["hist"], P["val"]["z"],
                                     cfg.flow_steps, g)
            pv = wp_std.inverse(t_hat.cpu().numpy()).reshape(-1, N_FUTURE, 2)
            d = np.linalg.norm(pv - splits["val"].traj, axis=-1)
            val_ade, val_fde = float(d.mean()), float(d[:, -1].mean())
        rec = {"epoch": epoch, "traj_loss": float(np.mean(tl)), "val_ade": val_ade,
               "val_fde": val_fde}
        if fl:
            rec["flow_loss"] = float(np.mean(fl))
        log.append(rec)
        if epoch % 5 == 0 or epoch == cfg.epochs - 1:
            print(f"  epoch {epoch:3d}  " + (f"flow={rec['flow_loss']:.4f}  " if fl else "")
                  + f"traj={rec['traj_loss']:.4f}  val_ADE={val_ade:.4f}  val_FDE={val_fde:.4f}"
                  f"  [{time.time() - t0:.1f}s]")
        if stopper.step(val_ade, epoch, model.state_dict()):
            print(f"  early stopping at epoch {epoch} (best val_ADE={stopper.best:.4f} "
                  f"at epoch {stopper.best_epoch})")
            break
    model.load_state_dict(stopper.best_state)

    # report
    report = {
        "family": "direct", "ablation": cfg.ablation, "seed": seed,
        "text_field": cfg.text_field if cfg.uses_text else None,
        "what_it_sees": {"image": cfg.uses_image, "past_can_window": True,
                         "caption_into_head": ("ground truth (oracle)" if cfg.gt_text_input else
                                               "generated by the flow head" if cfg.uses_flow
                                               else "none"),
                         "flow_conditioned_on": (None if not cfg.uses_flow else
                                                 "state only" if cfg.flow_cond_state_only
                                                 else "image + state")},
        "model_size": size, "best_epoch": stopper.best_epoch, "history_log": log,
        "data": info,
        "baselines": {k: baselines_for(v) for k, v in splits.items()},
    }
    if text:
        report["pca"] = pca_report(text.latent, {k: v.cap_emb for k, v in splits.items()
                                                 if v.has_text})
    print("\n  FINAL EVALUATION (best checkpoint, epoch %d)" % stopper.best_epoch)
    for name, sp in splits.items():
        if cfg.gt_text_input and not sp.has_text:
            continue                       # the oracle needs a GT caption
        r = evaluate(model, cfg, P[name], sp, wp_std, text, device, sweep=(name == "test"))
        report[name] = r
        print(f"  [{name}] n={r['n']}")
        print_traj(r)
        mk = f"min_ade_k{cfg.n_samples_eval}"
        if mk in r:
            print(f"    min-ADE(K={cfg.n_samples_eval}) {r[mk]:.4f}")
        if "text" in r:
            print_text(r["text"])
        b = report["baselines"][name]
        best = min(b.items(), key=lambda kv: kv[1]["ade_m"])
        print(f"    best 0-param baseline: {best[0].upper()} ADE {best[1]['ade_m']:.4f}")

    ckpt = {"model_state": model.state_dict(), "config": {k: str(v) for k, v in vars(cfg).items()},
            "hist_mean": hist_std.mean, "hist_std": hist_std.std,
            "wp_mean": wp_std.mean, "wp_std": wp_std.std}
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
    ap.add_argument("--ablation", default="d1_flow_generated", choices=list(DIRECT_ABLATIONS))
    add_common_args(ap)
    args = ap.parse_args()

    todo = list(DIRECT_ABLATIONS) if args.all else [args.ablation]
    for ab in todo:
        reports = []
        for seed in args.seeds:
            cfg = apply_common_args(Config(family="direct", ablation=ab), args)
            reports.append(run_one(cfg, args, seed))
        split_names = [k for k in ("train", "val", "test", "nusc_val") if k in reports[0]]
        s = write_summary(run_dir(cfg, 0).parent, reports, split_names,
                          {"family": "direct", "ablation": ab,
                           "text_field": cfg.text_field if cfg.uses_text else None,
                           "seeds": args.seeds})
        t = s.get("test", {})
        if t:
            print(f"\n  == {ab}: test ADE {t['ade_m']['mean']:.4f} +- {t['ade_m']['std']:.4f} "
                  f"over {s['n_seeds']} seed(s)")
    print("\nrun `python compare.py` for the tables")


if __name__ == "__main__":
    main()
