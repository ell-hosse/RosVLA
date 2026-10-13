"""Training loops for the CoVLA world-model VLA (ablation b_vla_diffusion).

shared/joint with the config's p_sampled_z reproduces ../world_model_head/
train.py: action pretraining, jerk / d-kappa penalties, the two-group
optimiser and early stopping on validation ADE after pretraining.

Staged mode:

    phase A   trunk + flow head on the flow loss, early-stopped on val flow loss
    freeze    flow head and its trunk
    phase B   action head (+ residual head, + control trunk if split) with a
              sampled latent at every step (p_sampled_z = 1.0)

Action pretraining happens at the start of phase B.
"""
from __future__ import annotations

import random
import time

import numpy as np
import torch
import torch.nn.functional as F

from dynamics import rollout                      # ../world_model_head/dynamics.py
from metrics import ade_fde                       # ../world_model_head/metrics.py
from modes import (
    RunningCosine, pcgrad_backward, set_requires_grad, split_epoch_budget, trainable,
)
from utils import EarlyStopper                    # ../world_model_head/utils.py
from wm_heads import (
    conditions, flow_cond_only, make_optimiser, phase_a_modules, phase_b_modules,
)


# losses (same as the baseline)
def _flow_loss(model, cond_flow, bz):
    b = bz.shape[0]
    s = torch.rand(b, 1, device=bz.device)
    eps = torch.randn_like(bz)
    z_s = (1 - s) * eps + s * bz
    return F.mse_loss(model.flow(z_s, s.squeeze(1), cond_flow), bz - eps)


def _draw_z(model, cond_flow, bz, p_sampled_z, cfg):
    if torch.rand(()) < p_sampled_z:
        with torch.no_grad():
            return model.flow.sample(cond_flow, cfg.flow_steps)
    return bz


def _control_loss(cfg, model, cond_ctrl, z_in, bp, bv, btr, bga, act_scale, pretrain):
    """Action-side loss. Returns (loss, kind, value) for logging."""
    actions, residual = model.act(cond_ctrl, z_in, bp)
    if pretrain:
        # supervise on the inverse-dynamics actions
        loss = F.mse_loss(actions / act_scale, bga / act_scale)
        return loss, "action", float(loss.detach())
    traj = rollout(bv, actions)
    if residual is not None:
        traj = traj + residual
    loss_traj = F.smooth_l1_loss(traj, btr)
    da = actions[:, 1:, 0] - actions[:, :-1, 0]
    dk = actions[:, 1:, 1] - actions[:, :-1, 1]
    loss_smooth = cfg.w_jerk * (da ** 2).mean() + cfg.w_dkappa * (dk ** 2).mean()
    return cfg.w_traj * loss_traj + loss_smooth, "traj", float(loss_traj.detach())


# validation
@torch.no_grad()
def _val_ade(model, cfg, VP, val_traj, device, epoch):
    model.eval()
    gen = torch.Generator(device=device).manual_seed(cfg.seed + epoch)
    _, traj_hat, _, _ = model.predict(
        VP["img"], VP["hist"], VP["v0"], VP["prior_pre"], cfg.flow_steps, gen
    )
    ade, fde = ade_fde(traj_hat.cpu().numpy(), val_traj)
    model.train()
    return ade, fde


@torch.no_grad()
def _val_flow_loss(model, VP, z_va, batch: int = 4096):
    """Validation flow loss at fixed flow times and noise, comparable across epochs."""
    model.eval()
    img, hist = VP["img"], VP["hist"]
    tot, n = 0.0, 0
    for i in range(0, len(hist), batch):
        bh, bz = hist[i:i + batch], z_va[i:i + batch]
        bi = img[i:i + batch] if img is not None else None
        cond = (model.flow_condition(bi, bh) if hasattr(model, "flow_condition")
                else model.condition(bi, bh))
        g = torch.Generator(device=bz.device).manual_seed(12345)
        eps = torch.randn(bz.shape, device=bz.device, generator=g)
        s = torch.linspace(0.0, 1.0, len(bz), device=bz.device).unsqueeze(1)
        z_s = (1 - s) * eps + s * bz
        tot += float(F.mse_loss(model.flow(z_s, s.squeeze(1), cond), bz - eps,
                                reduction="sum"))
        n += bz.numel()
    model.train()
    return tot / max(n, 1)


# joint / pcgrad
def _train_joint(cfg, model, loader, VP, val_traj, z_va, act_scale, device, mode,
                 p_sampled_z, shared_params, verbose=True):
    params = list(model.parameters())
    opt = make_optimiser(cfg, model)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.epochs)
    stopper = EarlyStopper(cfg.early_stop_patience, cfg.early_stop_min_delta)

    probe = RunningCosine(stride=25) if (mode.probe_grad_cosine and shared_params) else None
    rng = random.Random(cfg.seed)
    history, cosine_log = [], []
    n_proj = n_pairs = 0

    for epoch in range(cfg.epochs):
        model.train()
        t0 = time.time()
        pretrain = epoch < cfg.pretrain_epochs
        flow_losses, ctrl_losses = [], []
        if probe:
            probe.reset()

        for bi, bh, bv, bp, btr, bga, bz in loader:
            img_in = bi if cfg.uses_image else None
            cond_flow, cond_ctrl = conditions(model, img_in, bh)

            term_flow = cfg.w_flow * _flow_loss(model, cond_flow, bz)
            z_in = _draw_z(model, cond_flow, bz, p_sampled_z, cfg)
            term_ctrl, kind, val = _control_loss(
                cfg, model, cond_ctrl, z_in, bp, bv, btr, bga, act_scale, pretrain
            )

            if probe:
                probe.maybe(term_flow, term_ctrl, shared_params)

            opt.zero_grad(set_to_none=True)
            if mode.grad_mode == "pcgrad":
                st = pcgrad_backward([term_flow, term_ctrl], shared_params, params,
                                     rng=rng, grad_clip=cfg.grad_clip)
                n_proj += st["n_projections"]
                n_pairs += st["n_pairs"]
            else:
                (term_flow + term_ctrl).backward()
                torch.nn.utils.clip_grad_norm_(trainable(params), cfg.grad_clip)
            opt.step()

            flow_losses.append(float(term_flow.detach()) / max(cfg.w_flow, 1e-12))
            ctrl_losses.append(val)
        sched.step()

        val_ade, val_fde = _val_ade(model, cfg, VP, val_traj, device, epoch)
        rec = {"epoch": epoch, "phase": "joint", "pretrain": bool(pretrain),
               "flow_loss": float(np.mean(flow_losses)),
               f"{kind}_loss": float(np.mean(ctrl_losses)),
               "val_ade": val_ade, "val_fde": val_fde}
        if probe:
            rec.update(probe.report())
            cosine_log.append(probe.report())
        history.append(rec)
        if verbose:
            extra = (f"  cos={rec['grad_cosine_mean']:+.3f}"
                     if "grad_cosine_mean" in rec else "")
            tag = " (pretrain)" if pretrain else ""
            print(f"  epoch {epoch:3d}  flow={rec['flow_loss']:.4f}  "
                  f"{kind}={np.mean(ctrl_losses):.4f}{tag}  val_ADE={val_ade:.4f}"
                  f"{extra}  [{time.time() - t0:.1f}s]")

        # early stopping starts after the warm start, as in the baseline
        if not pretrain and stopper.step(val_ade, epoch, model.state_dict()):
            if verbose:
                print(f"  early stop at epoch {epoch} "
                      f"(best val_ADE={stopper.best:.4f} @ {stopper.best_epoch})")
            break

    if stopper.best_state is not None:
        model.load_state_dict(stopper.best_state)

    log = {"history_log": history, "best_epoch": stopper.best_epoch,
           "best_val_ade": round(stopper.best, 4),
           "epochs_stage_a": None, "epochs_stage_b": None}
    if cosine_log:
        log["grad_cosine_mean_over_training"] = round(
            float(np.mean([c["grad_cosine_mean"] for c in cosine_log])), 4)
        log["frac_conflicting_over_training"] = round(
            float(np.mean([c["frac_conflicting_batches"] for c in cosine_log])), 4)
    if mode.grad_mode == "pcgrad":
        log["pcgrad_projection_rate"] = round(n_proj / max(n_pairs, 1), 4)
    return log


# staged
def _train_staged(cfg, model, loader, VP, val_traj, z_va, act_scale, device, mode,
                  p_sampled_z, shared_params, verbose=True):
    a_epochs, b_epochs = split_epoch_budget(cfg.epochs, mode.stage_a_frac)
    a_mods = phase_a_modules(model, mode)
    b_mods = phase_b_modules(model, mode)
    history = []

    # phase A
    set_requires_grad(model.children(), False)
    set_requires_grad(a_mods, True)
    opt_a = make_optimiser(cfg, model, a_mods)
    a_params = trainable(model.parameters())
    sched_a = torch.optim.lr_scheduler.CosineAnnealingLR(opt_a, T_max=a_epochs)
    stop_a = EarlyStopper(cfg.early_stop_patience, cfg.early_stop_min_delta)

    if verbose:
        print(f"  --- phase A: flow head + its trunk, {a_epochs} epochs max ---")
    for epoch in range(a_epochs):
        model.train()
        t0 = time.time()
        losses = []
        for bi, bh, _bv, _bp, _btr, _bga, bz in loader:
            img_in = bi if cfg.uses_image else None
            cond_flow, _ = conditions(model, img_in, bh)
            loss_flow = _flow_loss(model, cond_flow, bz)
            opt_a.zero_grad(set_to_none=True)
            (cfg.w_flow * loss_flow).backward()
            torch.nn.utils.clip_grad_norm_(a_params, cfg.grad_clip)
            opt_a.step()
            losses.append(loss_flow.item())
        sched_a.step()
        vfl = _val_flow_loss(model, VP, z_va)
        history.append({"epoch": epoch, "phase": "A",
                        "flow_loss": float(np.mean(losses)), "val_flow_loss": vfl})
        if verbose:
            print(f"  A {epoch:3d}  flow={np.mean(losses):.4f}  val_flow={vfl:.4f}  "
                  f"[{time.time() - t0:.1f}s]")
        if stop_a.step(vfl, epoch, model.state_dict()):
            if verbose:
                print(f"     early stop (best val_flow={stop_a.best:.4f} "
                      f"@ {stop_a.best_epoch})")
            break
    if stop_a.best_state is not None:
        model.load_state_dict(stop_a.best_state)

    # freeze the flow head and the trunk that feeds it
    set_requires_grad(a_mods, False)
    model.flow.eval()
    set_requires_grad(b_mods, True)

    # phase B
    opt_b = make_optimiser(cfg, model, b_mods)
    b_params = trainable(model.parameters())
    sched_b = torch.optim.lr_scheduler.CosineAnnealingLR(opt_b, T_max=b_epochs)
    stop_b = EarlyStopper(cfg.early_stop_patience, cfg.early_stop_min_delta)

    if verbose:
        print(f"  --- phase B: action head on a FROZEN flow, {b_epochs} epochs max, "
              f"p_sampled_z={p_sampled_z}, {cfg.pretrain_epochs} pretrain epochs ---")
    for epoch in range(b_epochs):
        model.train()
        model.flow.eval()
        t0 = time.time()
        pretrain = epoch < cfg.pretrain_epochs
        losses = []
        kind = "traj"
        for bi, bh, bv, bp, btr, bga, bz in loader:
            img_in = bi if cfg.uses_image else None
            cond_flow = flow_cond_only(model, img_in, bh)
            cond_ctrl = model.condition(img_in, bh)
            z_in = _draw_z(model, cond_flow, bz, p_sampled_z, cfg)
            loss, kind, val = _control_loss(
                cfg, model, cond_ctrl, z_in, bp, bv, btr, bga, act_scale, pretrain
            )
            opt_b.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(b_params, cfg.grad_clip)
            opt_b.step()
            losses.append(val)
        sched_b.step()

        val_ade, val_fde = _val_ade(model, cfg, VP, val_traj, device, epoch)
        history.append({"epoch": a_epochs + epoch, "phase": "B",
                        "pretrain": bool(pretrain),
                        f"{kind}_loss": float(np.mean(losses)),
                        "val_ade": val_ade, "val_fde": val_fde})
        if verbose:
            tag = " (pretrain)" if pretrain else ""
            print(f"  B {epoch:3d}  {kind}={np.mean(losses):.4f}{tag}  "
                  f"val_ADE={val_ade:.4f}  [{time.time() - t0:.1f}s]")
        if not pretrain and stop_b.step(val_ade, epoch, model.state_dict()):
            if verbose:
                print(f"     early stop (best val_ADE={stop_b.best:.4f} "
                      f"@ {stop_b.best_epoch})")
            break
    if stop_b.best_state is not None:
        model.load_state_dict(stop_b.best_state)

    set_requires_grad(model.children(), True)
    return {
        "history_log": history,
        "best_epoch": a_epochs + max(stop_b.best_epoch, 0),
        "best_val_ade": round(stop_b.best, 4),
        "best_val_flow_loss": round(stop_a.best, 6),
        "epochs_stage_a": a_epochs,
        "epochs_stage_b": b_epochs,
        "stage_a_params": sum(p.numel() for m in a_mods for p in m.parameters()),
        "stage_b_params": sum(p.numel() for m in b_mods for p in m.parameters()),
    }


def train(cfg, model, loader, VP, val_traj, z_va, act_scale, device, mode,
          shared_params, verbose=True):
    p = mode.resolved_p_sampled_z(cfg.p_sampled_z)
    fn = _train_staged if mode.is_staged else _train_joint
    return fn(cfg, model, loader, VP, val_traj, z_va, act_scale, device, mode, p,
              shared_params, verbose=verbose)
