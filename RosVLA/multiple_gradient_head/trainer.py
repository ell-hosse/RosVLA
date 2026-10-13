"""Training loops, one per grad-mode.

train_fold mirrors ../velocity_head/run_cv.py's train_fold with an extra
`mode` argument; shared/joint reproduces the original loop.
"""
from __future__ import annotations

import random

import torch
import torch.nn.functional as Fn

from heads import (
    conditions, phase_a_modules, phase_b_modules, shared_parameters,
)
from modes import (
    RunningCosine, pcgrad_backward, set_requires_grad, split_epoch_budget,
    trainable,
)


def _control_losses(model, cond_ctrl, z_in, bv, bw, ba, class_weight):
    v_pred, w_pred, logits = model.head(cond_ctrl, z_in)
    return (
        Fn.smooth_l1_loss(v_pred, bv),
        Fn.smooth_l1_loss(w_pred, bw),
        Fn.cross_entropy(logits, ba, weight=class_weight),
    )


def _draw_z(model, cond_flow, bz, p_sampled_z, cfg):
    """Latent fed to the control head: a flow sample or the ground truth."""
    if torch.rand(()) < p_sampled_z:
        with torch.no_grad():
            return model.flow.sample(cond_flow, cfg.flow_steps)
    return bz


def _flow_loss(model, cond_flow, bz):
    """Rectified-flow loss."""
    b = bz.shape[0]
    s = torch.rand(b, 1)
    eps = torch.randn_like(bz)
    z_s = (1 - s) * eps + s * bz
    return Fn.mse_loss(model.flow(z_s, s.squeeze(1), cond_flow), bz - eps)


# joint / pcgrad
def _train_joint(cfg, model, tensors, class_weight, seed, mode, p_sampled_z):
    torch.manual_seed(seed)
    img, tel, z_true, v_true, w_true, action = tensors
    n = len(img)

    params = list(model.parameters())
    opt = torch.optim.AdamW(params, lr=cfg.lr, weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.epochs)

    shared = shared_parameters(model, mode)
    probe = RunningCosine(stride=10) if (mode.probe_grad_cosine and shared) else None
    rng = random.Random(seed)

    model.train()
    last, cosine_log, n_proj_total, n_pairs_total = {}, [], 0, 0
    for _ in range(cfg.epochs):
        perm = torch.randperm(n)
        acc = {"flow": 0.0, "v": 0.0, "w": 0.0, "act": 0.0}
        nb = 0
        if probe:
            probe.reset()
        for start in range(0, n, cfg.batch_size):
            idx = perm[start:start + cfg.batch_size]
            bi, bt, bz = img[idx], tel[idx], z_true[idx]
            bv, bw, ba = v_true[idx], w_true[idx], action[idx]

            cond_flow, cond_ctrl = conditions(model, bi, bt)
            loss_flow = _flow_loss(model, cond_flow, bz)
            z_in = _draw_z(model, cond_flow, bz, p_sampled_z, cfg)
            loss_v, loss_w, loss_act = _control_losses(
                model, cond_ctrl, z_in, bv, bw, ba, class_weight
            )

            term_flow = cfg.w_flow * loss_flow
            term_ctrl = (cfg.w_vel * loss_v + cfg.w_ang * loss_w
                         + cfg.w_action * loss_act)

            if probe:
                probe.maybe(term_flow, term_ctrl, shared)

            opt.zero_grad(set_to_none=True)
            if mode.grad_mode == "pcgrad":
                stats = pcgrad_backward([term_flow, term_ctrl], shared, params,
                                        rng=rng, grad_clip=1.0)
                n_proj_total += stats["n_projections"]
                n_pairs_total += stats["n_pairs"]
            else:
                (term_flow + term_ctrl).backward()
                torch.nn.utils.clip_grad_norm_(trainable(params), 1.0)
            opt.step()

            acc["flow"] += loss_flow.detach().item()
            acc["v"] += loss_v.detach().item()
            acc["w"] += loss_w.detach().item()
            acc["act"] += loss_act.detach().item()
            nb += 1
        sched.step()
        last = {k: round(x / max(nb, 1), 5) for k, x in acc.items()}
        if probe:
            cosine_log.append(probe.report())

    log = {"train_loss": last, "epochs_stage_a": None, "epochs_stage_b": None}
    if cosine_log:
        log["grad_cosine_first_epoch"] = cosine_log[0]
        log["grad_cosine_last_epoch"] = cosine_log[-1]
        log["grad_cosine_mean_over_training"] = round(
            sum(c["grad_cosine_mean"] for c in cosine_log) / len(cosine_log), 4
        )
        log["frac_conflicting_over_training"] = round(
            sum(c["frac_conflicting_batches"] for c in cosine_log) / len(cosine_log), 4
        )
    if mode.grad_mode == "pcgrad":
        log["pcgrad_projection_rate"] = round(
            n_proj_total / max(n_pairs_total, 1), 4
        )
    return log


# staged
def _train_staged(cfg, model, tensors, class_weight, seed, mode, p_sampled_z):
    """Phase A trains the flow, phase B the control head on the frozen flow.

    The epoch budget is split between the two phases.
    """
    torch.manual_seed(seed)
    img, tel, z_true, v_true, w_true, action = tensors
    n = len(img)
    a_epochs, b_epochs = split_epoch_budget(cfg.epochs, mode.stage_a_frac)

    a_mods = phase_a_modules(model, mode)
    b_mods = phase_b_modules(model, mode)

    # phase A
    set_requires_grad(model.children(), False)
    set_requires_grad(a_mods, True)
    a_params = trainable(model.parameters())
    opt_a = torch.optim.AdamW(a_params, lr=cfg.lr, weight_decay=cfg.weight_decay)
    sched_a = torch.optim.lr_scheduler.CosineAnnealingLR(opt_a, T_max=a_epochs)

    model.train()
    flow_last = 0.0
    for _ in range(a_epochs):
        perm = torch.randperm(n)
        tot, nb = 0.0, 0
        for start in range(0, n, cfg.batch_size):
            idx = perm[start:start + cfg.batch_size]
            cond_flow, _ = conditions(model, img[idx], tel[idx])
            loss_flow = _flow_loss(model, cond_flow, z_true[idx])
            opt_a.zero_grad(set_to_none=True)
            (cfg.w_flow * loss_flow).backward()
            torch.nn.utils.clip_grad_norm_(a_params, 1.0)
            opt_a.step()
            tot += loss_flow.detach().item()
            nb += 1
        sched_a.step()
        flow_last = tot / max(nb, 1)

    # freeze the flow head and the trunk that feeds it
    set_requires_grad(a_mods, False)
    model.flow.eval()
    set_requires_grad(b_mods, True)

    # phase B
    b_params = trainable(model.parameters())
    opt_b = torch.optim.AdamW(b_params, lr=cfg.lr, weight_decay=cfg.weight_decay)
    sched_b = torch.optim.lr_scheduler.CosineAnnealingLR(opt_b, T_max=b_epochs)

    last = {"flow": round(flow_last, 5), "v": 0.0, "w": 0.0, "act": 0.0}
    for _ in range(b_epochs):
        perm = torch.randperm(n)
        acc = {"v": 0.0, "w": 0.0, "act": 0.0}
        nb = 0
        for start in range(0, n, cfg.batch_size):
            idx = perm[start:start + cfg.batch_size]
            bi, bt, bz = img[idx], tel[idx], z_true[idx]
            bv, bw, ba = v_true[idx], w_true[idx], action[idx]

            with torch.no_grad():
                cond_flow = (model.flow_condition(bi, bt)
                             if hasattr(model, "flow_condition")
                             else model.condition(bi, bt))
            cond_ctrl = model.condition(bi, bt)
            z_in = _draw_z(model, cond_flow, bz, p_sampled_z, cfg)

            loss_v, loss_w, loss_act = _control_losses(
                model, cond_ctrl, z_in, bv, bw, ba, class_weight
            )
            loss = (cfg.w_vel * loss_v + cfg.w_ang * loss_w
                    + cfg.w_action * loss_act)
            opt_b.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(b_params, 1.0)
            opt_b.step()

            acc["v"] += loss_v.detach().item()
            acc["w"] += loss_w.detach().item()
            acc["act"] += loss_act.detach().item()
            nb += 1
        sched_b.step()
        last.update({k: round(x / max(nb, 1), 5) for k, x in acc.items()})

    # unfreeze everything before saving
    set_requires_grad(model.children(), True)
    return {
        "train_loss": last,
        "epochs_stage_a": a_epochs,
        "epochs_stage_b": b_epochs,
        "stage_a_params": sum(p.numel() for m in a_mods for p in m.parameters()),
        "stage_b_params": sum(p.numel() for m in b_mods for p in m.parameters()),
    }


def train_fold(cfg, model, tensors, class_weight, seed, mode):
    p = mode.resolved_p_sampled_z(cfg.p_sampled_z)
    fn = _train_staged if mode.is_staged else _train_joint
    return fn(cfg, model, tensors, class_weight, seed, mode, p)
