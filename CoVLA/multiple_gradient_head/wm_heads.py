"""Model variant and parameter groups for the CoVLA world-model head (ablation b).

SplitTrunkWorldModelVLA extends WorldModelVLA with a separate trunk for the
flow head; the action head, residual head, prior and rollout are unchanged.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from dynamics import rollout                          # ../world_model_head/dynamics.py
from model import VehicleStateEncoder, WorldModelVLA   # ../world_model_head/model.py


class SplitTrunkWorldModelVLA(WorldModelVLA):
    """WorldModelVLA (ablation b) with a separate trunk for the flow head.

    condition() is the control trunk used by the action and residual heads;
    flow_condition() is the flow head's own image projection and GRU.
    """

    def __init__(self, ablation: str, img_dim: int, history_feat_dim: int,
                 latent_dim: int, cfg):
        super().__init__(ablation, img_dim, history_feat_dim, latent_dim, cfg)
        if not self.uses_flow:
            raise ValueError(
                f"a split trunk only means something when there IS a flow head; "
                f"ablation {ablation!r} has none. Use b_vla_diffusion."
            )
        self.flow_vehicle_encoder = VehicleStateEncoder(
            history_feat_dim, cfg.vehicle_hidden_dim, cfg.vehicle_proj_dim
        )
        self.flow_img_proj = nn.Sequential(
            nn.Linear(img_dim, cfg.img_proj_dim), nn.SiLU()
        )

    def flow_condition(self, img: torch.Tensor, hist: torch.Tensor) -> torch.Tensor:
        return torch.cat(
            [self.flow_img_proj(img), self.flow_vehicle_encoder(hist)], dim=-1
        )

    @torch.no_grad()
    def predict(self, img, hist, v0, prior_pre, steps: int = 4, generator=None):
        """Returns z, traj, actions, residual, as in the base class."""
        z = self.flow.sample(self.flow_condition(img, hist), steps,
                             generator=generator)
        actions, residual = self.act(self.condition(img, hist), z, prior_pre)
        traj = rollout(v0, actions)
        if residual is not None:
            traj = traj + residual
        return z, traj, actions, residual


def build_model(ablation, img_dim, history_feat_dim, latent_dim, cfg, mode):
    cls = SplitTrunkWorldModelVLA if mode.is_split else WorldModelVLA
    return cls(ablation, img_dim, history_feat_dim, latent_dim, cfg)


# parameter groups
def conditions(model, img, hist):
    """Return (cond_for_flow, cond_for_control); the same tensor when trunk=shared."""
    ctrl = model.condition(img, hist)
    flow = model.flow_condition(img, hist) if hasattr(model, "flow_condition") else ctrl
    return flow, ctrl


def flow_cond_only(model, img, hist):
    """Flow conditioning under no_grad (phase B)."""
    with torch.no_grad():
        if hasattr(model, "flow_condition"):
            return model.flow_condition(img, hist)
        return model.condition(img, hist)


def control_trunk_modules(model, mode) -> list:
    return [model.img_proj, model.vehicle_encoder]


def flow_trunk_modules(model, mode) -> list:
    if mode.is_split:
        return [model.flow_img_proj, model.flow_vehicle_encoder]
    return [model.img_proj, model.vehicle_encoder]


def shared_parameters(model, mode) -> list:
    if mode.is_split:
        return []
    return [p for m in (model.img_proj, model.vehicle_encoder) for p in m.parameters()]


def phase_a_modules(model, mode) -> list:
    """Parameters trained in phase A: the flow head and its trunk."""
    return flow_trunk_modules(model, mode) + [model.flow]


def phase_b_modules(model, mode) -> list:
    """Parameters trained in phase B: action head, residual head and (split trunk only) the control trunk."""
    mods = [model.action_head]
    if model.residual_head is not None:
        mods.append(model.residual_head)
    if mode.is_split:
        mods = control_trunk_modules(model, mode) + mods
    return mods


def make_optimiser(cfg, model, modules=None):
    """AdamW with the baseline's two parameter groups (stronger decay on the residual head).

    Only currently trainable parameters are included.
    """
    if modules is None:
        pool = [p for p in model.parameters() if p.requires_grad]
    else:
        pool = [p for m in modules for p in m.parameters() if p.requires_grad]

    res_ids = {id(p) for p in (model.residual_head.parameters()
                               if model.residual_head is not None else [])}
    base = [p for p in pool if id(p) not in res_ids]
    res = [p for p in pool if id(p) in res_ids]

    groups = []
    if base:
        groups.append({"params": base, "weight_decay": cfg.weight_decay})
    if res:
        groups.append({"params": res, "weight_decay": cfg.residual_weight_decay})
    if not groups:
        raise ValueError("no trainable parameters were handed to the optimiser")
    return torch.optim.AdamW(groups, lr=cfg.lr)
