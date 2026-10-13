"""Model variants and parameter groups for the velocity-head VLA.

SplitTrunkVelocityVLA extends VelocityVLA with a separate conditioning trunk
for the flow head. With --trunk shared the original VelocityVLA is used.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from model import VelocityVLA          # ../velocity_head/model.py


class SplitTrunkVelocityVLA(VelocityVLA):
    """VelocityVLA with a separate image/telemetry projection for the flow head.

    condition() is the control trunk used by the velocity/action head and
    flow_condition() is the flow head's own trunk; they share no parameters.
    This adds one extra copy of the projections (~74k parameters).
    """

    def __init__(self, img_dim: int, tel_dim: int, latent_dim: int, cfg):
        super().__init__(img_dim, tel_dim, latent_dim, cfg)
        self.flow_img_proj = nn.Sequential(
            nn.Linear(img_dim, cfg.img_proj_dim), nn.SiLU()
        )
        self.flow_tel_proj = nn.Sequential(
            nn.Linear(tel_dim, cfg.tel_proj_dim), nn.SiLU()
        )

    def flow_condition(self, img, tel):
        return torch.cat([self.flow_img_proj(img), self.flow_tel_proj(tel)], dim=-1)

    @torch.no_grad()
    def predict(self, img, tel, steps: int, generator=None):
        """Returns z, v_chunk, w_chunk, action_logits, as in the base class."""
        z = self.flow.sample(self.flow_condition(img, tel), steps, generator=generator)
        v, w, logits = self.head(self.condition(img, tel), z)
        return z, v, w, logits


def build_model(img_dim: int, tel_dim: int, latent_dim: int, cfg, mode):
    cls = SplitTrunkVelocityVLA if mode.is_split else VelocityVLA
    return cls(img_dim, tel_dim, latent_dim, cfg)


# parameter groups
def conditions(model, img, tel):
    """Return (cond_for_flow, cond_for_control).

    With a shared trunk both are the same tensor, so one backward pass carries
    both losses into the trunk.
    """
    ctrl = model.condition(img, tel)
    flow = model.flow_condition(img, tel) if hasattr(model, "flow_condition") else ctrl
    return flow, ctrl


def control_trunk_modules(model, mode) -> list:
    return [model.img_proj, model.tel_proj]


def flow_trunk_modules(model, mode) -> list:
    if mode.is_split:
        return [model.flow_img_proj, model.flow_tel_proj]
    return [model.img_proj, model.tel_proj]


def shared_parameters(model, mode) -> list:
    """Parameters both objectives write to. Empty when the trunk is split."""
    if mode.is_split:
        return []
    return [p for m in (model.img_proj, model.tel_proj) for p in m.parameters()]


def phase_a_modules(model, mode) -> list:
    """Parameters trained in phase A: the flow head and its trunk."""
    return flow_trunk_modules(model, mode) + [model.flow]


def phase_b_modules(model, mode) -> list:
    """Parameters trained in phase B, after the flow is frozen.

    With a split trunk this includes the control trunk; with a shared trunk only
    the control head itself.
    """
    if mode.is_split:
        return control_trunk_modules(model, mode) + [model.head]
    return [model.head]
