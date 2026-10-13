r"""World-model VLA: one class for the three ablations.

    a_world_model_only          b_vla_diffusion              c_vla_no_text
    ------------------          ---------------              -------------
    hist -> GRU -> h            img -> proj -> f             img -> proj -> f
                                hist -> GRU -> h             hist -> GRU -> h
                                cond = [f, h]                cond = [f, h]
                                flow: noise -> z (PCA
                                       caption latent)
    cond = h                    action input = [cond, z]     action input = cond
              \                        |                            /
               \                       v                           /
                +------------> action head: 19 x (a, kappa) <-----+
                                       |
                          differentiable kinematic rollout   (dynamics.py)
                                       |
                          [+ learned per-step (dx, dy) residual]
                                       |
                            20 waypoints, next 2.85 s, metres

The action head predicts a correction to an analytic prior in pre-tanh space:

    a_k     = ACC_MAX   * tanh(raw_a_k + atanh(prior_a_k     / ACC_MAX))
    kappa_k = KAPPA_MAX * tanh(raw_k_k + atanh(prior_kappa_k / KAPPA_MAX))

Its output layer is zero-initialised, so the untrained model reproduces the
prior (cfg.prior, default `ca`).

The residual is scaled by a gate initialised to zero and has a stronger
weight decay (cfg.residual_weight_decay). metrics.action_report shows how much
displacement it carries.

The flow head (ablation b) is the same as in ../vla_trajecotry_head/model.py.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn

from config import ACC_MAX, KAPPA_MAX, N_STEPS
from dynamics import rollout


def timestep_embedding(s: torch.Tensor, dim: int) -> torch.Tensor:
    half = dim // 2
    freqs = torch.exp(
        -math.log(1000.0) * torch.arange(half, device=s.device, dtype=torch.float32) / half
    )
    ang = s[:, None] * freqs[None] * 1000.0
    return torch.cat([torch.sin(ang), torch.cos(ang)], dim=-1)


class VehicleStateEncoder(nn.Module):
    """(B, 20, 3) past vehicle-state window -> (B, proj_dim)."""

    def __init__(self, in_dim: int, hidden_dim: int, proj_dim: int):
        super().__init__()
        self.gru = nn.GRU(in_dim, hidden_dim, batch_first=True)
        self.proj = nn.Sequential(nn.Linear(hidden_dim, proj_dim), nn.SiLU())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, h_n = self.gru(x)
        return self.proj(h_n.squeeze(0))


class FiLMResBlock(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim)
        self.fc2 = nn.Linear(dim, dim)
        self.norm = nn.LayerNorm(dim)
        self.act = nn.SiLU()

    def forward(self, x, scale, shift):
        h = self.norm(self.fc1(x))
        h = h * (1 + scale) + shift
        return x + self.fc2(self.act(h))


class FlowTextHead(nn.Module):
    """Rectified-flow generator for the PCA caption latent (ablation b)."""

    def __init__(self, latent_dim: int, cond_dim: int, hidden: int, n_blocks: int):
        super().__init__()
        self.latent_dim = latent_dim
        self.time_dim = 32
        self.hidden, self.n_blocks = hidden, n_blocks
        n_film = 2 * hidden * n_blocks
        self.inp = nn.Linear(latent_dim, hidden)
        self.blocks = nn.ModuleList(FiLMResBlock(hidden) for _ in range(n_blocks))
        self.out = nn.Linear(hidden, latent_dim)
        self.film_cond = nn.Sequential(
            nn.Linear(cond_dim, hidden), nn.SiLU(), nn.Linear(hidden, n_film),
        )
        self.film_time = nn.Linear(self.time_dim, n_film)

    def cond_film(self, cond):
        return self.film_cond(cond)

    def _split(self, film):
        p = film.view(-1, self.n_blocks, 2, self.hidden)
        return p[:, :, 0], p[:, :, 1]

    def velocity(self, z_s, s, cond_film):
        film = cond_film + self.film_time(timestep_embedding(s, self.time_dim))
        scale, shift = self._split(film)
        h = self.inp(z_s)
        for i, blk in enumerate(self.blocks):
            h = blk(h, scale[:, i], shift[:, i])
        return self.out(h)

    def forward(self, z_s, s, cond):
        return self.velocity(z_s, s, self.cond_film(cond))

    @torch.no_grad()
    def sample(self, cond, steps: int, generator=None):
        b = cond.shape[0]
        cond_film = self.cond_film(cond)
        z = torch.randn(b, self.latent_dim, device=cond.device, generator=generator)
        dt = 1.0 / steps
        for i in range(steps):
            s = torch.full((b,), i * dt, device=cond.device)
            z = z + dt * self.velocity(z, s, cond_film)
        return z


class ActionHead(nn.Module):
    """conditioning -> 19 bounded (a, kappa) pairs, as a correction to the prior."""

    def __init__(self, in_dim: int, hidden: int, n_layers: int):
        super().__init__()
        layers = [nn.Linear(in_dim, hidden), nn.SiLU()]
        for _ in range(max(0, n_layers - 1)):
            layers += [nn.Linear(hidden, hidden), nn.SiLU()]
        self.trunk = nn.Sequential(*layers)
        self.out = nn.Linear(hidden, N_STEPS * 2)
        # zero init: the untrained head reproduces the prior
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, feat: torch.Tensor, prior_pre: torch.Tensor) -> torch.Tensor:
        """prior_pre: (B, 19, 2) prior actions in pre-tanh space."""
        raw = self.out(self.trunk(feat)).view(-1, N_STEPS, 2)
        pre = raw + prior_pre
        a = ACC_MAX * torch.tanh(pre[..., 0])
        kappa = KAPPA_MAX * torch.tanh(pre[..., 1])
        return torch.stack([a, kappa], dim=-1)


class ResidualHead(nn.Module):
    """conditioning -> per-step (dx, dy) added after the rollout, scaled by a gate."""

    def __init__(self, in_dim: int, hidden: int):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, hidden), nn.SiLU(),
                                 nn.Linear(hidden, N_STEPS * 2))
        # Default init for the inner net and a zero gate: the residual starts at
        # zero but the gate still gets a gradient (zero-initialising both would not).
        self.gate = nn.Parameter(torch.zeros(1))

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        """Returns (B, 20, 2); waypoint 0 stays at the origin."""
        d = self.net(feat).view(-1, N_STEPS, 2) * self.gate
        pad = torch.zeros(d.shape[0], 1, 2, device=d.device, dtype=d.dtype)
        return torch.cat([pad, d], dim=1)


class WorldModelVLA(nn.Module):
    def __init__(self, ablation: str, img_dim: int, history_feat_dim: int,
                 latent_dim: int, cfg):
        super().__init__()
        self.ablation = ablation
        self.uses_image = ablation in ("b_vla_diffusion", "c_vla_no_text")
        self.uses_flow = ablation == "b_vla_diffusion"
        self.latent_dim = latent_dim if self.uses_flow else 0

        self.vehicle_encoder = VehicleStateEncoder(
            history_feat_dim, cfg.vehicle_hidden_dim, cfg.vehicle_proj_dim
        )
        cond_dim = cfg.vehicle_proj_dim
        if self.uses_image:
            self.img_proj = nn.Sequential(nn.Linear(img_dim, cfg.img_proj_dim), nn.SiLU())
            cond_dim += cfg.img_proj_dim
        self.cond_dim = cond_dim

        if self.uses_flow:
            self.flow = FlowTextHead(latent_dim, cond_dim, cfg.hidden_dim, cfg.n_flow_blocks)

        act_in = cond_dim + self.latent_dim
        self.action_head = ActionHead(act_in, cfg.hidden_dim, cfg.n_action_layers)
        self.residual_head = ResidualHead(act_in, cfg.hidden_dim) if cfg.use_residual else None

    def condition(self, img: torch.Tensor | None, hist: torch.Tensor) -> torch.Tensor:
        h = self.vehicle_encoder(hist)
        if not self.uses_image:
            return h
        return torch.cat([self.img_proj(img), h], dim=-1)

    def act(self, cond, z, prior_pre):
        feat = cond if z is None else torch.cat([cond, z], dim=-1)
        actions = self.action_head(feat, prior_pre)
        residual = self.residual_head(feat) if self.residual_head is not None else None
        return actions, residual

    def forward(self, img, hist, z, v0, prior_pre):
        """Returns (traj_m, actions, residual); traj_m is in metres."""
        cond = self.condition(img, hist)
        actions, residual = self.act(cond, z, prior_pre)
        traj = rollout(v0, actions)
        if residual is not None:
            traj = traj + residual
        return traj, actions, residual

    @torch.no_grad()
    def predict(self, img, hist, v0, prior_pre, steps: int = 4, generator=None):
        """Inference. For ablation b the caption latent is generated by the flow head."""
        cond = self.condition(img, hist)
        z = self.flow.sample(cond, steps, generator=generator) if self.uses_flow else None
        actions, residual = self.act(cond, z, prior_pre)
        traj = rollout(v0, actions)
        if residual is not None:
            traj = traj + residual
        return z, traj, actions, residual


def parameter_report(model: nn.Module) -> dict:
    per_part = {name: sum(p.numel() for p in m.parameters())
                for name, m in model.named_children()}
    total = sum(p.numel() for p in model.parameters())
    rep = {
        "total_params": total,
        "by_module": per_part,
        "size_fp32_mb": round(total * 4 / 1e6, 3),
        "size_fp16_mb": round(total * 2 / 1e6, 3),
        "rollout_params": 0,
        "rollout_note": "the kinematic rollout in dynamics.py has no parameters at all",
    }
    if getattr(model, "uses_flow", False):
        hoisted = sum(p.numel() for p in model.flow.film_cond.parameters())
        rep["flow_per_step_params"] = sum(p.numel() for p in model.flow.parameters()) - hoisted
        rep["flow_hoisted_params"] = hoisted
    return rep
