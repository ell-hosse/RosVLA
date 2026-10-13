"""Models. Both families share the conditioning trunk and flow head of
CoVLA/vla_trajecotry_head/model.py and CoVLA/world_model_head/model.py.

    CAM_FRONT keyframe --frozen MobileNetV3-Small--> f (576) --proj--> 128
    past 2.85 s CAN window (20 x 4) --GRU--> h (64)
                                   cond = [proj(f), h]
                                          |
                    flow head (rectified flow): noise --Euler--> z  (PCA caption latent)
                                          |
     family "direct": trajectory head [cond, z] -> 6 (x, y) waypoints     (DirectVLA)
     family "wm"    : action head     [cond, z] -> 6 (a, kappa) -> kinematic rollout
                                                   (+ gated residual)     (WorldModelVLA)

Ablations
---------
  direct / d1_flow_generated   flow(cond) -> z_hat -> traj head  (CoVLA baseline)
  direct / d2_gt_caption       GT z -> traj head (oracle); flow trained for text metrics
  direct / d3_no_text          cond -> traj head; no flow, no caption
  direct / d4_flow_state_only  flow(h) -> z_hat; traj head gets [cond, z_hat]
  wm / a_world_model_only      h -> actions -> rollout (no image, no text)
  wm / b_vla_diffusion         flow(cond) -> z_hat -> actions -> rollout
  wm / c_vla_no_text           cond -> actions -> rollout
  wm / d_vla_gt_caption        GT z -> actions -> rollout (oracle)
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn

from config import ACC_MAX, KAPPA_MAX, N_FUTURE, N_STEPS
from dynamics import rollout


def timestep_embedding(s: torch.Tensor, dim: int) -> torch.Tensor:
    half = dim // 2
    freqs = torch.exp(-math.log(1000.0) *
                      torch.arange(half, device=s.device, dtype=torch.float32) / half)
    ang = s[:, None] * freqs[None] * 1000.0
    return torch.cat([torch.sin(ang), torch.cos(ang)], dim=-1)


class VehicleStateEncoder(nn.Module):
    """(B, 20, 4) past state window -> (B, proj_dim)."""

    def __init__(self, in_dim: int, hidden_dim: int, proj_dim: int):
        super().__init__()
        self.gru = nn.GRU(in_dim, hidden_dim, batch_first=True)
        self.proj = nn.Sequential(nn.Linear(hidden_dim, proj_dim), nn.SiLU())

    def forward(self, x):
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
    """Rectified-flow generator of the PCA caption latent, FiLM-conditioned."""

    def __init__(self, latent_dim: int, cond_dim: int, hidden: int, n_blocks: int):
        super().__init__()
        self.latent_dim = latent_dim
        self.time_dim = 32
        self.hidden, self.n_blocks = hidden, n_blocks
        n_film = 2 * hidden * n_blocks
        self.inp = nn.Linear(latent_dim, hidden)
        self.blocks = nn.ModuleList(FiLMResBlock(hidden) for _ in range(n_blocks))
        self.out = nn.Linear(hidden, latent_dim)
        self.film_cond = nn.Sequential(nn.Linear(cond_dim, hidden), nn.SiLU(),
                                       nn.Linear(hidden, n_film))
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


def _mlp(in_dim: int, hidden: int, n_layers: int) -> nn.Sequential:
    layers = [nn.Linear(in_dim, hidden), nn.SiLU()]
    for _ in range(max(0, n_layers - 1)):
        layers += [nn.Linear(hidden, hidden), nn.SiLU()]
    return nn.Sequential(*layers)


class _Trunk(nn.Module):
    """Shared conditioning + optional flow head. Subclasses add the trajectory side."""

    def __init__(self, cfg, img_dim: int, hist_feat_dim: int, latent_dim: int):
        super().__init__()
        self.uses_image = cfg.uses_image
        self.uses_flow = cfg.uses_flow
        self.gt_text_input = cfg.gt_text_input
        self.flow_state_only = cfg.flow_cond_state_only
        self.z_dim = latent_dim if cfg.uses_flow else 0

        self.vehicle_encoder = VehicleStateEncoder(hist_feat_dim, cfg.vehicle_hidden_dim,
                                                   cfg.vehicle_proj_dim)
        cond_dim = cfg.vehicle_proj_dim
        if self.uses_image:
            # optional dropout on the CNN feature (small training set)
            self.img_proj = nn.Sequential(nn.Dropout(cfg.img_dropout),
                                          nn.Linear(img_dim, cfg.img_proj_dim), nn.SiLU())
            cond_dim += cfg.img_proj_dim
        self.cond_dim = cond_dim
        if self.uses_flow:
            flow_cond_dim = cfg.vehicle_proj_dim if self.flow_state_only else cond_dim
            self.flow = FlowTextHead(latent_dim, flow_cond_dim, cfg.hidden_dim, cfg.n_flow_blocks)

    def encode(self, img, hist):
        """-> (cond for the trajectory side, cond for the flow head)."""
        h = self.vehicle_encoder(hist)
        cond = torch.cat([self.img_proj(img), h], dim=-1) if self.uses_image else h
        return cond, (h if self.flow_state_only else cond)

    def flow_loss(self, flow_cond, z_true):
        b = z_true.shape[0]
        s = torch.rand(b, 1, device=z_true.device)
        eps = torch.randn_like(z_true)
        z_s = (1 - s) * eps + s * z_true
        return nn.functional.mse_loss(self.flow(z_s, s.squeeze(1), flow_cond), z_true - eps)

    def text_latent_for_head(self, flow_cond, z_true, steps, generator=None, training=False,
                             p_sampled=0.5):
        """Latent fed to the trajectory/action head.

        Oracle ablations always use the ground-truth latent. Generative ablations
        use a sample at evaluation and a sample with probability p during training.
        """
        if not self.uses_flow:
            return None
        if self.gt_text_input:
            return z_true
        if training and torch.rand(()) >= p_sampled:
            return z_true
        return self.flow.sample(flow_cond, steps, generator=generator)


class DirectVLA(_Trunk):
    """Position regressor: [cond, z] -> 6 standardised (x, y) waypoints."""

    def __init__(self, cfg, img_dim: int, hist_feat_dim: int, latent_dim: int):
        super().__init__(cfg, img_dim, hist_feat_dim, latent_dim)
        self.traj_trunk = _mlp(self.cond_dim + self.z_dim, cfg.hidden_dim, cfg.n_traj_layers)
        self.traj_out = nn.Linear(cfg.hidden_dim, N_FUTURE * 2)

    def traj_head(self, cond, z):
        x = cond if z is None else torch.cat([cond, z], dim=-1)
        return self.traj_out(self.traj_trunk(x))

    @torch.no_grad()
    def predict(self, img, hist, z_true, steps, generator=None):
        cond, fcond = self.encode(img, hist)
        z_hat = self.flow.sample(fcond, steps, generator=generator) if self.uses_flow else None
        z_in = z_true if self.gt_text_input else z_hat
        return z_hat, self.traj_head(cond, z_in)


class ActionHead(nn.Module):
    """feat -> 6 bounded (a, kappa), as a correction to a prior in pre-tanh space.

    The output layer is zero-initialised, so an untrained head reproduces the prior.
    """

    def __init__(self, in_dim: int, hidden: int, n_layers: int):
        super().__init__()
        self.trunk = _mlp(in_dim, hidden, n_layers)
        self.out = nn.Linear(hidden, N_STEPS * 2)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, feat, prior_pre):
        pre = self.out(self.trunk(feat)).view(-1, N_STEPS, 2) + prior_pre
        return torch.stack([ACC_MAX * torch.tanh(pre[..., 0]),
                            KAPPA_MAX * torch.tanh(pre[..., 1])], dim=-1)


class ResidualHead(nn.Module):
    """feat -> per-step (dx, dy) added after the rollout, scaled by a gate starting at 0.

    The inner net uses the default init so that the gate still receives a gradient.
    """

    def __init__(self, in_dim: int, hidden: int):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, hidden), nn.SiLU(),
                                 nn.Linear(hidden, N_STEPS * 2))
        self.gate = nn.Parameter(torch.zeros(1))

    def forward(self, feat):
        d = self.net(feat).view(-1, N_STEPS, 2) * self.gate
        pad = torch.zeros(d.shape[0], 1, 2, device=d.device, dtype=d.dtype)
        return torch.cat([pad, d], dim=1)


class WorldModelVLA(_Trunk):
    def __init__(self, cfg, img_dim: int, hist_feat_dim: int, latent_dim: int):
        super().__init__(cfg, img_dim, hist_feat_dim, latent_dim)
        act_in = self.cond_dim + self.z_dim
        self.action_head = ActionHead(act_in, cfg.hidden_dim, cfg.n_action_layers)
        self.residual_head = ResidualHead(act_in, cfg.hidden_dim) if cfg.use_residual else None

    def act(self, cond, z, prior_pre):
        feat = cond if z is None else torch.cat([cond, z], dim=-1)
        actions = self.action_head(feat, prior_pre)
        residual = self.residual_head(feat) if self.residual_head is not None else None
        return actions, residual

    def roll(self, v0, actions, residual):
        traj = rollout(v0, actions)                 # (B, 7, 2) incl. origin
        if residual is not None:
            traj = traj + residual
        return traj

    @torch.no_grad()
    def predict(self, img, hist, z_true, v0, prior_pre, steps, generator=None):
        cond, fcond = self.encode(img, hist)
        z_hat = self.flow.sample(fcond, steps, generator=generator) if self.uses_flow else None
        z_in = z_true if self.gt_text_input else z_hat
        actions, residual = self.act(cond, z_in, prior_pre)
        return z_hat, self.roll(v0, actions, residual), actions, residual


def parameter_report(model: nn.Module) -> dict:
    per_part = {name: sum(p.numel() for p in m.parameters())
                for name, m in model.named_children()}
    total = sum(p.numel() for p in model.parameters())
    rep = {"total_params": total, "by_module": per_part,
           "size_fp32_mb": round(total * 4 / 1e6, 3),
           "size_fp16_mb": round(total * 2 / 1e6, 3),
           "note": "the frozen MobileNetV3-Small (0.93 M) and MiniLM encoders are not counted"}
    if getattr(model, "uses_flow", False):
        hoisted = sum(p.numel() for p in model.flow.film_cond.parameters())
        rep["flow_per_step_params"] = sum(p.numel() for p in model.flow.parameters()) - hoisted
        rep["flow_hoisted_params"] = hoisted
    return rep
