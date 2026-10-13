"""Flow head on its own (the text half of the CoVLA / DriveLM VLAs).

    clip frames (k x 576, frozen MobileNetV3-Small) --Linear+SiLU--> 128   [f1, f3]
    clip signals (32 x 5) --GRU--> 64                                      [f1, f2]
                              cond = concat(...)
                                   |
             flow head: rectified flow, FiLM-conditioned residual MLP
             noise --Euler--> z  (PCA latent of the MiniLM caption)

FiLMResBlock / FlowTextHead / VehicleStateEncoder match
CoVLA/vla_trajecotry_head/model.py.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn


def timestep_embedding(s: torch.Tensor, dim: int) -> torch.Tensor:
    half = dim // 2
    freqs = torch.exp(-math.log(1000.0) *
                      torch.arange(half, device=s.device, dtype=torch.float32) / half)
    ang = s[:, None] * freqs[None] * 1000.0
    return torch.cat([torch.sin(ang), torch.cos(ang)], dim=-1)


class VehicleStateEncoder(nn.Module):
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
        return self.velocity(z_s, s, self.film_cond(cond))

    @torch.no_grad()
    def sample(self, cond, steps: int, generator=None):
        b = cond.shape[0]
        cond_film = self.film_cond(cond)
        z = torch.randn(b, self.latent_dim, device=cond.device, generator=generator)
        dt = 1.0 / steps
        for i in range(steps):
            s = torch.full((b,), i * dt, device=cond.device)
            z = z + dt * self.velocity(z, s, cond_film)
        return z


class FlowCaptioner(nn.Module):
    def __init__(self, cfg, img_dim: int, state_dim: int, latent_dim: int):
        super().__init__()
        self.uses_image, self.uses_state = cfg.uses_image, cfg.uses_state
        cond_dim = 0
        if self.uses_image:
            self.img_proj = nn.Sequential(nn.Dropout(cfg.img_dropout),
                                          nn.Linear(img_dim, cfg.img_proj_dim), nn.SiLU())
            cond_dim += cfg.img_proj_dim
        if self.uses_state:
            self.vehicle_encoder = VehicleStateEncoder(state_dim, cfg.vehicle_hidden_dim,
                                                       cfg.vehicle_proj_dim)
            cond_dim += cfg.vehicle_proj_dim
        self.flow = FlowTextHead(latent_dim, cond_dim, cfg.hidden_dim, cfg.n_flow_blocks)

    def condition(self, img, state):
        parts = []
        if self.uses_image:
            parts.append(self.img_proj(img))
        if self.uses_state:
            parts.append(self.vehicle_encoder(state))
        return torch.cat(parts, dim=-1)

    def flow_loss(self, img, state, z):
        cond = self.condition(img, state)
        s = torch.rand(z.shape[0], 1, device=z.device)
        eps = torch.randn_like(z)
        z_s = (1 - s) * eps + s * z
        return nn.functional.mse_loss(self.flow(z_s, s.squeeze(1), cond), z - eps)

    @torch.no_grad()
    def generate(self, img, state, steps: int, generator=None):
        return self.flow.sample(self.condition(img, state), steps, generator=generator)


def parameter_report(model: nn.Module) -> dict:
    total = sum(p.numel() for p in model.parameters())
    hoisted = sum(p.numel() for p in model.flow.film_cond.parameters())
    return {"total_params": total,
            "by_module": {n: sum(p.numel() for p in m.parameters())
                          for n, m in model.named_children()},
            "size_fp32_mb": round(total * 4 / 1e6, 3),
            "flow_per_step_params": sum(p.numel() for p in model.flow.parameters()) - hoisted,
            "flow_hoisted_params": hoisted,
            "note": "frozen MobileNetV3-Small and MiniLM are not counted"}
