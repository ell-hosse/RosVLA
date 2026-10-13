"""CoVLA flow-matching VLA.

    2 Hz RGB frame  --frozen MobileNetV3-Small-->  f  (576)
                                                    |
    past 3 s vehicle state (20 x 3)  --GRU-->       h  (vehicle_proj_dim)
                                                    |
                                    cond = [proj(f), h]
                                                    |
                               flow head (rectified flow): noise --Euler--> z
                                                    |
                                     [cond, z] --> trajectory head
                                                    |
                                   20 future (x, y) waypoints, next 3.0 s

Same flow / FiLM design as RosVLA/defussion_text_generator/model.py, with a
GRU over the vehicle-state window and no action head (CoVLA has no per-frame
action labels).
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn


def timestep_embedding(s: torch.Tensor, dim: int) -> torch.Tensor:
    """Sinusoidal embedding of the scalar flow time s in [0, 1]. (B,) -> (B, dim)."""
    half = dim // 2
    freqs = torch.exp(
        -math.log(1000.0) * torch.arange(half, device=s.device, dtype=torch.float32) / half
    )
    ang = s[:, None] * freqs[None] * 1000.0
    return torch.cat([torch.sin(ang), torch.cos(ang)], dim=-1)


class VehicleStateEncoder(nn.Module):
    """(B, 20, 3) vehicle-state window -> (B, vehicle_proj_dim) via a GRU."""

    def __init__(self, in_dim: int, hidden_dim: int, proj_dim: int):
        super().__init__()
        self.gru = nn.GRU(in_dim, hidden_dim, batch_first=True)
        self.proj = nn.Sequential(nn.Linear(hidden_dim, proj_dim), nn.SiLU())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, h_n = self.gru(x)          # h_n: (1, B, hidden_dim)
        return self.proj(h_n.squeeze(0))


class FiLMResBlock(nn.Module):
    """Residual MLP block with FiLM conditioning: y = x + W2 act(FiLM(W1 x))."""

    def __init__(self, dim: int):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim)
        self.fc2 = nn.Linear(dim, dim)
        self.norm = nn.LayerNorm(dim)
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor, scale: torch.Tensor, shift: torch.Tensor):
        h = self.norm(self.fc1(x))
        h = h * (1 + scale) + shift
        return x + self.fc2(self.act(h))


class FlowTextHead(nn.Module):
    """Generates the PCA text latent z, conditioned on [image, vehicle state]."""

    def __init__(self, latent_dim: int, cond_dim: int, hidden: int, n_blocks: int):
        super().__init__()
        self.latent_dim = latent_dim
        self.time_dim = 32
        self.hidden, self.n_blocks = hidden, n_blocks
        n_film = 2 * hidden * n_blocks

        self.inp = nn.Linear(latent_dim, hidden)
        self.blocks = nn.ModuleList(FiLMResBlock(hidden) for _ in range(n_blocks))
        self.out = nn.Linear(hidden, latent_dim)

        # per-frame branch (computed once per sample() call)
        self.film_cond = nn.Sequential(
            nn.Linear(cond_dim, hidden), nn.SiLU(), nn.Linear(hidden, n_film),
        )
        # per-step branch
        self.film_time = nn.Linear(self.time_dim, n_film)

    def cond_film(self, cond: torch.Tensor) -> torch.Tensor:
        """(B, cond_dim) -> (B, 2*hidden*n_blocks). Compute once per frame."""
        return self.film_cond(cond)

    def _split(self, film: torch.Tensor):
        p = film.view(-1, self.n_blocks, 2, self.hidden)
        return p[:, :, 0], p[:, :, 1]                       # scale, shift

    def velocity(self, z_s: torch.Tensor, s: torch.Tensor, cond_film: torch.Tensor):
        film = cond_film + self.film_time(timestep_embedding(s, self.time_dim))
        scale, shift = self._split(film)
        h = self.inp(z_s)
        for i, blk in enumerate(self.blocks):
            h = blk(h, scale[:, i], shift[:, i])
        return self.out(h)

    def forward(self, z_s: torch.Tensor, s: torch.Tensor, cond: torch.Tensor):
        """Training path."""
        return self.velocity(z_s, s, self.cond_film(cond))

    @torch.no_grad()
    def sample(self, cond: torch.Tensor, steps: int, generator=None) -> torch.Tensor:
        """Euler-integrate the straight path from noise (s=0) to data (s=1)."""
        b = cond.shape[0]
        cond_film = self.cond_film(cond)
        z = torch.randn(b, self.latent_dim, device=cond.device, generator=generator)
        dt = 1.0 / steps
        for i in range(steps):
            s = torch.full((b,), i * dt, device=cond.device)
            z = z + dt * self.velocity(z, s, cond_film)
        return z


class TrajectoryHead(nn.Module):
    """[image, vehicle state, generated text latent] -> 20 future (x, y) waypoints."""

    def __init__(self, cond_dim: int, latent_dim: int, hidden: int,
                 traj_dim: int, n_layers: int):
        super().__init__()
        layers = [nn.Linear(cond_dim + latent_dim, hidden), nn.SiLU()]
        for _ in range(max(0, n_layers - 1)):
            layers += [nn.Linear(hidden, hidden), nn.SiLU()]
        self.trunk = nn.Sequential(*layers)
        self.out = nn.Linear(hidden, traj_dim)

    def forward(self, cond: torch.Tensor, z: torch.Tensor):
        return self.out(self.trunk(torch.cat([cond, z], dim=-1)))


class CoVLAFlowModel(nn.Module):
    def __init__(self, img_dim: int, history_len: int, history_feat_dim: int,
                 latent_dim: int, cfg):
        super().__init__()
        self.history_len = history_len

        self.img_proj = nn.Sequential(nn.Linear(img_dim, cfg.img_proj_dim), nn.SiLU())
        self.vehicle_encoder = VehicleStateEncoder(
            history_feat_dim, cfg.vehicle_hidden_dim, cfg.vehicle_proj_dim
        )
        cond_dim = cfg.img_proj_dim + cfg.vehicle_proj_dim

        self.flow = FlowTextHead(latent_dim, cond_dim, cfg.hidden_dim, cfg.n_flow_blocks)
        self.traj_head = TrajectoryHead(
            cond_dim, latent_dim, cfg.hidden_dim, cfg.traj_dim, cfg.n_traj_layers
        )
        self.latent_dim = latent_dim

    def condition(self, img: torch.Tensor, hist: torch.Tensor) -> torch.Tensor:
        return torch.cat([self.img_proj(img), self.vehicle_encoder(hist)], dim=-1)

    def forward(self, img, hist, z):
        return self.traj_head(self.condition(img, hist), z)

    @torch.no_grad()
    def predict(self, img, hist, steps: int, generator=None):
        cond = self.condition(img, hist)
        z = self.flow.sample(cond, steps, generator=generator)
        traj = self.traj_head(cond, z)
        return z, traj


def parameter_report(model: nn.Module) -> dict:
    """Trainable parameter count, size, and per-Euler-step cost.

    The frozen CNN is not counted. flow_per_step_params excludes the
    per-frame film_cond branch.
    """
    per_part = {
        name: sum(p.numel() for p in module.parameters())
        for name, module in model.named_children()
    }
    total = sum(p.numel() for p in model.parameters())

    flow = model.flow
    hoisted = sum(p.numel() for p in flow.film_cond.parameters())
    per_step = sum(p.numel() for p in flow.parameters()) - hoisted

    return {
        "total_params": total,
        "by_module": per_part,
        "size_fp32_mb": round(total * 4 / 1e6, 3),
        "size_fp16_mb": round(total * 2 / 1e6, 3),
        "flow_per_step_params": per_step,
        "flow_hoisted_params": hoisted,
    }
