"""Gradient-routing modes for the two-head VLA.

    img, state --[trunk]--> cond --+--> flow head  --> z   (caption latent)
                                   |
                                   +--> control head(cond, z) --> commands / trajectory

The control head does not backpropagate into the flow head (z is the ground
truth latent or a no_grad sample), so the only shared tensor is cond.

  --trunk {shared, split}
        shared : one trunk read by both heads
        split  : the flow head has its own trunk

  --grad-mode {joint, staged, pcgrad}
        joint  : one summed loss, one optimiser
        staged : phase A trains trunk + flow on the flow loss and freezes them;
                 phase B trains the control head
        pcgrad : joint training with gradient surgery on the shared trunk
                 (Yu et al., 2020)

  --p-sampled-z FLOAT
        Fraction of training steps where the control head gets a sampled z
        instead of the ground-truth latent. At evaluation it always gets a
        sample. staged defaults to 1.0.

staged freezes the trunk together with the flow head, since updating the
trunk in phase B would shift the conditioning the flow head was trained on.
With --trunk split the control head still has its own trainable trunk.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field

import torch

TRUNKS = ("shared", "split")
GRAD_MODES = ("joint", "staged", "pcgrad")


# mode definition
@dataclass
class Mode:
    """A validated (trunk, grad_mode) pair plus the knobs each one implies."""

    trunk: str = "shared"
    grad_mode: str = "joint"
    p_sampled_z: float | None = None   # None = take the config default
    stage_a_frac: float = 0.4          # fraction of epochs for phase A
    probe_grad_cosine: bool = True     # log flow-vs-control gradient agreement

    def __post_init__(self):
        if self.trunk not in TRUNKS:
            raise ValueError(f"--trunk must be one of {TRUNKS}, got {self.trunk!r}")
        if self.grad_mode not in GRAD_MODES:
            raise ValueError(
                f"--grad-mode must be one of {GRAD_MODES}, got {self.grad_mode!r}"
            )
        if self.grad_mode == "pcgrad" and self.trunk == "split":
            raise ValueError(
                "--grad-mode pcgrad with --trunk split is a no-op: a split trunk "
                "shares no parameters between the two heads, so there are no "
                "conflicting gradients to project. Use --trunk shared."
            )
        if not 0.0 < self.stage_a_frac < 1.0:
            raise ValueError("--stage-a-frac must lie strictly between 0 and 1")
        # with a frozen flow, train the control head on sampled latents only
        if self.p_sampled_z is None and self.grad_mode == "staged":
            self.p_sampled_z = 1.0
        # the gradient cosine needs shared parameters
        if self.trunk == "split" or self.grad_mode == "staged":
            self.probe_grad_cosine = False

    @property
    def name(self) -> str:
        return f"{self.trunk}-{self.grad_mode}"

    @property
    def is_staged(self) -> bool:
        return self.grad_mode == "staged"

    @property
    def is_split(self) -> bool:
        return self.trunk == "split"

    def resolved_p_sampled_z(self, cfg_default: float) -> float:
        return cfg_default if self.p_sampled_z is None else float(self.p_sampled_z)

    def describe(self, cfg_default_p: float) -> str:
        bits = [
            f"trunk={self.trunk}",
            f"grad_mode={self.grad_mode}",
            f"p_sampled_z={self.resolved_p_sampled_z(cfg_default_p)}",
        ]
        if self.is_staged:
            bits.append(f"stage_a_frac={self.stage_a_frac}")
        return "  ".join(bits)

    def to_dict(self, cfg_default_p: float) -> dict:
        return {
            "trunk": self.trunk,
            "grad_mode": self.grad_mode,
            "p_sampled_z": self.resolved_p_sampled_z(cfg_default_p),
            "stage_a_frac": self.stage_a_frac if self.is_staged else None,
        }


def add_mode_arguments(ap) -> None:
    """Attach the shared CLI surface to an argparse parser."""
    ap.add_argument("--trunk", choices=TRUNKS, default="shared",
                    help="shared: one conditioning trunk (baseline). "
                         "split: the flow head gets a private trunk.")
    ap.add_argument("--grad-mode", choices=GRAD_MODES, default="joint",
                    help="joint: one summed loss (baseline). "
                         "staged: train+freeze the flow, then the control head. "
                         "pcgrad: joint with gradient surgery at the shared trunk.")
    ap.add_argument("--p-sampled-z", type=float, default=None,
                    help="fraction of steps the control head sees a SAMPLED z "
                         "rather than the ground-truth latent. Defaults to the "
                         "config value, or 1.0 under --grad-mode staged.")
    ap.add_argument("--stage-a-frac", type=float, default=0.4,
                    help="share of the epoch budget spent on phase A (flow only) "
                         "under --grad-mode staged. The total budget is unchanged, "
                         "so staged and joint runs cost the same.")
    ap.add_argument("--no-grad-probe", action="store_true",
                    help="skip the flow-vs-control gradient cosine measurement")


def mode_from_args(args) -> Mode:
    return Mode(
        trunk=args.trunk,
        grad_mode=args.grad_mode,
        p_sampled_z=args.p_sampled_z,
        stage_a_frac=args.stage_a_frac,
        probe_grad_cosine=not args.no_grad_probe,
    )


# parameter groups
def set_requires_grad(modules, flag: bool) -> None:
    for m in modules:
        if m is None:
            continue
        for p in m.parameters():
            p.requires_grad_(flag)


def trainable(params) -> list:
    return [p for p in params if p.requires_grad]


def split_epoch_budget(total_epochs: int, stage_a_frac: float) -> tuple[int, int]:
    """Split the epoch budget between phase A and B (at least one epoch each)."""
    total = max(2, int(total_epochs))
    a = max(1, min(total - 1, int(round(total * stage_a_frac))))
    return a, total - a


# flow-vs-control gradient cosine
def _flatten(grads, params) -> torch.Tensor:
    return torch.cat([
        (torch.zeros_like(p) if g is None else g).reshape(-1)
        for g, p in zip(grads, params)
    ])


@torch.no_grad()
def _cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    na, nb = a.norm(), b.norm()
    if na < 1e-12 or nb < 1e-12:
        return 0.0
    return float((a @ b) / (na * nb))


def grad_cosine(loss_flow, loss_control, shared_params) -> float:
    """Cosine between the flow and control gradients at the shared trunk.

    Positive means the two objectives agree, negative means they conflict.
    Call it before the real backward(), while both graphs are still alive.
    """
    params = trainable(shared_params)
    if not params:
        return 0.0
    g_flow = torch.autograd.grad(loss_flow, params, retain_graph=True,
                                 allow_unused=True)
    g_ctrl = torch.autograd.grad(loss_control, params, retain_graph=True,
                                 allow_unused=True)
    return _cosine(_flatten(g_flow, params), _flatten(g_ctrl, params))


class RunningCosine:
    """Mean gradient cosine over an epoch, measured every `stride` batches."""

    def __init__(self, stride: int = 10):
        self.stride = max(1, stride)
        self.reset()

    def reset(self) -> None:
        self._sum = 0.0
        self._n = 0
        self._neg = 0
        self._batch = 0

    def maybe(self, loss_flow, loss_control, shared_params) -> None:
        take = (self._batch % self.stride) == 0
        self._batch += 1
        if not take:
            return
        c = grad_cosine(loss_flow, loss_control, shared_params)
        self._sum += c
        self._n += 1
        self._neg += int(c < 0)

    @property
    def mean(self) -> float:
        return self._sum / self._n if self._n else 0.0

    @property
    def frac_conflicting(self) -> float:
        """Fraction of measured batches with a negative cosine."""
        return self._neg / self._n if self._n else 0.0

    def report(self) -> dict:
        return {
            "grad_cosine_mean": round(self.mean, 4),
            "frac_conflicting_batches": round(self.frac_conflicting, 4),
            "n_measured": self._n,
        }


# PCGrad
def pcgrad_backward(losses, shared_params, all_params, rng: random.Random | None = None,
                    grad_clip: float | None = None) -> dict:
    """Backward pass with PCGrad applied to the shared parameters only.

    For each pair of task gradients (g_i, g_j) with g_i . g_j < 0, g_i is
    projected onto the normal plane of g_j, and the results are summed.
    Parameters outside the shared set get the usual summed-loss gradient.
    PCGrad is not scale-invariant, so the losses are projected as passed in.
    """
    rng = rng or random.Random(0)
    shared = trainable(shared_params)
    total = sum(losses)

    if not shared or len(losses) < 2:
        total.backward()
        stats = {"n_projections": 0, "n_pairs": 0}
    else:
        flats = []
        for i, l in enumerate(losses):
            g = torch.autograd.grad(l, shared, retain_graph=True, allow_unused=True)
            flats.append(_flatten(g, shared))

        projected = [f.clone() for f in flats]
        n_proj, n_pairs = 0, 0
        order = list(range(len(flats)))
        for i in range(len(flats)):
            rng.shuffle(order)
            for j in order:
                if i == j:
                    continue
                n_pairs += 1
                dot = torch.dot(projected[i], flats[j])
                if dot < 0:
                    n_proj += 1
                    projected[i] = projected[i] - dot * flats[j] / (
                        flats[j].dot(flats[j]) + 1e-12
                    )
        merged = torch.stack(projected).sum(0)

        # regular backward fills .grad for all parameters
        total.backward()

        # then overwrite the shared ones with the projected gradient
        offset = 0
        with torch.no_grad():
            for p in shared:
                n = p.numel()
                p.grad = merged[offset:offset + n].view_as(p).clone()
                offset += n
        stats = {"n_projections": n_proj, "n_pairs": n_pairs}

    if grad_clip:
        torch.nn.utils.clip_grad_norm_(trainable(all_params), grad_clip)
    return stats
