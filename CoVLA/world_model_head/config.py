"""Configuration for the CoVLA world-model trajectory head.

Instead of regressing waypoints directly, the head predicts per-step
longitudinal acceleration `a` and curvature `kappa`, and the waypoints come
from a differentiable kinematic rollout (dynamics.py). This follows
RosVLA/trajectory_predictor/kinematic_ridge.py (rollout + learned residual).

history.py, encoders.py, latent.py and utils.py are copies of the files in
../vla_trajecotry_head, so inputs and early stopping match.

Ablations (train.py --ablation)
-------------------------------
  a_world_model_only   past 3 s state -> actions -> rollout. No image, no text.
                       Reported with the CV / CA / CTRV / CTRA baselines.
  b_vla_diffusion      image + state --rectified flow--> PCA caption latent;
                       [image, state, latent] -> actions -> rollout.
  c_vla_no_text        image + state -> actions -> rollout. No caption at all.

Sampling geometry (same as ../vla_trajecotry_head)
--------------------------------------------------
CoVLA is 20 Hz and covla_preprocessed keeps every 10th frame (2 Hz anchors).
Each anchor has 20 future waypoints taken every 3rd raw frame from the
anchor, so waypoint 0 is the origin, waypoint k is at t = 0.15 k s and
waypoint 19 at 2.85 s. Column 0 is forward and column 1 lateral. This gives
19 rollout steps with dt = 0.15 s.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

HERE = Path(__file__).resolve().parent

# sampling geometry
FPS = 20.0
ANCHOR_STRIDE = 10
ANCHOR_FPS = FPS / ANCHOR_STRIDE          # 2.0 Hz
VIDEO_FRAMES = 600

FUTURE_RAW_LEN = 60
FUTURE_DOWNSAMPLE = 20                    # waypoints including the origin
FUTURE_STRIDE = FUTURE_RAW_LEN // FUTURE_DOWNSAMPLE          # 3 raw frames
FUTURE_DOWN_IDX = list(range(0, FUTURE_RAW_LEN, FUTURE_STRIDE))

DT = FUTURE_STRIDE / FPS                  # 0.15 s between consecutive waypoints
N_WAYPOINTS = FUTURE_DOWNSAMPLE           # 20
N_STEPS = N_WAYPOINTS - 1                 # 19 rollout steps / action pairs
HORIZON_S = N_STEPS * DT                  # 2.85 s

# past vehicle-state window, ending on the anchor frame
HISTORY_RAW_LEN = 60
HISTORY_DOWNSAMPLE = 20
HISTORY_STRIDE = HISTORY_RAW_LEN // HISTORY_DOWNSAMPLE
HISTORY_DOWN_IDX = list(range(2, HISTORY_RAW_LEN, HISTORY_STRIDE))
MIN_ANCHOR_FOR_HISTORY = (HISTORY_RAW_LEN - 1) - HISTORY_DOWN_IDX[0]      # 57

STATE_FIELDS = ("vEgo", "aEgo", "steeringAngleDeg")

# dynamics envelope
# action bounds, enforced with tanh
ACC_MAX = 8.0        # m/s^2
KAPPA_MAX = 0.2      # 1/m (min radius 5 m)
V_MAX = 45.0         # m/s, clamp on the integrated speed
V_MIN = 0.0          # no reversing

ABLATIONS = ("a_world_model_only", "b_vla_diffusion", "c_vla_no_text")


def _first_existing(env_var: str, candidates: tuple[Path, ...]) -> Path:
    env = os.environ.get(env_var)
    if env:
        return Path(env)
    for c in candidates:
        if c.exists():
            return c
    return candidates[0]


def _default_preprocessed_dir() -> Path:
    return _first_existing(
        "COVLA_PREPROCESSED_DIR",
        (Path(r"D:\hf\covla_preprocessed"), Path("/mnt/d/hf/covla_preprocessed")),
    )


def _default_meta_dir() -> Path:
    return _first_existing(
        "COVLA_META_DIR",
        (Path(r"D:\hf\CoVLA-metadata"), Path("/mnt/d/hf/CoVLA-metadata")),
    )


@dataclass
class Config:
    # which ablation
    ablation: str = "a_world_model_only"

    # paths
    preprocessed_dir: Path = field(default_factory=_default_preprocessed_dir)
    meta_dir: Path = field(default_factory=_default_meta_dir)
    out_dir: Path = HERE / "runs"            # set per ablation in train.py
    cache_dir: Path = HERE / "cache"

    # data sizing
    # same caps as the other CoVLA runs; None = full split
    max_train: int | None = 40000
    max_val: int | None = 4000
    max_test: int | None = 4000

    # frozen image encoder (ablations b, c only)
    cnn_name: str = "mobilenet_v3_small"
    image_size: int = 224
    img_feat_dim: int = 576

    # text (ablation b only)
    text_emb_dim: int = 384
    pca_dim: int = 32

    # model dims
    img_proj_dim: int = 128
    vehicle_hidden_dim: int = 64
    vehicle_proj_dim: int = 64
    hidden_dim: int = 256
    n_flow_blocks: int = 4
    n_action_layers: int = 3

    # world model
    # analytic prior the action head corrects. Test ADE of the priors:
    # ca 1.255, cv 1.351, ctra 1.481, ctrv 1.538.
    prior: str = "ca"                 # none | cv | ca | ctrv | ctra
    use_residual: bool = True         # learned per-step (dx, dy) added after rollout
    residual_weight_decay: float = 1e-2
    w_jerk: float = 1e-3              # penalty on (a_k - a_{k-1})^2
    w_dkappa: float = 1e-2            # penalty on (kappa_k - kappa_{k-1})^2
    pretrain_epochs: int = 0          # epochs supervised on inverse-dynamics actions

    # flow matching (ablation b only)
    flow_steps: int = 4
    step_sweep: tuple = (1, 2, 4, 8)
    n_samples_eval: int = 5

    # training
    epochs: int = 50
    batch_size: int = 256
    lr: float = 3e-4
    weight_decay: float = 1e-4
    w_flow: float = 1.0
    w_traj: float = 1.0
    p_sampled_z: float = 0.5
    early_stop_patience: int = 5
    early_stop_min_delta: float = 1e-4
    grad_clip: float = 1.0
    num_workers: int = 4
    device: str = "cuda"
    seed: int = 0

    def __post_init__(self):
        if self.ablation not in ABLATIONS:
            raise ValueError(f"ablation must be one of {ABLATIONS}, got {self.ablation!r}")
        self.preprocessed_dir = Path(self.preprocessed_dir)
        self.meta_dir = Path(self.meta_dir)
        self.out_dir = Path(self.out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.cache_dir = Path(self.cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    # inputs used by the ablation
    @property
    def uses_image(self) -> bool:
        return self.ablation in ("b_vla_diffusion", "c_vla_no_text")

    @property
    def uses_flow(self) -> bool:
        return self.ablation == "b_vla_diffusion"

    @property
    def uses_text(self) -> bool:
        """True if a caption embedding (generated or ground truth) reaches the action head."""
        return self.ablation == "b_vla_diffusion"

    @property
    def history_dim(self) -> int:
        return HISTORY_DOWNSAMPLE * len(STATE_FIELDS)     # 60
