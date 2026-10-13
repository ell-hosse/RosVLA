"""Configuration for ablation 1: GT_caption_embedding.

Same as ../../vla_trajecotry_head, except that the trajectory head is
conditioned on the ground-truth caption embedding (PCA-encoded into the same
latent) instead of the flow-generated one.

    image + past 3 s vehicle state  --flow matching-->  PCA caption embedding   (diagnostic only)
    image + past 3 s vehicle state + ground-truth PCA caption embedding  -->  future trajectory

The flow head is still trained with the same loss so its text metrics remain
comparable, but it does not feed the trajectory head. The trajectory metrics
are therefore an upper bound for a perfect text generator. See
../../vla_trajecotry_head/config.py for the data layout.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

HERE = Path(__file__).resolve().parent

# sampling geometry (same as ../../vla_trajecotry_head/config.py)
FPS = 20.0                          # CoVLA native frame rate
ANCHOR_STRIDE = 10                  # covla_preprocessed samples every 10th raw frame
ANCHOR_FPS = FPS / ANCHOR_STRIDE    # 2 Hz
VIDEO_FRAMES = 600                  # raw frames per 30 s clip

FUTURE_RAW_LEN = 60
FUTURE_DOWNSAMPLE = 20
FUTURE_STRIDE = FUTURE_RAW_LEN // FUTURE_DOWNSAMPLE                 # 3
FUTURE_DOWN_IDX = list(range(0, FUTURE_RAW_LEN, FUTURE_STRIDE))     # [0,3,...,57] len 20

HISTORY_RAW_LEN = 60
HISTORY_DOWNSAMPLE = 20
HISTORY_STRIDE = HISTORY_RAW_LEN // HISTORY_DOWNSAMPLE               # 3
HISTORY_DOWN_IDX = list(range(2, HISTORY_RAW_LEN, HISTORY_STRIDE))   # [2,5,...,59] len 20
MIN_ANCHOR_FOR_HISTORY = (HISTORY_RAW_LEN - 1) - HISTORY_DOWN_IDX[0]   # = 57

STATE_FIELDS = ("vEgo", "aEgo", "steeringAngleDeg")


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
    # paths
    preprocessed_dir: Path = field(default_factory=_default_preprocessed_dir)
    meta_dir: Path = field(default_factory=_default_meta_dir)
    out_dir: Path = HERE / "runs"
    cache_dir: Path = HERE / "cache"

    # data sizing
    max_train: int | None = 40000
    max_val: int | None = 4000
    max_test: int | None = 4000

    # frozen image encoder
    cnn_name: str = "mobilenet_v3_small"
    image_size: int = 224
    img_feat_dim: int = 576

    # text
    # captions are pre-embedded with all-MiniLM-L6-v2 (384-d); the PCA-encoded
    # ground truth conditions the trajectory head
    text_emb_dim: int = 384
    pca_dim: int = 32

    # model dims
    img_proj_dim: int = 128
    vehicle_hidden_dim: int = 64
    vehicle_proj_dim: int = 64
    hidden_dim: int = 256
    n_flow_blocks: int = 4
    n_traj_layers: int = 3

    # flow matching (trained for its text metrics only; no p_sampled_z here
    # because the trajectory head always sees the ground-truth latent)
    flow_steps: int = 4
    step_sweep: tuple = (1, 2, 4, 8)

    # training
    epochs: int = 50
    batch_size: int = 256
    lr: float = 3e-4
    weight_decay: float = 1e-4
    w_flow: float = 1.0
    w_traj: float = 1.0
    early_stop_patience: int = 5
    early_stop_min_delta: float = 1e-4
    grad_clip: float = 1.0
    num_workers: int = 4
    device: str = "cuda"
    seed: int = 0

    def __post_init__(self):
        self.preprocessed_dir = Path(self.preprocessed_dir)
        self.meta_dir = Path(self.meta_dir)
        self.out_dir = Path(self.out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.cache_dir = Path(self.cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    @property
    def traj_dim(self) -> int:
        return FUTURE_DOWNSAMPLE * 2                      # 40

    @property
    def history_dim(self) -> int:
        return HISTORY_DOWNSAMPLE * len(STATE_FIELDS)      # 60 (20 steps x 3 features)
