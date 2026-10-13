"""Configuration for the flaw_matching_based_on_states_only ablation.

    past 3 s vehicle state  --flow matching-->  PCA caption embedding
    image + past 3 s vehicle state + generated caption embedding  -->  future trajectory

Same as ../../vla_trajecotry_head except that the flow head is conditioned on
the vehicle-state window only, without the image. The first sentence of a
CoVLA caption ("The ego vehicle is moving slowly with deceleration and
turning right...") describes vEgo / aEgo / steering, which the model already
has as input. If caption fidelity stays the same without the image, the flow
head is mostly verbalising the telemetry.

The trajectory head keeps [image, vehicle state], so ADE/FDE remains
comparable. --flow-uses-image reproduces the baseline (control run) and
--traj-state-only runs a fully state-only pipeline.

Data locations (CLI flags or environment variables):

    <preprocessed_dir>   default D:\\hf\\covla_preprocessed   (COVLA_PREPROCESSED_DIR)
        cache_2hz/{train,val,test}.npz      per 2 Hz anchor: state (N, 7), traj (N, 20, 2)
        cache_2hz/paths_{split}.npz         image path per row
        cache_2hz/captions_{split}.npz      unique captions + per-row code
        cache_2hz/capemb_minilm_384.npz     all-MiniLM-L6-v2 caption embeddings
        frames/<video_id>/<frame_idx>.jpg   RGB frames

    <meta_dir>           default D:\\hf\\CoVLA-metadata       (COVLA_META_DIR)
        states/<video_id>.jsonl             raw 20 Hz vehicle state (see history.py)

cache/ was copied from ../GT_caption_embedding/cache (same seed, sizes and
encoder); caches are validated against the row selection and rebuilt if needed.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

HERE = Path(__file__).resolve().parent

# sampling geometry
FPS = 20.0                          # CoVLA native frame rate
ANCHOR_STRIDE = 10                  # covla_preprocessed samples every 10th raw frame
ANCHOR_FPS = FPS / ANCHOR_STRIDE    # 2 Hz
VIDEO_FRAMES = 600                  # raw frames per 30 s clip

# future trajectory: 3.0 s, 60 raw waypoints -> 20
# (cache_2hz "traj" is already decimated this way; documented here only)
FUTURE_RAW_LEN = 60
FUTURE_DOWNSAMPLE = 20
FUTURE_STRIDE = FUTURE_RAW_LEN // FUTURE_DOWNSAMPLE                 # 3
FUTURE_DOWN_IDX = list(range(0, FUTURE_RAW_LEN, FUTURE_STRIDE))     # [0,3,...,57] len 20

# past vehicle-state window: 3.0 s, 60 raw -> 20, ending on the anchor frame
HISTORY_RAW_LEN = 60
HISTORY_DOWNSAMPLE = 20
HISTORY_STRIDE = HISTORY_RAW_LEN // HISTORY_DOWNSAMPLE               # 3
HISTORY_DOWN_IDX = list(range(2, HISTORY_RAW_LEN, HISTORY_STRIDE))   # [2,5,...,59] len 20
# smallest anchor frame_idx whose full history window is inside the video
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
    # full splits: 398,805 / 23,170 / 23,685 rows. None = use all rows,
    # an int = seeded uniform subsample.
    max_train: int | None = 40000
    max_val: int | None = 4000
    max_test: int | None = 4000

    # frozen image encoder
    cnn_name: str = "mobilenet_v3_small"     # torchvision, ImageNet weights, frozen
    image_size: int = 224
    img_feat_dim: int = 576                  # pooled MobileNetV3-Small feature width

    # text
    # captions are pre-embedded with all-MiniLM-L6-v2 (384-d)
    text_emb_dim: int = 384
    pca_dim: int = 32                # text latent size

    # model dims
    img_proj_dim: int = 128
    vehicle_hidden_dim: int = 64     # GRU hidden size
    vehicle_proj_dim: int = 64
    hidden_dim: int = 256            # flow head / trajectory head width
    n_flow_blocks: int = 4
    n_traj_layers: int = 3

    # ablation switches. Baseline = (True, True), this ablation = (False, True).
    # --flow-uses-image gives the control run, --traj-state-only gives (False, False).
    flow_uses_image: bool = False
    traj_uses_image: bool = True

    # flow matching (rectified flow)
    flow_steps: int = 4
    step_sweep: tuple = (1, 2, 4, 8)
    n_samples_eval: int = 5          # K for min-ADE / min-FDE

    # training
    epochs: int = 50
    batch_size: int = 256
    lr: float = 3e-4
    weight_decay: float = 1e-4
    w_flow: float = 1.0
    w_traj: float = 1.0
    p_sampled_z: float = 0.5         # fraction of steps the traj head sees a sampled z
    early_stop_patience: int = 5     # epochs without val-ADE improvement
    early_stop_min_delta: float = 1e-4
    grad_clip: float = 1.0
    num_workers: int = 4             # image feature extraction
    device: str = "cuda"             # falls back to cpu
    seed: int = 0

    def __post_init__(self):
        self.preprocessed_dir = Path(self.preprocessed_dir)
        self.meta_dir = Path(self.meta_dir)
        self.out_dir = Path(self.out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.cache_dir = Path(self.cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    @property
    def variant(self) -> str:
        """Run tag used as the runs/ subdirectory name."""
        f = "imgstate" if self.flow_uses_image else "state"
        t = "imgstate" if self.traj_uses_image else "state"
        return f"flow-{f}__traj-{t}"

    @property
    def traj_dim(self) -> int:
        return FUTURE_DOWNSAMPLE * 2                      # 40

    @property
    def history_dim(self) -> int:
        return HISTORY_DOWNSAMPLE * len(STATE_FIELDS)      # 60 (20 steps x 3 features)
