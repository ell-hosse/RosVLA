"""Configuration for the BDD-X flow-matching caption-embedding experiment.

Only the flow head of the CoVLA / DriveLM VLAs is trained here, since BDD-X
has no trajectory waypoints:

    clip frames --frozen MobileNetV3-Small--> f
    clip signals (32 x [speed, accel, course, curvature, duration]) --GRU--> h
    cond = [proj(f), h] --rectified flow--> z = PCA latent of the MiniLM
                                              embedding of "<action> <justification>"

Data (`data_root`: --data-root, $BDDX_ROOT, or D:\\BDD-X-Dataset):

    BDD-X-metadata/BDD-X-Annotations_v1.csv   annotation start/end times
    BDD-X-metadata/{train,val,test}.txt       official video-level split
    .../BDDX_Processed/{info,video}/          16,392 train clips (RAG-Driver release):
                                              32 frames and 32 signal samples per clip
    .../BDDX_Test/{info,video}/               2,130 test clips

The folders are searched for anywhere under data_root.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

HERE = Path(__file__).resolve().parent

N_CLIP_FRAMES = 32
CACHED_FRAME_IDX = (2, 6, 10, 14, 18, 22, 26, 30)    # centres of 8 equal bins
FRAME_SUBSETS = {1: (3,), 2: (1, 5), 4: (1, 3, 5, 7), 8: tuple(range(8))}   # indices into the cached 8
STATE_FIELDS = ("speed", "acceleration", "course", "curvature")
# plus a 5th channel: clip duration in seconds

TEXT_FIELDS = ("action_justification", "action", "justification")
ABLATIONS = ("f1_img_state", "f2_state_only", "f3_img_only")


def default_data_root() -> Path:
    env = os.environ.get("BDDX_ROOT")
    if env:
        return Path(env)
    for c in (Path(r"D:\BDD-X-Dataset"), Path("/mnt/d/BDD-X-Dataset"), HERE.parent / "BDD-X-Dataset"):
        if c.exists():
            return c
    return Path(r"D:\BDD-X-Dataset")


@dataclass
class Config:
    ablation: str = "f1_img_state"

    data_root: Path = field(default_factory=default_data_root)
    cache_dir: Path = HERE / "cache"

    # split
    # official test videos -> test; official train videos -> train / val by video
    val_frac: float = 0.15
    split_seed: int = 0

    # frames
    n_frames: int = 4                 # 1 | 2 | 4 | 8
    frame_pool: str = "concat"        # concat (keeps temporal order) | mean
    cnn_name: str = "mobilenet_v3_small"
    image_size: int = 224
    img_feat_dim: int = 576

    # text
    text_field: str = "action_justification"
    text_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    text_emb_dim: int = 384
    pca_dim: int = 32

    # model dims (same as CoVLA)
    img_proj_dim: int = 128
    img_dropout: float = 0.0
    vehicle_hidden_dim: int = 64
    vehicle_proj_dim: int = 64
    hidden_dim: int = 256
    n_flow_blocks: int = 4

    # flow
    flow_steps: int = 4
    step_sweep: tuple = (1, 2, 4, 8)
    n_samples_eval: int = 5

    # training (CoVLA defaults)
    epochs: int = 50
    batch_size: int = 256
    lr: float = 3e-4
    weight_decay: float = 1e-4
    early_stop_patience: int = 5
    early_stop_min_delta: float = 1e-4
    grad_clip: float = 1.0
    knn_k: int = 10                   # kNN baseline
    num_workers: int = 0
    device: str = "cuda"
    seed: int = 0

    def __post_init__(self):
        if self.ablation not in ABLATIONS:
            raise ValueError(f"ablation must be one of {ABLATIONS}")
        if self.text_field not in TEXT_FIELDS:
            raise ValueError(f"text_field must be one of {TEXT_FIELDS}")
        if self.n_frames not in FRAME_SUBSETS:
            raise ValueError(f"n_frames must be one of {sorted(FRAME_SUBSETS)}")
        self.data_root = Path(self.data_root)
        self.cache_dir = Path(self.cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    @property
    def uses_image(self) -> bool:
        return self.ablation in ("f1_img_state", "f3_img_only")

    @property
    def uses_state(self) -> bool:
        return self.ablation in ("f1_img_state", "f2_state_only")
