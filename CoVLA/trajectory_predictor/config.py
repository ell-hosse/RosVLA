"""Configuration for the CoVLA linear trajectory predictor.

Data sources
------------
Reads the CoVLA metadata (not the Rosmaster dataset used by
RosVLA/trajectory_predictor):

    <meta_dir>/                        default D:\\hf\\CoVLA-metadata
        states/<video_id>.jsonl        one line per frame with vEgo, aEgo,
                                       steeringAngleDeg and the 60-point
                                       future trajectory
        traj_video_60f_traj10.jsonl    600,000 rows (10,000 videos x 60 anchors);
                                       used to list (video_id, frame_idx) pairs
        10k_data_splits.pt             official row-level split (windows of one
                                       video can appear in several splits);
                                       --split official

    <preprocessed_dir>/                default D:\\hf\\covla_preprocessed
        index/{train,val,test}.jsonl   video-level split (6,938 / 402 / 412
                                       videos, no overlap); default
                                       (--split preprocessed)

--split video uses a seeded random video-level holdout computed locally.

Paths can be overridden with --meta-dir / --preprocessed-dir or the
COVLA_META_DIR / COVLA_PREPROCESSED_DIR environment variables.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

HERE = Path(__file__).resolve().parent

# sampling geometry: 3 s = 60 raw samples, decimated to 20
FPS = 20.0
DT = 1.0 / FPS                    # 0.05 s per frame_idx step
RAW_LEN = 60                      # 3.0 s at 20 Hz
DOWNSAMPLE = 20                   # target resolution after decimation
STRIDE = RAW_LEN // DOWNSAMPLE    # every 3rd raw sample
assert RAW_LEN % DOWNSAMPLE == 0, "RAW_LEN must be an exact multiple of DOWNSAMPLE"
DOWN_IDX = list(range(0, RAW_LEN, STRIDE))   # [0, 3, 6, ..., 57], len == DOWNSAMPLE

ANCHOR_STRIDE = 10                 # anchor spacing baked into traj_video_60f_traj10.jsonl
VIDEO_FRAMES = 600                 # frames per clip (30 s at 20 Hz)
MAX_ANCHOR = VIDEO_FRAMES - RAW_LEN  # last anchor with a full future window

STATE_FIELDS = ("vEgo", "aEgo", "steeringAngleDeg")

SPLIT_SOURCES = ("preprocessed", "video", "official")


def _first_existing(env_var: str, candidates: tuple[Path, ...]) -> Path:
    env = os.environ.get(env_var)
    if env:
        return Path(env)
    for cand in candidates:
        if cand.exists():
            return cand
    return candidates[0]   # existence is checked in data.py


def _default_meta_dir() -> Path:
    return _first_existing(
        "COVLA_META_DIR",
        (Path(r"D:\hf\CoVLA-metadata"), Path("/mnt/d/hf/CoVLA-metadata")),
    )


def _default_preprocessed_dir() -> Path:
    return _first_existing(
        "COVLA_PREPROCESSED_DIR",
        (Path(r"D:\hf\covla_preprocessed"), Path("/mnt/d/hf/covla_preprocessed")),
    )


@dataclass
class Config:
    meta_dir: Path = field(default_factory=_default_meta_dir)
    preprocessed_dir: Path = field(default_factory=_default_preprocessed_dir)

    # caches / outputs, all next to this file
    row_index_cache: Path = HERE / "row_index_cache.npz"
    cache_path: Path = HERE / "covla_window_cache.npz"
    coeffs_path: Path = HERE / "coefficients.json"
    metrics_path: Path = HERE / "metrics.json"

    # ridge strength grid searched on the val split (see train.py)
    alphas: tuple = (0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0, 300.0, 1000.0)

    # fallback synthetic video-level holdout, only used by --split video
    video_split_ratios: tuple = (0.8, 0.1, 0.1)   # train, val, test
    seed: int = 0

    def __post_init__(self):
        self.meta_dir = Path(self.meta_dir)
        self.preprocessed_dir = Path(self.preprocessed_dir)
        self.row_index_cache = Path(self.row_index_cache)
        self.cache_path = Path(self.cache_path)
        self.coeffs_path = Path(self.coeffs_path)
        self.metrics_path = Path(self.metrics_path)

    @property
    def n_features(self) -> int:
        """Flattened input window: DOWNSAMPLE steps x (vEgo, aEgo, steeringAngleDeg)."""
        return DOWNSAMPLE * len(STATE_FIELDS)     # 20 * 3 = 60

    @property
    def n_targets(self) -> int:
        """Flattened output window: DOWNSAMPLE steps x (x, y)."""
        return DOWNSAMPLE * 2                     # 20 * 2 = 40
