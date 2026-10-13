"""Settings for the no-text ablation.

Same as ../defussion_text_generator/config.py without the text latent, so the
only difference between the two runs is whether a generated text embedding is
fed to the trajectory and action heads.
"""

from dataclasses import dataclass
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
SIBLING_CACHE = HERE.parent / "defussion_text_generator" / "cache"


@dataclass
class Config:
    # paths
    dataset_dir: Path = REPO_ROOT / "dataset"
    out_dir: Path = HERE / "runs"
    # reuse the main model's CNN feature cache when available
    cache_dir: Path = SIBLING_CACHE if SIBLING_CACHE.exists() else HERE / "cache"

    # data
    # read by data.py only; not used as an input or a target
    text_field: str = "trajectory_text"
    # 6 waypoints at 0.5 s spacing, same horizon as the main model
    n_waypoints: int = 6
    drop_first_waypoint: bool = False

    telemetry_keys: tuple = (
        "odom_linear_velocity",           # 3
        "odom_angular_velocity",          # 3
        "imu_linear_acceleration",        # 3
        "imu_attitude",                   # 3
        "wheel_speed_encoder_per_sec",    # 4
    )                                     # -> 16 dims

    # frozen encoder
    cnn_name: str = "mobilenet_v3_small"
    image_size: int = 224

    # model
    img_proj_dim: int = 128
    tel_proj_dim: int = 32
    hidden_dim: int = 256
    n_blocks: int = 3          # unused here; kept so the configs line up

    # --match-params: widen the trunk to match the full model's parameter count
    match_params: int = 0

    # training
    epochs: int = 300
    batch_size: int = 32
    lr: float = 1e-3
    weight_decay: float = 1e-4
    w_traj: float = 1.0
    w_action: float = 1.0

    seed: int = 0
    device: str = "cpu"

    action_classes: tuple = (
        "FORWARD", "RIGHT_TURN", "LEFT_TURN", "STOP", "REVERSE_LEFT",
    )

    def __post_init__(self):
        self.out_dir = Path(self.out_dir)
        self.cache_dir = Path(self.cache_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    @property
    def n_actions(self) -> int:
        return len(self.action_classes)

    @property
    def traj_dim(self) -> int:
        return self.n_waypoints * 2
