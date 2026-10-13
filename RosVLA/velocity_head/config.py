"""Settings for the velocity-head VLA.

Same structure as ../defussion_text_generator/config.py, with three changes:
the text target is action_text, the telemetry input is a window of past
commands, and the head predicts a chunk of future commands. Waypoints are
obtained from the frozen stage-2 calculator in ../trajectory_predictor.
"""

from dataclasses import dataclass
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
SIBLING_CACHE = HERE.parent / "defussion_text_generator" / "cache"


STAGE2_ROOT = HERE.parent / "trajectory_predictor"


def find_calculator_dir(root: Path = STAGE2_ROOT) -> Path:
    """Find the directory holding the exported stage-2 fold_*.json files."""
    if not root.is_dir():
        return root / "models"
    candidates = [root] + sorted(p for p in root.iterdir() if p.is_dir())
    for path in candidates:
        if any(path.glob("fold_*.json")):
            return path
    return root / "models"


@dataclass
class Config:
    # paths
    dataset_dir: Path = REPO_ROOT / "dataset"
    out_dir: Path = HERE / "runs"
    # reuse the main model's CNN feature cache when available
    cache_dir: Path = SIBLING_CACHE if SIBLING_CACHE.exists() else HERE / "cache"
    # exported stage-2 fold models; None = search ../trajectory_predictor
    calculator_dir: Path = None

    # data
    # 8 unique strings (trajectory_text has 9, caption_detailed ~240).
    # Nested fields can be given with a dot, e.g. "action.action_text".
    text_field: str = "action_text"

    # input: the H commands before the current frame, u[t-H .. t-1]
    hist: int = 6                 # 3.0 s at 2 Hz

    # output: the F commands starting at the current frame, u[t .. t+F-1].
    # This is the alignment that produces the recorded waypoints (see README).
    horizon: int = 6              # 3.0 s, 6 waypoints

    # frozen encoders
    cnn_name: str = "mobilenet_v3_small"
    image_size: int = 224
    text_encoder: str = "sentence-transformers/all-MiniLM-L6-v2"

    # text latent space
    # refit per fold on training texts, capped at n_unique_train_texts - 1
    pca_dim: int = 8

    # model
    img_proj_dim: int = 128
    tel_proj_dim: int = 32
    hidden_dim: int = 256
    n_blocks: int = 3

    # flow matching
    flow_steps: int = 4
    step_sweep: tuple = (1, 2, 4, 8)
    n_samples_eval: int = 5       # for min-ADE / min-FDE over K samples

    # training
    epochs: int = 300
    batch_size: int = 32
    lr: float = 1e-3
    weight_decay: float = 1e-4
    w_flow: float = 1.0
    w_vel: float = 1.0
    w_ang: float = 1.0
    w_action: float = 0.5         # auxiliary action-class loss
    p_sampled_z: float = 0.5

    seed: int = 0
    device: str = "cpu"

    action_classes: tuple = (
        "FORWARD", "RIGHT_TURN", "LEFT_TURN", "STOP", "REVERSE_LEFT",
    )

    def __post_init__(self):
        self.out_dir = Path(self.out_dir)
        self.cache_dir = Path(self.cache_dir)
        self.calculator_dir = (
            Path(self.calculator_dir) if self.calculator_dir is not None
            else find_calculator_dir()
        )
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    @property
    def n_actions(self) -> int:
        return len(self.action_classes)

    @property
    def tel_dim(self) -> int:
        """Flattened input window: H frames x (v, w)."""
        return self.hist * 2

    @property
    def vel_dim(self) -> int:
        """Flattened output chunk: F steps x (v, w)."""
        return self.horizon * 2
