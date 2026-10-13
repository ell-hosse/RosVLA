"""Configuration for the DriveLM-nuScenes flow-matching VLA.

Same design as the CoVLA models (frozen image encoder, PCA MiniLM caption
latent, rectified-flow text head with FiLM, GRU over past vehicle state), with
two trajectory heads:

    family "direct"  -> waypoint regression               (as CoVLA/vla_trajecotry_head)
    family "wm"      -> (a, kappa) controls -> kinematic rollout  (as CoVLA/world_model_head)

Data layout, under `data_root` ($DRIVELM_ROOT, else D:\\DriveLM; or --data-root):

    v1_1_train_nus.json            DriveLM QA per keyframe
    drivelm_nus_imgs_train.zip     nuScenes keyframe images
    v1.0-trainval_meta.tgz         nuScenes metadata
    can_bus.zip                    nuScenes CAN bus expansion
    drivelm_nus_imgs_val.zip       optional; adds an evaluation split on the
                                   official nuScenes val scenes

    nuscenes/                      written by prepare_data.py
        samples/CAM_FRONT/*.jpg
        v1.0-trainval/{sample,scene,log,ego_pose}.json
        can_bus/scene-XXXX_{pose,steeranglefeedback}.json

Sampling (nuScenes keyframes are 2 Hz):
    future trajectory : ego pose at the next 6 keyframes (0.5 ... 3.0 s) in the
                        current ego frame (x forward, y left, metres), as in
                        UniAD / VAD
    past state window : 20 CAN samples, 0.15 s apart, ending at the keyframe;
                        speed, longitudinal acceleration, steering angle, yaw rate
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

HERE = Path(__file__).resolve().parent

# sampling geometry
KEYFRAME_HZ = 2.0
DT = 0.5                                  # seconds between future waypoints
N_FUTURE = 6                              # 0.5 ... 3.0 s
HORIZON_S = N_FUTURE * DT                 # 3.0 s
N_STEPS = N_FUTURE                        # world-model rollout steps
N_WAYPOINTS = N_FUTURE + 1                # incl. the origin, for the rollout
L2_TIMES_S = (1.0, 2.0, 3.0)              # nuScenes planning report points
L2_IDX = tuple(int(round(t / DT)) - 1 for t in L2_TIMES_S)     # (1, 3, 5)

HISTORY_LEN = 20
HISTORY_STRIDE_S = 0.15
HISTORY_SPAN_S = (HISTORY_LEN - 1) * HISTORY_STRIDE_S          # 2.85 s
STATE_FIELDS = ("v_mps", "a_mps2", "steer_deg", "yaw_rate_rps")

# dynamics envelope
ACC_MAX = 8.0          # m/s^2
KAPPA_MAX = 0.25       # 1/m -> 4 m radius; nuScenes has tight urban turns
V_MAX = 40.0           # m/s
V_MIN = 0.0

# text variants
# behavior      : DriveLM "Predict the behavior of the ego vehicle." answer
#                 (~21 unique strings)
# caption       : default; behavior + target action + traffic signal + objects
#                 in front (~2-3k unique strings)
# caption_full  : caption + scene description, summary and planner comment
TEXT_FIELDS = ("behavior", "caption", "caption_full")

DIRECT_ABLATIONS = ("d1_flow_generated", "d2_gt_caption", "d3_no_text", "d4_flow_state_only")
WM_ABLATIONS = ("a_world_model_only", "b_vla_diffusion", "c_vla_no_text", "d_vla_gt_caption")


def default_data_root() -> Path:
    """Location of the DriveLM / nuScenes downloads: $DRIVELM_ROOT, the project's
    parent folder, or D:\\DriveLM.
    """
    env = os.environ.get("DRIVELM_ROOT")
    if env:
        return Path(env)
    for c in (HERE.parent, Path(r"D:\DriveLM"), Path("/mnt/d/DriveLM")):
        if (c / "v1_1_train_nus.json").exists():
            return c
    return Path(r"D:\DriveLM")


@dataclass
class Config:
    family: str = "direct"                  # direct | wm
    ablation: str = "d1_flow_generated"

    # paths
    data_root: Path = field(default_factory=default_data_root)
    out_dir: Path = HERE / "runs"
    cache_dir: Path = HERE / "cache"

    # split (scene-level; DriveLM only publishes a train split)
    split_fracs: tuple = (0.70, 0.15, 0.15)
    split_seed: int = 0

    # frozen image encoder
    cnn_name: str = "mobilenet_v3_small"
    image_size: int = 224
    img_feat_dim: int = 576

    # text
    text_field: str = "caption"
    text_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    text_emb_dim: int = 384
    pca_dim: int = 32                 # capped at n_unique_train_texts - 1

    # model dims (same as CoVLA)
    img_proj_dim: int = 128
    img_dropout: float = 0.0          # dropout on the CNN feature
    vehicle_hidden_dim: int = 64
    vehicle_proj_dim: int = 64
    hidden_dim: int = 256
    n_flow_blocks: int = 4
    n_traj_layers: int = 3            # direct family
    n_action_layers: int = 3          # wm family

    # world model
    prior: str = "auto"               # auto | none | cv | ca | ctrv | ctra
    use_residual: bool = True
    residual_weight_decay: float = 1e-2
    w_jerk: float = 1e-3
    w_dkappa: float = 1e-2
    pretrain_epochs: int = 0

    # flow matching
    flow_steps: int = 4
    step_sweep: tuple = (1, 2, 4, 8)
    n_samples_eval: int = 5

    # training
    # ~2.8k training rows (CoVLA: 40k), hence smaller batches and more epochs
    epochs: int = 150
    batch_size: int = 64
    lr: float = 3e-4
    weight_decay: float = 1e-4
    w_flow: float = 1.0
    w_traj: float = 1.0
    p_sampled_z: float = 0.5
    early_stop_patience: int = 15
    early_stop_min_delta: float = 1e-4
    grad_clip: float = 1.0
    num_workers: int = 0              # safe default on Windows
    device: str = "cuda"
    seed: int = 0

    def __post_init__(self):
        valid = DIRECT_ABLATIONS if self.family == "direct" else WM_ABLATIONS
        if self.family not in ("direct", "wm"):
            raise ValueError(f"family must be 'direct' or 'wm', got {self.family!r}")
        if self.ablation not in valid:
            raise ValueError(f"ablation for family {self.family!r} must be one of {valid}")
        if self.text_field not in TEXT_FIELDS:
            raise ValueError(f"text_field must be one of {TEXT_FIELDS}")
        self.data_root = Path(self.data_root)
        self.out_dir = Path(self.out_dir)
        self.cache_dir = Path(self.cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    # inputs used by the ablation
    @property
    def uses_image(self) -> bool:
        return self.ablation != "a_world_model_only"

    @property
    def uses_flow(self) -> bool:
        return self.ablation in ("d1_flow_generated", "d2_gt_caption", "d4_flow_state_only",
                                 "b_vla_diffusion", "d_vla_gt_caption")

    @property
    def uses_text(self) -> bool:
        """A caption embedding is needed (as a flow target and/or an input)."""
        return self.uses_flow

    @property
    def gt_text_input(self) -> bool:
        """True if the ground-truth caption latent is fed to the trajectory/action head (oracle)."""
        return self.ablation in ("d2_gt_caption", "d_vla_gt_caption")

    @property
    def flow_cond_state_only(self) -> bool:
        return self.ablation == "d4_flow_state_only"

    @property
    def nuscenes_dir(self) -> Path:
        return self.data_root / "nuscenes"
