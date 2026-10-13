"""Published open-loop planning results on nuScenes, shown by compare.py.

Notes:
1. All rows are on the official nuScenes val split (150 scenes). Our `test`
   split is a held-out 15 % of DriveLM-nuScenes scenes (nuScenes train
   scenes). Only `nusc_val` uses val scenes, and only the DriveLM subset.
2. "stp3" = mean L2 over waypoints up to t (ST-P3, VAD, BEV-Planner,
   OpenDriveVLA, EMMA); "uniad" = L2 at t (UniAD, GPT-Driver). They differ by
   ~2x at 3 s.
3. Ego status has a large effect on nuScenes (Li et al., CVPR 2024, "Is Ego
   Status All You Need..."). Our models use CAN-bus history, so the rows with
   ego=yes are the relevant references.
4. Collision rate is not reproduced.
"""

# (method, ego status used, stp3 L2 (1s, 2s, 3s, avg) or None, uniad L2 or None, note)
SOTA_NUSCENES = [
    ("ST-P3", "no", (1.33, 2.11, 2.90, 2.11), None, "camera BEV planner"),
    ("UniAD", "yes (in BEV)", (0.44, 0.67, 0.96, 0.69), (0.48, 0.96, 1.65, 1.03),
     "stp3 numbers as re-evaluated in the OpenDriveVLA table; Li et al. report 0.35/0.63/0.99"),
    ("VAD-Base", "yes", (0.17, 0.34, 0.60, 0.37), None, "official checkpoint"),
    ("AD-MLP", "ONLY ego-state history", (0.20, 0.26, 0.41, 0.29), None,
     "no camera, no LiDAR -- the reference point for a state-only model; its README table "
     "mixes conventions, so treat the convention of this row as uncertain"),
    ("Ego-MLP", "ONLY ego status", (0.15, 0.32, 0.59, 0.35), None,
     "from 'Is Ego Status All You Need'"),
    ("BEV-Planner++", "yes", (0.16, 0.32, 0.57, 0.35), None, ""),
    ("GPT-Driver", "yes", (0.20, 0.40, 0.70, 0.44), (0.27, 0.74, 1.52, 0.84), "LLM planner"),
    ("DriveVLM", "not stated", (0.18, 0.34, 0.68, 0.40), None, "Qwen-VL-7B"),
    ("EMMA", "yes", (0.14, 0.29, 0.54, 0.32), None, "Gemini-based"),
    ("OpenDriveVLA-7B", "yes", (0.15, 0.31, 0.55, 0.33), (0.20, 0.58, 1.21, 0.66),
     "Qwen2.5-VL-7B"),
]

# DriveLM paper (ECCV'24), Table 2, DriveLM-nuScenes val. Behavior accuracy is
# comparable to our behavior_accuracy (up to the split); the ADE horizon is not
# stated in the paper.
SOTA_DRIVELM = [
    # (method, behavior acc, speed acc, steer acc, ADE, collision %)
    ("Command Mean", None, None, None, 4.57, 5.72),
    ("UniAD-Single", None, None, None, 1.80, 2.62),
    ("BLIP-RT-2", None, None, None, 2.63, 2.77),
    ("DriveLM-Agent (no context)", 61.45, 72.20, 84.73, 1.39, 1.67),
    ("DriveLM-Agent (graph context)", 57.49, 69.89, 80.63, 1.74, 1.89),
    ("UniAD (full, privileged)", None, None, None, 0.80, 0.17),
]

SOURCES = {
    "OpenDriveVLA (table incl. ST-P3, UniAD, VAD, DriveVLM, EMMA)": "https://arxiv.org/abs/2503.23463",
    "Is Ego Status All You Need (Ego-MLP, BEV-Planner++)": "https://arxiv.org/abs/2312.03031",
    "AD-MLP": "https://arxiv.org/abs/2305.10430",
    "GPT-Driver": "https://arxiv.org/abs/2310.01415",
    "DriveLM": "https://arxiv.org/abs/2312.14150",
}
