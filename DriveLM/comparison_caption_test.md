# DriveLM-nuScenes VLA -- comparison (split `test`, text field `caption`)

Rows: train 2429, val 517, test 500 (scene-level split, split_seed 0). Horizon 3.0 s, 6 waypoints at 0.5 s. Mean ± std over seeds where more than one seed was run.

## Zero-parameter baselines (kinematic rollout of the measured state)

| baseline | ADE (m) | FDE (m) | L2 1s / 2s / 3s (ST-P3 avg) | L2 avg ST-P3 | L2 avg UniAD |
|---|---|---|---|---|---|
| CV | 0.897 | 1.955 | 0.220/0.502/0.897 | 0.540 | 1.081 |
| CA | 0.787 | 1.783 | 0.197/0.424/0.787 | 0.469 | 0.957 |
| CTRV | 0.797 | 1.747 | 0.201/0.445/0.797 | 0.481 | 0.962 |
| CTRA | 0.699 | 1.620 | 0.177/0.368/0.699 | 0.414 | 0.854 |

## Direct trajectory head (no world model)

| model | params | ADE (m) | FDE (m) | min-ADE K=5 | L2 1s / 2s / 3s (ST-P3 avg) | L2 avg ST-P3 | L2 avg UniAD | text cosine (scaled) | behavior acc % |
|---|---|---|---|---|---|---|---|---|---|
| D1  flow VLA, generated caption  (the model) | 1,472,108 | 0.720 ± 0.035 | 1.507 ± 0.049 | 0.590 ± 0.025 | 0.251/0.434/0.720 | 0.468 ± 0.031 | 0.858 ± 0.039 | 0.957 ± 0.002 (0.03 ± 0.04) | 52.0 ± 2.2 (maj 26.4) |
| D2  flow VLA, GT caption  (oracle, leaks future) | 1,472,108 | 0.570 ± 0.015 | 1.187 ± 0.027 | - | 0.205/0.347/0.570 | 0.374 ± 0.011 | 0.677 ± 0.015 | 0.958 ± 0.000 (0.07 ± 0.01) | 56.9 ± 1.2 (maj 26.4) |
| D3  no flow, no caption  (control) | 275,532 | 0.698 ± 0.013 | 1.468 ± 0.018 | - | 0.251/0.420/0.698 | 0.456 ± 0.011 | 0.832 ± 0.015 | - | - |
| D4  flow on CAN window only | 1,439,340 | 0.717 ± 0.002 | 1.509 ± 0.009 | 0.584 ± 0.006 | 0.246/0.429/0.717 | 0.464 ± 0.003 | 0.854 ± 0.002 | 0.946 ± 0.002 (-0.21 ± 0.05) | 52.9 ± 1.9 (maj 26.4) |

## World-model trajectory head

| model | params | ADE (m) | FDE (m) | min-ADE K=5 | L2 1s / 2s / 3s (ST-P3 avg) | L2 avg ST-P3 | L2 avg UniAD | text cosine (scaled) | behavior acc % | residual (m) |
|---|---|---|---|---|---|---|---|---|---|---|
| A   world model, CAN window only | 188,633 | 0.441 ± 0.001 | 1.036 ± 0.002 | - | 0.111/0.229/0.441 | 0.260 ± 0.001 | 0.540 ± 0.001 | - | - | 0.087 ± 0.002 |
| B   world model in the flow VLA  (the model) | 1,532,793 | 0.557 ± 0.007 | 1.289 ± 0.019 | 0.474 ± 0.005 | 0.138/0.293/0.557 | 0.330 ± 0.004 | 0.680 ± 0.009 | 0.939 ± 0.002 (-0.37 ± 0.04) | 30.5 ± 0.9 (maj 26.4) | 0.035 ± 0.005 |
| C   world model + image, no caption  (control) | 328,025 | 0.541 ± 0.003 | 1.255 ± 0.009 | - | 0.137/0.286/0.541 | 0.321 ± 0.002 | 0.661 ± 0.004 | - | - | 0.006 ± 0.002 |
| D   world model, GT caption  (oracle, leaks future) | 1,532,793 | 0.493 ± 0.010 | 1.137 ± 0.031 | - | 0.132/0.265/0.493 | 0.296 ± 0.005 | 0.602 ± 0.013 | 0.949 ± 0.003 (-0.15 ± 0.06) | 39.9 ± 3.0 (maj 26.4) | 0.108 ± 0.009 |

## What the language pathway is worth

- direct head, generated caption vs no caption: D1  flow VLA, generated caption ADE 0.7197 vs D3  no flow, no caption ADE 0.6981 -> **-0.0216 m** (-3.1 % of the second).
- world model, generated caption vs no caption: B   world model in the flow VLA ADE 0.5573 vs C   world model + image, no caption ADE 0.5415 -> **-0.0158 m** (-2.9 % of the second).
- direct head, headroom if the generator were perfect: D2  flow VLA, GT caption ADE 0.5704 vs D1  flow VLA, generated caption ADE 0.7197 -> **+0.1493 m** (+20.7 % of the second).
- world model, headroom if the generator were perfect: D   world model, GT caption ADE 0.4932 vs B   world model in the flow VLA ADE 0.5573 -> **+0.0641 m** (+11.5 % of the second).
- world model, what the image adds over the CAN window alone: C   world model + image, no caption ADE 0.5415 vs A   world model, CAN window only ADE 0.4407 -> **-0.1008 m** (-22.9 % of the second).

A positive number means the first model is better. With ~100 test scenes, differences smaller than about two seed-stds are noise.

## Published results for context

**Not the same split.** All rows below are on the official nuScenes *val* scenes; `test` here is a held-out part of nuScenes *train* scenes. Only `--split nusc_val` is on the SOTA scenes (and even then on DriveLM's subset of val keyframes). Collision rate is not reproduced.

| method | ego status as input | L2 1s / 2s / 3s (ST-P3 avg) | L2 avg ST-P3 | L2 1s / 2s / 3s (UniAD at-t) | L2 avg UniAD | note |
|---|---|---|---|---|---|---|
| ST-P3 | no | 1.33/2.11/2.90 | 2.11 | - | - | camera BEV planner |
| UniAD | yes (in BEV) | 0.44/0.67/0.96 | 0.69 | 0.48/0.96/1.65 | 1.03 | stp3 numbers as re-evaluated in the OpenDriveVLA table; Li et al. report 0.35/0.63/0.99 |
| VAD-Base | yes | 0.17/0.34/0.60 | 0.37 | - | - | official checkpoint |
| AD-MLP | ONLY ego-state history | 0.20/0.26/0.41 | 0.29 | - | - | no camera, no LiDAR -- the reference point for a state-only model; its README table mixes conventions, so treat the convention of this row as uncertain |
| Ego-MLP | ONLY ego status | 0.15/0.32/0.59 | 0.35 | - | - | from 'Is Ego Status All You Need' |
| BEV-Planner++ | yes | 0.16/0.32/0.57 | 0.35 | - | - |  |
| GPT-Driver | yes | 0.20/0.40/0.70 | 0.44 | 0.27/0.74/1.52 | 0.84 | LLM planner |
| DriveVLM | not stated | 0.18/0.34/0.68 | 0.40 | - | - | Qwen-VL-7B |
| EMMA | yes | 0.14/0.29/0.54 | 0.32 | - | - | Gemini-based |
| OpenDriveVLA-7B | yes | 0.15/0.31/0.55 | 0.33 | 0.20/0.58/1.21 | 0.66 | Qwen2.5-VL-7B |

DriveLM paper, Table 2 (DriveLM-nuScenes val):

| method | behavior acc | speed acc | steer acc | ADE | collision % |
|---|---|---|---|---|---|
| Command Mean | - | - | - | 4.57 | 5.72 |
| UniAD-Single | - | - | - | 1.80 | 2.62 |
| BLIP-RT-2 | - | - | - | 2.63 | 2.77 |
| DriveLM-Agent (no context) | 61.45 | 72.20 | 84.73 | 1.39 | 1.67 |
| DriveLM-Agent (graph context) | 57.49 | 69.89 | 80.63 | 1.74 | 1.89 |
| UniAD (full, privileged) | - | - | - | 0.80 | 0.17 |

Sources: [OpenDriveVLA (table incl. ST-P3, UniAD, VAD, DriveVLM, EMMA)](https://arxiv.org/abs/2503.23463); [Is Ego Status All You Need (Ego-MLP, BEV-Planner++)](https://arxiv.org/abs/2312.03031); [AD-MLP](https://arxiv.org/abs/2305.10430); [GPT-Driver](https://arxiv.org/abs/2310.01415); [DriveLM](https://arxiv.org/abs/2312.14150)
