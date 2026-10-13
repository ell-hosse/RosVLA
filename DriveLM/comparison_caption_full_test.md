# DriveLM-nuScenes VLA -- comparison (split `test`, text field `caption_full`)

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
| D1  flow VLA, generated caption  (the model) | 1,472,108 | 0.686 ± 0.003 | 1.467 ± 0.019 | 0.584 ± 0.008 | 0.221/0.402/0.686 | 0.436 ± 0.003 | 0.820 ± 0.006 | 0.930 ± 0.002 (-0.11 ± 0.03) | 37.5 ± 1.5 (maj 26.4) |
| D2  flow VLA, GT caption  (oracle, leaks future) | 1,472,108 | 0.635 ± 0.010 | 1.351 ± 0.022 | - | 0.217/0.378/0.635 | 0.410 ± 0.008 | 0.760 ± 0.014 | 0.932 ± 0.000 (-0.08 ± 0.01) | 38.4 ± 1.0 (maj 26.4) |
| D3  no flow, no caption  (control) | 275,532 | 0.698 ± 0.013 | 1.468 ± 0.018 | - | 0.251/0.420/0.698 | 0.456 ± 0.011 | 0.832 ± 0.015 | - | - |
| D4  flow on CAN window only | 1,439,340 | 0.698 ± 0.006 | 1.473 ± 0.020 | 0.591 ± 0.004 | 0.241/0.418/0.698 | 0.452 ± 0.005 | 0.833 ± 0.008 | 0.916 ± 0.001 (-0.37 ± 0.02) | 38.2 ± 1.1 (maj 26.4) |

## World-model trajectory head

| model | params | ADE (m) | FDE (m) | min-ADE K=5 | L2 1s / 2s / 3s (ST-P3 avg) | L2 avg ST-P3 | L2 avg UniAD | text cosine (scaled) | behavior acc % | residual (m) |
|---|---|---|---|---|---|---|---|---|---|---|
| A   world model, CAN window only | 188,633 | 0.441 ± 0.001 | 1.036 ± 0.002 | - | 0.111/0.229/0.441 | 0.260 ± 0.001 | 0.540 ± 0.001 | - | - | 0.087 ± 0.002 |
| B   world model in the flow VLA  (the model) | 1,532,793 | 0.559 ± 0.005 | 1.294 ± 0.009 | 0.486 ± 0.011 | 0.139/0.295/0.559 | 0.331 ± 0.003 | 0.683 ± 0.006 | 0.905 ± 0.005 (-0.58 ± 0.10) | 23.5 ± 5.0 (maj 26.4) | 0.025 ± 0.014 |
| C   world model + image, no caption  (control) | 328,025 | 0.541 ± 0.003 | 1.255 ± 0.009 | - | 0.137/0.286/0.541 | 0.321 ± 0.002 | 0.661 ± 0.004 | - | - | 0.006 ± 0.002 |
| D   world model, GT caption  (oracle, leaks future) | 1,532,793 | 0.532 ± 0.010 | 1.237 ± 0.024 | - | 0.135/0.279/0.532 | 0.315 ± 0.006 | 0.650 ± 0.013 | 0.899 ± 0.002 (-0.70 ± 0.04) | 21.3 ± 4.0 (maj 26.4) | 0.016 ± 0.006 |

## What the language pathway is worth

- direct head, generated caption vs no caption: D1  flow VLA, generated caption ADE 0.6859 vs D3  no flow, no caption ADE 0.6981 -> **+0.0122 m** (+1.7 % of the second).
- world model, generated caption vs no caption: B   world model in the flow VLA ADE 0.5592 vs C   world model + image, no caption ADE 0.5415 -> **-0.0177 m** (-3.3 % of the second).
- direct head, headroom if the generator were perfect: D2  flow VLA, GT caption ADE 0.6349 vs D1  flow VLA, generated caption ADE 0.6859 -> **+0.0510 m** (+7.4 % of the second).
- world model, headroom if the generator were perfect: D   world model, GT caption ADE 0.5317 vs B   world model in the flow VLA ADE 0.5592 -> **+0.0275 m** (+4.9 % of the second).
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
