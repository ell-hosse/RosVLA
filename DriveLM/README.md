# DriveLM-nuScenes VLA: flow-matching caption embedding + trajectory, with and without a world model

The CoVLA models (`../CoVLA`) applied to DriveLM-nuScenes. It has two trajectory heads:

| family | script | mirrors |
|---|---|---|
| **direct**: a position regressor, no world model | `train_direct.py` | `CoVLA/vla_trajecotry_head` + `CoVLA/ablation_studies/*` |
| **wm**: a world model that predicts (a, κ) controls and integrates them with a kinematic rollout | `train_wm.py` | `CoVLA/world_model_head` |

Both families use the same conditioning trunk and flow head as the CoVLA code:
a frozen MobileNetV3-Small image encoder, a GRU over a 20-step past-state
window, a rectified-flow head with FiLM conditioning that generates the
PCA-compressed **all-MiniLM-L6-v2** caption embedding (the same sentence
encoder as CoVLA), and the same dimensions and losses. The reports are also the
same: ADE/FDE, min-ADE K=5, text cosine vs. PCA ceiling vs. mean baseline,
retrieval, zero-parameter baselines, oracle actions and residual size. Each
report adds the standard nuScenes planning metrics and DriveLM's behavior
accuracy, so the numbers can be compared with published work (see [SOTA](#sota)).

```
 CAM_FRONT keyframe ──[frozen MobileNetV3-Small]──► f (576) ──proj──► 128
 past 2.85 s CAN window (20 × [v, a, steer, yaw-rate]) ──[GRU]──► h (64)
                                   cond = [proj(f), h]
                                          │
                    ┌─────────────────────┴───────────────────────┐
                    │ FLOW HEAD (rectified flow)  noise ─Euler─► z │  z = generated PCA
                    └─────────────────────┬───────────────────────┘      latent of the
                                          │                               MiniLM caption
     direct: TRAJECTORY HEAD [cond, z] ─► 6 (x, y) waypoints, 0.5 … 3.0 s
     wm:     ACTION HEAD     [cond, z] ─► 6 (a, κ) ─► kinematic rollout (0 params)
                                                     + gated residual (starts at 0)
```

---

## 1. Setup

```bat
cd DriveLM
pip install -r requirements.txt
```

The data folder (default `D:\DriveLM`, or `--data-root` / `DRIVELM_ROOT`)
should contain `v1_1_train_nus.json`, `drivelm_nus_imgs_train.zip`,
`v1.0-trainval_meta.tgz` and `can_bus.zip`; `prepare_data.py` extracts the
`nuscenes\` folder. `cache/` and `runs/` are written next to the code.

**Optional, for the official val scenes:** also download
`drivelm_nus_imgs_val.zip` (705 MB) from the same Hugging Face repo and put it
in `D:\DriveLM`:

```bat
hf download OpenDriveLab/DriveLM drivelm_nus_imgs_val.zip --repo-type dataset --local-dir D:\DriveLM
```

Its frames come from the official nuScenes val scenes, used by published results. `prepare_data.py` turns them into an extra evaluation-only
split called `nusc_val`. If you add the zip after the data was already
prepared, rebuild with `python prepare_data.py --force`. The image cache
refreshes itself on the next run, and the models must be retrained so their
reports include the new split.

## 2. Prepare the data (once, numpy only, about 3 min)

```bat
python prepare_data.py
```

This script:

- extracts only what is needed: 4 metadata tables, the CAM_FRONT keyframes,
  and the pose and steering CAN files of the scenes used. Everything goes into
  `D:\DriveLM\nuscenes\` in the standard nuScenes layout.
- builds one row per DriveLM keyframe:
  - the **future**: the ego pose at the next 6 keyframes (t = 0.5 … 3.0 s) in
    the current ego frame. This is the same planning target UniAD, VAD and
    OpenDriveVLA are scored on.
  - the **past**: a 20-sample CAN window every 0.15 s ending on the keyframe
    (speed, longitudinal acceleration, steering-wheel angle, yaw rate). This is
    the same 20-step / 0.15 s window as CoVLA, with yaw rate added.
  - three **caption** variants (section 4).
- writes `cache/dataset.npz` and `cache/dataset_summary.json`.

Dataset statistics:

| | |
|---|---|
| DriveLM keyframes | 4 072 in 696 scenes |
| dropped: future leaves the scene (last 3 s of a scene) | 544 |
| dropped: scene has no CAN data (15 scenes) or no window | 82 |
| **usable rows** | **3 446** (Boston 1 916, Singapore 1 530) |
| scene-level split (seed 0) | train 2 429 rows / 477 scenes, val 517 / 102, test 500 / 102 |
| CAN window edge-padded (keyframe < 2.85 s into its scene) | 605 rows, flagged, reported as a subgroup |
| consistency check: corr(v₀·3 s, x at 3 s) | 0.972 |
| unique texts over all usable rows: `behavior` / `caption` / `caption_full` | 21 / 2 626 / 3 424 |

## 3. Train and compare

```bat
:: world model: all four ablations, 3 seeds each
python train_wm.py --all --seeds 0 1 2

:: direct head: all four ablations, 3 seeds each
python train_direct.py --all --seeds 0 1 2

:: tables -> comparison_caption_test.md (+ .json)
python compare.py
python compare.py --split nusc_val        :: only if drivelm_nus_imgs_val.zip was prepared
```

`run_all.bat` runs everything, including the other two text fields.
Useful flags: `--text-field behavior|caption|caption_full`,
`--epochs`, `--patience`, `--pca-dim`, `--split-seed`, `--img-dropout 0.2`,
`--device cpu`; for the world model only: `--prior ca`, `--no-residual`,
`--pretrain-epochs 3`, `--self-test`.

The first run with images makes one MobileNet pass over 3 446 frames (about 1
minute on a GPU), and the first run with text embeds the unique captions with
MiniLM. Both are cached in `cache/`, so later runs skip them. After that, each
run is roughly 1 to 3 minutes on a GPU.

Outputs go to `runs/<family>/<ablation>/<text-field or no_text>/seed_<k>/{report.json, model.pt}`
plus `summary.json` (mean ± std over seeds) one level up.

## 4. Ablations

**Direct family** (`train_direct.py`):

| ablation | image | flow head | caption reaching the trajectory head |
|---|---|---|---|
| `d1_flow_generated` | ✓ | ✓ (sees image + state) | **generated** (this is the model; same as `CoVLA/vla_trajecotry_head`) |
| `d2_gt_caption` | ✓ | ✓, trained for its text metrics only | **ground truth**: oracle / upper bound (`CoVLA/ablation_studies/GT_caption_embedding`) |
| `d3_no_text` | ✓ | – | none |
| `d4_flow_state_only` | ✓ | ✓ (sees the CAN window only) | generated (`CoVLA/ablation_studies/flaw_matching_based_on_states_only`) |

**World-model family** (`train_wm.py`), the same four as CoVLA plus an oracle:

| ablation | image | flow head | caption reaching the action head |
|---|---|---|---|
| `a_world_model_only` | – | – | none; the CAN window only |
| `b_vla_diffusion` | ✓ | ✓ | **generated** (the model) |
| `c_vla_no_text` | ✓ | – | none |
| `d_vla_gt_caption` | ✓ | ✓ | ground truth (oracle) |

**Caption variants** (`--text-field`):

- `behavior`: DriveLM's answer to *"Predict the behavior of the ego vehicle"*,
  e.g. *"The ego vehicle is steering to the left. The ego vehicle is driving
  slowly."* There are only 21 unique strings, so this is effectively a
  classification task (as with Rosmaster's `trajectory_text`). Its PCA ceiling
  is trivially about 1.0.
- `caption` (**default**): a CoVLA-style multi-slot caption built from answers
  about the front view: behavior, target action, traffic signal, and objects in
  front. Example: *"The ego vehicle is going straight. The ego vehicle is
  driving fast. Target action: Go straight. Traffic signal: Green light. There
  are many obstacles in front of the ego car."*
- `caption_full`: `caption` plus the current-scene description, the
  scene-level summary and the planner comment. This is the richest variant, but
  much of it cannot be seen from CAM_FRONT.

## 5. Reading the numbers

- **ADE / FDE** are over the 6 future waypoints. The origin is not counted,
  unlike CoVLA's waypoint 0.
- **L2 @ 1 s / 2 s / 3 s** is reported in *both* conventions used in the
  literature. `stp3` (ST-P3, VAD, OpenDriveVLA, EMMA) is the mean L2 over all
  waypoints up to t. `uniad` (UniAD, GPT-Driver's second table) is the L2 at
  exactly t. They differ by about 2× at 3 s, so only compare like with like.
- **Text metrics**, same as CoVLA: the generated-embedding cosine is read
  against the PCA ceiling and the train-mean baseline. `scaled` is 1.0 at the
  ceiling and 0.0 when the model is no better than predicting the mean
  caption. Retrieval is measured against the train caption bank; `coverage` is
  the share of test captions that exist in that bank at all.
- **Behavior accuracy (Acc / Speed / Steer)**: the behavior sentences of the
  retrieved caption vs. the ground truth. These are the three numbers DriveLM
  reports for DriveLM-Agent. Read them against the majority-class baseline
  printed next to them.
- **World model**: compare ADE with the best analytic baseline and with
  `oracle_actions.clipped` (about 0 here), and check
  `actions.residual.mean_abs_m`, the displacement carried by the residual.
- `subgroups` breaks ADE down by city (Singapore drives on the left) and by
  whether the CAN window was edge-padded.

### Baselines and checks

The zero-parameter baselines, test split:

| baseline | ADE | FDE | L2 1/2/3 s (stp3) | avg stp3 | avg uniad |
|---|---|---|---|---|---|
| CV | 0.897 | 1.955 | 0.220 / 0.502 / 0.897 | 0.540 | 1.081 |
| CA | 0.787 | 1.783 | 0.197 / 0.424 / 0.787 | 0.469 | 0.957 |
| CTRV | 0.798 | 1.747 | 0.201 / 0.445 / 0.798 | 0.481 | 0.962 |
| **CTRA** | **0.699** | **1.620** | 0.177 / 0.368 / 0.699 | 0.415 | 0.854 |

Other checks:

- **World model A** (CAN window only, 188 k params, 3 seeds): **ADE 0.441 ± 0.001**,
  FDE 1.036; L2 stp3 0.111 / 0.229 / 0.441, avg **0.260**; uniad avg 0.540.
  Residual 0.09 m.
- The integrator round trip is exact: oracle ADE 8e-4 m, from rows where the
  car is stopped.
- The untrained world model reproduces its prior exactly (val 0.7419 = CTRA 0.7419).
- The measured κ₀ = yaw rate / v explains R² = 0.92 of the first-step
  curvature. On CoVLA, the steering map managed only 0.28, which is why the
  auto-selected prior here is CTRA and on CoVLA it was CA.

## 6. <a name="sota"></a>Published results

| method | ego status as input | L2 1/2/3 s (stp3) | avg | L2 1/2/3 s (uniad) | avg |
|---|---|---|---|---|---|
| ST-P3 | no | 1.33 / 2.11 / 2.90 | 2.11 | – | – |
| UniAD | yes (in BEV) | 0.44 / 0.67 / 0.96 | 0.69 | 0.48 / 0.96 / 1.65 | 1.03 |
| VAD-Base | yes | 0.17 / 0.34 / 0.60 | 0.37 | – | – |
| AD-MLP (ego history only, no sensors) | only ego | 0.20 / 0.26 / 0.41 | 0.29 | – | – |
| Ego-MLP | only ego | 0.15 / 0.32 / 0.59 | 0.35 | – | – |
| BEV-Planner++ | yes | 0.16 / 0.32 / 0.57 | 0.35 | – | – |
| GPT-Driver | yes | 0.20 / 0.40 / 0.70 | 0.44 | 0.27 / 0.74 / 1.52 | 0.84 |
| DriveVLM | not stated | 0.18 / 0.34 / 0.68 | 0.40 | – | – |
| EMMA | yes | 0.14 / 0.29 / 0.54 | 0.32 | – | – |
| OpenDriveVLA-7B | yes | 0.15 / 0.31 / 0.55 | 0.33 | 0.20 / 0.58 / 1.21 | 0.66 |

DriveLM paper, Table 2 (DriveLM-nuScenes val): DriveLM-Agent reaches
behavior accuracy **61.45** (speed 72.20, steer 84.73), ADE 1.39, collision
1.67 %. UniAD-Single reaches ADE 1.80. UniAD with privileged inputs reaches
ADE 0.80.

**Caveats when comparing:**

1. **Split.** All rows above are on the official nuScenes-*val* scenes.
   `test` here is a held-out 15 % of DriveLM scenes, which are nuScenes-*train*
   scenes. Only `--split nusc_val` is on SOTA scenes, and even that uses
   DriveLM's subset of val keyframes, not all ~6 k.
2. **Ego status.** All models here use CAN history. On nuScenes ego status
   alone is a strong signal: AD-MLP and Ego-MLP match perception-based
   planners (Li et al., CVPR'24). The relevant references are the rows with
   ego status, and the internal comparisons B vs. C and D1 vs. D3 show what
   the generated caption adds.
3. **Collision rate is not computed.** It needs the ST-P3 occupancy code and
   the annotated boxes.
4. **DriveLM's behavior sentence is derived from the future path**, so feeding
   the ground-truth caption (`d2`, `d`) leaks the answer; these rows are
   upper bounds only. The other ablations generate the caption.

Sources:
[OpenDriveVLA](https://arxiv.org/abs/2503.23463) (its Table 2 collects ST-P3, UniAD, VAD, DriveVLM, EMMA),
[Is Ego Status All You Need](https://arxiv.org/abs/2312.03031),
[AD-MLP](https://arxiv.org/abs/2305.10430),
[GPT-Driver](https://arxiv.org/abs/2310.01415),
[DriveLM](https://arxiv.org/abs/2312.14150).

## 7. Data splits and fitting

- The split is by **scene** (2 Hz keyframes of one scene are near-duplicates),
  and an assert checks that no scene appears in two splits. `nusc_val` rows are
  filtered to official val scenes, so they cannot overlap DriveLM-train.
- The standardisers, the PCA text latent (mean, basis, scale), the retrieval
  bank, the majority-behavior baseline and the world-model prior choice
  (`--prior auto`) are all fitted on **train** only.
- MobileNet and MiniLM are pretrained and frozen, so extracting features up
  front is not leakage.
- Early stopping uses validation ADE with a generated caption latent.

## 8. Differences from the CoVLA code

| | CoVLA | here | why |
|---|---|---|---|
| future | 20 waypoints / 0.15 s incl. origin, 2.85 s | 6 waypoints / 0.5 s, 3.0 s | the nuScenes planning target |
| state window | (v, a, steer) | (v, a, steer, **yaw rate**) | nuScenes has it; it makes κ₀ reliable |
| world-model prior | `ca` | `auto` (picked on train; CTRA) | measured κ₀ is good here |
| caption embeddings | precomputed | computed once with the same MiniLM | DriveLM ships text only |
| data size | 40 k train rows | 2.4 k | batch 64, 150 epochs, patience 15 |
| `--img-dropout` | – | available, default 0 | regularisation for the small training set |

## 9. Files

| file | role |
|---|---|
| `prepare_data.py` | extraction plus the per-keyframe future, past and captions → `cache/dataset.npz` |
| `config.py` | geometry, envelope, text fields, ablation registry, all hyper-parameters |
| `data.py` | scene-level split, cached image features and text embeddings |
| `encoders.py` | frozen MobileNetV3-Small and MiniLM, both cached |
| `latent.py` | PCA text latent (train only) |
| `model.py` | GRU, rectified-flow head, direct head, action head, residual, ablation wiring |
| `dynamics.py` | world model: rollout, exact inverse, priors, self-test |
| `metrics.py` | ADE/FDE, nuScenes L2 (both conventions), text fidelity, retrieval, behavior accuracy |
| `common.py` | shared CLI, text side of the report, seed aggregation |
| `train_direct.py` / `train_wm.py` | the two families |
| `compare.py` | `comparison_<field>_<split>.md` with the SOTA table |
| `sota.py` | the published numbers, with sources and caveats |
| `nusc_splits.py` | the 150 official nuScenes val scenes |
