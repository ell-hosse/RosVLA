# CoVLA world-model trajectory head

Instead of regressing the 20 future waypoints directly, this head predicts a
sequence of controls (longitudinal acceleration `a` and curvature `kappa`,
one pair per step) and obtains the waypoints from a differentiable kinematic
rollout. This mirrors `../../RosVLA/trajectory_predictor/kinematic_ridge.py`
(fixed rollout plus a learned residual). Curvature is used instead of yaw
rate because a car cannot turn in place and `kappa` stays well defined as
`v -> 0`.

`history.py`, `encoders.py`, `latent.py` and `utils.py` are copies of the
files in `../vla_trajecotry_head`.

## Model

```
   2 Hz RGB frame --[frozen MobileNetV3-Small]--> f (576)     [ablations b, c]
   past 3 s state window (20 x 3) --[GRU]------> h            [all ablations]
                                     cond = [proj(f), h]
                                            |
                  flow head (rectified flow): noise --Euler--> z   [ablation b]
                                            |
                         action head: 19 x (a, kappa), bounded by tanh
                                            |
                  differentiable kinematic rollout (no parameters):
                     v_k    = clamp(v_{k-1} + a_k dt, 0, 45)
                     dpsi_k = v_k kappa_k dt
                     x_k    = x_{k-1} + v_k dt cos(psi_{k-1} + dpsi_k / 2)
                     y_k    = y_{k-1} + v_k dt sin(psi_{k-1} + dpsi_k / 2)
                     psi_k  = psi_{k-1} + dpsi_k
                                            |
                         + gated learned residual (dx, dy), initialised at 0
                                            |
                          20 waypoints, next 2.85 s, metres
```

The action head predicts a correction to an analytic prior in pre-tanh space:

```
a_k     = ACC_MAX   * tanh(raw_a_k + atanh(prior_a_k     / ACC_MAX))
kappa_k = KAPPA_MAX * tanh(raw_k_k + atanh(prior_kappa_k / KAPPA_MAX))
```

The output layer is zero-initialised, so the untrained model reproduces the
prior (default `ca`: constant measured aEgo, straight). The residual is
multiplied by a gate initialised to zero and has a stronger weight decay;
the report shows how much displacement it carries.

## Ablations

| flag | image | flow head | caption embedding to the action head |
|---|---|---|---|
| `a_world_model_only` | no | no | none |
| `b_vla_diffusion` | yes | yes | generated |
| `c_vla_no_text` | yes | no | none |

- **a**: past state window -> actions -> rollout, compared with four analytic
  baselines that use the same rollout.
- **b**: the flow head of `../vla_trajecotry_head` with this world model as the
  trajectory head. The caption latent is generated at evaluation time.
- **c**: image + state -> actions -> rollout, without any caption. (Unlike
  `../ablation_studies/no_text_embedding`, which uses the ground-truth caption.)

## Data conventions

Checked on `covla_preprocessed/cache_2hz`:

- `traj[:, 0]` is exactly the origin, so 19 steps are predicted.
- `dt = 0.15 s` (every 3rd raw frame at 20 Hz); the last waypoint is at 2.85 s.
- Column 0 is forward, column 1 lateral
  (corr(traj[:, -1, 0], vEgo * 2.85) = 0.98, 0.01 for column 1).

Analytic baselines through this rollout (full splits):

| baseline | val ADE | val FDE | test ADE | test FDE |
|---|---|---|---|---|
| CV (constant speed, straight) | 0.998 | 2.618 | 1.351 | 3.344 |
| CA (constant aEgo, straight) | 0.875 | 2.264 | 1.255 | 3.068 |
| CTRV (constant speed and curvature) | 1.169 | 3.246 | 1.538 | 4.003 |
| CTRA (constant aEgo and curvature) | 1.084 | 3.011 | 1.481 | 3.838 |

CA is the default prior. The steering -> curvature map fitted on a single
frame is noisy (R^2 ~ 0.28 on the first-step curvature); it is still fitted
and reported, and used by `--prior ctra`.

## Running

```bash
pip install -r requirements.txt

python train.py --self-test                      # integrator round trip
python train.py --ablation a_world_model_only    # no image features needed
python train.py --ablation b_vla_diffusion
python train.py --ablation c_vla_no_text
python train.py --all
python compare.py                                # comparison.md / comparison.json
```

Other options:

```bash
python train.py --all --max-train 100000 --max-val 10000 --max-test 10000
python train.py --ablation a_world_model_only --prior cv        # none|cv|ca|ctrv|ctra
python train.py --ablation a_world_model_only --no-residual
python train.py --ablation c_vla_no_text --pretrain-epochs 3    # warm start on inverse-dynamics actions
python train.py --preprocessed-dir D:\hf\covla_preprocessed --meta-dir D:\hf\CoVLA-metadata
python train.py --device cpu
```

Defaults are 40,000 / 4,000 / 4,000 rows, as in the other CoVLA runs. Each
ablation writes to `runs_<ablation>/`; `cache/` is shared.

## Report

| field | description |
|---|---|
| `ade_m` / `fde_m` | over the 20 waypoints / at t+2.85 s (m) |
| `per_step_displacement_m` | error per waypoint |
| `min_ade_k5` / `min_fde_k5` | best of 5 flow samples (ablation b) |
| `baselines` | CV / CA / CTRV / CTRA through the same rollout |
| `oracle_actions.exact` | inverse-dynamics actions rolled out (should be ~0) |
| `oracle_actions.clipped_to_envelope` | the same actions clipped to the action bounds |
| `identified_steering_map` | kappa = deg2rad(steer) / (L_eff (1 + K_us v^2)), fitted on train |
| `actions` | action ranges, tanh saturation, jerk, residual displacement |
| `text` | caption cosine vs. PCA ceiling and mean baseline (ablation b) |

Standardisers, the PCA latent and the steering map are fitted on the
training split. Row selection is the same as in `../vla_trajecotry_head`, and
the image-exists filter is applied in all ablations so they use the same rows.

## Notes

- The horizon is 2.85 s (from the covla_preprocessed decimation), not 3.0 s.
- About 40 of 23,685 test rows have implausible paths (final waypoint more
  than 30 m beyond vEgo * 2.85). They are kept, as in the other CoVLA runs;
  compare `ade_m` with `ade_median_m`.

## Files

| file | content |
|---|---|
| `config.py` | settings, sampling, action bounds, ablations |
| `dynamics.py` | rollout, inverse dynamics, steering fit, baselines, self-test |
| `model.py` | GRU state encoder, flow head, action head, residual |
| `data.py` | splits (same rows as `../vla_trajecotry_head`) |
| `history.py`, `encoders.py`, `latent.py`, `utils.py` | copies from `../vla_trajecotry_head` |
| `metrics.py` | trajectory, action and text metrics |
| `train.py` | entry point (`--ablation`, `--all`, `--self-test`) |
| `compare.py` | comparison tables, including the other CoVLA models |
