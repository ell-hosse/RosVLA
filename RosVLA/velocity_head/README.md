# Velocity-head VLA (RosVLA)

A VLA that predicts velocity commands instead of waypoints. The waypoints are
obtained by passing the commands through the identified plant model in
`../trajectory_predictor`.

```
 RGB frame  --frozen MobileNetV3-Small-->  f  --.
                                                |--> cond --> flow head --> z
 u[t-6 .. t-1]  (past 3 s of commands) -------->  '                         |
                                                                            v
                                             cond + z --> velocity head --> u[t .. t+5]
                                                          action head   --> class of u[t]
                                                                            |
                              ../trajectory_predictor (4 frozen parameters) |
                                                                            v
                                                       6 waypoints, robot frame, 3 s
```

Stage 1 (this folder) has ~1.07 M trainable parameters and follows
`../defussion_text_generator` (rectified flow, FiLM, PCA text latent).
Stage 2 is the frozen plant model.

## Running

Export the per-fold stage-2 models first:

```bash
cd RosVLA/trajectory_predictor
python trajectory_calculator.py --export trajectory_predictors

cd ../velocity_head
python run_cv.py
```

`run_cv.py` uses the first directory under `RosVLA/trajectory_predictor/`
that contains `fold_*.json`; `--calculator-dir` overrides this. Other options:
`--epochs N`, `--folds N`, `--hist N`, `--text-field caption_detailed`.

Fold *i* holds out scenario *S* and uses `fold_*_S.json`, the plant model
identified without *S*.

## Setup

- Text target: `action.action_text` (8 unique strings).
- Input: the six previous commands `u[t-6 .. t-1]` (3 s at 2 Hz).
- Output: `u[t .. t+5]`. `future_waypoints_robot_frame[k]` is the pose at
  t+k+1, produced by the command at t, so the chunk starts at the current
  frame. Plant residual for different alignments:

  | chunk | plant residual |
  |---|---|
  | `u[t .. t+5]` | 0.037 m |
  | `u[t+1 .. t+6]` | 0.072 m |
  | `u[t-1 .. t+4]` | 0.044 m |

- Windowing removes `hist + horizon` = 12 frames per run, leaving 211
  windows out of 331 frames.

## Outputs

`runs/fold_XX_<scenario>/metrics.json` and `runs/summary.json`:

- `velocity.v`, `velocity.w`: MAE, RMSE, R^2, max and per-step error (m/s and
  rad/s, reported separately)
- `velocity.snapped_exact_match`: exact match after snapping predictions to
  the 10 distinct training commands
- `ade`, `fde`: trajectory error of the predicted commands
- `ade_oracle_commands`: the plant model with the true commands (~0.04 m)
- `ade_hold_last_command`: baseline that repeats the last command (~0.18 m)
- training losses, action accuracy, text metrics and the Euler-step sweep

## Command distribution

| label | v | w | frames |
|---|---|---|---|
| FORWARD | 0.50 | 0.00 | 100 |
| STOP | 0.00 | 0.00 | 58 |
| RIGHT_TURN | 0.12 | -0.433 (6 variants) | 153 |
| LEFT_TURN | 0.12 | +0.40 | 11 |
| REVERSE_LEFT | -0.20 | +0.44 | 9 |

Replacing each command by its class prototype changes the stage-2 ADE by
0.0001 m.

## Notes

- `VLA_CNN_WEIGHTS=none` / `VLA_TEXT_WEIGHTS=none` skip the pretrained
  downloads (smoke tests only).
- Image features are cached by `scenario:frame` and shared with
  `../defussion_text_generator/cache`.
- `action.linear_velocity_target` / `angular_velocity_target` equal the h5
  `cmd_vel` channels; they are commands, not measured velocities.
- `make_paper_figure.py` (after `export_predictions.py`) produces the
  qualitative figure in `paper_figures/`.
