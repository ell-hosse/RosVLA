# Trunk sharing for the world-model VLA (CoVLA)

Experiments on how the flow (caption) head and the action head of the CoVLA
world-model VLA (`../world_model_head`, ablation `b_vla_diffusion`) share the
conditioning trunk. Data, inverse dynamics, steering map, prior, rollout,
action bounds and `evaluate()` are imported from `../world_model_head`, so
`--trunk shared --grad-mode joint` reproduces
`train.py --ablation b_vla_diffusion`.

## Modes

    --trunk shared      one conditioning trunk for both heads (baseline)
    --trunk split       separate img_proj + GRU for the flow head

    --grad-mode joint   one summed loss (baseline)
    --grad-mode staged  phase A: trunk + flow on the flow loss, then frozen;
                        phase B: action head with p_sampled_z = 1
    --grad-mode pcgrad  joint training with PCGrad on the shared trunk (Yu et al., 2020)

`staged` freezes the trunk together with the flow head; with `--trunk split`
the action head keeps its own trainable trunk. Phase A early-stops on
validation flow loss and phase B on validation ADE, and the epoch budget is
split between the two (`--stage-a-frac`, default 0.4).

The baseline training details are kept: action pretraining on
inverse-dynamics actions (at the start of phase B in staged mode), the
zero-initialised action head, the jerk and d-kappa penalties, the two
optimiser groups with stronger weight decay on the residual head, and early
stopping on validation ADE after pretraining.

## Running

    python train_world_model_modes.py                                   # shared / joint
    python train_world_model_modes.py --trunk split --grad-mode staged
    python compare_wm.py                                                # shared-joint and split-staged
    python compare_wm.py --all                                          # all five modes
    python compare_wm.py --max-train 8000 --epochs 15                   # quick check

Results go to `runs_wm/<trunk>-<grad_mode>/report.json`. `compare_wm.py`
writes `comparison_world_model.json`, includes the reference run from
`../world_model_head/runs_b_vla_diffusion/report.json`, and prints the share
of the test ADE carried by the learned residual.

Data paths follow `../world_model_head/config.py` (`--preprocessed-dir`,
`--meta-dir`, or `COVLA_PREPROCESSED_DIR` / `COVLA_META_DIR`).

## Files

    modes.py                     mode definition, freezing, PCGrad, gradient-cosine probe
    wm_heads.py                  SplitTrunkWorldModelVLA, parameter groups, optimiser
    wm_trainer.py                joint / pcgrad / staged training loops
    train_world_model_modes.py   driver
    compare_wm.py                runs and tabulates the modes

`modes.py` is shared with `../../RosVLA/multiple_gradient_head/`.
