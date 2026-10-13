# Gradient routing between the caption and control heads (RosVLA)

Experiments on how the flow (caption) head and the control head share the
conditioning trunk of the velocity-head VLA. Data, encoders, PCA latent,
plant model and metrics are imported from `../velocity_head`; only the
training loop changes. `--trunk shared --grad-mode joint` reproduces
`../velocity_head/run_cv.py` and is the baseline.

In the baseline the control head receives either the ground-truth latent or
a `no_grad` flow sample, so the two objectives only share the trunk
(`img_proj`, `tel_proj`, ~74k parameters). During training the control head
sees the ground-truth latent in part of the batches (`p_sampled_z = 0.5`),
while at inference it always gets a sample.

## Modes

    --trunk shared      one conditioning trunk for both heads (baseline)
    --trunk split       separate trunk for the flow head

    --grad-mode joint   one summed loss (baseline)
    --grad-mode staged  phase A: trunk + flow on the flow loss, then frozen;
                        phase B: control head with p_sampled_z = 1
    --grad-mode pcgrad  joint training with PCGrad on the shared trunk (Yu et al., 2020)

`staged` freezes the trunk together with the flow head so the conditioning
seen by the flow head does not change in phase B. With `--trunk split` the
control head keeps its own trainable trunk. `pcgrad` requires a shared
trunk. The staged runs split the same epoch budget (`--stage-a-frac`,
default 0.4).

Shared-trunk joint runs also log the cosine between the flow and control
gradients on the shared parameters and the fraction of batches where it is
negative.

## Running

    python run_modes.py                                    # baseline
    python run_modes.py --trunk split --grad-mode staged
    python run_modes.py --grad-mode pcgrad
    python run_modes.py --folds 2 --epochs 20              # quick check

    python compare.py                                      # all five modes
    python compare.py --folds 3 --epochs 100
    python compare.py --modes shared-joint split-staged
    python compare.py --skip-existing

Results go to `runs/<trunk>-<grad_mode>/` with the same layout as the
baseline; `compare.py` writes `comparison.json`. The stage-2 models are
looked up under `../trajectory_predictor/`; use `--calculator-dir` to
override.

## Files

    modes.py       mode definition, freezing, PCGrad, gradient-cosine probe
    heads.py       SplitTrunkVelocityVLA and parameter groups
    trainer.py     joint / pcgrad / staged training loops
    run_modes.py   cross-validation driver
    compare.py     runs all modes and tabulates them

`modes.py` is shared with `../../CoVLA/multiple_gradient_head/`.
