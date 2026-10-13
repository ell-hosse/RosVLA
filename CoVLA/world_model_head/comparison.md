# World-model trajectory head -- ablation comparison

Horizon 2.85 s, 20 waypoints (index 0 = the origin), dt = 0.15 s.
Rows: {'train': 35784, 'val': 3567, 'test': 3587}. Prior: `ca`.

## Zero-parameter baselines (same rollout, no learned parameters)

| baseline | test ADE (m) | test FDE (m) |
|---|---|---|
| CV | 1.3019 | 3.0658 |
| CA | 1.1966 | 2.7576 |
| CTRV | 1.4082 | 3.5207 |
| CTRA | 1.3380 | 3.3495 |

Action-envelope ceiling (inverse-dynamics actions clipped to |a|<=8, |kappa|<=0.2): **ADE 0.5546 m**. Integrator round-trip check: 3.42e-01 m (should be ~0).

## Test split

| model | params | ADE (m) | FDE (m) |
|---|---|---|---|
| A. world model alone (state only) | 201,805 | 0.8711 | 1.9167 |
| B. world model in the diffusion VLA loop | 1,545,965 | 0.9073 | 1.9942 |
| C. world model in the VLA loop, no text | 341,197 | 0.9120 | 2.0088 |
| _flow VLA, generated caption (position regressor)_ | 1,479,112 | 0.9647 | 2.1172 |
| _flow VLA, GT caption (position regressor)_ | 1,479,112 | 0.9518 | 2.0732 |
| _no diffusion, GT caption (position regressor)_ | 323,560 | 0.9549 | 2.0902 |
| _ridge state->positions (no rollout)_ | 2,560 | 23.4209 | 25.2221 |

## Validation split

| model | params | ADE (m) | FDE (m) |
|---|---|---|---|
| A. world model alone (state only) | 201,805 | 0.5620 | 1.4422 |
| B. world model in the diffusion VLA loop | 1,545,965 | 0.5956 | 1.5147 |
| C. world model in the VLA loop, no text | 341,197 | 0.6008 | 1.5263 |

## What the diffusion/text pathway is worth

Ablation B (with the generated caption latent) test ADE 0.9073 m; ablation C (no text of any kind) 0.9120 m.
Difference: **+0.0047 m** (+0.52% of C's ADE) for 1,204,768 extra parameters.

## How much of the path is physics vs. the learned residual

| ablation | test ADE (m) | mean abs residual (m) | accel p99 (m/s2) | curvature p99 (1/m) |
|---|---|---|---|---|
| A. world model alone (state only) | 0.8711 | 0.0984 | 1.717 | 0.07068 |
| B. world model in the diffusion VLA loop | 0.9073 | 0.2361 | 1.193 | 0.05382 |
| C. world model in the VLA loop, no text | 0.9120 | 0.1095 | 1.347 | 0.06849 |

A residual that approaches the ADE means the kinematic model has stopped doing the work and the head has become a position regressor with extra steps.
