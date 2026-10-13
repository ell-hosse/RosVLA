# Ablation: flow head conditioned on vehicle state only

Same model and data as [`../../vla_trajecotry_head`](../../vla_trajecotry_head),
except that the flow (caption) head only sees the past 3 s vehicle-state
window, not the image. The trajectory head is unchanged.

```
   2 Hz RGB frame --[frozen MobileNetV3-Small]--> f  (576)
   past 3 s vehicle state (20 x 3) --[GRU]-----> h  (64)

     cond_flow = h  (no image)              cond_traj = [proj(f), h]
           |                                         |
     flow head: noise --Euler--> z                   |
           |                                         |
           +--------------> [cond_traj, z] <---------+
                                   |
                            trajectory head --> 20 future (x, y) waypoints
```

## Motivation

The first sentence of a CoVLA caption describes the ego motion, e.g.

> "The ego vehicle is moving slowly with deceleration and turning right.
> There is a traffic light near the ego vehicle displaying a red signal. ..."

which corresponds to vEgo, aEgo and steeringAngleDeg, already part of the
input. This ablation measures how much of the caption fidelity comes from the
image. The effect of the caption latent on the trajectory is covered by
[`../GT_caption_embedding`](../GT_caption_embedding).

## Differences from the baseline

| | baseline | this ablation (default) |
|---|---|---|
| flow head conditioning | `[proj(f), h]`, dim 192 | `h`, dim 64 |
| trajectory head conditioning | `[proj(f), h]`, dim 192 | `[proj(f), h]`, dim 192 |
| conditioning method | `model.condition()` | `model.conditions()` -> `(cond_flow, cond_traj)` |
| output location | `runs/` | `runs/<variant>/` |

`data.py`, `encoders.py`, `history.py`, `latent.py`, `metrics.py` and
`utils.py` are the same as in the baseline. With `--flow-uses-image` the
model has the same parameters as the baseline (1,479,112). `cache/` was
copied from `../GT_caption_embedding/cache`.

## Running

```bash
python train.py --flow-uses-image    # control (baseline conditioning)
python train.py                      # ablation (state-only flow head)
python compare.py                    # comparison table

python train.py --traj-state-only    # optional: fully state-only pipeline
```

Each run writes `runs/<variant>/model.pt` and `runs/<variant>/report.json`.
Other options are the same as in the baseline.

## Metrics

The main metric is `text.scaled_score` (1 = PCA ceiling, 0 = mean caption
baseline), since raw sentence-embedding cosines are all close to 1. For
reference, the image + state model in `../GT_caption_embedding` reaches a
test `cosine_model` of 0.9526 (ceiling 0.9951, mean baseline 0.9405,
scaled score 0.2214). ADE/FDE is reported as well; the trajectory head is
the same in all runs.
