# No-text ablation (RosVLA)

The main model (`../defussion_text_generator`) without the flow head and the
generated text embedding. Everything else is kept the same: cached CNN
features, telemetry, folds and their order, seeds, epochs, optimiser,
schedule, class weights, head shape and the 6-waypoint (3.0 s) horizon.

```
   full model                          this ablation
   image --CNN--> f --.                image --CNN--> f --.
   telemetry ---> t --+ cond           telemetry ---> t --+ cond
                      |                                   |
              flow head --> z                             |
                      |                                   |
          [cond, z] --+--> waypoints          [cond] -----+--> waypoints
                      +--> action                         +--> action
```

## Running

```bash
python RosVLA/no_text_ablation/run_cv.py
python RosVLA/no_text_ablation/run_cv.py --match-params 1073332
python RosVLA/no_text_ablation/compare.py
```

The feature cache of `../defussion_text_generator/cache/` is reused when it
exists. No sentence encoder is needed.

Removing the flow head also removes most of the parameters, so the ablation
is run twice: once as is, and once with the head trunk widened to match the
full model's parameter count.

| Variant | Trainable parameters |
|---|---|
| full model | 1,073,332 |
| no-text ablation | 184,749 |
| no-text, capacity matched (`hidden=916`) | 1,073,769 |

## compare.py

Both runs hold out the same scenarios, so `compare.py` compares them per
fold. For ADE, FDE and action accuracy it prints the per-scenario values and
differences, the number of folds where the text model is better, and a
paired t-statistic (|t| > 2.26 corresponds to p < 0.05 for 10 folds).

## Files

| File | Difference from the full model |
|---|---|
| `data.py` | identical |
| `encoders.py` | no sentence encoder |
| `config.py` | no text / PCA / flow settings; adds `match_params` |
| `model.py` | no flow head; the head only takes `cond` |
| `metrics.py` | no text metrics, no min-ADE (deterministic model) |
| `run_cv.py` | no PCA, flow loss or sampling |
| `compare.py` | paired comparison with the full model |
