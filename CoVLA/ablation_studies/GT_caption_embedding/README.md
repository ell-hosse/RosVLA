# Ablation: ground-truth caption embedding

Same model and data as [`../../vla_trajecotry_head`](../../vla_trajecotry_head),
except that the trajectory head is conditioned on the ground-truth caption
embedding (PCA-encoded) instead of the flow-generated one. The resulting
ADE/FDE is an upper bound for a perfect text generator.

```
   2 Hz RGB frame --[frozen MobileNetV3-Small]--> f  (576)
   past 3 s vehicle state (20 x 3) --[GRU]-----> h  (vehicle_proj_dim)
                                     cond = [proj(f), h]
                                            |
                  flow head: noise --Euler--> z_hat   (trained, reported only)

               [cond, z_GT] --> trajectory head --> 20 future (x, y) waypoints
```

The flow head is still trained with the same loss so its text metrics can be
compared with the baseline.

## Differences from the baseline

- `model.py`, `data.py`, `encoders.py`, `history.py`, `latent.py`,
  `metrics.py`, `utils.py` are the same as in the baseline.
- The trajectory head always gets the ground-truth latent, in training and
  evaluation (no `p_sampled_z`).
- No min-ADE / min-FDE, since the prediction is deterministic.
- The report includes `diagnostic_generated_z_ade` / `_fde`: the same model
  evaluated with the flow-generated latent.

## Running

```bash
pip install -r requirements.txt
python train.py
```

Same options as the baseline. Outputs go to `runs/model.pt` and
`runs/report.json`, caches to `cache/`.

## Report

- `ade`, `fde`: with the ground-truth latent
- `diagnostic_generated_z_ade`, `diagnostic_generated_z_fde`: with the generated latent
- `text.*`: flow head text metrics
- `pca.*`: PCA reconstruction quality
