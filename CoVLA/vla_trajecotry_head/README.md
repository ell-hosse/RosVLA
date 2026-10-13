# CoVLA flow-matching VLA

The RosVLA design (`../../RosVLA/defussion_text_generator`: frozen image
encoder, PCA text latent, rectified-flow head with FiLM) applied to CoVLA.
The vehicle state is a 3 s history window (60 samples at 20 Hz, decimated to
20) and all data is used at 2 Hz.

```
   2 Hz RGB frame --[frozen MobileNetV3-Small]--> f  (576)
   past 3 s vehicle state (20 x 3) --[GRU]-----> h  (vehicle_proj_dim)
                                     cond = [proj(f), h]
                                            |
                  flow head (rectified flow): noise --Euler--> z   (PCA caption latent)
                                            |
                                [cond, z] --> trajectory head
                                            |
                             20 future (x, y) waypoints, next 3.0 s
```

The caption embeddings (384-d, all-MiniLM-L6-v2) are precomputed in
`covla_preprocessed`, so no text encoder runs here.

## Running

```bash
pip install -r requirements.txt
python train.py
python train.py --max-train 100000 --max-val 10000 --max-test 10000
python train.py --pca-dim 32
python train.py --device cpu
python train.py --preprocessed-dir D:\hf\covla_preprocessed --meta-dir D:\hf\CoVLA-metadata
python train.py --rebuild-history-cache --rebuild-image-cache
```

Defaults: 40,000 / 4,000 / 4,000 train/val/test rows, up to 50 epochs with
early stopping (patience 5). The state windows and image features are
cached in `cache/` on the first run.

Outputs: `runs/model.pt` (weights, standardisers, PCA) and
`runs/report.json` (metrics and training log).

## Data

Paths can also be set with `COVLA_PREPROCESSED_DIR` / `COVLA_META_DIR`.

- `covla_preprocessed/cache_2hz/*.npz`: one row per 2 Hz anchor frame, with
  the future trajectory `traj` (20 waypoints), image paths, captions and the
  caption embeddings (`capemb_minilm_384.npz`). The video-level train/val/test
  split is used as is.
- `CoVLA-metadata/states/<video_id>.jsonl`: raw 20 Hz vehicle state, used to
  build the past window (`history.py`). If it is missing, the six previous
  2 Hz anchors are resampled to 20 points instead; this is flagged in the
  console and in `report.json["approximate_history"]`.

## Sampling

CoVLA is recorded at 20 Hz.

- Anchors: every 10th raw frame (2 Hz).
- Future trajectory: 60 raw samples over 3 s, every 3rd kept
  (`FUTURE_DOWN_IDX = [0, 3, ..., 57]`).
- Past state: 60 raw samples, every 3rd kept, ending at the anchor
  (`HISTORY_DOWN_IDX = [2, 5, ..., 59]`), giving a (20, 3) window of vEgo,
  aEgo and steeringAngleDeg.
- Anchors within 57 raw frames of the start of a video have no full history
  and are dropped (`MIN_ANCHOR_FOR_HISTORY`).

## Metrics

| Metric | Description |
|---|---|
| ADE / FDE | over the 20 waypoints / at t+3.0 s (m) |
| min-ADE / min-FDE (K=5) | best of 5 flow samples |
| text cosine (model / PCA ceiling / mean baseline) | generated vs. true caption embedding |
| scaled score | (model - mean) / (ceiling - mean) |
| PCA reconstruction cosine / MSE | information kept by the PCA latent |

Image, state and trajectory standardisers and the PCA are fitted on the
training split only.

## Files

| File | Content |
|---|---|
| `config.py` | settings and sampling constants |
| `history.py` | past vehicle-state windows (cached) |
| `encoders.py` | frozen MobileNetV3-Small features (cached) |
| `latent.py` | PCA text latent and `pca_report()` |
| `data.py` | assembles each split |
| `model.py` | GRU state encoder, flow head, trajectory head |
| `metrics.py` | trajectory and text metrics |
| `utils.py` | standardisation, early stopping |
| `train.py` | training and evaluation |
