# CoVLA linear trajectory model

Closed-form ridge regression from a 3 s window of ego state (vEgo, aEgo,
steeringAngleDeg) to the (x, y) trajectory over the same window. Fitted on the
training split and evaluated on train / val / test.

- Input: each signal sampled at 20 Hz over 3 s (60 samples) and decimated to
  20, giving a 60-d vector.
- Output: the (x, y) path over the same window, decimated to 20 points (40-d).
- Model: `y = bias + W @ standardise(x)`, with an unregularised bias.
- Inference: one standardisation and one (61, 40) matrix product.

## Data

Paths: `--meta-dir` / `--preprocessed-dir`, or `COVLA_META_DIR` /
`COVLA_PREPROCESSED_DIR`.

- `CoVLA-metadata/states/<video_id>.jsonl`: per frame (600 per video, 20 Hz),
  vEgo, aEgo, steeringAngleDeg and the next 60 poses in the frame's own
  coordinates. Used for both inputs and targets.
- `CoVLA-metadata/traj_video_60f_traj10.jsonl`: used to list the
  (video_id, frame_idx) anchors (frame_idx = 0, 10, ..., 590).
- Anchors with frame_idx > 540 are dropped (their future window is padded).

The windows are built video by video and cached in `covla_window_cache.npz`
(the states files are ~31 GB). Use `--rebuild-cache` after changing
`RAW_LEN` / `DOWNSAMPLE`.

## Splits

- `--split preprocessed` (default): the video-level split of
  `covla_preprocessed/index/{train,val,test}.jsonl` (6,938 / 402 / 412
  videos, no overlap). Videos outside it are excluded.
- `--split video`: seeded random 80/10/10 split by video.
- `--split official`: `CoVLA-metadata/10k_data_splits.pt`. This is a
  row-level split, so overlapping windows of the same video can end up in
  different splits; it is included for comparison only.

## Usage

```bash
python train.py --limit-videos 100     # quick test
python train.py                        # full run
python train.py --alpha 10             # fixed ridge strength
python train.py --split video
python train.py --split official
```

The ridge strength is chosen on validation ADE. Outputs:
`coefficients.json` (mean, std, alpha and the (61, 40) weights) and
`metrics.json` (ADE / FDE / RMSE / MAE per split).

## Files

- `config.py`: paths and sampling constants
- `data.py`: builds and caches the windowed dataset and split labels
- `model.py`: `LinearWorldModel` and error metrics
- `train.py`: alpha search, fit, evaluation
