# Flow-matching text-embedding VLA (RosVLA)

The main RosVLA model. A frozen CNN encodes the camera frame, a flow-matching
head generates the sentence embedding of the frame's caption, and that
embedding together with the telemetry is used by the trajectory and action
heads. No language model is run at training or inference time.

```
   RGB frame --[frozen MobileNetV3-Small]--> f  (576)
   telemetry -------------------------------> t  (16)
                                    cond = [f, t]
                                         |
                      flow head: noise --Euler steps--> z   (PCA text latent, <= 8)
                                         |
                              [cond, z] --> 6 waypoints (x, y)
                                        --> action class (5)
```

## Running

```bash
pip install -r RosVLA/defussion_text_generator/requirements.txt
python RosVLA/defussion_text_generator/run_cv.py
```

The first run downloads the ImageNet MobileNetV3 weights and the MiniLM
sentence encoder and caches the extracted features in `cache/`. Options:
`--epochs N`, `--text-field caption_detailed`, `--folds N` (debug).

## Evaluation

Leave-one-scenario-out cross-validation over the 10 scenarios (frames are
0.5 s apart, so splitting by frame would leak). Results are reported as
mean +- std over folds:

| Metric | Description |
|---|---|
| ADE / FDE | displacement error over the 6 waypoints / at t+3.0 s (m) |
| min-ADE / min-FDE (K=5) | best of 5 flow samples |
| action accuracy | 5-way, with the majority-class baseline |
| cosine (model / PCA ceiling / train mean) | generated vs. true text embedding |
| retrieval accuracy, bank coverage | nearest training text vs. the true text |
| step sweep | metrics and ms/sample for 1, 2, 4, 8 Euler steps |

Outputs: `runs/fold_XX_<scenario>/{model.pt, metrics.json}` and
`runs/summary.json` (including the model size).

The PCA, standardisers, waypoint scaling, class weights, retrieval bank and
mean-embedding baseline are all fitted on the training scenarios of each
fold. The image and text encoders are pretrained and frozen.

### Horizon

Each frame has 6 future waypoints at t+0.5, 1.0, ..., 3.0 s
(waypoint k of frame i is the pose at frame i+k+1).

### Text metrics

`trajectory_text` has only 9 unique strings in the dataset, so the PCA
ceiling is close to 1 and predicting the text is essentially a 9-way
classification. The cosine should be read against the train-mean baseline
and retrieval accuracy against the majority-text rate; both are printed.
`explain_text_metric.py` rescales the cosine between these two references.
`--text-field caption_detailed` (~240 unique strings) gives a richer target.

## Visualisation

```bash
python RosVLA/defussion_text_generator/visualize.py            # last fold
python RosVLA/defussion_text_generator/visualize.py --fold 9
python RosVLA/defussion_text_generator/visualize.py --calib-sweep
```

Writes `runs/fold_XX_<scenario>/<scenario>.mp4` with the ground-truth and
predicted trajectories on the camera image, a bird's-eye panel, and the
true/predicted action. The dataset has no camera calibration, so use
`--calib-sweep` (or `--solve`) to choose the camera height and pitch for the
image overlay; the bird's-eye panel does not depend on it.

## Not included

- Mirror augmentation: `REVERSE_LEFT` has no right-hand counterpart.
- Warm starting between consecutive frames.

## Files

| File | Content |
|---|---|
| `config.py` | settings |
| `data.py` | `.h5` + annotations -> per-frame records |
| `encoders.py` | frozen CNN and sentence encoder (cached) |
| `latent.py` | per-fold PCA text latent |
| `model.py` | flow head, trajectory/action heads, parameter report |
| `metrics.py` | ADE/FDE, action and text metrics |
| `run_cv.py` | cross-validation (entry point) |
| `explain_text_metric.py` | text cosine relative to its baselines |
| `model_size.py` | parameter count without training |
| `visualize.py` | per-fold videos |
| `export_predictions.py` | per-frame test predictions for the paper figure |
