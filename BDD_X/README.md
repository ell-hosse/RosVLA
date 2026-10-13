# BDD-X: flow-matching caption head

BDD-X has **no trajectory waypoints**. Its processed release has speed,
acceleration, a per-sample turning value and curvature, but no positions (see
section 6). So this folder tests only the **text half** of the VLA: the
rectified-flow head that generates the caption embedding. It is the same head
as `CoVLA/vla_trajecotry_head` and `DriveLM/`, with no trajectory or action
head.

```
 clip frames (4 of 32, frozen MobileNetV3-Small, 4 × 576) ──Linear+SiLU──► 128  ┐
 clip vehicle signals (32 × [speed, accel, course, curvature, duration]) ─GRU─► 64 ┤ cond
                                                                                  ┘
        FLOW HEAD (rectified flow, FiLM, 4 Euler steps): noise ──► z
        z = PCA latent of MiniLM("<action> <justification>")
```

The target text is the concatenation of the **action** and **justification**
annotations. For example: *"The car slows down because the light ahead turned
red."* Use `--text-field action` or `--text-field justification` to train on
either sentence on its own.

## 1. Setup and data

```bat
cd BDD_X
pip install -r requirements.txt
python prepare_data.py          :: writes cache\dataset.npz
```

The data folder defaults to `D:\BDD-X-Dataset` (or `--data-root` /
`BDDX_ROOT`) and should contain the release as unpacked:

- `BDD-X-metadata\BDD-X-Annotations_v1.csv` and `train.txt`, `val.txt`, `test.txt`
- `video_process\video_process\BDDX_Processed\{info,video}`: 16,392 clips of
  the official **train** videos
- `video_process\video_process\BDDX_Test\{info,video}`: 2,130 clips; 2,123 of
  them are from the official **test** videos. This is the same 2,123-clip test
  set that RAG-Driver and DriveGPT4 report on.

Dataset statistics:

| | |
|---|---|
| clips / videos | 18,522 / 4,048, all valid, no missing mp4 |
| split (by **video**, seed 0) | train 13,931 clips / 3,002 videos, val 2,461 / 530, **test 2,123 / 511** |
| 7 test-release clips from official *val* videos | left out |
| real clip duration recovered from the CSV | 18,345 clips; p10 / p50 / p90 = 2 / 6 / 19 s. The rest get the train median and are flagged. |
| unique texts: action+justification / action / justification | 13,048 / 4,453 / 9,314 |

Every clip is one annotated action. Each clip has 32 frames at 455×256 and 32
samples of speed (m/s), acceleration (m/s²), `course` and `curvature`, spread
over the clip. The mp4 headers all report 10.67 s, so the real duration is
taken from the annotation CSV and used as a fifth signal channel.

## 2. Run

```bat
python train.py --all --seeds 0 1 2      :: 3 ablations x 3 seeds
python compare.py                        :: -> comparison_action_justification.md
```

`run_all.bat` does the above, plus a single-frame run and the action-only and
justification-only text fields.

- **First run:** decodes every clip once with OpenCV and runs MobileNet on 8
  frames per clip, about 10–20 min. It also embeds about 13 k captions with
  MiniLM, about 1 min. Both are cached in `cache\`. `--num-workers 4`
  speeds up decoding.
- **Broken videos:** a few mp4s in the release are truncated. For example,
  `1036b362-92000280_11763.mp4` is exactly 256 KiB and has no `moov` atom.
  These are skipped, listed in `cache\undecodable_clips.txt`, and dropped from
  *every* ablation, so all runs use identical rows. The count is saved in each
  report as `undecodable_clips_dropped`.
- **Interrupted runs:** the feature pass saves a checkpoint every ~2,000 clips,
  so if it is interrupted it resumes where it stopped.
- **Later runs:** about 1–3 min each on a GPU.
- **Useful flags:** `--n-frames 1|2|4|8` (default 4; 1 = CoVLA-style single
  frame), `--frame-pool concat|mean`, `--text-field action|justification`,
  `--pca-dim`, `--img-dropout`, `--split-seed`, `--epochs`, `--patience`,
  `--device cpu`.
- **Outputs:** `runs\<ablation>\<text-field>[__<k>f-<pool>]\seed_<k>\{report.json, model.pt}`,
  plus a `summary.json` with mean ± std one level up.

## 3. Ablations

| ablation | the flow head is conditioned on | CoVLA counterpart |
|---|---|---|
| `f1_img_state` | clip frames + clip vehicle signals | `vla_trajecotry_head` (its flow head) |
| `f2_state_only` | vehicle signals only | `ablation_studies/flaw_matching_based_on_states_only` |
| `f3_img_only` | clip frames only | – |

## 4. Metrics

**Embedding fidelity**, identical to CoVLA and DriveLM:

- **Cosine** of the generated embedding to the true caption embedding.
- **PCA ceiling:** the true embedding round-tripped through the latent. This is
  the best any model on this latent can score.
- **Mean baseline:** always predict the average training caption.
- **Scaled score:** 1 means the model reaches the ceiling; 0 means it is no
  better than the mean baseline.
- **Best of 5:** the best cosine over 5 flow samples.
- **kNN baseline:** the average embedding of the 10 nearest training clips in
  the same input space.

**Retrieval-decoded captions.** The generated embedding is mapped to the
nearest training caption, which is then scored on:

- **Action-class accuracy:** stop, slow, accelerate, turn left, turn right,
  lane change, forward, taken from keywords in the action sentence. This is the
  BDD-X analogue of DriveLM's behavior accuracy. Read it against the
  majority-class accuracy.
- **BLEU-4 and CIDEr-D** of the retrieved action and justification against the
  ground truth, ×100 as in the papers. The implementation was checked against
  `pycocoevalcap` and gives identical values (B4 39.75, CIDEr 274.38 on a
  shared test case).
- **Reference rows:** `oracle`, which decodes the *true* embedding (the ceiling
  of this decoder), and `mean`, which returns one caption for every clip.

## 5. Published results (BDD-X test, same 2,123 clips)

RAG-Driver (RSS 2024), Table I. B4 = BLEU-4, C = CIDEr, M = METEOR, all ×100.

| method | action B4 | action C | action M | justification B4 | justification C | justification M |
|---|---|---|---|---|---|---|
| S2VT | 30.2 | 179.8 | 27.5 | 6.3 | 53.4 | 11.2 |
| SAA | 31.8 | 214.8 | 29.1 | 7.1 | 66.1 | 12.2 |
| WAA | 32.3 | 215.8 | 29.2 | 7.3 | 69.5 | 12.2 |
| ADAPT | 34.6 | 247.5 | 30.6 | 11.4 | 102.6 | 15.2 |
| DriveGPT4 | 30.0 | 214.0 | 29.8 | 9.4 | 102.7 | 14.6 |
| RAG-Driver | 34.3 | 260.8 | 30.7 | 11.1 | 109.1 | 14.8 |

**Caveats when comparing:**

- Those models *generate* free text with a video captioner or a 7B multimodal
  LLM. This head generates a 32-d embedding. Its caption scores come from
  retrieving a training caption, so they are capped by the `oracle` row.
- METEOR is not computed here.
- The main metric is the cosine relative to the ceiling, mean baseline and
  kNN baseline, as on CoVLA. For reference,
  CoVLA test scored cosine 0.952 against a ceiling of 0.995 and a mean baseline
  of 0.941 (scaled 0.22).
- BDD-X text is free-form human writing, and about 1,000 videos have two annotators
  (they describe the same moments differently), so the PCA ceiling and mean
  baseline are lower than for CoVLA's templated captions.

## 6. Why there is no trajectory head

- The info JSONs have no GPS or positions.
- `course` starts at 0 in every clip and behaves like a per-sample turn, with
  dropouts. 21 % of moving clips are all zeros, and left turns only add up to
  about −53°.
- The units of `curvature` don't integrate to plausible headings.
- Speed and acceleration are consistent with the recovered timing (∫a·dt
  matches Δv, median ratio 0.99), so a forward-distance target would be
  possible. The sideways part of a rebuilt path would not be reliable.

## 7. Data splits and fitting

- Train, val and test are split by **video**, and an assert checks that no
  video appears in two splits. Test is the official test split.
- The standardisers, the PCA latent (mean, basis, scale), the caption bank, the
  kNN index and the majority class are fitted on **train** only.
- MobileNet and MiniLM are pretrained and frozen.
- Early stopping uses the validation cosine of the *generated* embedding.

## Files

| file | role |
|---|---|
| `prepare_data.py` | info JSONs + CSV → `cache/dataset.npz` (texts, 32×4 signals, real duration, split) |
| `config.py` | paths, frame subsets, text fields, hyper-parameters |
| `encoders.py` | OpenCV decode → frozen MobileNetV3-Small (8 frames/clip, cached); MiniLM text |
| `data.py` | video-level split, frame selection and pooling, signals + duration channel |
| `latent.py`, `utils.py` | PCA latent, standardiser, early stopping (same as CoVLA) |
| `model.py` | GRU, frame projection, rectified-flow head |
| `metrics.py` | cosine / ceiling / mean, retrieval decoding, action classes, BLEU-4, CIDEr-D |
| `train.py` | training, all baselines, report |
| `compare.py` | `comparison_<text-field>.md` with the published table |
