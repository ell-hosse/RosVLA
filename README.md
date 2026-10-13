# RosVLA

Code for **"RosVLA: Flow-Matching Latent Language Approximation with a
Kinematic World Model for Efficient Vision–Language–Action Models"**
(E. Hosseini, P. Barati, H. Hassani, S. Nikan).

RosVLA is a compact VLA model (~1.5 M trainable parameters). Instead of
writing a caption with a language model, a rectified-flow head generates the
caption's sentence embedding in a low-dimensional PCA space. A velocity head
then predicts a short chunk of controls, and a parameter-free kinematic world
model integrates them into a trajectory. The model is evaluated on the RosVLA
dataset (Yahboom ROSMASTER X3), CoVLA, DriveLM-nuScenes and BDD-X.

## Setup

```bash
pip install torch torchvision numpy h5py matplotlib sentence-transformers
```

or the `requirements.txt` of the folder you want to run. The image backbone
(MobileNetV3-Small) and the sentence encoder (all-MiniLM-L6-v2) are
downloaded on first use; features are cached in each folder's `cache/`.

## Data

### RosVLA dataset

331 human-verified keyframes from ten scenario runs with RGB-D images,
telemetry, action labels, 3 s waypoints, captions and 3,972 QA pairs.

Download: **[https://huggingface.co/datasets/ell-hosse/RosVLA]**

Place the files so that the repository contains

```
dataset/
  annotations/scenario_*_vla_annotations_FINAL_APPROVED.json
  data/scenario_*.h5
```

The RosVLA scripts look for the data in `<repo>/dataset`.
`python dataset/tools/data_audit.py --root dataset` recomputes the dataset
statistics.

### Public datasets

These are not redistributed; download them from their official sources.

- **CoVLA**: `covla_preprocessed` and `CoVLA-metadata` (see `CoVLA/vla_trajecotry_head/README.md`).
- **DriveLM-nuScenes**: DriveLM v1.1 and the nuScenes metadata and CAN bus (see `DriveLM/README.md`).
- **BDD-X**: annotations and the RAG-Driver processed clips (see `BDD_X/README.md`).

Paths are set with command-line flags or environment variables
(`COVLA_PREPROCESSED_DIR`, `COVLA_META_DIR`, `DRIVELM_ROOT`, `BDDX_ROOT`).

## Reproducing the main results

```bash
# RosVLA (leave-one-scenario-out, 10 folds)
python RosVLA/defussion_text_generator/run_cv.py
python RosVLA/trajectory_predictor/trajectory_calculator.py --export RosVLA/trajectory_predictor/trajectory_predictors
python RosVLA/velocity_head/run_cv.py
python RosVLA/no_text_ablation/run_cv.py
python RosVLA/no_text_ablation/run_cv.py --match-params 1073332
python RosVLA/multiple_gradient_head/compare.py

# CoVLA
python CoVLA/vla_trajecotry_head/train.py
python CoVLA/world_model_head/train.py --all
python CoVLA/ablation_studies/GT_caption_embedding/train.py
python CoVLA/ablation_studies/flaw_matching_based_on_states_only/train.py
python CoVLA/trajectory_predictor/train.py

# DriveLM-nuScenes
python DriveLM/prepare_data.py
python DriveLM/train_wm.py --all --seeds 0 1 2
python DriveLM/train_direct.py --all --seeds 0 1 2

# BDD-X
python BDD_X/prepare_data.py
python BDD_X/train.py --all --seeds 0 1 2
```

See each folder's README for options and outputs.

<!--
## Citation

```bibtex
@article{hosseini2026rosvla,
  title   = {RosVLA: Flow-Matching Latent Language Approximation with a Kinematic
             World Model for Efficient Vision--Language--Action Models},
  author  = {Hosseini, Elaheh and Barati, Parnian and Hassani, Hossein and Nikan, Soodeh},
  journal = {IEEE Transactions on Intelligent Transportation Systems},
  note    = {under review},
  year    = {2026}
}
```
-->
