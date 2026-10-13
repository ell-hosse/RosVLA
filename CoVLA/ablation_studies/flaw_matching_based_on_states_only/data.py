"""Assemble one CoVLA split: image features, past state window, future targets.

Per split:
  1. load the cache_2hz/*.npz rows (video-level train/val/test split);
  2. subsample to cfg.max_{train,val,test} rows if set;
  3. build or load the past state windows for those rows (history.py);
  4. drop rows without a full history window or a frame image on disk;
  5. extract (or load cached) frozen image features for the remaining rows.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

import encoders
import history as history_mod
from config import Config


@dataclass
class Split:
    name: str
    img_feat: np.ndarray        # (N, img_feat_dim) frozen CNN features
    hist: np.ndarray            # (N, 20, 3) past vehicle-state window (vEgo, aEgo, steer)
    traj: np.ndarray            # (N, 20, 2) future trajectory target, metres
    cap_emb: np.ndarray         # (N, 384) target caption embedding
    cap_text: np.ndarray        # (N,) the ground-truth caption strings
    approximate_history: bool   # True if the 2 Hz fallback was used
    video_id: np.ndarray
    frame_idx: np.ndarray


def _load_cache_2hz(cfg: Config, split: str) -> dict:
    base = cfg.preprocessed_dir / "cache_2hz"
    d = np.load(base / f"{split}.npz", allow_pickle=True)
    p = np.load(base / f"paths_{split}.npz", allow_pickle=True)
    c = np.load(base / f"captions_{split}.npz", allow_pickle=True)
    e = np.load(base / "capemb_minilm_384.npz", allow_pickle=True)

    texts, pool = c["texts"], e[split]
    if len(texts) != pool.shape[0]:
        raise ValueError(
            f"{split}: captions_{split}.npz has {len(texts)} unique texts but "
            f"capemb_minilm_384.npz['{split}'] has {pool.shape[0]} rows -- "
            "these are expected to be aligned 1:1 by index in covla_preprocessed."
        )
    return {
        "video_code": d["video_code"], "video_names": d["video_names"],
        "frame_idx": d["frame_idx"], "state": d["state"], "traj": d["traj"],
        "paths": p["paths"], "cap_codes": c["codes"], "cap_texts": texts,
        "cap_emb_pool": pool,
    }


def _subsample(n: int, max_n: int | None, seed: int) -> np.ndarray:
    if max_n is None or max_n >= n:
        return np.arange(n)
    rng = np.random.default_rng(seed)
    idx = rng.choice(n, size=max_n, replace=False)
    idx.sort()
    return idx


def _cuda_available() -> bool:
    return torch.cuda.is_available()


def load_split(
    cfg: Config, split: str, force_history: bool = False, force_images: bool = False,
) -> Split:
    raw = _load_cache_2hz(cfg, split)
    n = len(raw["frame_idx"])
    max_n = {"train": cfg.max_train, "val": cfg.max_val, "test": cfg.max_test}[split]
    idx = _subsample(n, max_n, seed=cfg.seed)
    print(f"[{split}] {len(idx)}/{n} rows selected")

    video_code = raw["video_code"][idx]
    frame_idx = raw["frame_idx"][idx]
    state = raw["state"][idx]
    traj = raw["traj"][idx]
    paths_rel = raw["paths"][idx]
    cap_text = raw["cap_texts"][raw["cap_codes"][idx]]
    cap_emb = raw["cap_emb_pool"][raw["cap_codes"][idx]]
    video_id = raw["video_names"][video_code]

    hist, hist_valid, approximate = history_mod.build_or_load_history(
        cfg, split, raw["video_names"], video_code, frame_idx, state, force=force_history,
    )

    paths_abs = [cfg.preprocessed_dir / str(p) for p in paths_rel]
    img_exists = np.array([p.exists() for p in paths_abs], dtype=bool)
    n_missing = int((~img_exists).sum())
    if n_missing:
        print(f"  [{split}] {n_missing}/{len(idx)} frame images missing on disk -> dropped")

    keep = hist_valid & img_exists
    print(f"  [{split}] {int(keep.sum())}/{len(idx)} rows kept after history + image "
          f"filtering ({'approximate 2 Hz' if approximate else 'true 20 Hz'} history)")

    kept_paths = [p for p, k in zip(paths_abs, keep) if k]
    img_feat = encoders.extract_image_features(
        kept_paths, cfg,
        cache_path=cfg.cache_dir / f"imgfeat_{split}.npz",
        device=("cuda" if _cuda_available() else "cpu"),
        force=force_images,
    )

    return Split(
        name=split,
        img_feat=img_feat,
        hist=hist[keep],
        traj=traj[keep],
        cap_emb=cap_emb[keep],
        cap_text=cap_text[keep],
        approximate_history=approximate,
        video_id=video_id[keep],
        frame_idx=frame_idx[keep],
    )
