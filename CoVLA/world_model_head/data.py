"""Assemble one CoVLA split for the world-model head.

Row selection is the same as ../vla_trajecotry_head/data.py, so all runs are
evaluated on the same rows. In addition:

  * anchor_state (N, 3): vEgo, aEgo, steeringAngleDeg at the anchor, used to
    initialise the rollout and fit the steering->curvature map;
  * traj is shifted to start at the origin if needed;
  * image features and caption embeddings are only loaded when the ablation
    uses them (the image-exists filter is always applied).
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
    img_feat: np.ndarray | None   # (N, img_feat_dim) or None if unused
    hist: np.ndarray              # (N, 20, 3) past vehicle-state window
    anchor_state: np.ndarray      # (N, 3)  vEgo, aEgo, steeringAngleDeg at t=0
    traj: np.ndarray              # (N, 20, 2) future trajectory, metres
    cap_emb: np.ndarray | None    # (N, 384) or None if unused
    cap_text: np.ndarray | None
    approximate_history: bool
    video_id: np.ndarray
    frame_idx: np.ndarray

    def __len__(self) -> int:
        return len(self.traj)

    @property
    def v0(self) -> np.ndarray:
        return self.anchor_state[:, 0].astype(np.float64)

    @property
    def a0(self) -> np.ndarray:
        return self.anchor_state[:, 1].astype(np.float64)

    @property
    def steer0(self) -> np.ndarray:
        return self.anchor_state[:, 2].astype(np.float64)


def _load_cache_2hz(cfg: Config, split: str, want_text: bool) -> dict:
    base = cfg.preprocessed_dir / "cache_2hz"
    d = np.load(base / f"{split}.npz", allow_pickle=True)
    p = np.load(base / f"paths_{split}.npz", allow_pickle=True)
    out = {
        "video_code": d["video_code"], "video_names": d["video_names"],
        "frame_idx": d["frame_idx"], "state": d["state"], "traj": d["traj"],
        "paths": p["paths"], "cap_codes": None, "cap_texts": None, "cap_emb_pool": None,
    }
    if want_text:
        c = np.load(base / f"captions_{split}.npz", allow_pickle=True)
        e = np.load(base / "capemb_minilm_384.npz", allow_pickle=True)
        texts, pool = c["texts"], e[split]
        if len(texts) != pool.shape[0]:
            raise ValueError(
                f"{split}: captions_{split}.npz has {len(texts)} unique texts but "
                f"capemb_minilm_384.npz['{split}'] has {pool.shape[0]} rows"
            )
        out["cap_codes"], out["cap_texts"], out["cap_emb_pool"] = c["codes"], texts, pool
    return out


def _subsample(n: int, max_n: int | None, seed: int) -> np.ndarray:
    if max_n is None or max_n >= n:
        return np.arange(n)
    rng = np.random.default_rng(seed)
    idx = rng.choice(n, size=max_n, replace=False)
    idx.sort()
    return idx


def load_split(
    cfg: Config, split: str, force_history: bool = False, force_images: bool = False,
) -> Split:
    want_text = cfg.uses_text
    raw = _load_cache_2hz(cfg, split, want_text)
    n = len(raw["frame_idx"])
    max_n = {"train": cfg.max_train, "val": cfg.max_val, "test": cfg.max_test}[split]
    idx = _subsample(n, max_n, seed=cfg.seed)
    print(f"[{split}] {len(idx)}/{n} rows selected")

    video_code = raw["video_code"][idx]
    frame_idx = raw["frame_idx"][idx]
    state = raw["state"][idx]
    traj = raw["traj"][idx].astype(np.float32)
    paths_rel = raw["paths"][idx]
    video_id = raw["video_names"][video_code]

    # the rollout starts at (0, 0)
    origin_off = float(np.abs(traj[:, 0]).max())
    if origin_off > 1e-6:
        print(f"  [{split}] traj[:, 0] is not the origin (max |.| = {origin_off:.4g}) "
              f"-> re-anchoring every path to its first waypoint")
        traj = traj - traj[:, :1]

    cap_text = cap_emb = None
    if want_text:
        cap_text = raw["cap_texts"][raw["cap_codes"][idx]]
        cap_emb = raw["cap_emb_pool"][raw["cap_codes"][idx]]

    hist, hist_valid, approximate = history_mod.build_or_load_history(
        cfg, split, raw["video_names"], video_code, frame_idx, state, force=force_history,
    )

    # always apply the image-exists filter so all ablations use the same rows
    paths_abs = [cfg.preprocessed_dir / str(p) for p in paths_rel]
    img_exists = np.array([p.exists() for p in paths_abs], dtype=bool)
    n_missing = int((~img_exists).sum())
    if n_missing:
        print(f"  [{split}] {n_missing}/{len(idx)} frame images missing on disk -> dropped")

    keep = hist_valid & img_exists
    print(f"  [{split}] {int(keep.sum())}/{len(idx)} rows kept after history + image "
          f"filtering ({'approximate 2 Hz' if approximate else 'true 20 Hz'} history)")

    img_feat = None
    if cfg.uses_image:
        kept_paths = [p for p, k in zip(paths_abs, keep) if k]
        img_feat = encoders.extract_image_features(
            kept_paths, cfg,
            cache_path=cfg.cache_dir / f"imgfeat_{split}.npz",
            device=("cuda" if torch.cuda.is_available() else "cpu"),
            force=force_images,
        )
    else:
        print(f"  [{split}] ablation {cfg.ablation!r} uses no image -> frozen CNN skipped")

    return Split(
        name=split,
        img_feat=img_feat,
        hist=hist[keep],
        anchor_state=state[keep][:, :3].astype(np.float32),
        traj=traj[keep],
        cap_emb=(cap_emb[keep] if cap_emb is not None else None),
        cap_text=(cap_text[keep] if cap_text is not None else None),
        approximate_history=approximate,
        video_id=video_id[keep],
        frame_idx=frame_idx[keep],
    )
