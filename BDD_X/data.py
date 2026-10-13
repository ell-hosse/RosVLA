"""Load cache/dataset.npz into train / val / test.

    test        all clips of the official BDD-X test videos (BDDX_Test), as
                used by RAG-Driver / DriveGPT4
    train / val official train videos (BDDX_Processed), split by video with
                cfg.val_frac / cfg.split_seed

Test-release clips whose video is in the official val list are excluded.
All ablations use the same rows.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

import encoders
from config import FRAME_SUBSETS, HERE, Config


@dataclass
class Split:
    name: str
    img: np.ndarray | None        # (N, n_frames*576) or (N, 576) pooled clip features
    state: np.ndarray             # (N, 32, 5) speed, accel, course, curvature, duration
    text: np.ndarray              # (N,) the target text (cfg.text_field)
    action: np.ndarray            # (N,)
    justification: np.ndarray     # (N,)
    emb: np.ndarray | None        # (N, 384) MiniLM embedding of `text`
    video_id: np.ndarray
    name_: np.ndarray
    duration_known: np.ndarray

    def __len__(self):
        return len(self.text)


def _video_split(videos: np.ndarray, frac: float, seed: int) -> set:
    uniq = np.array(sorted(set(videos)))
    rng = np.random.default_rng(seed)
    rng.shuffle(uniq)
    return set(uniq[: int(round(frac * len(uniq)))])


def load_splits(cfg: Config, need_text: bool = True, force_images: bool = False,
                verbose: bool = True) -> tuple[dict, dict]:
    path = HERE / "cache" / "dataset.npz"
    if not path.exists():
        raise SystemExit(f"{path} not found -- run `python prepare_data.py` first")
    d = np.load(path, allow_pickle=True)
    rel, osp = d["release"], d["official_split"]
    usable = d["valid"] & (((rel == "train_release") & (osp == "train")) |
                           ((rel == "test_release") & (osp == "test")))
    usable = usable.copy()
    all_idx = np.where(usable)[0]
    dev = "cuda" if torch.cuda.is_available() and cfg.device == "cuda" else "cpu"

    # clip features (cached); always loaded so undecodable clips are dropped
    # from every ablation
    paths = [cfg.data_root / str(d["mp4"][i]) for i in all_idx]
    feats, decodable = encoders.extract_clip_features(
        paths, [str(d["name"][i]) for i in all_idx], cfg,
        cache_path=cfg.cache_dir / "clipfeat_mnv3s_8f.npz", device=dev, force=force_images)
    pos = {int(r): k for k, r in enumerate(all_idx)}
    n_bad = int((~decodable).sum())
    if n_bad:
        usable[all_idx[~decodable]] = False
        print(f"  {n_bad} clip(s) with an undecodable mp4 dropped (see cache/undecodable_clips.txt)")
    all_idx = np.where(usable)[0]
    if not cfg.uses_image:
        feats = None

    # text embeddings of the chosen field
    texts = d[cfg.text_field]
    emb_of = None
    if need_text:
        uniq = sorted({str(texts[i]) for i in all_idx})
        E = encoders.embed_texts(uniq, cfg, cfg.cache_dir / f"textemb_{cfg.text_field}.npz", dev)
        emb_of = dict(zip(uniq, E))

    # splits
    val_videos = _video_split(d["video_id"][usable & (rel == "train_release")],
                              cfg.val_frac, cfg.split_seed)
    is_val = np.isin(d["video_id"], list(val_videos))
    masks = {"train": usable & (rel == "train_release") & ~is_val,
             "val": usable & (rel == "train_release") & is_val,
             "test": usable & (rel == "test_release")}

    # unknown durations get the training median
    dur = d["duration_s"].astype(np.float32)
    med = float(np.nanmedian(dur[masks["train"]]))

    splits, info = {}, {"undecodable_clips_dropped": n_bad,
                        "text_field": cfg.text_field, "val_frac": cfg.val_frac,
                        "split_seed": cfg.split_seed, "n_frames": cfg.n_frames,
                        "frame_pool": cfg.frame_pool, "duration_fill_median_s": med}
    sub = list(FRAME_SUBSETS[cfg.n_frames])
    for name, m in masks.items():
        idx = np.where(m)[0]
        img = None
        if feats is not None:
            f = feats[[pos[int(i)] for i in idx]][:, sub].astype(np.float32)   # (n, k, 576)
            img = f.reshape(len(idx), -1) if cfg.frame_pool == "concat" else f.mean(1)
        du = dur[idx]
        known = np.isfinite(du)
        du = np.where(known, du, med)
        state = np.concatenate([d["signals"][idx],
                                np.repeat(du[:, None, None], d["signals"].shape[1], 1)], -1)
        tx = np.array([str(texts[i]) for i in idx], dtype=object)
        splits[name] = Split(
            name=name, img=img, state=state.astype(np.float32), text=tx,
            action=np.array([str(d["action"][i]) for i in idx], dtype=object),
            justification=np.array([str(d["justification"][i]) for i in idx], dtype=object),
            emb=(np.stack([emb_of[t] for t in tx]).astype(np.float32) if emb_of else None),
            video_id=d["video_id"][idx], name_=d["name"][idx], duration_known=known)
        info[f"clips_{name}"] = int(len(idx))
        info[f"videos_{name}"] = int(len(set(d["video_id"][idx])))
        info[f"duration_known_{name}"] = int(known.sum())

    v = [set(s.video_id) for s in splits.values()]
    assert not (v[0] & v[1] or v[0] & v[2] or v[1] & v[2]), "video leakage between splits"
    if verbose:
        print("  splits: " + "  ".join(f"{k}={len(s)} clips / {info[f'videos_{k}']} videos"
                                       for k, s in splits.items()))
    return splits, info
