"""Load cache/dataset.npz (from prepare_data.py) into train/val/test splits.

  1. keep rows with a full 3 s future, a CAN window and a CAM_FRONT image;
  2. split DriveLM rows by scene into train/val/test (70/15/15, seeded by
     cfg.split_seed). DriveLM only has a train split, so these are all
     nuScenes-train scenes;
  3. if drivelm_nus_imgs_val.zip was available, its rows (official nuScenes
     val scenes) form an extra evaluation-only split `nusc_val` (trajectory
     metrics only, since DriveLM has no public val answers).

All ablations use the same rows.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import torch

import encoders
from config import HERE, Config


@dataclass
class Split:
    name: str
    img_feat: np.ndarray | None
    hist: np.ndarray               # (N, 20, 4)
    anchor_state: np.ndarray       # (N, 4) v, a, steer_deg, yaw_rate at t = 0
    traj: np.ndarray               # (N, 6, 2) future waypoints, ego frame, metres
    cap_text: np.ndarray           # (N,) caption of cfg.text_field ('' if unavailable)
    behavior: np.ndarray           # (N,) DriveLM behavior answer ('' if unavailable)
    cap_emb: np.ndarray | None     # (N, 384) or None
    has_text: bool
    hist_padded: np.ndarray        # (N,) CAN window edge-padded at scene start
    scene_name: np.ndarray
    sample_token: np.ndarray
    location: np.ndarray

    def __len__(self):
        return len(self.traj)

    @property
    def v0(self):
        return self.anchor_state[:, 0].astype(np.float64)

    @property
    def a0(self):
        return self.anchor_state[:, 1].astype(np.float64)

    @property
    def steer0(self):
        return self.anchor_state[:, 2].astype(np.float64)

    @property
    def yaw_rate0(self):
        return self.anchor_state[:, 3].astype(np.float64)

    @property
    def traj7(self):
        """(N, 7, 2) trajectory with the origin prepended."""
        return np.concatenate([np.zeros((len(self.traj), 1, 2), np.float32), self.traj], axis=1)


def _scene_split(scenes: np.ndarray, fracs, seed: int) -> dict[str, set]:
    uniq = np.array(sorted(set(scenes)))
    rng = np.random.default_rng(seed)
    rng.shuffle(uniq)
    n = len(uniq)
    n_tr = int(round(fracs[0] * n))
    n_va = int(round(fracs[1] * n))
    return {"train": set(uniq[:n_tr]), "val": set(uniq[n_tr:n_tr + n_va]),
            "test": set(uniq[n_tr + n_va:])}


def load_splits(cfg: Config, force_images: bool = False, verbose: bool = True) -> tuple[dict, dict]:
    path = HERE / "cache" / "dataset.npz"
    if not path.exists():
        raise SystemExit(f"{path} not found -- run `python prepare_data.py` first")
    d = np.load(path, allow_pickle=True)

    usable = d["traj_valid"] & d["hist_valid"] & d["img_exists"]
    src = d["source"]
    info = {
        "rows_in_file": int(len(usable)),
        "dropped_no_full_future": int((~d["traj_valid"]).sum()),
        "dropped_no_can_window": int((d["traj_valid"] & ~d["hist_valid"]).sum()),
        "dropped_no_image": int((d["traj_valid"] & d["hist_valid"] & ~d["img_exists"]).sum()),
        "usable": int(usable.sum()),
        "text_field": cfg.text_field,
        "split_seed": cfg.split_seed,
    }

    img_dir = cfg.nuscenes_dir / "samples" / "CAM_FRONT"
    all_idx = np.where(usable)[0]

    # frozen image features for all usable rows
    feats_all = None
    if cfg.uses_image:
        paths = [img_dir / str(d["img"][i]) for i in all_idx]
        feats_all = encoders.extract_image_features(
            paths, cfg, cache_path=cfg.cache_dir / "imgfeat_cam_front.npz",
            device=("cuda" if torch.cuda.is_available() and cfg.device == "cuda" else "cpu"),
            force=force_images)
    row_to_feat = {int(r): k for k, r in enumerate(all_idx)}

    # caption embeddings of the chosen field (unique strings only)
    texts = d[cfg.text_field]
    emb_of = None
    if cfg.uses_text:
        uniq = sorted({str(texts[i]) for i in all_idx if str(texts[i])})
        emb = encoders.embed_texts(
            uniq, cfg, cache_path=cfg.cache_dir / f"textemb_{cfg.text_field}.npz",
            device=("cuda" if torch.cuda.is_available() and cfg.device == "cuda" else "cpu"))
        emb_of = {t: e for t, e in zip(uniq, emb)}

    # scene-level split
    is_drivelm = src == "drivelm_train"
    parts = _scene_split(d["scene_name"][usable & is_drivelm], cfg.split_fracs, cfg.split_seed)
    selections = {k: usable & is_drivelm & np.isin(d["scene_name"], list(v))
                  for k, v in parts.items()}
    # official nuScenes val scenes (disjoint from the DriveLM train scenes)
    val_img = usable & (src == "drivelm_val_images") & (d["official_split"] == "val")
    if val_img.any():
        selections["nusc_val"] = val_img

    splits = {}
    for name, mask in selections.items():
        idx = np.where(mask)[0]
        has_text = name != "nusc_val"
        cap = np.array([str(texts[i]) for i in idx], dtype=object)
        cap_emb = None
        if cfg.uses_text and has_text:
            cap_emb = np.stack([emb_of[t] for t in cap]).astype(np.float32)
        splits[name] = Split(
            name=name,
            img_feat=(feats_all[[row_to_feat[int(i)] for i in idx]] if feats_all is not None
                      else None),
            hist=d["hist"][idx].astype(np.float32),
            anchor_state=d["anchor_state"][idx].astype(np.float32),
            traj=d["traj"][idx].astype(np.float32),
            cap_text=cap,
            behavior=np.array([str(d["behavior"][i]) for i in idx], dtype=object),
            cap_emb=cap_emb,
            has_text=has_text,
            hist_padded=d["hist_padded"][idx],
            scene_name=d["scene_name"][idx],
            sample_token=d["sample_token"][idx],
            location=d["location"][idx],
        )
        info[f"rows_{name}"] = int(len(idx))
        info[f"scenes_{name}"] = int(len(set(d["scene_name"][idx])))

    # check that no scene appears in two splits
    names = [set(s.scene_name) for k, s in splits.items() if k != "nusc_val"]
    assert not (names[0] & names[1] or names[0] & names[2] or names[1] & names[2]), \
        "scene leakage between splits"
    if verbose:
        print("  split sizes: " + "  ".join(f"{k}={len(v)} ({info[f'scenes_{k}']} scenes)"
                                           for k, v in splits.items()))
    return splits, info
