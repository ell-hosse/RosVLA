"""Build (state window -> trajectory window) samples from CoVLA.

One sample per anchor frame t:

    X[i] : vEgo, aEgo, steeringAngleDeg over t..t+59, decimated to 20 -> (60,)
    Y[i] : states[t]["trajectory"][:, :2], decimated to 20         -> (40,)

X and Y cover the same 3 s window, so this fits the vehicle response to a
known speed / acceleration / steering history (not forecasting).

Each window carries three split labels; train.py chooses one with --split:

    split_preprocessed   covla_preprocessed/index/*.jsonl (video-level, default)
    split_video          seeded random video-level holdout
    split_official       CoVLA-metadata/10k_data_splits.pt (row-level)

Parsing states/*.jsonl (~31 GB) is slow, so the result is cached to
config.cache_path. Use --rebuild-cache to re-parse.
"""
from __future__ import annotations

import json
import pickle
import re
import zipfile
from collections import defaultdict
from pathlib import Path

import numpy as np

from config import (
    Config,
    DOWN_IDX,
    MAX_ANCHOR,
    RAW_LEN,
    STATE_FIELDS,
    VIDEO_FRAMES,
)

_ROW_RE = re.compile(rb'"video_id":\s*"([0-9a-fA-F]+)",\s*"frame_idx":\s*(\d+)')
_VIDEO_ID_RE = re.compile(rb'"video_id":\s*"([0-9a-fA-F]+)"')

SPLIT_KEYS = ("split_official", "split_video", "split_preprocessed")


# 0. sanity-check the dataset location
def _require(path: Path, what: str):
    if not path.exists():
        raise FileNotFoundError(
            f"expected {what} at {path}\n"
            "  pass --meta-dir / --preprocessed-dir, or set COVLA_META_DIR / "
            "COVLA_PREPROCESSED_DIR."
        )
    return path


# 1. row index: row number in traj_video_60f_traj10.jsonl -> (video_id, frame_idx)
#    (parsed with a regex; the bundled trajectory field is not needed)
def build_row_index(cfg: Config):
    if cfg.row_index_cache.exists():
        d = np.load(cfg.row_index_cache, allow_pickle=False)
        return d["video_id"], d["frame_idx"]

    path = _require(cfg.meta_dir / "traj_video_60f_traj10.jsonl", "traj_video_60f_traj10.jsonl")
    vids, idxs = [], []
    with open(path, "rb") as f:
        for line_no, line in enumerate(f):
            m = _ROW_RE.search(line)
            if not m:
                raise ValueError(f"{path}:{line_no}: unexpected row format: {line[:120]!r}")
            vids.append(m.group(1).decode("ascii"))
            idxs.append(int(m.group(2)))

    video_id = np.array(vids, dtype="U16")
    frame_idx = np.array(idxs, dtype=np.int32)
    cfg.row_index_cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cfg.row_index_cache, video_id=video_id, frame_idx=frame_idx)
    return video_id, frame_idx


# 2a. official row-level split
def _load_pt_dict(path: Path) -> dict:
    """Read a {"train": [...], "val": [...], "test": [...]} torch.save() file.

    Falls back to reading data.pkl from the archive directly when torch is not
    installed (the split files only contain plain Python lists).
    """
    try:
        import torch  # noqa: PLC0415
        return torch.load(path, map_location="cpu", weights_only=False)
    except ModuleNotFoundError:
        with zipfile.ZipFile(path) as z:
            name = next(n for n in z.namelist() if n.endswith("data.pkl"))
            return pickle.loads(z.read(name))


def load_official_split(cfg: Config, n_rows: int) -> np.ndarray:
    """-> int8 array of length n_rows: 0=train, 1=val, 2=test, -1=unused."""
    path = _require(cfg.meta_dir / "10k_data_splits.pt", "10k_data_splits.pt")
    d = _load_pt_dict(path)
    split_id = np.full(n_rows, -1, dtype=np.int8)
    for code, key in ((0, "train"), (1, "val"), (2, "test")):
        idx = np.asarray(list(d[key]), dtype=np.int64)
        split_id[idx] = code
    return split_id


# 2b. seeded random video-level holdout (--split video)
def assign_video_split(unique_video_ids: np.ndarray, cfg: Config) -> dict:
    rng = np.random.default_rng(cfg.seed)
    order = rng.permutation(len(unique_video_ids))
    train_r, val_r, _ = cfg.video_split_ratios
    n = len(unique_video_ids)
    n_train = int(round(n * train_r))
    n_val = int(round(n * val_r))
    codes = np.empty(n, dtype=np.int8)
    codes[order[:n_train]] = 0
    codes[order[n_train:n_train + n_val]] = 1
    codes[order[n_train + n_val:]] = 2
    return {vid: int(c) for vid, c in zip(unique_video_ids, codes)}


# 2c. covla_preprocessed video-level split (default). Only the
#     video_id -> split assignment is used; the state windows come from
#     <meta_dir>/states/.
def _unique_video_ids(path: Path) -> set:
    ids = set()
    with open(path, "rb") as f:
        for line in f:
            m = _VIDEO_ID_RE.search(line)
            if m:
                ids.add(m.group(1).decode("ascii"))
    return ids


def load_preprocessed_split(cfg: Config) -> dict:
    """-> {video_id: 0/1/2} for every video_id that appears in the index."""
    idx_dir = _require(cfg.preprocessed_dir / "index", "covla_preprocessed/index")
    split: dict = {}
    for code, name in ((0, "train"), (1, "val"), (2, "test")):
        fp = _require(idx_dir / f"{name}.jsonl", f"covla_preprocessed/index/{name}.jsonl")
        for vid in _unique_video_ids(fp):
            split[vid] = code
    return split


# 3. parse one video's states/<id>.jsonl once, for the anchors we need from it
def _parse_states_file(path: Path, anchors_needed: set):
    v = np.full(VIDEO_FRAMES, np.nan, dtype=np.float32)
    a = np.full(VIDEO_FRAMES, np.nan, dtype=np.float32)
    s = np.full(VIDEO_FRAMES, np.nan, dtype=np.float32)
    traj_xy = {}   # frame_idx -> (RAW_LEN, 2) float32, only for anchors_needed
    n_seen = 0

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            (k, rec), = obj.items()
            fi = int(k)
            n_seen = max(n_seen, fi + 1)
            if fi >= VIDEO_FRAMES:
                continue
            v[fi] = rec[STATE_FIELDS[0]]
            a[fi] = rec[STATE_FIELDS[1]]
            s[fi] = rec[STATE_FIELDS[2]]
            if fi in anchors_needed:
                tj = np.asarray(rec["trajectory"], dtype=np.float32)  # (RAW_LEN, 3): x, y, yaw
                if tj.shape[0] < RAW_LEN:
                    continue  # malformed record, anchor is skipped
                traj_xy[fi] = tj[:RAW_LEN, :2]

    return v, a, s, traj_xy, n_seen


def _downsample_window(arr_60: np.ndarray) -> np.ndarray:
    return arr_60[DOWN_IDX]


# 4. full dataset builder (streams video-by-video, caches the result)
def build_dataset(cfg: Config, force: bool = False, limit_videos: int | None = None,
                   progress_every: int = 500) -> dict:
    if cfg.cache_path.exists() and not force and limit_videos is None:
        return load_cache(cfg.cache_path)

    video_id_row, frame_idx_row = build_row_index(cfg)
    n_rows = len(video_id_row)
    split_official_row = load_official_split(cfg, n_rows)

    states_dir = _require(cfg.meta_dir / "states", "the states/ directory")

    rows_by_video = defaultdict(list)
    for row in range(n_rows):
        rows_by_video[video_id_row[row]].append(row)

    unique_videos = np.array(sorted(rows_by_video.keys()))
    video_split_random = assign_video_split(unique_videos, cfg)
    video_split_preprocessed = load_preprocessed_split(cfg)
    n_not_in_preprocessed = sum(1 for vid in unique_videos if vid not in video_split_preprocessed)
    print(f"video-level split source 'preprocessed': {len(video_split_preprocessed)} videos "
          f"matched, {n_not_in_preprocessed} of this dataset's {len(unique_videos)} videos "
          "have no entry there (windows from those get split_preprocessed = -1, i.e. unused "
          "under --split preprocessed).")

    X_parts, Y_parts = [], []
    split_off_parts, split_rand_parts, split_pre_parts = [], [], []
    vid_parts, frame_parts = [], []
    n_skipped_nan, n_skipped_short, n_missing_file = 0, 0, 0

    n_done = 0
    for vid in unique_videos:
        rows = rows_by_video[vid]
        anchors = frame_idx_row[rows]
        keep = anchors <= MAX_ANCHOR
        if not keep.any():
            continue
        rows = [r for r, k in zip(rows, keep) if k]
        anchors = anchors[keep]

        fp = states_dir / f"{vid}.jsonl"
        if not fp.exists():
            n_missing_file += len(rows)
            continue

        v, a, s, traj_xy, n_seen = _parse_states_file(fp, set(int(x) for x in anchors))
        pre_code = video_split_preprocessed.get(str(vid), -1)

        for row, anchor in zip(rows, anchors):
            anchor = int(anchor)
            if anchor + RAW_LEN > n_seen:
                n_skipped_short += 1
                continue
            v_win, a_win, s_win = (v[anchor:anchor + RAW_LEN],
                                    a[anchor:anchor + RAW_LEN],
                                    s[anchor:anchor + RAW_LEN])
            if anchor not in traj_xy:
                n_skipped_short += 1
                continue
            if np.isnan(v_win).any() or np.isnan(a_win).any() or np.isnan(s_win).any():
                n_skipped_nan += 1
                continue

            feat = np.concatenate([
                _downsample_window(v_win),
                _downsample_window(a_win),
                _downsample_window(s_win),
            ]).astype(np.float32)
            targ = _downsample_window(traj_xy[anchor]).reshape(-1).astype(np.float32)

            X_parts.append(feat)
            Y_parts.append(targ)
            split_off_parts.append(split_official_row[row])
            split_rand_parts.append(video_split_random[vid])
            split_pre_parts.append(pre_code)
            vid_parts.append(vid)
            frame_parts.append(anchor)

        n_done += 1
        if limit_videos and n_done >= limit_videos:
            break
        if n_done % progress_every == 0:
            print(f"  ... {n_done} videos processed, {len(X_parts)} windows so far")

    if not X_parts:
        raise RuntimeError("no windows were built -- check --meta-dir/--preprocessed-dir "
                            "and the dataset layout")

    result = {
        "X": np.stack(X_parts),
        "Y": np.stack(Y_parts),
        "split_official": np.array(split_off_parts, dtype=np.int8),
        "split_video": np.array(split_rand_parts, dtype=np.int8),
        "split_preprocessed": np.array(split_pre_parts, dtype=np.int8),
        "video_id": np.array(vid_parts, dtype="U16"),
        "frame_idx": np.array(frame_parts, dtype=np.int32),
    }

    print(f"built {len(result['X'])} windows from {n_done} videos "
          f"(skipped: {n_skipped_short} short/padded-tail, {n_skipped_nan} NaN, "
          f"{n_missing_file} missing states file)")
    for key in SPLIT_KEYS:
        counts = {c: int((result[key] == c).sum()) for c in (0, 1, 2, -1)}
        print(f"  {key:<20} train={counts[0]:<7} val={counts[1]:<7} "
              f"test={counts[2]:<7} unused={counts[-1]}")

    if limit_videos is None:
        cfg.cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cfg.cache_path, **result)
        print(f"cached -> {cfg.cache_path}")

    return result


def load_cache(path: Path) -> dict:
    d = np.load(path, allow_pickle=False)
    return {k: d[k] for k in ("X", "Y", *SPLIT_KEYS, "video_id", "frame_idx")}
