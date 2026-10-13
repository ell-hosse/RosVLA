"""Past 3 s vehicle-state window for every row of covla_preprocessed's cache_2hz.

The window holds vEgo, aEgo and steeringAngleDeg (60 raw samples at 20 Hz,
decimated to 20). covla_preprocessed only stores the state at each 2 Hz
anchor, so the history is read from CoVLA-metadata/states/<video_id>.jsonl.

If CoVLA-metadata is not available (set COVLA_META_DIR or pass --meta-dir),
the six preceding 2 Hz anchors are resampled to 20 points instead. This is a
lower-fidelity approximation; which source was used is recorded in the
report.

Windows are cached to <cache_dir>/history_<split>.npz, keyed on the selected
(video_code, frame_idx) rows, and rebuilt when the selection changes.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from config import (
    Config,
    HISTORY_DOWN_IDX,
    HISTORY_RAW_LEN,
    MIN_ANCHOR_FOR_HISTORY,
    STATE_FIELDS,
    VIDEO_FRAMES,
)


def _cache_path(cfg: Config, split: str) -> Path:
    return cfg.cache_dir / f"history_{split}.npz"


def _parse_states_file(path: Path) -> dict[str, np.ndarray]:
    """One video's states/<id>.jsonl -> {field: (VIDEO_FRAMES,) float32, NaN where missing}."""
    out = {k: np.full(VIDEO_FRAMES, np.nan, dtype=np.float32) for k in STATE_FIELDS}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            (k, rec), = obj.items()
            fi = int(k)
            if fi >= VIDEO_FRAMES:
                continue
            for field_name in STATE_FIELDS:
                if field_name in rec:
                    out[field_name][fi] = rec[field_name]
    return out


def _build_from_raw_states(
    cfg: Config, video_names: np.ndarray, video_code: np.ndarray, frame_idx: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    n = len(frame_idx)
    hist = np.zeros((n, len(HISTORY_DOWN_IDX), len(STATE_FIELDS)), dtype=np.float32)
    valid = np.zeros(n, dtype=bool)

    # group rows by video so each states/<id>.jsonl is parsed once
    order = np.argsort(video_code, kind="stable")
    states_dir = cfg.meta_dir / "states"

    cur_code = None
    cur_arrays = None
    missing_files = 0
    n_done_videos = 0

    for pos in order:
        code = int(video_code[pos])
        if code != cur_code:
            cur_code = code
            vid = video_names[code]
            fp = states_dir / f"{vid}.jsonl"
            cur_arrays = _parse_states_file(fp) if fp.exists() else None
            if cur_arrays is None:
                missing_files += 1
            n_done_videos += 1
            if n_done_videos % 1000 == 0:
                print(f"    ...parsed states for {n_done_videos} videos")

        anchor = int(frame_idx[pos])
        if cur_arrays is None or anchor < MIN_ANCHOR_FOR_HISTORY:
            continue

        raw_idx = anchor - (HISTORY_RAW_LEN - 1) + np.asarray(HISTORY_DOWN_IDX)
        window = np.stack([cur_arrays[f][raw_idx] for f in STATE_FIELDS], axis=1)  # (20, 3)
        if np.isnan(window).any():
            continue

        hist[pos] = window
        valid[pos] = True

    if missing_files:
        print(f"  [history] {missing_files} videos had no states/*.jsonl under {states_dir} "
              f"-> their rows have no history and are dropped downstream.")
    return hist, valid


def _build_from_2hz_fallback(
    state_2hz: np.ndarray, video_code: np.ndarray, frame_idx: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Fallback: resample the 6 preceding 0.5 s anchors to 20 points.

    state_2hz is the (N, 7) "state" column of cache_2hz; columns 0-2 are vEgo,
    aEgo and steeringAngleDeg.
    """
    n = len(frame_idx)
    k = len(HISTORY_DOWN_IDX)   # 20
    hist = np.zeros((n, k, len(STATE_FIELDS)), dtype=np.float32)
    valid = np.zeros(n, dtype=bool)
    field_cols = [0, 1, 2]

    by_video: dict[int, list[int]] = {}
    for i, c in enumerate(video_code):
        by_video.setdefault(int(c), []).append(i)

    src_t = np.linspace(-2.5, 0.0, 6)      # 6 anchors, 0.5 s apart
    dst_t = np.linspace(-2.85, 0.0, k)     # 20 points over ~3 s

    for _, idxs in by_video.items():
        idxs = sorted(idxs, key=lambda i: frame_idx[i])
        for pos_in_video, i in enumerate(idxs):
            if pos_in_video < 5:
                continue   # fewer than 6 preceding anchors
            window_rows = idxs[pos_in_video - 5: pos_in_video + 1]
            raw = state_2hz[window_rows][:, field_cols]     # (6, 3)
            if np.isnan(raw).any():
                continue
            resampled = np.stack(
                [np.interp(dst_t, src_t, raw[:, j]) for j in range(3)], axis=1
            )
            hist[i] = resampled.astype(np.float32)
            valid[i] = True
    return hist, valid


def build_or_load_history(
    cfg: Config, split: str, video_names: np.ndarray, video_code: np.ndarray,
    frame_idx: np.ndarray, state_2hz: np.ndarray, force: bool = False,
) -> tuple[np.ndarray, np.ndarray, bool]:
    cache = _cache_path(cfg, split)
    if cache.exists() and not force:
        d = np.load(cache, allow_pickle=False)
        same = (
            d["frame_idx"].shape == frame_idx.shape
            and np.array_equal(d["frame_idx"], frame_idx)
            and np.array_equal(d["video_code"], video_code)
        )
        if same:
            print(f"  [cache] history window ({split}) <- {cache.name}")
            return d["hist"], d["valid"], bool(d["approximate"])
        print(f"  [history:{split}] cached row selection is stale "
              f"(different max_{split}/seed?) -> rebuilding")

    states_dir = cfg.meta_dir / "states"
    if states_dir.exists():
        print(f"  [history:{split}] building true 20 Hz history windows from {states_dir}")
        hist, valid = _build_from_raw_states(cfg, video_names, video_code, frame_idx)
        approximate = False
    else:
        print(f"  [history:{split}] {states_dir} not found -> using the 2 Hz fallback "
              f"(set COVLA_META_DIR to real CoVLA-metadata for genuine 20 Hz history)")
        hist, valid = _build_from_2hz_fallback(state_2hz, video_code, frame_idx)
        approximate = True

    print(f"  [history:{split}] {int(valid.sum())}/{len(valid)} rows have a full "
          f"3 s history window")
    np.savez_compressed(
        cache, hist=hist, valid=valid, approximate=approximate,
        frame_idx=frame_idx, video_code=video_code,
    )
    return hist, valid, approximate
