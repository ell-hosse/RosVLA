#!/usr/bin/env python3
"""One-time BDD-X preparation (numpy + stdlib only).

    python prepare_data.py                          # data in D:\\BDD-X-Dataset
    python prepare_data.py --data-root E:\\BDD-X
    python prepare_data.py --force                  # rebuild

Reads each clip's info JSON (train = BDDX_Processed, test = BDDX_Test) and
writes cache/dataset.npz with, per clip:

    action, justification             human annotations
    signals (32, 4)                   speed (m/s), acceleration (m/s^2), course,
                                      curvature, aligned with the 32 frames
    duration_s                        clip length from the annotation CSV
                                      (the mp4 headers all report 10.67 s)
    video_id, official_split, mp4 path

Clips are matched to CSV rows by their consecutive numeric suffixes; a match
is only accepted if all action strings agree.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import HERE, N_CLIP_FRAMES, STATE_FIELDS, default_data_root   # noqa: E402


def find_dir(root: Path, name: str) -> Path:
    for dirpath, dirnames, _ in os.walk(root):
        if name in dirnames:
            return Path(dirpath) / name
        if dirpath.count(os.sep) - str(root).count(os.sep) > 4:
            dirnames[:] = []
    sys.exit(f"could not find a folder named {name!r} under {root}")


def read_split_lists(meta: Path) -> dict[str, str]:
    out = {}
    for sp in ("train", "val", "test"):
        for line in (meta / f"{sp}.txt").read_text().split():
            out[line.split("_", 1)[1]] = sp
    return out


def read_csv(meta: Path) -> dict[str, list[list[tuple]]]:
    rows = defaultdict(list)
    with open(meta / "BDD-X-Annotations_v1.csv", newline="", encoding="utf-8",
              errors="replace") as f:
        for row in csv.DictReader(f):
            vid = row["Input.Video"].split("/")[-1].rsplit(".", 1)[0]
            ann = []
            for k in range(1, 16):
                a = (row.get(f"Answer.{k}action") or "").strip()
                if not a:
                    continue
                try:
                    s, e = float(row[f"Answer.{k}start"]), float(row[f"Answer.{k}end"])
                except (TypeError, ValueError):
                    s = e = None
                ann.append((s, e, a))
            rows[vid].append(ann)
    return rows


def clean(s: str) -> str:
    return " ".join((s or "").replace("\n", " ").split())


def join_text(action: str, justification: str) -> str:
    a = clean(action).rstrip(" .")
    j = clean(justification).rstrip(" .")
    return (f"{a} {j}." if j else f"{a}.").strip()


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-root", default=None)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    root = Path(args.data_root) if args.data_root else default_data_root()
    out_npz = HERE / "cache" / "dataset.npz"
    if out_npz.exists() and not args.force:
        print(f"{out_npz} already exists -- use --force to rebuild")
        return
    out_npz.parent.mkdir(parents=True, exist_ok=True)

    meta = find_dir(root, "BDD-X-metadata")
    folders = {"train_release": find_dir(root, "BDDX_Processed"),
               "test_release": find_dir(root, "BDDX_Test")}
    print(f"data root {root}\n  metadata {meta}\n  " +
          "\n  ".join(f"{k}: {v}" for k, v in folders.items()))
    split_of = read_split_lists(meta)
    csv_rows = read_csv(meta)

    rec = []
    for release, folder in folders.items():
        files = sorted((folder / "info").glob("*.json"))
        print(f"  {release}: {len(files)} clips")
        for p in files:
            d = json.loads(p.read_text(encoding="utf-8"))
            act = just = ""
            for c in d.get("comment", []):
                act = c.get("action", act)
                just = c.get("justification", just)
            ci = d.get("car_info", {})
            try:
                sig = np.stack([np.asarray(ci[k], dtype=np.float32) for k in STATE_FIELDS], -1)
            except (KeyError, ValueError):
                sig = np.full((N_CLIP_FRAMES, 4), np.nan, np.float32)
            if sig.shape != (N_CLIP_FRAMES, 4):
                sig = np.full((N_CLIP_FRAMES, 4), np.nan, np.float32)
            name = p.stem
            vid, suf = name.rsplit("_", 1)
            mp4 = folder / "video" / f"{name}.mp4"
            rec.append({"name": name, "video_id": vid, "suffix": int(suf), "release": release,
                        "official_split": split_of.get(vid, "unknown"),
                        "action": clean(act), "justification": clean(just),
                        "signals": sig, "mp4": str(mp4.relative_to(root)),
                        "mp4_exists": mp4.exists()})

    # durations from the CSV
    by_vid = defaultdict(list)
    for i, r in enumerate(rec):
        by_vid[(r["release"], r["video_id"])].append(i)
    duration = np.full(len(rec), np.nan, np.float32)
    start = np.full(len(rec), np.nan, np.float32)
    annot_row = np.full(len(rec), -1, np.int32)
    for (_, vid), idxs in by_vid.items():
        idxs.sort(key=lambda i: rec[i]["suffix"])
        runs = [[idxs[0]]]
        for i in idxs[1:]:
            (runs[-1].append(i) if rec[i]["suffix"] == rec[runs[-1][-1]]["suffix"] + 1
             else runs.append([i]))
        for k, (run, ann) in enumerate(zip(runs, csv_rows.get(vid, []))):
            if len(run) != len(ann):
                continue
            if not all(rec[i]["action"] == clean(a[2]) for i, a in zip(run, ann)):
                continue
            for i, (s, e, _) in zip(run, ann):
                if s is not None and e is not None and e > s:
                    duration[i], start[i], annot_row[i] = e - s, s, k

    sig = np.stack([r["signals"] for r in rec])
    text = {"action": [r["action"] for r in rec],
            "justification": [r["justification"] for r in rec]}
    text["action_justification"] = [join_text(a, j) for a, j in
                                     zip(text["action"], text["justification"])]
    valid = (np.isfinite(sig).all(axis=(1, 2)) & np.array([r["mp4_exists"] for r in rec])
             & np.array([bool(a) for a in text["action"]])
             & np.array([bool(j) for j in text["justification"]]))

    part = out_npz.with_name("dataset.part.npz")
    np.savez_compressed(
        part,
        name=np.array([r["name"] for r in rec]), video_id=np.array([r["video_id"] for r in rec]),
        release=np.array([r["release"] for r in rec]),
        official_split=np.array([r["official_split"] for r in rec]),
        mp4=np.array([r["mp4"] for r in rec]),
        signals=sig, duration_s=duration, start_s=start, annot_row=annot_row, valid=valid,
        **{k: np.array(v, dtype=object) for k, v in text.items()},
    )
    os.replace(part, out_npz)

    moving = sig[..., 0].mean(1) > 2
    summary = {
        "clips": len(rec),
        "by_release_and_official_split": {
            f"{rel}/{sp}": int(sum(r["release"] == rel and r["official_split"] == sp for r in rec))
            for rel in folders for sp in ("train", "val", "test", "unknown")},
        "valid": int(valid.sum()),
        "missing_mp4": int(sum(not r["mp4_exists"] for r in rec)),
        "videos": len({r["video_id"] for r in rec}),
        "duration_recovered_from_csv": int(np.isfinite(duration).sum()),
        "duration_s_p10_p50_p90": [round(float(x), 1) for x in
                                   np.nanpercentile(duration, [10, 50, 90])],
        "unique_texts": {k: len(set(v)) for k, v in text.items()},
        "speed_mps_max": round(float(np.nanmax(sig[..., 0])), 2),
        "moving_clips_with_all_zero_course": round(float(
            (np.abs(sig[moving, :, 2]).max(1) == 0).mean()), 3),
        "example": {"action": text["action"][0], "justification": text["justification"][0],
                    "action_justification": text["action_justification"][0]},
    }
    (HERE / "cache" / "dataset_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    print(f"\n-> {out_npz}")


if __name__ == "__main__":
    main()
