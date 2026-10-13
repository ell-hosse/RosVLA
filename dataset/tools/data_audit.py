#!/usr/bin/env python3
"""Dataset audit for the Rosmaster X3 VLA project.

Reports:
  - sample counts, duplicate files, sampling rate
  - action / trajectory class balance (and the effect of mirroring)
  - task-instruction diversity
  - depth validity per image row
  - a scenario-level split proposal

Usage:
    python tools/data_audit.py [--root .] [--no-depth]

Requires numpy and h5py.
"""

import argparse
import collections
import glob
import hashlib
import json
import os
import sys

import numpy as np

try:
    import h5py
except ImportError:
    h5py = None


BAR = "=" * 74
SUB = "-" * 74


def header(title):
    print(f"\n{BAR}\n{title}\n{BAR}")


# 1. Files and duplicates
def audit_files(root):
    header("1. FILES")
    ann = sorted(glob.glob(os.path.join(root, "annotations", "*.json")))
    h5s = sorted(glob.glob(os.path.join(root, "data", "*.h5")))
    print(f"annotation files : {len(ann)}")
    print(f"h5 data files    : {len(h5s)}")

    # content-hash the annotations to catch duplicated runs
    hashes = collections.defaultdict(list)
    for fp in ann:
        with open(fp, "rb") as f:
            hashes[hashlib.md5(f.read()).hexdigest()].append(os.path.basename(fp))

    dupes = {h: names for h, names in hashes.items() if len(names) > 1}
    if dupes:
        print("\n  !! DUPLICATE annotation files (identical content):")
        for names in dupes.values():
            print(f"     {names}")
        print("     -> drop one copy before splitting, or it leaks across train/test")
    else:
        print("\n  no duplicate annotation files")

    return ann, h5s, dupes


# 2. Annotations: labels, balance, language diversity
def audit_annotations(ann_files, dupes):
    header("2. ANNOTATIONS")

    seen_hashes = set()
    unique_files = []
    for fp in ann_files:
        with open(fp, "rb") as f:
            h = hashlib.md5(f.read()).hexdigest()
        if h in seen_hashes:
            print(f"  (skipping duplicate: {os.path.basename(fp)})")
            continue
        seen_hashes.add(h)
        unique_files.append(fp)

    actions = collections.Counter()
    trajs = collections.Counter()
    zones = collections.Counter()
    marks = collections.Counter()
    instrs = collections.Counter()
    caps = collections.Counter()
    per_file = {}
    lin, ang = [], []

    for fp in unique_files:
        data = json.load(open(fp))
        per_file[os.path.basename(fp)] = len(data)
        for s in data:
            actions[s["action"]["action_label"]] += 1
            trajs[s["trajectory"]["trajectory_type"]] += 1
            zones[s["map_context"]["zone"]] += 1
            marks[s["map_context"]["nearest_landmark"]] += 1
            instrs[s["task_instruction"]] += 1
            caps[s["caption_short"]] += 1
            lin.append(s["action"]["linear_velocity_target"])
            ang.append(s["action"]["angular_velocity_target"])

    total = sum(per_file.values())
    print(f"\nunique runs      : {len(unique_files)}")
    print(f"total samples    : {total}")
    print(f"samples per run  : min {min(per_file.values())}  "
          f"max {max(per_file.values())}  mean {total/len(per_file):.1f}")

    print(f"\n{SUB}\nACTION LABEL BALANCE")
    for k, v in actions.most_common():
        print(f"  {k:<14} {v:5d}  {100*v/total:5.1f}%  {'#' * int(60*v/total)}")

    majority = actions.most_common(1)[0]
    print(f"\n  Constant-predictor baseline ('{majority[0]}'): "
          f"{100*majority[1]/total:.1f}%")
    print("  -> any model must beat this to mean anything")

    # what mirroring would do
    mirror_map = {"RIGHT_TURN": "LEFT_TURN", "LEFT_TURN": "RIGHT_TURN",
                  "REVERSE_LEFT": "REVERSE_RIGHT", "REVERSE_RIGHT": "REVERSE_LEFT"}
    mirrored = collections.Counter(actions)
    for k, v in actions.items():
        mirrored[mirror_map.get(k, k)] += v
    print(f"\n  After horizontal-mirror augmentation "
          f"(flip_x -> negate angular velocity):")
    for k, v in mirrored.most_common():
        print(f"    {k:<16} {v:5d}")

    print(f"\n{SUB}\nTRAJECTORY TYPES")
    for k, v in trajs.most_common():
        print(f"  {k:<22} {v:5d}")

    print(f"\n{SUB}\nLANGUAGE DIVERSITY  (decides if language-conditioning is possible)")
    print(f"  unique task_instruction : {len(instrs)}")
    for k, v in instrs.most_common(5):
        print(f"      {v:4d} x  {k[:80]}")
    if len(instrs) == 1:
        print("\n  !! ONE unique instruction across the whole dataset.")
        print("     Language input carries ZERO information -- cross-attention on it")
        print("     can only learn a constant bias. Either vary instructions at")
        print("     collection time, or use language as TRAINING-TIME supervision only.")
    print(f"\n  unique caption_short    : {len(caps)}  (these are OUTPUTS, not inputs --")
    print( "                                     feeding them at inference is leakage)")

    print(f"\n{SUB}\nAUX LABEL COVERAGE  (free extra supervision per frame)")
    print(f"  unique zones      : {len(zones)}")
    print(f"  unique landmarks  : {len(marks)}  {dict(marks.most_common(4))}")
    print(f"  labels per frame  : ~5 (action, trajectory, zone, landmark, caption)")

    lin, ang = np.array(lin), np.array(ang)
    print(f"\n{SUB}\nCONTROL TARGETS")
    print(f"  v_lin  : [{lin.min():.3f}, {lin.max():.3f}]  mean {lin.mean():.3f}  "
          f"zeros {(lin==0).sum()}")
    print(f"  v_ang  : [{ang.min():.3f}, {ang.max():.3f}]  mean {ang.mean():.3f}  "
          f"zeros {(ang==0).sum()}")
    if ang.mean() < -0.05:
        print("  -> mean angular velocity is strongly negative: right-turn bias confirmed")

    return unique_files, per_file


# 3. H5: sampling rate and depth validity by row
def audit_h5(h5_files, check_depth=True):
    header("3. SENSOR DATA")
    if h5py is None:
        print("  h5py not installed -- skipping (pip install h5py)")
        return

    dts, durations, nframes = [], [], []
    for fp in h5_files:
        with h5py.File(fp, "r") as f:
            t = f["sample_wall_times"][:]
            if len(t) > 1:
                dts.append(np.diff(t))
            durations.append(t[-1] - t[0])
            nframes.append(len(t))

    alldt = np.concatenate(dts)
    print(f"  frames total     : {sum(nframes)}")
    print(f"  dt               : mean {alldt.mean():.3f}s  "
          f"min {alldt.min():.3f}  max {alldt.max():.3f}")
    print(f"  effective rate   : {1/alldt.mean():.1f} Hz")
    print(f"  total duration   : {sum(durations):.1f}s "
          f"({sum(durations)/60:.1f} min of driving)")
    print(f"\n  -> supervision exists only at {1/alldt.mean():.1f} Hz.")
    print( "     Higher-rate CONTROL requires action chunking, not faster inference.")

    if not check_depth:
        return

    print(f"\n{SUB}\nDEPTH VALIDITY BY IMAGE ROW  (is depth usable for the road?)")
    valid = near = None
    count = 0
    for fp in h5_files:
        with h5py.File(fp, "r") as f:
            if "depth_images" not in f:
                continue
            d = f["depth_images"][:]
            if valid is None:
                valid = np.zeros(d.shape[1:], dtype=np.float64)
                near = np.zeros(d.shape[1:], dtype=np.float64)
            valid += (d > 0).sum(axis=0)
            near += ((d > 0) & (d < 2000)).sum(axis=0)
            count += len(d)

    if valid is None:
        print("  no depth_images found")
        return

    vr = (valid / count).mean(axis=1)
    nr = (near / count).mean(axis=1)
    H = len(vr)
    band = max(1, H // 8)

    print(f"\n  {'rows':<12} {'valid depth':>12} {'within 2m':>12}")
    for a in range(0, H, band):
        b = min(a + band, H)
        print(f"  {f'{a}-{b-1}':<12} {100*vr[a:b].mean():11.1f}% "
              f"{100*nr[a:b].mean():11.1f}%")

    lower = vr[int(0.75 * H):].mean()
    print(f"\n  bottom 25% of rows, valid depth: {100*lower:.1f}%")
    print(f"  whole frame, pixels within 2m  : {100*(near/count).mean():.1f}%")
    if lower < 0.10:
        print("\n  !! Depth is BLIND in the lower frame -- exactly where the road is.")
        print("     Depth-weighted attention is not implementable from this sensor.")
        print("     Use the planar homography instead (camera is rigid, floor is flat):")
        print("       Z(v) = h_cam / tan( theta_pitch + arctan((v - c_y)/f_y) )")
        print("     -> exact per-row distance, precomputed once, zero runtime cost.")


# 4. Split proposal
def propose_split(per_file):
    header("4. SPLIT PROPOSAL")
    print("  Frames 0.5s apart are near-duplicates. A random frame-level split")
    print("  leaks train into test and will report an inflated, meaningless score.")
    print("  SPLIT BY SCENARIO.\n")

    runs = sorted(per_file.items(), key=lambda kv: -kv[1])
    n = len(runs)
    n_test = max(1, round(0.2 * n))
    n_val = max(1, round(0.1 * n))

    # interleave by size so splits are size-balanced
    test = [runs[i][0] for i in range(0, n, max(1, n // n_test))][:n_test]
    rest = [r for r in runs if r[0] not in test]
    val = [rest[i][0] for i in range(0, len(rest), max(1, len(rest) // n_val))][:n_val]
    train = [r[0] for r in rest if r[0] not in val]

    for name, group in (("TRAIN", train), ("VAL", val), ("TEST", test)):
        cnt = sum(per_file[g] for g in group)
        print(f"  {name:<6} {len(group)} runs, {cnt:4d} samples")
        for g in group:
            print(f"           {g}")
    print("\n  Also report the random-frame-split number, labeled as the leakage")
    print("  baseline -- the gap between the two is a useful result in itself.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))))
    ap.add_argument("--no-depth", action="store_true",
                    help="skip the depth scan (it reads every frame)")
    args = ap.parse_args()

    print(f"Auditing dataset at: {args.root}")
    ann, h5s, dupes = audit_files(args.root)
    if not ann:
        print("No annotations found -- check --root")
        sys.exit(1)

    _, per_file = audit_annotations(ann, dupes)
    audit_h5(h5s, check_depth=not args.no_depth)
    propose_split(per_file)

    header("DONE")
    print("  See VLA_DESIGN_PLAN.md for what these numbers imply for the design.")


if __name__ == "__main__":
    main()
