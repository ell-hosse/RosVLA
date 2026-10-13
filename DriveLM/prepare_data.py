#!/usr/bin/env python3
"""One-time data preparation for DriveLM-nuScenes (numpy + stdlib only).

    python prepare_data.py                         # data in D:\\DriveLM (or $DRIVELM_ROOT)
    python prepare_data.py --data-root D:\\DriveLM
    python prepare_data.py --force                 # rebuild cache/dataset.npz

Steps (each skipped if its output exists):

  1. extract the sample, scene, log and ego_pose tables from
     v1.0-trainval_meta.tgz -> nuscenes/v1.0-trainval/
  2. extract the CAM_FRONT keyframes from drivelm_nus_imgs_train.zip (and
     drivelm_nus_imgs_val.zip if present) -> nuscenes/samples/CAM_FRONT/
  3. extract the pose and steering CAN-bus files of the needed scenes
     -> nuscenes/can_bus/
  4. for each DriveLM keyframe build the future trajectory (next 6 keyframes
     in the current ego frame), the past 2.85 s CAN window
     (20 x speed, accel, steering, yaw rate), the anchor state, and the three
     caption variants (config.TEXT_FIELDS)
  5. save to cache/dataset.npz and cache/dataset_summary.json

Rows without a full future or CAN window are kept with a validity flag and
filtered in data.py.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import tarfile
import time
import zipfile
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import (HISTORY_LEN, HISTORY_STRIDE_S, HERE, N_FUTURE,  # noqa: E402
                    default_data_root)
from nusc_splits import VAL_SCENES                                     # noqa: E402

META_TABLES = ("sample", "scene", "log", "ego_pose")
CAN_MSGS = ("pose", "steeranglefeedback")
TAG_RE = re.compile(r"<c\d+,[A-Z_]+,[\d.\-]+,[\d.\-]+>")


def extract_meta(tgz: Path, out: Path) -> None:
    want = {f"v1.0-trainval/{t}.json" for t in META_TABLES}
    marker = out / "v1.0-trainval" / ".drivelm_vla_meta_ok"
    if marker.exists():
        print("  [meta] already extracted")
        return
    if not tgz.exists():
        sys.exit(f"missing {tgz}")
    print(f"  [meta] streaming {tgz.name} (sequential gzip read, ~1-3 min) ...")
    t0 = time.time()
    with tarfile.open(tgz, "r:gz") as tf:
        for m in tf:
            if m.name in want:
                dst = out / m.name
                if dst.exists() and dst.stat().st_size == m.size:
                    want.discard(m.name)
                    if not want:
                        break
                    continue
                dst.parent.mkdir(parents=True, exist_ok=True)
                # write to .part and rename so an interrupted run leaves no partial file
                part = dst.with_name(dst.name + ".part")
                with tf.extractfile(m) as src, open(part, "wb") as f:
                    while True:
                        buf = src.read(1 << 24)
                        if not buf:
                            break
                        f.write(buf)
                os.replace(part, dst)
                print(f"    extracted {m.name} ({m.size / 1e6:.0f} MB)")
                want.discard(m.name)
                if not want:
                    break
    if want:
        sys.exit(f"these tables were not found in {tgz.name}: {sorted(want)}")
    marker.write_text("ok")
    print(f"  [meta] done in {time.time() - t0:.0f} s")


def extract_cam_front(zip_path: Path, out: Path) -> int:
    """Copy nuscenes/samples/CAM_FRONT/*.jpg out of a DriveLM image zip."""
    if not zip_path.exists():
        return 0
    dst_dir = out / "samples" / "CAM_FRONT"
    dst_dir.mkdir(parents=True, exist_ok=True)
    n_new = n_all = 0
    with zipfile.ZipFile(zip_path) as z:
        for info in z.infolist():
            if "/CAM_FRONT/" not in info.filename or info.is_dir():
                continue
            n_all += 1
            dst = dst_dir / Path(info.filename).name
            if dst.exists() and dst.stat().st_size == info.file_size:
                continue
            with z.open(info) as src, open(dst, "wb") as f:
                f.write(src.read())
            n_new += 1
    print(f"  [images] {zip_path.name}: {n_all} CAM_FRONT frames ({n_new} newly extracted)")
    return n_all


def extract_can(zip_path: Path, out: Path, scene_names: set[str]) -> None:
    dst_dir = out / "can_bus"
    dst_dir.mkdir(parents=True, exist_ok=True)
    todo = [f"can_bus/{s}_{m}.json" for s in sorted(scene_names) for m in CAN_MSGS]
    todo = [t for t in todo if not (out / t).exists()]
    if not todo:
        print("  [can_bus] already extracted")
        return
    if not zip_path.exists():
        sys.exit(f"missing {zip_path}")
    with zipfile.ZipFile(zip_path) as z:
        names = set(z.namelist())
        n_missing = 0
        for t in todo:
            if t not in names:
                n_missing += 1
                continue
            (out / t).write_bytes(z.read(t))
    print(f"  [can_bus] extracted {len(todo) - n_missing} files; "
          f"{n_missing} requested files do not exist in the CAN expansion "
          f"(nuScenes has no CAN data for a few scenes -- those rows get dropped)")


def load_table(nusc: Path, name: str):
    with open(nusc / "v1.0-trainval" / f"{name}.json", "r", encoding="utf-8") as f:
        return json.load(f)


def stream_json_array(path: Path, chunk: int = 1 << 24):
    """Stream the objects of a large top-level JSON array (ego_pose.json is 646 MB)."""
    dec = json.JSONDecoder()
    with open(path, "r", encoding="utf-8") as f:
        buf = f.read(chunk)
        pos = buf.index("[") + 1
        while True:
            while True:  # skip whitespace and commas
                while pos < len(buf) and buf[pos] in " \t\r\n,":
                    pos += 1
                if pos < len(buf):
                    break
                more = f.read(chunk)
                if not more:
                    return
                buf, pos = buf[pos:] + more, 0
            if buf[pos] == "]":
                return
            try:
                obj, end = dec.raw_decode(buf, pos)
            except json.JSONDecodeError:
                more = f.read(chunk)
                if not more:
                    raise
                buf, pos = buf[pos:] + more, 0
                continue
            yield obj
            pos = end


def quat_yaw(q) -> float:
    w, x, y, z = q
    return float(np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z)))


def load_ego_poses(nusc: Path):
    """Return (timestamps int64 sorted, x, y, yaw) for all ego poses."""
    cache = nusc / "v1.0-trainval" / "ego_pose_xyyaw.npz"
    if cache.exists():
        d = np.load(cache)
        return d["ts"], d["x"], d["y"], d["yaw"]
    print("  [ego_pose] streaming ego_pose.json once (~1-2 min) ...")
    t0 = time.time()
    ts, xs, ys, yaws = [], [], [], []
    for o in stream_json_array(nusc / "v1.0-trainval" / "ego_pose.json"):
        ts.append(o["timestamp"])
        xs.append(o["translation"][0])
        ys.append(o["translation"][1])
        yaws.append(quat_yaw(o["rotation"]))
    ts = np.asarray(ts, dtype=np.int64)
    order = np.argsort(ts, kind="stable")
    out = (ts[order], np.asarray(xs)[order], np.asarray(ys)[order], np.asarray(yaws)[order])
    part = cache.with_name("ego_pose_xyyaw.part.npz")
    np.savez(part, ts=out[0], x=out[1], y=out[2], yaw=out[3])
    os.replace(part, cache)
    print(f"  [ego_pose] {len(ts):,} poses in {time.time() - t0:.0f} s -> {cache.name}")
    return out


def pose_at(ts_sorted, x, y, yaw, t: int, tol_us: int = 5_000):
    i = int(np.searchsorted(ts_sorted, t))
    best = None
    for j in (i - 1, i):
        if 0 <= j < len(ts_sorted):
            d = abs(int(ts_sorted[j]) - t)
            if best is None or d < best[0]:
                best = (d, j)
    if best is None or best[0] > tol_us:
        return None
    j = best[1]
    return float(x[j]), float(y[j]), float(yaw[j])


class CanScene:
    def __init__(self, nusc: Path, scene_name: str):
        self.ok = False
        p_pose = nusc / "can_bus" / f"{scene_name}_pose.json"
        p_steer = nusc / "can_bus" / f"{scene_name}_steeranglefeedback.json"
        if not (p_pose.exists() and p_steer.exists()):
            return
        pose = json.loads(p_pose.read_text())
        steer = json.loads(p_steer.read_text())
        if len(pose) < 10 or len(steer) < 10:
            return
        pose.sort(key=lambda m: m["utime"])
        steer.sort(key=lambda m: m["utime"])
        self.t_pose = np.array([m["utime"] for m in pose], dtype=np.float64)
        self.v = np.array([m["vel"][0] for m in pose])
        self.a = np.array([m["accel"][0] for m in pose])
        self.yaw_rate = np.array([m["rotation_rate"][2] for m in pose])
        self.t_steer = np.array([m["utime"] for m in steer], dtype=np.float64)
        self.steer_deg = np.rad2deg(np.array([m["value"] for m in steer]))
        self.ok = True

    def window(self, t0_us: int, pad: bool, tol_s: float = 0.1):
        """Return (hist (20, 4), anchor (4,), valid, padded)."""
        q = t0_us - (np.arange(HISTORY_LEN)[::-1] * HISTORY_STRIDE_S * 1e6)
        lo = max(self.t_pose[0], self.t_steer[0])
        hi = min(self.t_pose[-1], self.t_steer[-1])
        tol = tol_s * 1e6
        covered = (q[0] >= lo - tol) and (q[-1] <= hi + tol)
        anchor_ok = (lo - tol) <= q[-1] <= (hi + tol)
        valid = covered or (pad and anchor_ok)
        padded = (not covered) and valid
        hist = np.stack([
            np.interp(q, self.t_pose, self.v),          # np.interp holds edge values
            np.interp(q, self.t_pose, self.a),
            np.interp(q, self.t_steer, self.steer_deg),
            np.interp(q, self.t_pose, self.yaw_rate),
        ], axis=-1).astype(np.float32)
        return hist, hist[-1].copy(), bool(valid), bool(padded)


def _clean(s: str) -> str:
    s = TAG_RE.sub("the object", s or "")
    return re.sub(r"\s+", " ", s).strip()


def _answer(qa: dict, section: str, question: str) -> str | None:
    for q in qa.get(section, []):
        if q["Q"].strip() == question:
            return _clean(q["A"])
    return None


def build_captions(frame: dict, scene_desc: str) -> dict:
    qa = frame["QA"]
    behavior = " ".join(_clean(q["A"]) for q in qa.get("behavior", [])) or None
    target = _answer(qa, "planning", "What is the target action of the ego vehicle?")
    signal = _answer(qa, "planning",
                     "What is the traffic signal that the ego vehicle should pay attention to?")
    front = _answer(qa, "perception", "What are objects to the front of the ego car?")
    scene_now = _answer(qa, "perception", "Please describe the current scene.")
    comment = _answer(qa, "planning", "What's your comment on this scene?")

    parts = [behavior]
    if target:
        parts.append(f"Target action: {target.rstrip('.')}.")
    if signal:
        parts.append(f"Traffic signal: {signal.rstrip('.')}.")
    if front:
        parts.append(front)
    caption = " ".join(p for p in parts if p)
    full = " ".join(p for p in (caption, scene_now, _clean(scene_desc), comment) if p)
    return {"behavior": behavior or "", "caption": caption, "caption_full": full}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-root", default=None)
    ap.add_argument("--force", action="store_true", help="rebuild cache/dataset.npz")
    ap.add_argument("--no-history-pad", action="store_true",
                    help="drop keyframes in the first 2.85 s of a scene instead of "
                         "edge-padding their CAN window (default: pad + flag)")
    args = ap.parse_args()

    root = Path(args.data_root) if args.data_root else default_data_root()
    nusc = root / "nuscenes"
    out_npz = HERE / "cache" / "dataset.npz"
    if out_npz.exists() and not args.force:
        print(f"{out_npz} already exists -- use --force to rebuild")
        return
    out_npz.parent.mkdir(parents=True, exist_ok=True)
    print(f"data root: {root}")

    # 1-2. extract
    extract_meta(root / "v1.0-trainval_meta.tgz", nusc)
    n_train_img = extract_cam_front(root / "drivelm_nus_imgs_train.zip", nusc)
    n_val_img = extract_cam_front(root / "drivelm_nus_imgs_val.zip", nusc)
    if n_train_img == 0:
        sys.exit(f"no CAM_FRONT images found in {root / 'drivelm_nus_imgs_train.zip'}")

    # tables
    print("  [meta] loading sample / scene / log tables")
    samples = {s["token"]: s for s in load_table(nusc, "sample")}
    scenes = {s["token"]: s for s in load_table(nusc, "scene")}
    logs = {l["token"]: l for l in load_table(nusc, "log")}
    ego_ts, ego_x, ego_y, ego_yaw = load_ego_poses(nusc)

    # rows
    drivelm = json.loads((root / "v1_1_train_nus.json").read_text(encoding="utf-8"))
    rows = []
    for scene_tok, sc in drivelm.items():
        for sample_tok, fr in sc["key_frames"].items():
            cam = fr.get("image_paths", {}).get("CAM_FRONT")
            if cam is None or sample_tok not in samples:
                continue
            rows.append({"sample": sample_tok, "img": Path(cam).name,
                         "source": "drivelm_train",
                         "texts": build_captions(fr, sc.get("scene_description", ""))})
    print(f"  DriveLM train keyframes with a CAM_FRONT image: {len(rows)}")

    if n_val_img:
        # map each val CAM_FRONT file to its sample: same log, nearest sample timestamp
        by_log = defaultdict(list)
        for s in samples.values():
            sc = scenes[s["scene_token"]]
            by_log[logs[sc["log_token"]]["logfile"]].append((s["timestamp"], s["token"]))
        for k in by_log:
            by_log[k].sort()
        seen = {r["sample"] for r in rows}
        n_val_rows = 0
        with zipfile.ZipFile(root / "drivelm_nus_imgs_val.zip") as z:
            for nm in z.namelist():
                if "/CAM_FRONT/" not in nm or nm.endswith("/"):
                    continue
                fname = Path(nm).name
                log_name, _, ts = fname[:-4].split("__")
                cands = by_log.get(log_name, [])
                if not cands:
                    continue
                tss = np.array([c[0] for c in cands])
                j = int(np.argmin(np.abs(tss - int(ts))))
                if abs(int(tss[j]) - int(ts)) > 100_000:
                    continue
                tok = cands[j][1]
                if tok in seen:
                    continue
                seen.add(tok)
                rows.append({"sample": tok, "img": fname, "source": "drivelm_val_images",
                             "texts": {"behavior": "", "caption": "", "caption_full": ""}})
                n_val_rows += 1
        print(f"  DriveLM val images mapped to nuScenes samples: {n_val_rows} "
              f"(trajectory-only evaluation; DriveLM publishes no val answers)")

    scene_names = {scenes[samples[r["sample"]]["scene_token"]]["name"] for r in rows}
    extract_can(root / "can_bus.zip", nusc, scene_names)

    # per-row targets
    can_cache: dict[str, CanScene] = {}
    pad = not args.no_history_pad
    N = len(rows)
    traj = np.zeros((N, N_FUTURE, 2), np.float32)
    traj_valid = np.zeros(N, bool)
    hist = np.zeros((N, HISTORY_LEN, 4), np.float32)
    anchor = np.zeros((N, 4), np.float32)
    hist_valid = np.zeros(N, bool)
    hist_padded = np.zeros(N, bool)
    ego_found = np.zeros(N, bool)
    meta = defaultdict(list)

    for i, r in enumerate(rows):
        s = samples[r["sample"]]
        sc = scenes[s["scene_token"]]
        log = logs[sc["log_token"]]
        meta["sample_token"].append(r["sample"])
        meta["scene_token"].append(s["scene_token"])
        meta["scene_name"].append(sc["name"])
        meta["location"].append(log["location"])
        meta["timestamp"].append(s["timestamp"])
        meta["img"].append(r["img"])
        meta["source"].append(r["source"])
        meta["official_split"].append("val" if sc["name"] in VAL_SCENES else "train")
        for k in ("behavior", "caption", "caption_full"):
            meta[k].append(r["texts"][k])

        # future: next 6 keyframes, ego frame of the current keyframe
        p0 = pose_at(ego_ts, ego_x, ego_y, ego_yaw, s["timestamp"])
        if p0 is not None:
            ego_found[i] = True
            x0, y0, yaw0 = p0
            c, sn = np.cos(-yaw0), np.sin(-yaw0)
            tok, t_prev, ok = s["next"], s["timestamp"], True
            for k in range(N_FUTURE):
                if not tok:
                    ok = False
                    break
                sk = samples[tok]
                gap = (sk["timestamp"] - t_prev) / 1e6
                pk = pose_at(ego_ts, ego_x, ego_y, ego_yaw, sk["timestamp"])
                if pk is None or not (0.3 <= gap <= 0.7):
                    ok = False
                    break
                dx, dy = pk[0] - x0, pk[1] - y0
                traj[i, k] = (c * dx - sn * dy, sn * dx + c * dy)
                t_prev, tok = sk["timestamp"], sk["next"]
            traj_valid[i] = ok

        # past: CAN bus window
        name = sc["name"]
        if name not in can_cache:
            can_cache[name] = CanScene(nusc, name)
        cs = can_cache[name]
        if cs.ok:
            h, a0, v, pd = cs.window(s["timestamp"], pad=pad)
            hist[i], anchor[i], hist_valid[i], hist_padded[i] = h, a0, v, pd

    img_exists = np.array([(nusc / "samples" / "CAM_FRONT" / f).exists() for f in meta["img"]])

    np.savez_compressed(
        out_npz,
        traj=traj, traj_valid=traj_valid, hist=hist, anchor_state=anchor,
        hist_valid=hist_valid, hist_padded=hist_padded, ego_found=ego_found,
        img_exists=img_exists,
        **{k: np.asarray(v) for k, v in meta.items()},
    )

    src = np.asarray(meta["source"])
    usable = traj_valid & hist_valid & img_exists
    summary = {
        "rows": N,
        "by_source": {k: int((src == k).sum()) for k in np.unique(src)},
        "ego_pose_found": int(ego_found.sum()),
        "future_3s_inside_scene": int(traj_valid.sum()),
        "can_scenes_available": int(sum(c.ok for c in can_cache.values())),
        "can_scenes_total": len(can_cache),
        "history_valid": int(hist_valid.sum()),
        "history_edge_padded": int(hist_padded.sum()),
        "image_on_disk": int(img_exists.sum()),
        "usable_rows": int(usable.sum()),
        "usable_by_source": {k: int((usable & (src == k)).sum()) for k in np.unique(src)},
        "unique_texts_usable_train_source": {
            k: len(set(np.asarray(meta[k])[usable & (src == "drivelm_train")]))
            for k in ("behavior", "caption", "caption_full")},
        "locations": {k: int((np.asarray(meta["location"])[usable] == k).sum())
                      for k in np.unique(meta["location"])},
        "speed_mps_at_anchor": {
            "mean": round(float(anchor[usable, 0].mean()), 3),
            "max": round(float(anchor[usable, 0].max()), 3)},
        "final_waypoint_dist_m_mean": round(float(np.linalg.norm(traj[usable, -1], axis=-1).mean()), 3),
        "consistency_check_corr_(v0*3s, x_final)": round(float(np.corrcoef(
            anchor[usable, 0] * 3.0, traj[usable, -1, 0])[0, 1]), 4),
    }
    (HERE / "cache" / "dataset_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    print(f"\n-> {out_npz}")


if __name__ == "__main__":
    main()
