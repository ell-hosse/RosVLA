#!/usr/bin/env python3
"""Build comparison.md / comparison.json from all finished runs.

    python compare.py                          # text field 'caption', test split
    python compare.py --text-field behavior
    python compare.py --split nusc_val         # official nuScenes val scenes

Reads runs/<family>/<ablation>/<tag>/summary.json (mean +- std over seeds).
No-text ablations (d3, a, c) are stored under tag `no_text`. Published
numbers come from sota.py.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from config import DIRECT_ABLATIONS, HERE, WM_ABLATIONS
from sota import SOTA_DRIVELM, SOTA_NUSCENES, SOURCES

LABEL = {
    "d1_flow_generated": "D1  flow VLA, generated caption  (the model)",
    "d2_gt_caption": "D2  flow VLA, GT caption  (oracle, leaks future)",
    "d3_no_text": "D3  no flow, no caption  (control)",
    "d4_flow_state_only": "D4  flow on CAN window only",
    "a_world_model_only": "A   world model, CAN window only",
    "b_vla_diffusion": "B   world model in the flow VLA  (the model)",
    "c_vla_no_text": "C   world model + image, no caption  (control)",
    "d_vla_gt_caption": "D   world model, GT caption  (oracle, leaks future)",
}


def _load(p: Path):
    try:
        return json.loads(p.read_text())
    except Exception:
        return None


def _f(x, nd=3):
    if x is None:
        return "-"
    if isinstance(x, dict):
        if x.get("n", 1) > 1:
            return f"{x['mean']:.{nd}f} ± {x['std']:.{nd}f}"
        return f"{x['mean']:.{nd}f}"
    return f"{x:.{nd}f}"


def collect(family, ablations, text_field, tag_suffix=""):
    rows = {}
    for ab in ablations:
        for tag in (text_field + tag_suffix, "no_text" + tag_suffix):
            s = _load(HERE / "runs" / family / ab / tag / "summary.json")
            if s:
                rows[ab] = s
                break
    return rows


def any_report(family, ab, tag):
    base = HERE / "runs" / family / ab / tag
    for p in sorted(base.glob("seed_*/report.json")):
        r = _load(p)
        if r:
            return r
    return None


def table(rows: dict, split: str, family: str) -> list[str]:
    L = ["| model | params | ADE (m) | FDE (m) | min-ADE K=5 | L2 1s / 2s / 3s (ST-P3 avg) | "
         "L2 avg ST-P3 | L2 avg UniAD | text cosine (scaled) | behavior acc % |"
         + (" residual (m) |" if family == "wm" else ""),
         "|---|---|---|---|---|---|---|---|---|---|" + ("---|" if family == "wm" else "")]
    for ab, s in rows.items():
        t = s.get(split)
        if not t:
            continue
        mk = next((k for k in t if k.startswith("min_ade_k")), None)
        l2 = "/".join(f"{t[f'l2_stp3_{k}']['mean']:.3f}" for k in ("1s", "2s", "3s"))
        txt = (f"{_f(t['text_cosine_model'])} ({_f(t['text_scaled_score'], 2)})"
               if "text_cosine_model" in t else "-")
        ba = t.get("text_behavior_accuracy")
        beh = (f"{100 * ba['mean']:.1f}" + (f" ± {100 * ba['std']:.1f}" if ba.get("n", 1) > 1 else "")
               + f" (maj {100 * t['behavior_majority_accuracy']:.1f})"
               if "text_behavior_accuracy" in t else "-")
        line = (f"| {LABEL[ab]} | {s['params']:,} | {_f(t['ade_m'])} | {_f(t['fde_m'])} | "
                f"{_f(t[mk]) if mk else '-'} | {l2} | {_f(t['l2_stp3_avg'])} | "
                f"{_f(t['l2_uniad_avg'])} | {txt} | {beh} |")
        if family == "wm":
            line += f" {_f(t.get('residual_mean_abs_m'))} |"
        L.append(line)
    return L


def delta_line(rows, a, b, split, what):
    if a in rows and b in rows and split in rows[a] and split in rows[b]:
        x, y = rows[a][split]["ade_m"]["mean"], rows[b][split]["ade_m"]["mean"]
        return (f"- {what}: {LABEL[a].split('(')[0].strip()} ADE {x:.4f} vs "
                f"{LABEL[b].split('(')[0].strip()} ADE {y:.4f} -> **{y - x:+.4f} m** "
                f"({100 * (y - x) / max(y, 1e-9):+.1f} % of the second).")
    return None


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--text-field", default="caption")
    ap.add_argument("--split", default="test", choices=["train", "val", "test", "nusc_val"])
    ap.add_argument("--tag-suffix", default="",
                    help="compare non-default runs, e.g. __imgdrop-0.2 or __split-1 "
                         "(the folder suffix train_*.py appended)")
    args = ap.parse_args()
    tf, sp = args.text_field, args.split

    direct = collect("direct", DIRECT_ABLATIONS, tf, args.tag_suffix)
    wm = collect("wm", WM_ABLATIONS, tf, args.tag_suffix)
    if not direct and not wm:
        print("no finished runs found under runs/ -- train something first")
        return

    # any report, for data counts and baselines
    ref = None
    for fam, rows in (("wm", wm), ("direct", direct)):
        for ab, s in rows.items():
            tag = (tf if s.get("text_field") else "no_text") + args.tag_suffix
            ref = any_report(fam, ab, tag)
            if ref:
                break
        if ref:
            break

    L = [f"# DriveLM-nuScenes VLA -- comparison (split `{sp}`, text field `{tf}`)", ""]
    d = ref["data"]
    L += [f"Rows: train {d.get('rows_train')}, val {d.get('rows_val')}, test {d.get('rows_test')}"
          + (f", nusc_val {d['rows_nusc_val']}" if "rows_nusc_val" in d else "")
          + f" (scene-level split, split_seed {d['split_seed']}). Horizon 3.0 s, 6 waypoints "
          "at 0.5 s. Mean ± std over seeds where more than one seed was run.", ""]

    if sp in ref.get("baselines", {}):
        L += ["## Zero-parameter baselines (kinematic rollout of the measured state)", "",
              "| baseline | ADE (m) | FDE (m) | L2 1s / 2s / 3s (ST-P3 avg) | L2 avg ST-P3 | "
              "L2 avg UniAD |", "|---|---|---|---|---|---|"]
        for k, v in ref["baselines"][sp].items():
            s3, u = v["l2"]["stp3"], v["l2"]["uniad"]
            L.append(f"| {k.upper()} | {v['ade_m']:.3f} | {v['fde_m']:.3f} | "
                     f"{s3['1s']:.3f}/{s3['2s']:.3f}/{s3['3s']:.3f} | {s3['avg']:.3f} | {u['avg']:.3f} |")
        L.append("")

    if direct:
        L += ["## Direct trajectory head (no world model)", ""] + table(direct, sp, "direct") + [""]
    if wm:
        L += ["## World-model trajectory head", ""] + table(wm, sp, "wm") + [""]

    L += ["## What the language pathway is worth", ""]
    for line in (delta_line(direct, "d1_flow_generated", "d3_no_text", sp,
                            "direct head, generated caption vs no caption"),
                 delta_line(wm, "b_vla_diffusion", "c_vla_no_text", sp,
                            "world model, generated caption vs no caption"),
                 delta_line(direct, "d2_gt_caption", "d1_flow_generated", sp,
                            "direct head, headroom if the generator were perfect"),
                 delta_line(wm, "d_vla_gt_caption", "b_vla_diffusion", sp,
                            "world model, headroom if the generator were perfect"),
                 delta_line(wm, "c_vla_no_text", "a_world_model_only", sp,
                            "world model, what the image adds over the CAN window alone")):
        if line:
            L.append(line)
    L += ["", "A positive number means the first model is better. With ~100 test scenes, "
          "differences smaller than about two seed-stds are noise.", ""]

    # SOTA
    L += ["## Published results for context", "",
          "**Not the same split.** All rows below are on the official nuScenes *val* "
          "scenes; `test` here is a held-out part of nuScenes *train* scenes. Only "
          "`--split nusc_val` is on the SOTA scenes (and even then on DriveLM's subset "
          "of val keyframes). Collision rate is not reproduced.", "",
          "| method | ego status as input | L2 1s / 2s / 3s (ST-P3 avg) | L2 avg ST-P3 | "
          "L2 1s / 2s / 3s (UniAD at-t) | L2 avg UniAD | note |",
          "|---|---|---|---|---|---|---|"]
    for name, ego, s3, u, note in SOTA_NUSCENES:
        a = "/".join(f"{x:.2f}" for x in s3[:3]) if s3 else "-"
        b = "/".join(f"{x:.2f}" for x in u[:3]) if u else "-"
        L.append(f"| {name} | {ego} | {a} | {s3[3]:.2f} | {b} | "
                 f"{f'{u[3]:.2f}' if u else '-'} | {note} |")
    L += ["", "DriveLM paper, Table 2 (DriveLM-nuScenes val):", "",
          "| method | behavior acc | speed acc | steer acc | ADE | collision % |",
          "|---|---|---|---|---|---|"]
    for name, ba, sa, st, ade, col in SOTA_DRIVELM:
        fmt = lambda x: "-" if x is None else f"{x:.2f}"   # noqa: E731
        L.append(f"| {name} | {fmt(ba)} | {fmt(sa)} | {fmt(st)} | {fmt(ade)} | {fmt(col)} |")
    L += ["", "Sources: " + "; ".join(f"[{k}]({v})" for k, v in SOURCES.items()), ""]

    stem = f"comparison_{tf}_{sp}{args.tag_suffix}"
    out_md = HERE / f"{stem}.md"
    out_md.write_text("\n".join(L), encoding="utf-8")
    (HERE / f"{stem}.json").write_text(json.dumps(
        {"direct": direct, "wm": wm, "baselines": ref.get("baselines", {}).get(sp)},
        indent=2, default=str))
    print("\n".join(L))
    print(f"\nwritten -> {out_md}")


if __name__ == "__main__":
    main()
