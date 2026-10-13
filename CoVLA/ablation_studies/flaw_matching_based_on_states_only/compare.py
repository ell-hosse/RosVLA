#!/usr/bin/env python3
"""Side-by-side table of all runs in runs/.

    python compare.py                 # all variants under runs/
    python compare.py --split test    # one split only
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
CONTROL = "flow-imgstate__traj-imgstate"
ABLATION = "flow-state__traj-imgstate"

LABEL = {
    CONTROL: "control (baseline: image+state -> z)",
    ABLATION: "ABLATION (state only -> z)",
    "flow-state__traj-state": "fully state-only pipeline",
}


def load_runs() -> dict[str, dict]:
    runs = {}
    for p in sorted((HERE / "runs").glob("*/report.json")):
        try:
            runs[p.parent.name] = json.loads(p.read_text())
        except json.JSONDecodeError:
            print(f"  [!] {p} is not valid JSON -- skipped")
    return runs


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", default=None, choices=["train", "val", "test"])
    args = ap.parse_args()

    runs = load_runs()
    if not runs:
        print("No runs found under runs/. Run the control and the ablation first:\n"
              "    python train.py --flow-uses-image\n"
              "    python train.py")
        return

    splits = [args.split] if args.split else ["train", "val", "test"]
    hdr = (f"{'variant':<38} {'split':<6} {'ADE':>7} {'FDE':>7} "
           f"{'minADE':>7} {'minFDE':>7} {'cos':>7} {'scaled':>7}")
    print("\n" + hdr)
    print("-" * len(hdr))
    for variant, rep in runs.items():
        label = LABEL.get(variant, variant)
        for split in splits:
            r = rep.get(split)
            if not r:
                continue
            k_ade = next((k for k in r if k.startswith("min_ade_k")), None)
            k_fde = next((k for k in r if k.startswith("min_fde_k")), None)
            t = r.get("text", {})
            print(f"{label:<38} {split:<6} {r['ade']:>7.4f} {r['fde']:>7.4f} "
                  f"{r.get(k_ade, float('nan')):>7.4f} {r.get(k_fde, float('nan')):>7.4f} "
                  f"{t.get('cosine_model', float('nan')):>7.4f} "
                  f"{t.get('scaled_score', float('nan')):>7.4f}")
        print()

    if CONTROL in runs and ABLATION in runs:
        print("=" * len(hdr))
        print("VERDICT (test split)")
        c, a = runs[CONTROL].get("test", {}), runs[ABLATION].get("test", {})
        cs, as_ = c.get("text", {}).get("scaled_score"), a.get("text", {}).get("scaled_score")
        if cs is not None and as_ is not None:
            keep = (as_ / cs * 100) if abs(cs) > 1e-9 else float("nan")
            print(f"  caption scaled_score : control {cs:.4f}  ->  state-only {as_:.4f}"
                  f"   ({keep:.0f}% retained without ever seeing the frame)")
            print("  " + (
                "The image contributes little to caption generation: the flow head is\n"
                "  largely paraphrasing the telemetry it is already given. Treat the\n"
                "  caption branch as a state verbaliser, not as vision-language grounding."
                if keep > 80 else
                "The image contributes materially to caption generation -- the drop is\n"
                "  real, so the caption latent is not merely a restatement of telemetry."
            ))
        if "ade" in c and "ade" in a:
            print(f"  trajectory ADE       : control {c['ade']:.4f} m  ->  "
                  f"state-only z {a['ade']:.4f} m   (delta {a['ade'] - c['ade']:+.4f} m)")
            print("     the trajectory head is identical in both runs, so this delta is\n"
                  "     entirely the cost of a worse (or no worse) caption latent.")
        print()


if __name__ == "__main__":
    main()
