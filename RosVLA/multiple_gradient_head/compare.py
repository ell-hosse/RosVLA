#!/usr/bin/env python3
"""Run each gradient-routing mode on the same folds and tabulate the results.

    python compare.py                        # all five modes, full CV
    python compare.py --folds 3 --epochs 60  # quicker
    python compare.py --modes shared-joint split-staged

Each mode runs in its own process and writes runs/<mode>/summary.json.
shared-joint is the baseline (the ../velocity_head setup).
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent

ALL_MODES = [
    ("shared", "joint"),     # baseline
    ("shared", "staged"),    # train the flow, freeze it, then train the head
    ("split", "joint"),      # separate flow trunk, joint training
    ("split", "staged"),     # separate flow trunk, staged training
    ("shared", "pcgrad"),    # joint training with PCGrad on the shared trunk
]

ROWS = [
    ("ADE (m)", "ade", True),
    ("FDE (m)", "fde", True),
    ("min-ADE K=5 (m)", "min_ade_k5", True),
    ("v MAE (m/s)", "velocity.v.mae", True),
    ("w MAE (rad/s)", "velocity.w.mae", True),
    ("action accuracy", "accuracy", False),
    ("text cosine", "text_cosine_model", False),
    ("text retrieval acc", "text_retrieval_accuracy", False),
    ("grad cosine (flow vs control)", "grad_cosine_mean_over_training", None),
]


def mode_name(trunk: str, grad: str) -> str:
    return f"{trunk}-{grad}"


def parse_modes(tokens) -> list[tuple[str, str]]:
    out = []
    for t in tokens:
        if t.count("-") != 1:
            raise SystemExit(f"bad --modes entry {t!r}; expected e.g. split-staged")
        trunk, grad = t.split("-")
        out.append((trunk, grad))
    return out


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--modes", nargs="*", default=None,
                    help="subset, e.g. --modes shared-joint split-staged")
    ap.add_argument("--folds", type=int, default=None)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--stage-a-frac", type=float, default=None)
    ap.add_argument("--calculator-dir", default=None)
    ap.add_argument("--skip-existing", action="store_true",
                    help="reuse runs/<mode>/summary.json when it already exists")
    ap.add_argument("--out", default=str(HERE / "comparison.json"))
    args = ap.parse_args()

    modes = parse_modes(args.modes) if args.modes else ALL_MODES
    results = {}

    for trunk, grad in modes:
        name = mode_name(trunk, grad)
        summary_path = HERE / "runs" / name / "summary.json"
        if args.skip_existing and summary_path.exists():
            print(f"[skip] {name} (summary.json present)")
            results[name] = json.loads(summary_path.read_text())
            continue

        cmd = [sys.executable, str(HERE / "run_modes.py"),
               "--trunk", trunk, "--grad-mode", grad, "--quiet"]
        for flag, val in (("--folds", args.folds), ("--epochs", args.epochs),
                          ("--stage-a-frac", args.stage_a_frac),
                          ("--calculator-dir", args.calculator_dir)):
            if val is not None:
                cmd += [flag, str(val)]

        print(f"\n{'=' * 72}\n[run] {name}\n{'=' * 72}")
        proc = subprocess.run(cmd, cwd=str(HERE))
        if proc.returncode != 0 or not summary_path.exists():
            print(f"[!] {name} failed (exit {proc.returncode}); skipping it in the table")
            continue
        results[name] = json.loads(summary_path.read_text())

    if not results:
        raise SystemExit("no modes completed; nothing to compare")

    names = list(results)
    width = max(len(n) for n in names) + 2

    print("\n" + "=" * 72)
    print("COMPARISON   (mean +/- std over folds; lower is better where marked v)")
    print("=" * 72)
    header = "  " + "metric".ljust(34) + "".join(n.rjust(width) for n in names)
    print(header)
    print("  " + "-" * (len(header) - 2))

    table = {}
    for label, key, lower_better in ROWS:
        cells, present = [], False
        for n in names:
            avg = results[n]["average"]
            if key in avg:
                present = True
                cells.append(f"{avg[key]['mean']:.4f}".rjust(width))
                table.setdefault(key, {})[n] = avg[key]
            else:
                cells.append("-".rjust(width))
        if not present:
            continue
        mark = " v" if lower_better else ("" if lower_better is False else " ~")
        print("  " + (label + mark).ljust(34) + "".join(cells))

    base = "shared-joint"
    if base in results and "ade" in results[base]["average"]:
        b = results[base]["average"]["ade"]["mean"]
        print("\n  ADE relative to the shared-joint baseline "
              "(negative = better than baseline):")
        for n in names:
            avg = results[n]["average"]
            if n == base or "ade" not in avg:
                continue
            d = avg["ade"]["mean"] - b
            print(f"    {n:<20s} {d:+.4f} m  ({100 * d / b:+.1f}%)")

    payload = {
        "modes": names,
        "table": table,
        "per_mode": {n: results[n]["average"] for n in names},
    }
    Path(args.out).write_text(json.dumps(payload, indent=2, default=str))
    print(f"\n  comparison -> {args.out}")

    cos_key = "grad_cosine_mean_over_training"
    if base in results and cos_key in results[base]["average"]:
        c = results[base]["average"][cos_key]["mean"]
        verdict = ("the two objectives COOPERATE at the shared trunk -- gradient "
                   "surgery has little to fix here"
                   if c > 0.05 else
                   "the two objectives genuinely CONFLICT at the shared trunk"
                   if c < -0.05 else
                   "the two objectives are near-orthogonal at the shared trunk")
        print(f"\n  flow-vs-control gradient cosine = {c:+.4f}: {verdict}.")


if __name__ == "__main__":
    main()
