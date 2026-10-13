#!/usr/bin/env python3
"""Run the gradient-routing modes on the world-model VLA and compare with
ablation b of ../world_model_head (test ADE 0.9073 m).

    python compare_wm.py                                   # shared-joint and split-staged
    python compare_wm.py --modes shared-joint shared-staged split-joint split-staged shared-pcgrad
    python compare_wm.py --max-train 8000 --epochs 15      # quick check
    python compare_wm.py --skip-existing                   # reuse finished runs

The reference row is read from ../world_model_head/runs_b_vla_diffusion/
report.json when available.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
PUBLISHED = HERE.parent / "world_model_head" / "runs_b_vla_diffusion" / "report.json"
PUBLISHED_ADE = 0.9073          # fallback if that report is not on disk

DEFAULT_MODES = [("shared", "joint"), ("split", "staged")]
ALL_MODES = [("shared", "joint"), ("shared", "staged"), ("split", "joint"),
             ("split", "staged"), ("shared", "pcgrad")]

ROWS = [
    ("test ADE (m)", ("test", "ade_m"), True),
    ("test FDE (m)", ("test", "fde_m"), True),
    ("test min-ADE K=5 (m)", ("test", "min_ade_k5"), True),
    ("val ADE (m)", ("val", "ade_m"), True),
    ("residual carries (m)", ("test", "actions", "residual", "mean_abs_m"), True),
    ("test text cosine", ("test", "text", "cosine_model"), False),
    ("test text scaled score", ("test", "text", "scaled_score"), False),
    ("params", ("model_size", "total_params"), None),
    ("best epoch", ("best_epoch",), None),
    ("grad cosine (flow vs action)", ("grad_cosine_mean_over_training",), None),
]


def dig(d, path):
    for k in path:
        if not isinstance(d, dict) or k not in d:
            return None
        d = d[k]
    return d if isinstance(d, (int, float)) else None


def parse_modes(tokens):
    out = []
    for t in tokens:
        if t.count("-") != 1:
            raise SystemExit(f"bad --modes entry {t!r}; expected e.g. split-staged")
        out.append(tuple(t.split("-")))
    return out


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--modes", nargs="*", default=None)
    ap.add_argument("--all", action="store_true", help="run all five modes")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--patience", type=int, default=None)
    ap.add_argument("--max-train", type=int, default=None)
    ap.add_argument("--max-val", type=int, default=None)
    ap.add_argument("--max-test", type=int, default=None)
    ap.add_argument("--pretrain-epochs", type=int, default=None)
    ap.add_argument("--stage-a-frac", type=float, default=None)
    ap.add_argument("--prior", default=None)
    ap.add_argument("--device", default=None, choices=["cuda", "cpu"])
    ap.add_argument("--preprocessed-dir", default=None)
    ap.add_argument("--meta-dir", default=None)
    ap.add_argument("--skip-existing", action="store_true")
    ap.add_argument("--out", default=str(HERE / "comparison_world_model.json"))
    args = ap.parse_args()

    modes = (parse_modes(args.modes) if args.modes
             else (ALL_MODES if args.all else DEFAULT_MODES))
    results = {}

    for trunk, grad in modes:
        name = f"{trunk}-{grad}"
        path = HERE / "runs_wm" / name / "report.json"
        if args.skip_existing and path.exists():
            print(f"[skip] {name} (report.json present)")
            results[name] = json.loads(path.read_text())
            continue

        cmd = [sys.executable, str(HERE / "train_world_model_modes.py"),
               "--trunk", trunk, "--grad-mode", grad]
        for flag, val in (
            ("--epochs", args.epochs), ("--patience", args.patience),
            ("--max-train", args.max_train), ("--max-val", args.max_val),
            ("--max-test", args.max_test), ("--pretrain-epochs", args.pretrain_epochs),
            ("--stage-a-frac", args.stage_a_frac), ("--prior", args.prior),
            ("--device", args.device),
            ("--preprocessed-dir", args.preprocessed_dir),
            ("--meta-dir", args.meta_dir),
        ):
            if val is not None:
                cmd += [flag, str(val)]

        print(f"\n{'=' * 72}\n[run] world model / {name}\n{'=' * 72}")
        proc = subprocess.run(cmd, cwd=str(HERE))
        if proc.returncode != 0 or not path.exists():
            print(f"[!] {name} failed (exit {proc.returncode}); skipped in the table")
            continue
        results[name] = json.loads(path.read_text())

    if not results:
        raise SystemExit("no modes completed; nothing to compare")

    published = None
    if PUBLISHED.exists():
        published = json.loads(PUBLISHED.read_text())
        results = {"published (ablation b)": published, **results}

    names = list(results)
    width = max(max(len(n) for n in names), 12) + 2

    print("\n" + "=" * 72)
    print("WORLD-MODEL VLA -- COMPARISON   (v = lower is better)")
    print("=" * 72)
    header = "  " + "metric".ljust(30) + "".join(n.rjust(width) for n in names)
    print(header)
    print("  " + "-" * (len(header) - 2))

    table = {}
    for label, path_, lower in ROWS:
        cells, present = [], False
        for n in names:
            v = dig(results[n], path_)
            if v is None:
                cells.append("-".rjust(width))
            else:
                present = True
                fmt = f"{v:,}" if label == "params" else f"{v:.4f}"
                cells.append(fmt.rjust(width))
                table.setdefault(".".join(path_), {})[n] = v
        if present:
            mark = " v" if lower else ("" if lower is False else " ~")
            print("  " + (label + mark).ljust(30) + "".join(cells))

    base_ade = (dig(published, ("test", "ade_m")) if published else None) or PUBLISHED_ADE
    print(f"\n  test ADE vs the published ablation-b baseline ({base_ade:.4f} m):")
    for n in names:
        if n.startswith("published"):
            continue
        v = dig(results[n], ("test", "ade_m"))
        if v is None:
            continue
        d = v - base_ade
        verdict = "BETTER" if d < 0 else "worse"
        print(f"    {n:<20s} {v:.4f} m   {d:+.4f} m  ({100 * d / base_ade:+.1f}%)  {verdict}")

    # residual share check (world model only)
    print("\n  residual sanity (it should stay well under the ADE -- if it "
          "approaches it,\n  the kinematic model has stopped doing the work):")
    for n in names:
        ade = dig(results[n], ("test", "ade_m"))
        res = dig(results[n], ("test", "actions", "residual", "mean_abs_m"))
        if ade and res is not None:
            flag = "  <-- CHECK THIS" if res > 0.5 * ade else ""
            print(f"    {n:<20s} residual {res:.4f} m  =  {100 * res / ade:.0f}% "
                  f"of ADE{flag}")

    Path(args.out).write_text(json.dumps(
        {"modes": names, "table": table, "baseline_test_ade": base_ade},
        indent=2, default=str))
    print(f"\n  comparison -> {args.out}")


if __name__ == "__main__":
    main()
