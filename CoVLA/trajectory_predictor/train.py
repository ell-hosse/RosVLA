#!/usr/bin/env python3
"""Fit and evaluate the CoVLA linear trajectory model.

    python train.py                        # build/load cache, fit, evaluate
    python train.py --limit-videos 200     # quick test on a subset
    python train.py --rebuild-cache        # re-parse states/*.jsonl
    python train.py --alpha 10             # fixed ridge strength
    python train.py --split official       # CoVLA-metadata row-level split
    python train.py --split video          # local random video holdout

The default split (--split preprocessed) is the video-level split from
covla_preprocessed/index/*.jsonl.
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np

from config import Config, SPLIT_SOURCES
from data import build_dataset
from model import LinearWorldModel, ade_fde, mae, rmse


def evaluate(model: LinearWorldModel, X: np.ndarray, Y: np.ndarray) -> dict:
    pred = model.predict(X)
    ade, fde = ade_fde(pred, Y)
    return {
        "n": int(len(X)),
        "ADE_m": ade,
        "FDE_m": fde,
        "RMSE_m": rmse(pred, Y),
        "MAE_m": mae(pred, Y),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--meta-dir", default=None,
                     help="path to CoVLA-metadata (default: config._default_meta_dir())")
    ap.add_argument("--preprocessed-dir", default=None,
                     help="path to covla_preprocessed (default: "
                          "config._default_preprocessed_dir())")
    ap.add_argument("--rebuild-cache", action="store_true",
                     help="ignore covla_window_cache.npz and re-parse the dataset")
    ap.add_argument("--limit-videos", type=int, default=None,
                     help="only process this many videos (debug / smoke test; "
                          "result is not cached)")
    ap.add_argument("--alpha", type=float, default=None,
                     help="skip the val-set ridge-strength search, fit with this alpha")
    ap.add_argument("--split", choices=SPLIT_SOURCES, default="preprocessed",
                     help="'preprocessed' (default): D:\\hf\\covla_preprocessed's own "
                          "video-level train/val/test lists -- verified leakage-free. "
                          "'video': a local seeded random video-level holdout, for "
                          "machines without covla_preprocessed. "
                          "'official': CoVLA-metadata's row-level split -- kept only "
                          "for comparability with other CoVLA heads in this repo; "
                          "windows from the same video can land in more than one of "
                          "train/val/test under this split (see README.md).")
    args = ap.parse_args()

    cfg = Config()
    if args.meta_dir:
        cfg.meta_dir = args.meta_dir
    if args.preprocessed_dir:
        cfg.preprocessed_dir = args.preprocessed_dir

    t0 = time.time()
    data = build_dataset(cfg, force=args.rebuild_cache, limit_videos=args.limit_videos)
    dt = time.time() - t0
    X, Y = data["X"], data["Y"]
    print(f"windows: {len(X)}  (features={X.shape[1]}, targets={Y.shape[1]})  "
          f"in {dt:.1f}s")

    split_id = data[f"split_{args.split}"]
    print(f"using the '{args.split}' split")

    Xtr, Ytr = X[split_id == 0], Y[split_id == 0]
    Xva, Yva = X[split_id == 1], Y[split_id == 1]
    Xte, Yte = X[split_id == 2], Y[split_id == 2]
    for name, x in (("train", Xtr), ("val", Xva), ("test", Xte)):
        if len(x) == 0:
            raise RuntimeError(f"empty {name} split under --split {args.split} -- "
                                "check the cache and the split source")
    print(f"split sizes: train={len(Xtr)}  val={len(Xva)}  test={len(Xte)}")

    if args.alpha is not None:
        best_alpha = args.alpha
        print(f"\nfitting with fixed alpha={best_alpha:g} (search skipped)")
    else:
        print("\nsearching ridge strength alpha on VAL (each candidate is fit on TRAIN only):")
        best_alpha, best_ade = None, float("inf")
        for alpha in cfg.alphas:
            m = LinearWorldModel(alpha).fit(Xtr, Ytr)
            ade, fde = ade_fde(m.predict(Xva), Yva)
            flag = ""
            if ade < best_ade:
                best_ade, best_alpha = ade, alpha
                flag = "  <- best so far"
            print(f"  alpha={alpha:<9g} val ADE {ade:.4f} m   val FDE {fde:.4f} m{flag}")
        print(f"-> selected alpha = {best_alpha:g}")

    model = LinearWorldModel(best_alpha).fit(Xtr, Ytr)
    print(f"\nfit on TRAIN only ({len(Xtr)} windows): "
          f"{model.n_params} coefficients ({model.n_params * 8 / 1024:.1f} KB fp64)")

    report = {
        "split": args.split,
        "alpha": best_alpha,
        "n_params": model.n_params,
        "n_features": X.shape[1],
        "n_targets": Y.shape[1],
    }
    for name, x, y in (("train", Xtr, Ytr), ("val", Xva, Yva), ("test", Xte, Yte)):
        report[name] = evaluate(model, x, y)

    print("\nerrors by split (ADE/FDE over the 20 downsampled waypoints, in metres):")
    for name in ("train", "val", "test"):
        r = report[name]
        print(f"  {name:<5} n={r['n']:6d}   ADE {r['ADE_m']:.4f}   FDE {r['FDE_m']:.4f}   "
              f"RMSE {r['RMSE_m']:.4f}   MAE {r['MAE_m']:.4f}")

    tr_ade, te_ade = report["train"]["ADE_m"], report["test"]["ADE_m"]
    print(f"\n  test/train ADE ratio = {te_ade / max(tr_ade, 1e-9):.2f}x "
          f"(close to 1x => train and held-out videos behave similarly)")

    model.save(cfg.coeffs_path)
    with open(cfg.metrics_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\ncoefficients -> {cfg.coeffs_path}")
    print(f"metrics      -> {cfg.metrics_path}")
    print("\ninference cost per window: 1 standardise + 1 "
          f"({model.mu.shape[0] + 1}x{model.Wb.shape[1]}) matmul -- "
          "fixed size, independent of dataset size (O(1)).")


if __name__ == "__main__":
    main()
