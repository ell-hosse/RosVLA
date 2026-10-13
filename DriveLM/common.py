"""Helpers shared by train_direct.py and train_wm.py: CLI arguments, text
metrics, seed aggregation and the analytic baselines.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from config import HERE, TEXT_FIELDS, Config
from dynamics import baseline_report, measured_kappa
from latent import TextLatent
from metrics import behavior_majority_baseline, retrieval_report, text_fidelity_report


def to_tensor(x, device):
    return torch.as_tensor(np.asarray(x), dtype=torch.float32, device=device)


def pick_device(cfg: Config) -> str:
    return cfg.device if (cfg.device == "cpu" or torch.cuda.is_available()) else "cpu"


def set_seed(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)


def add_common_args(ap: argparse.ArgumentParser):
    ap.add_argument("--all", action="store_true", help="run every ablation of this family")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0],
                    help="training seeds; >1 gives mean +- std in the reports (recommended: "
                         "0 1 2 -- the test split is only ~100 scenes)")
    ap.add_argument("--text-field", default=None, choices=list(TEXT_FIELDS))
    ap.add_argument("--data-root", default=None)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--patience", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--pca-dim", type=int, default=None)
    ap.add_argument("--split-seed", type=int, default=None)
    ap.add_argument("--img-dropout", type=float, default=None,
                    help="input dropout on the frozen CNN feature (default 0.0, as on CoVLA)")
    ap.add_argument("--rebuild-image-cache", action="store_true")
    ap.add_argument("--device", default=None, choices=["cuda", "cpu"])


def apply_common_args(cfg: Config, args) -> Config:
    for attr, val in (("epochs", args.epochs), ("early_stop_patience", args.patience),
                      ("batch_size", args.batch_size), ("lr", args.lr),
                      ("pca_dim", args.pca_dim), ("split_seed", args.split_seed),
                      ("device", args.device), ("text_field", args.text_field),
                      ("img_dropout", args.img_dropout)):
        if val is not None:
            setattr(cfg, attr, val)
    if args.data_root:
        cfg.data_root = Path(args.data_root)
    return cfg


def run_dir(cfg: Config, seed: int) -> Path:
    tag = cfg.text_field if cfg.uses_text else "no_text"
    if cfg.family == "wm":   # non-default world-model settings get their own run folder
        if cfg.prior != "auto":
            tag += f"__prior-{cfg.prior}"
        if not cfg.use_residual:
            tag += "__no-residual"
        if cfg.pretrain_epochs:
            tag += f"__pretrain-{cfg.pretrain_epochs}"
    if cfg.split_seed != 0:
        tag += f"__split-{cfg.split_seed}"
    if cfg.img_dropout > 0 and cfg.uses_image:
        tag += f"__imgdrop-{cfg.img_dropout:g}"
    return HERE / "runs" / cfg.family / cfg.ablation / tag / f"seed_{seed}"


class TextSide:
    """Text components fitted on training captions: PCA latent, mean embedding, retrieval bank."""

    def __init__(self, train, cfg: Config):
        self.latent = TextLatent(train.cap_emb, cfg.pca_dim)
        m = train.cap_emb.mean(axis=0)
        self.mean_emb = m / max(np.linalg.norm(m), 1e-8)
        bank = {}
        for t, e, b in zip(train.cap_text, train.cap_emb, train.behavior):
            bank.setdefault(t, (e, b))
        self.bank_texts = list(bank)
        self.bank_emb = np.stack([bank[t][0] for t in self.bank_texts])
        self.bank_beh = [bank[t][1] for t in self.bank_texts]
        self.train_behaviors = list(train.behavior)

    def z(self, split):
        return self.latent.encode(split.cap_emb).astype(np.float32)

    def report(self, z_hat: np.ndarray, split) -> dict:
        pred = self.latent.decode(z_hat)
        rep = text_fidelity_report(pred, split.cap_emb, self.latent.reconstruct(split.cap_emb),
                                   self.mean_emb)
        rep["retrieval"] = retrieval_report(pred, list(split.cap_text), list(split.behavior),
                                            self.bank_texts, self.bank_emb, self.bank_beh)
        rep["behavior_majority_baseline"] = behavior_majority_baseline(
            self.train_behaviors, list(split.behavior))
        return rep


def print_text(t: dict, indent: str = "    "):
    r = t["retrieval"]
    mb = t["behavior_majority_baseline"]
    print(f"{indent}text cosine: model={t['cosine_model']:.4f}  ceiling={t['cosine_pca_ceiling']:.4f}"
          f"  mean-baseline={t['cosine_mean_baseline']:.4f}  scaled={t['scaled_score']:.4f}")
    print(f"{indent}retrieval acc={r['retrieval_accuracy']:.4f} (bank {r['bank_size']}, "
          f"coverage {r['bank_coverage']:.3f})   behavior acc={r['behavior_accuracy']:.4f} "
          f"speed={r['behavior_speed_accuracy']:.4f} steer={r['behavior_steer_accuracy']:.4f}  "
          f"[majority: {mb['behavior_accuracy']:.4f}]")


def print_traj(r: dict, indent: str = "    "):
    l2s, l2u = r["l2"]["stp3"], r["l2"]["uniad"]
    print(f"{indent}ADE {r['ade_m']:.4f} m   FDE {r['fde_m']:.4f} m   "
          f"L2 1/2/3 s (ST-P3 avg) {l2s['1s']:.3f}/{l2s['2s']:.3f}/{l2s['3s']:.3f} avg {l2s['avg']:.3f}"
          f"   (UniAD at-t) {l2u['1s']:.3f}/{l2u['2s']:.3f}/{l2u['3s']:.3f} avg {l2u['avg']:.3f}")


def subgroup_report(pred6: np.ndarray, split) -> dict:
    out = {}
    for loc in sorted(set(split.location)):
        m = split.location == loc
        out[f"location={loc}"] = {"n": int(m.sum()),
                                  "ade_m": round(float(np.linalg.norm(pred6[m] - split.traj[m], axis=-1).mean()), 4)}
    for flag in (True, False):
        m = split.hist_padded == flag
        if m.any():
            out[f"history_edge_padded={flag}"] = {
                "n": int(m.sum()),
                "ade_m": round(float(np.linalg.norm(pred6[m] - split.traj[m], axis=-1).mean()), 4)}
    return out


def baselines_for(split) -> dict:
    return baseline_report(split.traj, split.v0, split.a0,
                           measured_kappa(split.v0, split.yaw_rate0))


def _agg(values):
    v = [x for x in values if isinstance(x, (int, float)) and np.isfinite(x)]
    if not v:
        return None
    return {"mean": round(float(np.mean(v)), 4), "std": round(float(np.std(v)), 4), "n": len(v)}


def aggregate_seeds(reports: list[dict], split_names) -> dict:
    """Mean / std over seeds of the main metrics."""
    out = {"n_seeds": len(reports), "params": reports[0]["model_size"]["total_params"]}
    for sp in split_names:
        rs = [r[sp] for r in reports if sp in r]
        if not rs:
            continue
        o = {k: _agg([r[k] for r in rs]) for k in ("ade_m", "fde_m")}
        for conv in ("stp3", "uniad"):
            for k in ("1s", "2s", "3s", "avg"):
                o[f"l2_{conv}_{k}"] = _agg([r["l2"][conv][k] for r in rs])
        mk = [k for k in rs[0] if k.startswith("min_ade_k")]
        if mk:
            o[mk[0]] = _agg([r[mk[0]] for r in rs])
        if "text" in rs[0]:
            for k in ("cosine_model", "cosine_pca_ceiling", "cosine_mean_baseline", "scaled_score"):
                o[f"text_{k}"] = _agg([r["text"][k] for r in rs])
            for k in ("retrieval_accuracy", "behavior_accuracy", "behavior_speed_accuracy",
                      "behavior_steer_accuracy"):
                o[f"text_{k}"] = _agg([r["text"]["retrieval"][k] for r in rs])
            o["behavior_majority_accuracy"] = rs[0]["text"]["behavior_majority_baseline"]["behavior_accuracy"]
        if "actions" in rs[0] and "residual" in rs[0]["actions"]:
            o["residual_mean_abs_m"] = _agg([r["actions"]["residual"]["mean_abs_m"] for r in rs])
        out[sp] = o
    return out


def write_summary(base: Path, reports: list[dict], split_names, extra: dict):
    base.mkdir(parents=True, exist_ok=True)
    s = {**extra, **aggregate_seeds(reports, split_names)}
    (base / "summary.json").write_text(json.dumps(s, indent=2, default=str))
    return s


class Timer:
    def __enter__(self):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self.t = time.perf_counter()
        return self

    def __exit__(self, *a):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self.dt = time.perf_counter() - self.t
