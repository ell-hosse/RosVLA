#!/usr/bin/env python3
"""Train and evaluate the flow head on BDD-X (no trajectory / action head).

    target text = "<action> <justification>", e.g.
                  "The car slows down because the light ahead turned red."

    python train.py                                  # f1_img_state, seed 0
    python train.py --all --seeds 0 1 2              # all three ablations, 3 seeds
    python train.py --ablation f2_state_only
    python train.py --n-frames 1                     # single frame
    python train.py --text-field action              # action sentence only

Ablations (flow head conditioning):
    f1_img_state    clip frames + vehicle signals
    f2_state_only   vehicle signals only
    f3_img_only     clip frames only

Reports also include a mean-embedding baseline, a kNN baseline on the same
inputs, and an oracle row that decodes the true embedding. Early stopping
uses the validation cosine of the generated embedding.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import ABLATIONS, HERE, TEXT_FIELDS, Config            # noqa: E402
from data import load_splits                                         # noqa: E402
from latent import TextLatent, pca_report                            # noqa: E402
from metrics import (CaptionBank, action_class, class_distribution,  # noqa: E402
                     cosine, decode_report, text_fidelity_report)
from model import FlowCaptioner, parameter_report                    # noqa: E402
from utils import EarlyStopper, Standardiser                         # noqa: E402


def T(x, dev):
    return torch.as_tensor(np.asarray(x), dtype=torch.float32, device=dev)


def run_dir(cfg: Config, seed: int) -> Path:
    tag = f"{cfg.text_field}__{cfg.n_frames}f-{cfg.frame_pool}" if cfg.uses_image else cfg.text_field
    if cfg.split_seed:
        tag += f"__split-{cfg.split_seed}"
    if cfg.img_dropout and cfg.uses_image:
        tag += f"__imgdrop-{cfg.img_dropout:g}"
    return HERE / "runs" / cfg.ablation / tag / f"seed_{seed}"


def knn_embed(train_x, train_emb, x, k, dev, chunk=2048, exclude_self=False):
    """Mean embedding of the k nearest training rows (Euclidean, standardised inputs).

    With exclude_self, each row's own match is dropped (x is the training set).
    """
    tx = T(train_x, dev)
    te = T(train_emb, dev)
    out = []
    for i in range(0, len(x), chunk):
        d = torch.cdist(T(x[i:i + chunk], dev), tx)
        idx = d.topk(k + int(exclude_self), largest=False).indices
        if exclude_self:
            idx = idx[:, 1:]
        out.append(te[idx].mean(1).cpu().numpy())
    return np.concatenate(out)


def run_one(cfg: Config, args, seed: int) -> dict:
    cfg.seed = seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    dev = cfg.device if (cfg.device == "cpu" or torch.cuda.is_available()) else "cpu"
    out = run_dir(cfg, seed)
    out.mkdir(parents=True, exist_ok=True)
    print("\n" + "#" * 78)
    print(f"# BDD-X flow head / {cfg.ablation}   text={cfg.text_field}   frames="
          f"{cfg.n_frames if cfg.uses_image else '-'} ({cfg.frame_pool})   seed={seed}   device={dev}")
    print("#" * 78)

    S, info = load_splits(cfg, force_images=args.rebuild_image_cache)
    tr = S["train"]

    # fit on train only
    img_std = Standardiser(tr.img) if cfg.uses_image else None
    st_std = Standardiser(tr.state)
    latent = TextLatent(tr.emb, cfg.pca_dim)
    m = tr.emb.mean(0)
    mean_emb = m / max(np.linalg.norm(m), 1e-8)
    bank = CaptionBank(tr)
    maj_cls = Counter(action_class(a) for a in tr.action).most_common(1)[0][0]
    print(f"  PCA latent dim {latent.dim} from {latent.n_unique_train} unique train texts, "
          f"explained variance {latent.explained:.4f}; caption bank {len(bank.text)}")

    def pack(sp):
        n = len(sp)
        img = T(img_std(sp.img), dev) if cfg.uses_image else torch.zeros(n, 1, device=dev)
        return img, T(st_std(sp.state), dev), T(latent.encode(sp.emb), dev)

    P = {k: pack(v) for k, v in S.items()}
    loader = DataLoader(TensorDataset(*P["train"]), batch_size=cfg.batch_size, shuffle=True,
                        generator=torch.Generator().manual_seed(seed))

    model = FlowCaptioner(cfg, img_dim=(tr.img.shape[1] if cfg.uses_image else 0),
                          state_dim=tr.state.shape[2], latent_dim=latent.dim).to(dev)
    size = parameter_report(model)
    print(f"  model: {size['total_params']:,} params; {size['by_module']}")
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.epochs)
    stopper = EarlyStopper(cfg.early_stop_patience, cfg.early_stop_min_delta)

    def gen_emb(name, steps=None, seed_=None):
        img, st, _ = P[name]
        g = torch.Generator(device=dev).manual_seed(cfg.seed if seed_ is None else seed_)
        z = model.generate(img if cfg.uses_image else None, st, steps or cfg.flow_steps, g)
        return latent.decode(z.cpu().numpy())

    log = []
    for epoch in range(cfg.epochs):
        model.train()
        t0 = time.time()
        losses = []
        for bi, bs, bz in loader:
            loss = model.flow_loss(bi if cfg.uses_image else None, bs, bz)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            opt.step()
            losses.append(loss.item())
        sched.step()
        model.eval()
        val_cos = float(cosine(gen_emb("val", seed_=seed + epoch), S["val"].emb).mean())
        log.append({"epoch": epoch, "flow_loss": float(np.mean(losses)), "val_cosine": val_cos})
        print(f"  epoch {epoch:3d}  flow_loss={np.mean(losses):.4f}  val_cos={val_cos:.4f}"
              f"  [{time.time() - t0:.1f}s]")
        if stopper.step(-val_cos, epoch, model.state_dict()):
            print(f"  early stopping at epoch {epoch} (best val_cos={-stopper.best:.4f} "
                  f"at epoch {stopper.best_epoch})")
            break
    model.load_state_dict(stopper.best_state)
    model.eval()

    # report
    report = {"ablation": cfg.ablation, "seed": seed, "text_field": cfg.text_field,
              "conditioned_on": {"frames": cfg.uses_image, "vehicle_signals": cfg.uses_state},
              "n_frames": cfg.n_frames if cfg.uses_image else None, "frame_pool": cfg.frame_pool,
              "model_size": size, "best_epoch": stopper.best_epoch, "history_log": log,
              "data": info,
              "pca": pca_report(latent, {k: v.emb for k, v in S.items()}),
              "action_class_distribution_test": class_distribution(S["test"].action),
              "action_class_majority": maj_cls}

    # kNN baseline on the ablation's own inputs
    def flat(sp):
        parts = []
        if cfg.uses_image:
            parts.append(img_std(sp.img))
        if cfg.uses_state:
            parts.append(st_std(sp.state).reshape(len(sp), -1))
        return np.concatenate(parts, 1)

    trx = flat(tr)
    print("\n  FINAL EVALUATION (best checkpoint, epoch %d)" % stopper.best_epoch)
    for name, sp in S.items():
        pred = gen_emb(name)
        r = text_fidelity_report(pred, sp.emb, latent.reconstruct(sp.emb), mean_emb)
        # best-of-K cosine over K flow samples
        cs = np.stack([cosine(gen_emb(name, seed_=1000 + k), sp.emb)
                       for k in range(cfg.n_samples_eval)])
        r[f"cosine_best_of_{cfg.n_samples_eval}"] = round(float(cs.max(0).mean()), 4)
        knn = knn_embed(trx, tr.emb, flat(sp), cfg.knn_k, dev, exclude_self=(name == "train"))
        r["cosine_knn_baseline"] = round(float(cosine(knn, sp.emb).mean()), 4)
        if name != "train":
            r["decode"] = decode_report(pred, sp, bank)
            r["decode_oracle_true_embedding"] = decode_report(sp.emb, sp, bank)
            r["decode_mean_baseline"] = decode_report(
                np.repeat(mean_emb[None], len(sp), 0), sp, bank)
            r["action_class_majority_accuracy"] = round(float(np.mean(
                [action_class(a) == maj_cls for a in sp.action])), 4)
        if name == "test":
            sweep = []
            for steps in cfg.step_sweep:
                if dev == "cuda":
                    torch.cuda.synchronize()
                t0 = time.perf_counter()
                p = gen_emb(name, steps=steps)
                if dev == "cuda":
                    torch.cuda.synchronize()
                sweep.append({"steps": steps,
                              "cosine_model": round(float(cosine(p, sp.emb).mean()), 4),
                              "ms_per_sample": round(1000 * (time.perf_counter() - t0) / len(sp), 4)})
            r["step_sweep"] = sweep
        report[name] = r
        print(f"  [{name}] n={r['n']}  cosine model={r['cosine_model']:.4f}  "
              f"ceiling={r['cosine_pca_ceiling']:.4f}  mean={r['cosine_mean_baseline']:.4f}  "
              f"kNN={r['cosine_knn_baseline']:.4f}  scaled={r['scaled_score']:.4f}  "
              f"best-of-{cfg.n_samples_eval}={r[f'cosine_best_of_{cfg.n_samples_eval}']:.4f}")
        if "decode" in r:
            dcd, orc = r["decode"], r["decode_oracle_true_embedding"]
            print(f"      decoded: action-class acc {dcd['action_class_accuracy']:.3f} "
                  f"(majority {r['action_class_majority_accuracy']:.3f}, oracle "
                  f"{orc['action_class_accuracy']:.3f})   action B4/CIDEr {dcd['action_bleu4']:.1f}/"
                  f"{dcd['action_cider']:.1f}   justification B4/CIDEr {dcd['justification_bleu4']:.1f}/"
                  f"{dcd['justification_cider']:.1f}")

    torch.save({"model_state": model.state_dict(),
                "config": {k: str(v) for k, v in vars(cfg).items()},
                "state_mean": st_std.mean, "state_std": st_std.std,
                **({"img_mean": img_std.mean, "img_std": img_std.std} if img_std else {}),
                "pca_mean": latent.mean, "pca_basis": latent.basis, "pca_scale": latent.scale,
                "mean_emb": mean_emb}, out / "model.pt")
    (out / "report.json").write_text(json.dumps(report, indent=2, default=str))
    print(f"  -> {out / 'report.json'}")
    return report


def _agg(v):
    v = [x for x in v if isinstance(x, (int, float)) and np.isfinite(x)]
    return {"mean": round(float(np.mean(v)), 4), "std": round(float(np.std(v)), 4), "n": len(v)} if v else None


def summarise(reports: list[dict]) -> dict:
    s = {"n_seeds": len(reports), "params": reports[0]["model_size"]["total_params"],
         "text_field": reports[0]["text_field"], "n_frames": reports[0]["n_frames"],
         "frame_pool": reports[0]["frame_pool"]}
    keys = ("cosine_model", "cosine_pca_ceiling", "cosine_mean_baseline", "cosine_knn_baseline",
            "scaled_score")
    for sp in ("train", "val", "test"):
        rs = [r[sp] for r in reports]
        o = {k: _agg([r[k] for r in rs]) for k in keys}
        bk = next(k for k in rs[0] if k.startswith("cosine_best_of_"))
        o[bk] = _agg([r[bk] for r in rs])
        if "decode" in rs[0]:
            for k in rs[0]["decode"]:
                if k != "bank_size":
                    o[f"decode_{k}"] = _agg([r["decode"][k] for r in rs])
            o["decode_oracle"] = rs[0]["decode_oracle_true_embedding"]
            o["decode_mean_baseline"] = rs[0]["decode_mean_baseline"]
            o["action_class_majority_accuracy"] = rs[0]["action_class_majority_accuracy"]
        s[sp] = o
    return s


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ablation", default="f1_img_state", choices=list(ABLATIONS))
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--text-field", default=None, choices=list(TEXT_FIELDS))
    ap.add_argument("--n-frames", type=int, default=None, choices=[1, 2, 4, 8])
    ap.add_argument("--frame-pool", default=None, choices=["concat", "mean"])
    ap.add_argument("--data-root", default=None)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--patience", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--pca-dim", type=int, default=None)
    ap.add_argument("--split-seed", type=int, default=None)
    ap.add_argument("--img-dropout", type=float, default=None)
    ap.add_argument("--num-workers", type=int, default=None,
                    help="video-decoding workers for the one-time feature pass")
    ap.add_argument("--rebuild-image-cache", action="store_true")
    ap.add_argument("--device", default=None, choices=["cuda", "cpu"])
    args = ap.parse_args()

    for ab in (ABLATIONS if args.all else [args.ablation]):
        reps = []
        for seed in args.seeds:
            cfg = Config(ablation=ab)
            for attr, val in (("text_field", args.text_field), ("n_frames", args.n_frames),
                              ("frame_pool", args.frame_pool), ("epochs", args.epochs),
                              ("early_stop_patience", args.patience), ("batch_size", args.batch_size),
                              ("lr", args.lr), ("pca_dim", args.pca_dim),
                              ("split_seed", args.split_seed), ("img_dropout", args.img_dropout),
                              ("num_workers", args.num_workers), ("device", args.device)):
                if val is not None:
                    setattr(cfg, attr, val)
            if args.data_root:
                cfg.data_root = Path(args.data_root)
            reps.append(run_one(cfg, args, seed))
        base = run_dir(cfg, 0).parent
        s = summarise(reps)
        (base / "summary.json").write_text(json.dumps(s, indent=2))
        t = s["test"]["cosine_model"]
        print(f"\n  == {ab}: test cosine {t['mean']:.4f} +- {t['std']:.4f} over {len(reps)} seed(s)")
    print("\nrun `python compare.py` for the table")


if __name__ == "__main__":
    main()
