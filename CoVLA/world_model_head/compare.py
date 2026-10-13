#!/usr/bin/env python3
"""Build comparison.md / comparison.json for the three world-model ablations.

    python compare.py

Also reads, if present, the reports of the other CoVLA trajectory models:

    ../vla_trajecotry_head/runs/report.json                   flow VLA, generated caption
    ../ablation_studies/GT_caption_embedding/runs/report.json flow VLA, GT caption
    ../ablation_studies/no_text_embedding/runs/report.json    no flow head, GT caption
    ../trajectory_predictor/metrics.json                      ridge on positions
"""
from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ABLATIONS = ("a_world_model_only", "b_vla_diffusion", "c_vla_no_text")
LABEL = {
    "a_world_model_only": "A. world model alone (state only)",
    "b_vla_diffusion":    "B. world model in the diffusion VLA loop",
    "c_vla_no_text":      "C. world model in the VLA loop, no text",
}


def _load(p: Path):
    try:
        return json.loads(p.read_text())
    except Exception:
        return None


def _row(name, params, split_report, key_ade="ade_m", key_fde="fde_m"):
    if split_report is None:
        return None
    return {
        "model": name,
        "params": params,
        "ade_m": split_report.get(key_ade, split_report.get("ade")),
        "fde_m": split_report.get(key_fde, split_report.get("fde")),
    }


def main():
    ours, rows_test, rows_val = {}, [], []
    for ab in ABLATIONS:
        rep = _load(HERE / f"runs_{ab}" / "report.json")
        if rep is None:
            print(f"  [skip] runs_{ab}/report.json not found -- run train.py --ablation {ab}")
            continue
        ours[ab] = rep
        p = rep["model_size"]["total_params"]
        rows_test.append(_row(LABEL[ab], p, rep["test"]))
        rows_val.append(_row(LABEL[ab], p, rep["val"]))

    if not ours:
        print("nothing to compare yet.")
        return

    ref = next(iter(ours.values()))
    base = ref["baselines"]
    oracle = ref["oracle_actions"]

    # direct-regression models
    sib = []
    for label, path, params in (
        ("flow VLA, generated caption (position regressor)",
         HERE.parent / "vla_trajecotry_head" / "runs" / "report.json", None),
        ("flow VLA, GT caption (position regressor)",
         HERE.parent / "ablation_studies" / "GT_caption_embedding" / "runs" / "report.json", None),
        ("no diffusion, GT caption (position regressor)",
         HERE.parent / "ablation_studies" / "no_text_embedding" / "runs" / "report.json", None),
    ):
        rep = _load(path)
        if rep:
            sib.append({"model": label,
                        "params": rep["model_size"]["total_params"],
                        "ade_m": rep["test"]["ade"], "fde_m": rep["test"]["fde"]})
    ridge = _load(HERE.parent / "trajectory_predictor" / "metrics.json")
    if ridge:
        sib.append({"model": "ridge state->positions (no rollout)",
                    "params": ridge.get("n_params"),
                    "ade_m": round(ridge["test"]["ADE_m"], 4),
                    "fde_m": round(ridge["test"]["FDE_m"], 4)})

    # markdown
    L = ["# World-model trajectory head -- ablation comparison", ""]
    L += [f"Horizon {ref['horizon_s']} s, {ref['n_waypoints']} waypoints "
          f"(index 0 = the origin), dt = {ref['dt_s']} s.",
          f"Rows: {ref['n_rows']}. Prior: `{ref['prior']}`.", ""]

    L += ["## Zero-parameter baselines (same rollout, no learned parameters)", "",
          "| baseline | test ADE (m) | test FDE (m) |", "|---|---|---|"]
    for k, v in base["test"].items():
        L.append(f"| {k.upper()} | {v['ade_m']:.4f} | {v['fde_m']:.4f} |")
    L += ["",
          f"Action-envelope ceiling (inverse-dynamics actions clipped to "
          f"|a|<=8, |kappa|<=0.2): **ADE "
          f"{oracle['test']['clipped_to_envelope']['ade_m']:.4f} m**. "
          f"Integrator round-trip check: "
          f"{oracle['test']['exact']['ade_m']:.2e} m (should be ~0).", ""]

    for title, rows in (("Test", rows_test), ("Validation", rows_val)):
        L += [f"## {title} split", "",
              "| model | params | ADE (m) | FDE (m) |", "|---|---|---|---|"]
        for r in rows:
            if r:
                L.append(f"| {r['model']} | {r['params']:,} | {r['ade_m']:.4f} | "
                         f"{r['fde_m']:.4f} |")
        if title == "Test" and sib:
            for r in sib:
                pr = f"{r['params']:,}" if r["params"] else "-"
                L.append(f"| _{r['model']}_ | {pr} | {r['ade_m']:.4f} | {r['fde_m']:.4f} |")
        L.append("")

    # effect of the language pathway
    if "b_vla_diffusion" in ours and "c_vla_no_text" in ours:
        b, c = ours["b_vla_diffusion"]["test"], ours["c_vla_no_text"]["test"]
        d = c["ade_m"] - b["ade_m"]
        L += ["## What the diffusion/text pathway is worth", "",
              f"Ablation B (with the generated caption latent) test ADE "
              f"{b['ade_m']:.4f} m; ablation C (no text of any kind) "
              f"{c['ade_m']:.4f} m.",
              f"Difference: **{d:+.4f} m** "
              f"({100 * d / max(c['ade_m'], 1e-9):+.2f}% of C's ADE) for "
              f"{ours['b_vla_diffusion']['model_size']['total_params'] - ours['c_vla_no_text']['model_size']['total_params']:,} "
              f"extra parameters.", ""]

    # contribution of the rollout vs. the residual
    L += ["## How much of the path is physics vs. the learned residual", "",
          "| ablation | test ADE (m) | mean abs residual (m) | accel p99 (m/s2) | "
          "curvature p99 (1/m) |", "|---|---|---|---|---|"]
    for ab, rep in ours.items():
        a = rep["test"]["actions"]
        res = a.get("residual", {}).get("mean_abs_m", 0.0)
        L.append(f"| {LABEL[ab]} | {rep['test']['ade_m']:.4f} | {res:.4f} | "
                 f"{a['accel_mps2']['p99']:.3f} | {a['curvature_1pm']['p99']:.5f} |")
    L += ["", "A residual that approaches the ADE means the kinematic model has "
              "stopped doing the work and the head has become a position "
              "regressor with extra steps.", ""]

    (HERE / "comparison.md").write_text("\n".join(L))
    (HERE / "comparison.json").write_text(json.dumps(
        {"ours": {k: {"test": v["test"], "val": v["val"],
                      "params": v["model_size"]["total_params"]} for k, v in ours.items()},
         "baselines_test": base["test"], "oracle_test": oracle["test"],
         "siblings_test": sib}, indent=2))
    print("\n".join(L))
    print(f"\nwritten -> {HERE / 'comparison.md'}  and  {HERE / 'comparison.json'}")


if __name__ == "__main__":
    main()
