"""Paper figure: predicted vs. ground-truth waypoints for the velocity-head VLA.

    python export_predictions.py      # once, re-runs the saved fold models
    python make_paper_figure.py

Writes to paper_figures/:
    grid_best16.pdf / .png     4x4 grid of low-ADE held-out windows
    frames/<scenario>_tXXX.png every held-out window
    best16_selection.csv       selected windows and their errors

Each tile is the camera frame at t with a bird's-eye inset of the 6 future
waypoints in the robot frame (no camera calibration, so nothing is projected
into the image). Windows are picked by lowest ADE per ground-truth action,
with a limit per scenario and a minimum frame gap within a run.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import cv2
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

HERE = Path(__file__).resolve().parent
OUT = HERE / "paper_figures"

GT_C = "#1A9850"      # ground truth  - green, solid, circles
PR_C = "#E66101"      # prediction    - orange, dashed, triangles

mpl.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "STIXGeneral", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 7,
    "pdf.fonttype": 42, "ps.fonttype": 42,     # editable text in the PDF
    "savefig.dpi": 300,
})


# bystander faces to blur, (x0, y0, x1, y1) in the 640x480 frame
FACE_BOXES = {
    ("scenario_8_run_03", 22): [(565, 150, 615, 205)],
    ("scenario_8_run_03", 29): [(85, 95, 145, 160)],
    ("scenario_8_run_03", 33): [(0, 45, 70, 125)],
    ("scenario_8_run_03", 36): [(165, 50, 245, 125), (10, 145, 70, 210)],
}


def anonymise(img, boxes):
    img = img.copy()
    for x0, y0, x1, y1 in boxes:
        roi = img[y0:y1, x0:x1]
        small = cv2.resize(roi, (max(1, roi.shape[1] // 10), max(1, roi.shape[0] // 10)),
                           interpolation=cv2.INTER_AREA)
        img[y0:y1, x0:x1] = cv2.resize(small, (roi.shape[1], roi.shape[0]),
                                       interpolation=cv2.INTER_NEAREST)
    return img


def load_frames():
    z = np.load(OUT / "_rgb_frames_jpg.npz", allow_pickle=True)
    idx = {str(k): i for i, k in enumerate(z["key"])}
    jpg = z["jpg"]
    # h5 rgb_images are already RGB
    return lambda scen, t: anonymise(cv2.imdecode(jpg[idx[f"{scen}:{t}"]], cv2.IMREAD_UNCHANGED),
                                     FACE_BOXES.get((scen, t), []))


def select_best(d, n=16, per_scen=2, min_gap=3):
    order = np.argsort(d["ade"])
    picked, count = [], {}

    def ok(i):
        s, t = d["scenario"][i], d["frame"][i]
        if count.get(s, 0) >= per_scen:
            return False
        return all(not (d["scenario"][j] == s and abs(d["frame"][j] - t) < min_gap)
                   for j in picked)

    for i in order:                                   # pass 1: best per scenario
        if d["scenario"][i] not in count:
            picked.append(i); count[d["scenario"][i]] = 1
    for i in order:                                   # pass 2: fill by ADE
        if len(picked) >= n:
            break
        if i not in picked and ok(i):
            picked.append(i); count[d["scenario"][i]] = count.get(d["scenario"][i], 0) + 1
    picked = picked[:n]
    return sorted(picked, key=lambda i: d["ade"][i])


# tiles per ground-truth action in the 4x4 grid
ACTION_QUOTA = (("FORWARD", 6), ("RIGHT_TURN", 6), ("LEFT_TURN", 1),
                ("STOP", 1), ("REVERSE_LEFT", 2))
ACTION_NAME = {"FORWARD": "Straight", "RIGHT_TURN": "Right turn", "LEFT_TURN": "Left turn",
               "STOP": "Stop", "REVERSE_LEFT": "Reverse-left"}


# windows excluded by hand (scenario, frame)
EXCLUDE = {("scenario_2_run_06", 24), ("scenario_2_run_06", 23)}


def select_diverse(d, quota=ACTION_QUOTA, per_scen=2, min_gap=3):
    """Lowest-ADE windows per ground-truth action.

    At most `per_scen` per scenario (relaxed if the action occurs in one run
    only) and at least `min_gap` frames apart within a run.
    """
    picked = []
    for action, n in quota:
        idx = np.where(d["true_action"] == action)[0]
        idx = idx[np.argsort(d["ade"][idx])]
        cap = per_scen if len(set(d["scenario"][idx])) > 1 else n
        mine, count = [], {}
        for i in idx:
            s, t = d["scenario"][i], d["frame"][i]
            if (str(s), int(t)) in EXCLUDE:
                continue
            if count.get(s, 0) >= cap:
                continue
            if any(d["scenario"][j] == s and abs(d["frame"][j] - t) < min_gap for j in mine):
                continue
            mine.append(i); count[s] = count.get(s, 0) + 1
            if len(mine) == n:
                break
        picked += mine
    return picked


def short(scen):                                      # scenario_9_run_02 -> S9
    return "S" + scen.split("_")[1]


def bev(ax, gt, pr):
    """Robot frame (x fwd, y left) -> plot (-y right, x up). Equal aspect."""
    for xy, c, ls, m in ((gt, GT_C, "-", "o"), (pr, PR_C, "--", "^")):
        p = np.vstack([[0, 0], xy])
        ax.plot(-p[:, 1], p[:, 0], ls=ls, c=c, lw=1.1, zorder=3)
        ax.plot(-xy[:, 1], xy[:, 0], ls="none", marker=m, ms=2.6, mfc=c,
                mec="white", mew=0.35, zorder=4)
    ax.plot(0, 0, marker=(3, 0, 0), ms=5, c="#222222", zorder=5)  # robot, heading up
    pts = np.vstack([[0, 0], gt, pr])
    lo, hi = pts.min(0), pts.max(0)
    span = max(hi[0] - lo[0], hi[1] - lo[1], 0.25) * 0.62 + 0.03
    cx, cy = -(lo[1] + hi[1]) / 2, (lo[0] + hi[0]) / 2
    ax.set_xlim(cx - span, cx + span)
    ax.set_ylim(cy - span, cy + span)
    ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_facecolor((1, 1, 1, 0.86))
    for s in ax.spines.values():
        s.set_color("#444444"); s.set_linewidth(0.5)
    # 10 cm scale bar
    x0 = cx - span * 0.88
    y0 = cy + span * 0.70
    ax.plot([x0, x0 + 0.10], [y0, y0], c="#222222", lw=1.0, solid_capstyle="butt")
    ax.text(x0 + 0.05, y0 + span * 0.07, "10 cm", ha="center", va="bottom", fontsize=4.6)


def tile(ax, img, gt, pr, label, ade, fde, inset_w=0.38):
    ax.imshow(img)
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    ins = ax.inset_axes([1 - inset_w - 0.015, 0.02, inset_w, inset_w * 4 / 3])
    bev(ins, gt, pr)
    ax.text(0.02, 0.97, label, transform=ax.transAxes, ha="left", va="top",
            fontsize=6, color="white",
            bbox=dict(boxstyle="round,pad=0.2", fc=(0, 0, 0, 0.55), ec="none"))
    ax.text(0.02, 0.03, f"ADE {100*ade:.1f} cm\nFDE {100*fde:.1f} cm",
            transform=ax.transAxes, ha="left", va="bottom", fontsize=5.6,
            color="white", linespacing=1.1,
            bbox=dict(boxstyle="round,pad=0.2", fc=(0, 0, 0, 0.55), ec="none"))


def legend_handles():
    return [Line2D([], [], c=GT_C, ls="-", marker="o", ms=3.5, mec="white", mew=0.4,
                   lw=1.2, label="Ground truth"),
            Line2D([], [], c=PR_C, ls="--", marker="^", ms=3.5, mec="white", mew=0.4,
                   lw=1.2, label="Predicted (ours)"),
            Line2D([], [], c="#222222", ls="none", marker=(3, 0, 0), ms=5,
                   label="Robot at $t$ (heading up)")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--width", type=float, default=7.0, help="figure width, inches")
    ap.add_argument("--no-frames", action="store_true", help="skip per-frame PNGs")
    args = ap.parse_args()

    d = dict(np.load(OUT / "test_predictions.npz", allow_pickle=True))
    get = load_frames()
    sel = select_diverse(d)

    # 4 x 4 grid
    W = args.width
    tw = W / 4
    th = tw * 3 / 4
    leg_h = 0.22
    fig = plt.figure(figsize=(W, 4 * th + leg_h))
    gs = fig.add_gridspec(4, 4, left=0, right=1, bottom=0, top=1 - leg_h / (4 * th + leg_h),
                          wspace=0.012, hspace=0.016)
    for k, i in enumerate(sel):
        ax = fig.add_subplot(gs[k // 4, k % 4])
        scen, t = str(d["scenario"][i]), int(d["frame"][i])
        tile(ax, get(scen, t), d["gt"][i], d["pred"][i],
             f"{short(scen)}, $t$={t} · {ACTION_NAME[str(d['true_action'][i])]}",
             d["ade"][i], d["fde"][i])
    fig.legend(handles=legend_handles(), loc="upper center", ncol=3, frameon=False,
               bbox_to_anchor=(0.5, 1.0), fontsize=7, handlelength=2.6,
               columnspacing=2.0)
    OUT.mkdir(exist_ok=True)
    fig.savefig(OUT / "grid_best16.pdf", bbox_inches="tight", pad_inches=0.01)
    fig.savefig(OUT / "grid_best16.png", bbox_inches="tight", pad_inches=0.01, dpi=300)
    plt.close(fig)

    with open(OUT / "best16_selection.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["rank", "scenario", "frame", "fold", "ade_m", "fde_m",
                    "gt_path_len_m", "true_action", "pred_action"])
        for r, i in enumerate(sel, 1):
            w.writerow([r, d["scenario"][i], int(d["frame"][i]), int(d["fold"][i]),
                        f"{d['ade'][i]:.4f}", f"{d['fde'][i]:.4f}",
                        f"{np.linalg.norm(np.diff(np.vstack([[0,0], d['gt'][i]]), axis=0), axis=1).sum():.3f}",
                        d["true_action"][i], d["pred_action"][i]])

    # every held-out window
    if not args.no_frames:
        fdir = OUT / "frames"
        fdir.mkdir(exist_ok=True)
        for i in range(len(d["ade"])):
            scen, t = str(d["scenario"][i]), int(d["frame"][i])
            fig = plt.figure(figsize=(3.4, 2.55 + 0.2))
            ax = fig.add_axes([0, 0, 1, 2.55 / 2.75])
            tile(ax, get(scen, t), d["gt"][i], d["pred"][i],
                 f"{short(scen)}, $t$={t}  |  GT {d['true_action'][i]}  Pred {d['pred_action'][i]}",
                 d["ade"][i], d["fde"][i])
            fig.legend(handles=legend_handles()[:2], loc="upper center", ncol=2,
                       frameon=False, fontsize=6.5, bbox_to_anchor=(0.5, 1.0))
            fig.savefig(fdir / f"{scen}_t{t:03d}.png", dpi=200)
            plt.close(fig)

    print("selected:", [(short(str(d['scenario'][i])), int(d['frame'][i]),
                         round(float(d['ade'][i]), 4)) for i in sel])
    print(f"wrote {OUT/'grid_best16.pdf'}")


if __name__ == "__main__":
    main()
