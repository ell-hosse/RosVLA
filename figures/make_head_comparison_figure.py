"""Qualitative comparison: direct trajectory head vs. control head + world
model (WM) against ground truth, on three datasets (2 samples each).

Inputs (from the saved checkpoints):
    CoVLA/figures/covla_direct_test_predictions.npz      CoVLA/export_test_predictions.py direct
    CoVLA/figures/covla_wm_test_predictions.npz          CoVLA/export_test_predictions.py wm
    DriveLM/figures/drivelm_test_predictions.npz         DriveLM/export_test_predictions.py
    RosVLA/velocity_head/paper_figures/test_predictions.npz          (velocity head + WM)
    RosVLA/defussion_text_generator/runs/test_predictions.npz        (direct waypoint head)

    python figures/make_head_comparison_figure.py                 # 2x3 figure
    python figures/make_head_comparison_figure.py --candidates    # candidate sheets

Each tile shows the camera frame at t with a bird's-eye inset of the future
trajectory (ego frame, x forward, y left). Nothing is projected into the image.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "figures"

# image roots (override on the command line if the data lives elsewhere)
COVLA_ROOT = Path(r"D:\hf\covla_preprocessed")
NUSC_CAM = Path(r"D:\DriveLM\nuscenes\samples\CAM_FRONT")
ROS_FRAMES = REPO / "RosVLA" / "velocity_head" / "paper_figures" / "_rgb_frames_jpg.npz"

GT_C, DIR_C, WM_C = "#1A9850", "#7B3294", "#E66101"
STYLE = {
    "gt":  dict(c=GT_C,  ls="-",  marker="o", label="Ground truth"),
    "dir": dict(c=DIR_C, ls=":",  marker="s", label="Direct trajectory head"),
    "wm":  dict(c=WM_C,  ls="--", marker="^", label="Control head + WM (ours)"),
}

mpl.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "STIXGeneral", "DejaVu Serif"],
    "mathtext.fontset": "stix", "font.size": 7,
    "pdf.fonttype": 42, "ps.fonttype": 42, "savefig.dpi": 300,
})

# picks
# selected from the candidate sheets (scenes with a turn or speed change)
PICKS = {
    "CoVLA":   [2879, 2333, 2054],
    "DriveLM": [73, 194, 184],
    "RosVLA":  [182, 108],
}


def load_covla():
    d = np.load(REPO / "CoVLA/figures/covla_direct_test_predictions.npz", allow_pickle=True)
    w = np.load(REPO / "CoVLA/figures/covla_wm_test_predictions.npz", allow_pickle=True)
    assert (d["rel"] == w["rel"]).all()
    return dict(gt=d["gt"], dir=d["pred"], wm=w["pred"], ade_dir=d["ade"], ade_wm=w["ade"],
                img=lambda i: cv2.imread(str(COVLA_ROOT / str(d["rel"][i])))[:, :, ::-1],
                name=lambda i: f"{str(d['video'][i])[:8]}, frame {int(d['frame'][i])}",
                v0=d["v0"], unit="m")


def load_drivelm():
    d = np.load(REPO / "DriveLM/figures/drivelm_test_predictions.npz", allow_pickle=True)
    return dict(gt=d["gt"], dir=d["pred_direct"], wm=d["pred_wm"],
                ade_dir=d["ade_direct"], ade_wm=d["ade_wm"],
                img=lambda i: cv2.imread(str(NUSC_CAM / str(d["img"][i])))[:, :, ::-1],
                name=lambda i: str(d["scene"][i]), v0=d["v0"], unit="m")


def load_rosvla():
    v = np.load(REPO / "RosVLA/velocity_head/paper_figures/test_predictions.npz", allow_pickle=True)
    d = np.load(REPO / "RosVLA/defussion_text_generator/runs/test_predictions.npz", allow_pickle=True)
    key = {(s, f): i for i, (s, f) in enumerate(zip(d["scenario"], d["frame"]))}
    j = np.array([key[(s, f)] for s, f in zip(v["scenario"], v["frame"])])
    assert np.allclose(d["gt"][j], v["gt"], atol=1e-5)
    z = np.load(ROS_FRAMES, allow_pickle=True)
    idx = {str(k): n for n, k in enumerate(z["key"])}
    return dict(gt=v["gt"], dir=d["pred"][j], wm=v["pred"], ade_dir=d["ade"][j], ade_wm=v["ade"],
                img=lambda i: cv2.imdecode(z["jpg"][idx[f"{v['scenario'][i]}:{v['frame'][i]}"]],
                                           cv2.IMREAD_UNCHANGED),       # stored RGB
                name=lambda i: f"S{str(v['scenario'][i]).split('_')[1]}, t={int(v['frame'][i])}",
                v0=None, unit="cm")


LOADERS = {"CoVLA": load_covla, "DriveLM": load_drivelm, "RosVLA": load_rosvla}


# draw
_ASPECT = {}   # id(axes) -> height/width of that axes in display units

def nice_bar(span):
    for b in (0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10, 20):
        if b >= span * 0.12:
            return b
    return 50


def bev(ax, trajs, origin_in_traj):
    pts = [np.zeros((1, 2))]
    for k in ("gt", "dir", "wm"):
        xy = trajs[k]
        p = xy if origin_in_traj else np.vstack([[0, 0], xy])
        s = STYLE[k]
        ax.plot(-p[:, 1], p[:, 0], ls=s["ls"], c=s["c"], lw=1.1, zorder=3 + (k == "wm"))
        ax.plot(-xy[:, 1], xy[:, 0], ls="none", marker=s["marker"], ms=2.4, mfc=s["c"],
                mec="white", mew=0.3, zorder=4 + (k == "wm"),
                markevery=(max(1, len(xy) // 6)) if len(xy) > 8 else 1)
        pts.append(p)
    ax.plot(0, 0, marker=(3, 0, 0), ms=5, c="#222222", zorder=6)
    P = np.vstack(pts)
    lo, hi = P.min(0), P.max(0)
    span = max(hi[0] - lo[0], hi[1] - lo[1]) * 0.6 + 1e-3
    span = max(span, 0.15)
    cx, cy = -(lo[1] + hi[1]) / 2, (lo[0] + hi[0]) / 2
    ax.set_xlim(cx - span, cx + span); ax.set_ylim(cy - span, cy + span)
    ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
    ax.set_facecolor((1, 1, 1, 0.88))
    for s in ax.spines.values():
        s.set_color("#444444"); s.set_linewidth(0.5)
    b = nice_bar(2 * span)
    # put the scale bar in whichever top/bottom-left/right corner is emptiest
    corners = []
    for sx in (-1, 1):
        for sy in (1, -1):
            x0 = cx - span * 0.88 if sx < 0 else cx + span * 0.88 - b
            y0 = cy + span * 0.72 if sy > 0 else cy - span * 0.86
            bx = np.array([x0 - span * 0.05, x0 + b + span * 0.05])
            by = np.array([y0 - span * 0.08, y0 + span * 0.25])
            n = ((-P[:, 1] >= bx[0]) & (-P[:, 1] <= bx[1]) &
                 (P[:, 0] >= by[0]) & (P[:, 0] <= by[1])).sum()
            # also penalise segments passing through: densify
            corners.append((n, x0, y0))
    dense = np.vstack([np.linspace(a, c, 12) for k in ("gt", "dir", "wm")
                       for a, c in zip(np.vstack([[0, 0], trajs[k]])[:-1],
                                       np.vstack([[0, 0], trajs[k]])[1:])])
    best = None
    for _, x0, y0 in corners:
        n = ((-dense[:, 1] >= x0 - span * 0.06) & (-dense[:, 1] <= x0 + b + span * 0.06) &
             (dense[:, 0] >= y0 - span * 0.18) & (dense[:, 0] <= y0 + span * 0.3)).sum()
        if best is None or n < best[0]:
            best = (n, x0, y0)
    _, x0, y0 = best
    ax.plot([x0, x0 + b], [y0, y0], c="#222222", lw=1.0, solid_capstyle="butt")
    lab = f"{b * 100:.0f} cm" if b < 1 else f"{b:g} m"
    ax.text(x0 + b / 2, y0 + span * 0.07, lab, ha="center", va="bottom", fontsize=4.8)


def legend(fig, y=1.0):
    hs = [Line2D([], [], c=s["c"], ls=s["ls"], marker=s["marker"], ms=3.5, mec="white",
                 mew=0.4, lw=1.2, label=s["label"]) for s in STYLE.values()]
    hs.append(Line2D([], [], c="#222222", ls="none", marker=(3, 0, 0), ms=5,
                     label="Ego at $t$ (heading up)"))
    fig.legend(handles=hs, loc="upper center", ncol=2, frameon=False, fontsize=7,
               bbox_to_anchor=(0.5, y), handlelength=2.6, columnspacing=1.4)


def figure(width=7.0, cols=("CoVLA", "DriveLM"), n_rows=3, stem="head_comparison_3x2"):
    """One column per dataset, n_rows samples each (rows x columns grid)."""
    data = {c: LOADERS[c]() for c in cols}
    nc = len(cols)
    tw = width / nc
    th = tw * 0.56
    head, leg = 0.2, 0.42
    H = n_rows * th + head + leg
    fig = plt.figure(figsize=(width, H))
    gap_w, gap_h = 0.005, 0.008
    for c_i, c in enumerate(cols):
        for r in range(n_rows):
            l = c_i / nc + gap_w / 2
            b = (n_rows - 1 - r) * th / H
            ax = fig.add_axes([l, b + gap_h / 2, 1 / nc - gap_w, th / H - gap_h])
            _ASPECT[id(ax)] = (th / H - gap_h) * H / ((1 / nc - gap_w) * width)
            tile_square(ax, data[c], PICKS[c][r], c == "CoVLA")
        fig.text(c_i / nc + 1 / (2 * nc), (n_rows * th + head * 0.45) / H, c,
                 ha="center", va="center", fontsize=9, fontweight="bold")
    legend(fig, y=1.0)
    OUT.mkdir(exist_ok=True)
    fig.savefig(OUT / f"{stem}.pdf", bbox_inches="tight", pad_inches=0.01)
    fig.savefig(OUT / f"{stem}.png", bbox_inches="tight", pad_inches=0.01, dpi=300)
    plt.close(fig)
    for c in cols:
        for i in PICKS[c][:n_rows]:
            D = data[c]
            print(f"{c:8s} #{i:<5d} {D['name'](i):28s} ADE direct {D['ade_dir'][i]:.3f}  "
                  f"ours {D['ade_wm'][i]:.3f}")


def figure_rows(width=7.0, rows=("CoVLA", "DriveLM"), n_cols=3, stem="head_comparison_2x3"):
    """One ROW per dataset, n_cols samples each; dataset name rotated on the left."""
    data = {r: LOADERS[r]() for r in rows}
    lab = 0.2                                     # inches for the row labels
    tw = (width - lab) / n_cols
    th = tw * 0.62
    leg = 0.26
    H = len(rows) * th + leg
    fig = plt.figure(figsize=(width, H))
    gap_w, gap_h = 0.004, 0.008
    x0 = lab / width
    cw = (1 - x0) / n_cols
    for r_i, r in enumerate(rows):
        b = (len(rows) - 1 - r_i) * th / H
        for c in range(n_cols):
            ax = fig.add_axes([x0 + c * cw + gap_w / 2, b + gap_h / 2, cw - gap_w, th / H - gap_h])
            _ASPECT[id(ax)] = (th / H - gap_h) * H / ((cw - gap_w) * width)
            tile_square(ax, data[r], PICKS[r][c], r == "CoVLA")
        fig.text(x0 * 0.45, b + th / (2 * H), r, rotation=90, ha="center", va="center",
                 fontsize=9, fontweight="bold")
    hs = [Line2D([], [], c=s["c"], ls=s["ls"], marker=s["marker"], ms=3.5, mec="white",
                 mew=0.4, lw=1.2, label=s["label"]) for s in STYLE.values()]
    hs.append(Line2D([], [], c="#222222", ls="none", marker=(3, 0, 0), ms=5,
                     label="Ego at $t$ (heading up)"))
    fig.legend(handles=hs, loc="upper center", ncol=4, frameon=False, fontsize=7,
               bbox_to_anchor=(0.5 + x0 / 2, 1.0), handlelength=2.6, columnspacing=1.6)
    OUT.mkdir(exist_ok=True)
    fig.savefig(OUT / f"{stem}.pdf", bbox_inches="tight", pad_inches=0.01)
    fig.savefig(OUT / f"{stem}.png", bbox_inches="tight", pad_inches=0.01, dpi=300)
    plt.close(fig)


def crop_to(img, asp):
    """Centre-crop to height/width = asp, so the photo is never stretched."""
    h, w = img.shape[:2]
    if h / w > asp:
        nh = int(round(w * asp)); y = (h - nh) // 2
        return img[y:y + nh]
    nw = int(round(h / asp)); x = (w - nw) // 2
    return img[:, x:x + nw]


def tile_square(ax, D, i, origin_in_traj, inset_h=0.62):
    """Same as tile() but the inset is a square in display units, `inset_h` of the tile height."""
    asp = _ASPECT.get(id(ax), 0.62)              # height / width
    img = crop_to(D["img"](i), asp)
    ax.imshow(img, aspect="auto")
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    iw = inset_h * asp
    ins = ax.inset_axes([1 - iw - 0.012, 0.025, iw, inset_h])
    bev(ins, {k: D[k][i] for k in ("gt", "dir", "wm")}, origin_in_traj)
    box = dict(boxstyle="round,pad=0.22", fc=(0, 0, 0, 0.58), ec="none")
    ax.text(0.015, 0.97, D["name"](i), transform=ax.transAxes, ha="left", va="top",
            fontsize=5.6, color="white", bbox=box)
    a_d, a_w = D["ade_dir"][i], D["ade_wm"][i]
    fmt = (lambda a: f"{100 * a:.1f} cm") if D["unit"] == "cm" else (lambda a: f"{a:.2f} m")
    ax.text(0.015, 0.035, f"ADE direct: {fmt(a_d)}", transform=ax.transAxes, ha="left",
            va="bottom", fontsize=5.6, color="white", bbox=box)
    ax.text(0.015, 0.185, f"ADE ours: {fmt(a_w)}", transform=ax.transAxes, ha="left",
            va="bottom", fontsize=5.6, color="#FFD8B0", bbox=box, fontweight="bold")


def candidates(ds, ids):
    D = LOADERS[ds]()
    n = len(ids)
    cols = 4
    rows = (n + cols - 1) // cols
    fig = plt.figure(figsize=(cols * 3, rows * 1.9))
    for k, i in enumerate(ids):
        ax = fig.add_axes([(k % cols) / cols, 1 - (k // cols + 1) / rows, 1 / cols - 0.005,
                           1 / rows - 0.01])
        _ASPECT[id(ax)] = (1 / rows - 0.01) * rows * 1.9 / ((1 / cols - 0.005) * cols * 3)
        tile_square(ax, D, i, ds == "CoVLA")
        ax.text(0.5, 0.97, f"#{i}", transform=ax.transAxes, color="yellow", ha="center",
                va="top", fontsize=9)
    legend(fig, y=1.03)
    fig.savefig(OUT / f"_candidates_{ds}.png", dpi=110, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates", action="store_true")
    ap.add_argument("--covla-root", default=None)
    ap.add_argument("--nusc-cam", default=None)
    a = ap.parse_args()
    if a.covla_root:
        COVLA_ROOT = Path(a.covla_root)
    if a.nusc_cam:
        NUSC_CAM = Path(a.nusc_cam)
    if a.candidates:
        candidates("CoVLA", [553, 2333, 1695, 1550, 2054, 3293, 2879, 1361, 3575, 2008, 1610, 2577])
        candidates("DriveLM", [97, 122, 36, 73, 424, 135, 362, 273, 23, 233])
        candidates("RosVLA", [66, 65, 31, 168, 108, 182, 102, 32, 13, 176, 115, 68])
    else:
        figure_rows()          # 2 x 3: rows = datasets, columns = samples
        figure()               # 3 x 2: rows = samples, columns = datasets
