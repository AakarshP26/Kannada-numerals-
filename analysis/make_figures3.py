"""Methodology schematic (original KNSD style, sane resolution) and
per-digit loop distribution grid."""
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from make_figures import _save, _load, KANNADA, FEAT  # noqa
from ktsd import preprocess_mnist  # noqa
from ktsd.features import _neighbor_counts, MIN_LOOP_AREA  # noqa

COL = {"input": "#3498db", "pre": "#9b59b6", "feat": "#2ecc71",
       "clf": "#e74c3c", "out": "#f39c12", "arrow": "#34495e"}


def fig_methodology(out):
    """KTSD pipeline in the original paper's schematic style, with real
    stage images embedded, at print-appropriate resolution."""
    import cv2
    from skimage.morphology import skeletonize
    X, y = _load("train")
    rng = np.random.RandomState(11)
    i = rng.choice(np.where(y == 5)[0], 30)[7]
    raw = X[i]
    up = cv2.resize(raw, (100, 100), interpolation=cv2.INTER_CUBIC)
    binary = preprocess_mnist(raw)
    skel = skeletonize(binary > 0).astype(np.uint8)

    fig = plt.figure(figsize=(12.5, 6.8))
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 16); ax.set_ylim(0, 9)
    ax.axis("off")

    def draw_box(x, yb, w, h, label, sub, color, fs=10.5, subfs=8):
        ax.add_patch(FancyBboxPatch((x, yb), w, h,
                     boxstyle="round,pad=0.03,rounding_size=0.15",
                     facecolor=color, edgecolor="white", lw=2, alpha=0.92))
        ax.text(x + w / 2, yb + h / 2 + (0.16 if sub else 0), label,
                fontsize=fs, fontweight="bold", ha="center", va="center", color="white")
        if sub:
            ax.text(x + w / 2, yb + h / 2 - 0.26, sub, fontsize=subfs,
                    ha="center", va="center", color="white")

    def arrow(x1, y1, x2, y2):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="->", color=COL["arrow"], lw=1.8))

    ax.text(8, 8.55, "KTSD: Kannada Topological Shape Descriptor Pipeline",
            fontsize=15, fontweight="bold", ha="center", color="#2c3e50")

    # stage images above the main boxes
    def show_img(x, yb, w, h, im, title, cmap="gray"):
        a = fig.add_axes([x / 16, yb / 9, w / 16, h / 9])
        a.imshow(im, cmap=cmap); a.set_xticks([]); a.set_yticks([])
        a.set_title(title, fontsize=8)
        for s in a.spines.values():
            s.set_color("#666666")

    show_img(0.75, 6.05, 1.5, 1.55, raw, "input $28\\times28$")
    show_img(3.60, 6.05, 1.5, 1.55, up, "upsampled $100\\times100$")
    show_img(5.45, 6.05, 1.5, 1.55, binary, "binarized")
    show_img(7.30, 6.05, 1.5, 1.55, skel, "skeleton")
    # overlay image with structural marks
    a = fig.add_axes([9.55 / 16, 6.05 / 9, 1.5 / 16, 1.55 / 9])
    a.imshow(binary, cmap="gray")
    cs, hh = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if hh is not None:
        for c_, h_ in zip(cs, hh[0]):
            if h_[3] != -1 and cv2.contourArea(c_) > MIN_LOOP_AREA:
                c_ = c_.squeeze()
                a.plot(np.append(c_[:, 0], c_[0, 0]), np.append(c_[:, 1], c_[0, 1]),
                       c="lime", lw=1.6)
    counts = _neighbor_counts(skel)
    ey, ex = np.where((skel > 0) & (counts == 1))
    jy, jx = np.where((skel > 0) & (counts >= 3))
    a.scatter(ex, ey, s=22, marker="o", facecolors="none", edgecolors="red", linewidths=1.2)
    a.scatter(jx, jy, s=22, marker="s", facecolors="none", edgecolors="orange", linewidths=1.2)
    a.set_xticks([]); a.set_yticks([]); a.set_title("structural events", fontsize=8)

    # main pipeline band
    yb, h = 4.55, 1.15
    draw_box(0.35, yb, 2.3, h, "INPUT", "grayscale numeral", COL["input"])
    draw_box(3.25, yb, 2.3, h, "PREPROCESSING", "upsample, binarize,\nclean", COL["pre"], subfs=7.5)
    draw_box(6.15, yb, 3.5, h, "FEATURE EXTRACTION", "KTSD, 29 features", COL["feat"])
    draw_box(10.45, yb, 2.6, h, "CLASSIFICATION", "SVM (RBF kernel)", COL["clf"])
    draw_box(13.85, yb, 1.9, h, "OUTPUT", "digit + explanation", COL["out"], subfs=7.5)
    ym = yb + h / 2
    arrow(2.65, ym, 3.21, ym); arrow(5.55, ym, 6.11, ym)
    arrow(9.65, ym, 10.41, ym); arrow(13.05, ym, 13.81, ym)

    # feature-group breakdown band
    labels = [
        ("LOOP\nFEATURES", "8 features", "count, areas,\ncentroids, CV", "#27ae60"),
        ("ENDPOINT\nFEATURES", "5 features", "count, positions,\nspread", "#16a085"),
        ("JUNCTION\nFEATURES", "3 features", "count,\nposition", "#1abc9c"),
        ("DIRECTION\nHISTOGRAM", "8 features", "stroke tangent\norientation bins", "#2980b9"),
        ("STROKE\nSTATISTICS", "5 features", "aspect, solidity,\nextent, circularity", "#8e44ad"),
    ]
    wf, gapf = 2.65, 0.35
    x0 = (16 - (5 * wf + 4 * gapf)) / 2
    for k, (lab, n, desc, col) in enumerate(labels):
        x = x0 + k * (wf + gapf)
        ax.add_patch(FancyBboxPatch((x, 1.25), wf, 1.9,
                     boxstyle="round,pad=0.03,rounding_size=0.15",
                     facecolor=col, edgecolor="white", lw=2, alpha=0.92))
        ax.text(x + wf / 2, 2.72, lab, fontsize=9.5, fontweight="bold",
                ha="center", va="center", color="white")
        ax.text(x + wf / 2, 2.18, n, fontsize=8.5, ha="center", va="center",
                color="white", fontweight="bold")
        ax.text(x + wf / 2, 1.70, desc, fontsize=7.5, ha="center", va="center", color="white")
        arrow(7.9, 4.50, x + wf / 2, 3.22)

    ax.text(8, 0.55, "The five groups concatenate into the 29-dimensional descriptor; "
            "loops and stroke statistics derive from the binary image, endpoints, junctions, "
            "and the direction histogram from the skeleton.",
            fontsize=9, ha="center", color="#2c3e50", style="italic")
    _save(fig, out, "fig_methodology")


def fig_loop_dist_grid(out):
    """Per-digit loop-count distribution, one panel per class."""
    d = np.load(os.path.join(FEAT, "train.npz")); F, y = d["X"], d["y"]
    loops = F[:, 0].astype(int)
    fig, axes = plt.subplots(2, 5, figsize=(11, 4.6), sharey=True)
    for c in range(10):
        ax = axes[c // 5, c % 5]
        v = loops[y == c]
        vals = [np.mean(v == k) * 100 for k in range(3)] + [np.mean(v >= 3) * 100]
        bars = ax.bar(range(4), vals, color="#4c78a8")
        for b, val in zip(bars, vals):
            if val > 2:
                ax.annotate(f"{val:.0f}", (b.get_x() + b.get_width() / 2, val),
                            ha="center", va="bottom", fontsize=7)
        ax.set_title(f"{KANNADA[c]}  ({c})", fontsize=11)
        ax.set_xticks(range(4)); ax.set_xticklabels(["0", "1", "2", "$\\geq$3"], fontsize=8)
        if c % 5 == 0:
            ax.set_ylabel("share (%)")
        ax.set_ylim(0, 105)
    fig.suptitle("Detected loop-count distribution for each Kannada numeral (training split)")
    fig.tight_layout()
    _save(fig, out, "fig_loop_dist_grid")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="figures")
    ap.add_argument("--which", default="all")
    a = ap.parse_args()
    fns = {"fig_methodology": fig_methodology, "fig_loop_dist_grid": fig_loop_dist_grid}
    todo = fns if a.which == "all" else {a.which: fns[a.which]}
    for n, f in todo.items():
        f(a.out)
