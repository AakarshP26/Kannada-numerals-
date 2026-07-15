"""Faithful code-drawn remakes of the original paper's Fig. 3 (KTSD Pipeline)
and Fig. 4 (Standard Approach vs KTSD Pipeline)."""
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle, FancyArrowPatch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from make_figures import _save, _load  # noqa
from ktsd import preprocess_mnist  # noqa
from ktsd.features import _neighbor_counts, MIN_LOOP_AREA  # noqa


def _box(ax, x, y, w, h, text, fc, ec="black", fs=9, tc="black", sub=None,
         subfs=7, lw=1.2, style="round,pad=0.02,rounding_size=0.08"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=style, fc=fc, ec=ec, lw=lw))
    ax.text(x + w / 2, y + h / 2 + (0.05 * h if sub else 0), text, ha="center",
            va="center", fontsize=fs, color=tc, fontweight="bold")
    if sub:
        ax.text(x + w / 2, y + h * 0.28, sub, ha="center", va="center",
                fontsize=subfs, color=tc)


def _arr(ax, x1, y1, x2, y2, lw=1.4, color="black", style="-|>"):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle=style,
                                 mutation_scale=12, lw=lw, color=color))


def fig_ktsd_pipeline(out):
    """Remake of the original Fig. 3: compact horizontal KTSD pipeline."""
    fig, ax = plt.subplots(figsize=(10.5, 2.9))
    ax.set_xlim(0, 21); ax.set_ylim(0, 6); ax.axis("off")

    _box(ax, 0.3, 2.1, 2.9, 1.8, "Kannada\nDigit", "#d5e8d4", fs=10,
         sub="Input Image (28$\\times$28)")
    _arr(ax, 3.2, 3.0, 4.0, 3.0)
    _box(ax, 4.0, 2.1, 2.9, 1.8, "Preprocessing", "#dae8fc", fs=10,
         sub="Binarization\nSkeletonization")
    _arr(ax, 6.9, 3.0, 7.7, 3.0)

    # dashed feature-extraction container with four inner boxes
    ax.add_patch(Rectangle((7.7, 0.7), 5.9, 4.6, fill=False, ls="--", lw=1.2))
    ax.text(10.65, 5.55, "Feature Extraction", ha="center", fontsize=10,
            fontweight="bold")
    inner = [("Loop\nTopology (8)", 7.95, 3.05, "#ffe6cc"),
             ("Curvature\nHistogram (8)", 10.85, 3.05, "#ffe6cc"),
             ("Endpoints \\&\nJunctions (8)", 7.95, 1.0, "#ffe6cc"),
             ("Stroke\nStatistics (5)", 10.85, 1.0, "#ffe6cc")]
    for t, x, y, c in inner:
        _box(ax, x, y, 2.6, 1.7, t, c, fs=8.5)
    _arr(ax, 13.6, 3.0, 14.35, 3.0)

    # 29-D vector strip
    ax.add_patch(Rectangle((14.35, 1.0), 0.95, 4.0, fc="#f5f5f5", ec="black", lw=1.1))
    for k, lab in enumerate(["$F_1$", "$F_2$", "$\\vdots$", "$F_{29}$"]):
        ax.text(14.82, 4.45 - k * 0.95, lab, ha="center", va="center", fontsize=8)
    ax.text(14.82, 0.55, "29-D Vector", ha="center", fontsize=8.5, fontweight="bold")
    _arr(ax, 15.3, 3.0, 16.1, 3.0)

    _box(ax, 16.1, 2.1, 2.5, 1.8, "SVM\nClassifier", "#f8cecc", fs=10, sub="RBF Kernel")
    _arr(ax, 18.6, 3.0, 19.25, 3.0)
    _box(ax, 19.25, 2.1, 1.55, 1.8, "Prediction\nClass: 6", "#d5e8d4", fs=8.5)
    _save(fig, out, "fig_ktsd_pipeline")


def fig_standard_vs_ktsd(out):
    """Remake of the original Fig. 4: standard CNN (black box) vs KTSD
    (white box), drawn entirely from code and real data."""
    import cv2
    from skimage.morphology import skeletonize
    X, y = _load("train")
    rng = np.random.RandomState(11)
    i = rng.choice(np.where(y == 0)[0], 20)[3]
    raw = X[i]
    up = cv2.resize(raw, (100, 100), interpolation=cv2.INTER_CUBIC)
    binary = preprocess_mnist(raw)
    skel = skeletonize(binary > 0).astype(np.uint8)

    fig = plt.figure(figsize=(13, 7.6))
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 26); ax.set_ylim(0, 15.2)
    ax.axis("off")

    # panel titles
    ax.add_patch(Rectangle((0.2, 14.2), 11.6, 0.85, fc="#404040"))
    ax.text(6.0, 14.62, "STANDARD CNN ARCHITECTURE (BLACK BOX APPROACH)",
            fontsize=9.5, color="white", ha="center", fontweight="bold")
    ax.add_patch(Rectangle((13.0, 14.2), 11.6, 0.85, fc="#1f6f43"))
    ax.text(18.8, 14.62, "PROPOSED KTSD PIPELINE (WHITE BOX APPROACH)",
            fontsize=9.5, color="white", ha="center", fontweight="bold")

    # center comparative band
    ax.add_patch(Rectangle((12.0, 0.4), 0.8, 13.6, fc="#eaeaea", ec="#888888", lw=0.8))
    ax.text(12.4, 7.2, "COMPARATIVE ANALYSIS OF ARCHITECTURAL TRANSPARENCY",
            rotation=90, ha="center", va="center", fontsize=7.5, fontweight="bold",
            color="#333333")
    # right interpretability band
    ax.add_patch(Rectangle((25.3, 0.4), 0.55, 13.6, fc="#dff0d8", ec="#1f6f43", lw=0.8))
    ax.text(25.57, 7.2, "HIGH INTERPRETABILITY / EXPLAINED DECISION MAKING",
            rotation=90, ha="center", va="center", fontsize=7, fontweight="bold",
            color="#1f6f43")

    def img_at(x, yb, w, h, im, title=None, cmap="gray"):
        a = fig.add_axes([x / 26, yb / 15.2, w / 26, h / 15.2])
        a.imshow(im, cmap=cmap); a.set_xticks([]); a.set_yticks([])
        if title:
            a.set_title(title, fontsize=6.5, pad=2)
        for s in a.spines.values():
            s.set_color("#555555")
        return a

    # ---------------- left: CNN ----------------
    img_at(0.6, 11.6, 1.9, 2.0, raw, "INPUT 28$\\times$28")
    _arr(ax, 2.7, 12.6, 3.4, 12.6)
    ax.text(6.9, 13.5, "HIERARCHICAL FEATURE LEARNING", fontsize=7.5,
            ha="center", fontweight="bold")
    # conv/pool stacks as layered rectangles
    x0 = 3.5
    depths = [(1.9, 2.6, "Conv2D+ReLU\n32@26$\\times$26"), (1.6, 2.2, "MaxPooling"),
              (1.35, 1.9, "Conv2D+ReLU\n64@11$\\times$11"), (1.1, 1.6, "MaxPooling")]
    for k, (w_, h_, lab) in enumerate(depths):
        for j in range(3):
            ax.add_patch(Rectangle((x0 + k * 2.0 + j * 0.09, 11.0 - j * 0.09 + (2.6 - h_) / 2),
                         w_, h_, fc=plt.cm.Blues(0.35 + 0.12 * k), ec="black", lw=0.6))
        ax.text(x0 + k * 2.0 + w_ / 2, 10.35, lab, fontsize=5.8, ha="center", va="top")
    _arr(ax, 3.0, 9.0, 3.0, 8.2)
    _box(ax, 0.8, 6.6, 4.6, 1.5, "DIMENSIONALITY REDUCTION\nFlatten (1024-D)",
         "#e8e8e8", fs=7.5)
    _arr(ax, 3.1, 6.6, 3.1, 5.8)
    _box(ax, 0.8, 4.2, 4.6, 1.5, "CLASSIFICATION BLOCK\nDense + Dropout + Softmax",
         "#e8e8e8", fs=7.5)
    # opaque features note
    _box(ax, 6.4, 4.2, 5.0, 3.9,
         "Abstract &\nOpaque\nFeatures", "#f2f2f2", fs=8,
         sub="learned filters are not\nhuman-nameable;\ndecision provenance\nis inaccessible", subfs=6.5)
    _arr(ax, 3.1, 4.2, 3.1, 3.4)
    _box(ax, 0.8, 1.9, 4.6, 1.4, "Predicted Class + Probabilities", "#dddddd", fs=7.5)
    ax.text(6.0, 1.0, "interpretation: post-hoc only (saliency)", fontsize=6.5,
            ha="center", style="italic", color="#555555")

    # ---------------- right: KTSD ----------------
    ax.text(14.2, 13.6, "PREPROCESSING BLOCK", fontsize=7.5, ha="left",
            fontweight="bold")
    img_at(13.4, 11.5, 1.7, 1.8, raw, "grayscale")
    img_at(15.4, 11.5, 1.7, 1.8, binary, "adaptive/global\nthreshold")
    img_at(17.4, 11.5, 1.7, 1.8, skel, "Zhang--Suen\nthinning")
    _arr(ax, 15.15, 12.4, 15.35, 12.4); _arr(ax, 17.15, 12.4, 17.35, 12.4)

    ax.text(18.8, 10.9, "TOPOLOGICAL FEATURE ENGINEERING", fontsize=7.5,
            ha="center", fontweight="bold")
    # 2x2 mini panels: loops overlay, endpoints/junctions, direction hist, global stats
    a1 = img_at(13.4, 8.4, 1.9, 2.0, binary, "Topology: loops")
    import cv2 as _cv
    cs, hh = _cv.findContours(binary, _cv.RETR_TREE, _cv.CHAIN_APPROX_SIMPLE)
    if hh is not None:
        for c_, h_ in zip(cs, hh[0]):
            if h_[3] != -1 and _cv.contourArea(c_) > MIN_LOOP_AREA:
                c_ = c_.squeeze()
                a1.plot(np.append(c_[:, 0], c_[0, 0]), np.append(c_[:, 1], c_[0, 1]),
                        c="lime", lw=1.4)
    a2 = img_at(15.7, 8.4, 1.9, 2.0, skel, "Graph: endpoints,\njunctions")
    cnt = _neighbor_counts(skel)
    ey, ex = np.where((skel > 0) & (cnt == 1)); jy, jx = np.where((skel > 0) & (cnt >= 3))
    a2.scatter(ex, ey, s=14, marker="o", fc="none", ec="red", linewidths=1.0)
    a2.scatter(jx, jy, s=14, marker="s", fc="none", ec="orange", linewidths=1.0)
    # direction histogram mini
    from ktsd.features import extract_curvature_histogram
    hvals = extract_curvature_histogram(skel)
    a3 = fig.add_axes([18.05 / 26, 8.55 / 15.2, 1.85 / 26, 1.75 / 15.2])
    a3.bar(range(8), hvals, color="#2980b9"); a3.set_xticks([]); a3.set_yticks([])
    a3.set_title("Curvature: angle\nhistogram (8)", fontsize=6.5, pad=2)
    # global stats mini
    from ktsd.features import extract_stroke_statistics
    svals = extract_stroke_statistics(binary)
    a4 = fig.add_axes([20.35 / 26, 8.55 / 15.2, 1.85 / 26, 1.75 / 15.2])
    a4.bar(range(5), svals, color="#8e44ad"); a4.set_xticks([]); a4.set_yticks([])
    a4.set_title("Global stats:\naspect, solidity...", fontsize=6.5, pad=2)

    _box(ax, 13.4, 6.3, 8.8, 1.3,
         "DIMENSIONALITY REDUCTION BLOCK", "#dff0d8", fs=7.5,
         sub="feature concatenation $\\rightarrow$ 29-D named vector", subfs=7)
    _arr(ax, 17.8, 8.35, 17.8, 7.65)
    _arr(ax, 17.8, 6.3, 17.8, 5.6)
    _box(ax, 13.4, 4.1, 8.8, 1.5, "CLASSIFICATION BLOCK", "#dff0d8", fs=7.5,
         sub="SVM with RBF kernel; margin on named features", subfs=7)
    _arr(ax, 17.8, 4.1, 17.8, 3.4)
    _box(ax, 13.4, 1.9, 8.8, 1.4,
         "Decision & Explanation: class $+$ per-feature attribution",
         "#d5e8d4", fs=7.5)
    ax.text(18.8, 1.0, "every stage inspectable; one large circular loop $\\Rightarrow$ class 0",
            fontsize=6.5, ha="center", style="italic", color="#1f6f43")

    # legend
    ax.add_patch(Rectangle((0.6, 0.35), 0.5, 0.3, fc="#e8e8e8", ec="black", lw=0.6))
    ax.text(1.25, 0.5, "Processing step", fontsize=6.5, va="center")
    _arr(ax, 3.4, 0.5, 4.0, 0.5)
    ax.text(4.15, 0.5, "Data flow", fontsize=6.5, va="center")
    ax.add_patch(Rectangle((5.8, 0.35), 0.5, 0.3, fc="#dff0d8", ec="#1f6f43", lw=0.6))
    ax.text(6.45, 0.5, "Visual insight", fontsize=6.5, va="center")
    _save(fig, out, "fig_standard_vs_ktsd")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="figures")
    ap.add_argument("--which", default="all")
    a = ap.parse_args()
    fns = {"fig_ktsd_pipeline": fig_ktsd_pipeline,
           "fig_standard_vs_ktsd": fig_standard_vs_ktsd}
    todo = fns if a.which == "all" else {a.which: fns[a.which]}
    for n, f in todo.items():
        f(a.out)
