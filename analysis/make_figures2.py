"""Additional manuscript figures mirroring the original paper's figure set.

Same conventions as make_figures.py: everything is generated from datasets
and machine-readable experiment artifacts; no external graphics.
"""
import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from make_figures import (_save, _load, KANNADA, DATA, FEAT, RESULTS)  # noqa
from ktsd import preprocess_mnist  # noqa
from ktsd.features import FEATURE_NAMES, FEATURE_GROUPS  # noqa


def fig_pipeline(out):
    """Standard OCR pipeline vs the KTSD pipeline (schematic, from code)."""
    fig, ax = plt.subplots(figsize=(11, 4.6))
    ax.axis("off")

    def box(x, y, w, h, text, fc, fontsize=9):
        ax.add_patch(mpatches.FancyBboxPatch(
            (x, y), w, h, boxstyle="round,pad=0.02",
            fc=fc, ec="black", lw=1))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
                fontsize=fontsize)

    def arrow(x1, y1, x2, y2):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="->", lw=1.4))

    # top lane: conventional appearance-based pipeline
    ax.text(0.02, 0.95, "(a) Conventional appearance-based pipeline",
            fontsize=11, weight="bold")
    steps1 = ["Input\nimage", "Normalization", "Appearance descriptor\n(pixels / HOG / CNN)",
              "Classifier", "Label only"]
    w1, gap1 = 0.155, 0.045
    x = 0.03
    xs = []
    for s in steps1:
        box(x, 0.68, w1, 0.20, s, "#dbe5f1", fontsize=9)
        xs.append(x)
        x += w1 + gap1
    for i in range(len(xs) - 1):
        arrow(xs[i] + w1, 0.78, xs[i + 1] - 0.004, 0.78)

    # bottom lane: KTSD with an explicit two-branch split
    ax.text(0.02, 0.56, "(b) KTSD pipeline (this work)", fontsize=11, weight="bold")
    wb = 0.13
    box(0.03, 0.22, wb, 0.20, "Input\nimage", "#dbe5f1", 9)
    box(0.20, 0.22, wb, 0.20, "Upsample to\n$100\\times100$,\nbinarize", "#dbe5f1", 8.5)
    box(0.385, 0.34, 0.17, 0.16, "Binary-image branch:\nloops, shape statistics", "#e2efda", 8.5)
    box(0.385, 0.10, 0.17, 0.16, "Skeleton branch:\nendpoints, junctions,\ndirection histogram", "#e2efda", 8.5)
    box(0.60, 0.22, wb, 0.20, "29-D named\nfeature vector", "#e2efda", 8.5)
    box(0.765, 0.22, 0.09, 0.20, "RBF-\nSVM", "#dbe5f1", 9)
    box(0.885, 0.22, 0.105, 0.20, "Label +\nreadable\nexplanation", "#e2efda", 8.5)
    arrow(0.03 + wb, 0.32, 0.196, 0.32)
    arrow(0.20 + wb, 0.37, 0.381, 0.42)   # to binary branch
    arrow(0.20 + wb, 0.27, 0.381, 0.18)   # to skeleton branch
    arrow(0.555, 0.42, 0.596, 0.35)       # binary branch -> vector
    arrow(0.555, 0.18, 0.596, 0.29)       # skeleton branch -> vector
    arrow(0.60 + wb, 0.32, 0.761, 0.32)
    arrow(0.765 + 0.09, 0.32, 0.881, 0.32)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    _save(fig, out, "fig_pipeline")


def fig_digit_profiles(out):
    """Per-digit mean standardized feature profile (10 x 29 heatmap)."""
    d = np.load(os.path.join(FEAT, "train.npz")); F, y = d["X"], d["y"]
    mu, sd = F.mean(0), F.std(0) + 1e-9
    prof = np.stack([((F[y == c] - mu) / sd).mean(0) for c in range(10)])
    fig, ax = plt.subplots(figsize=(11, 4.2))
    im = ax.imshow(prof, cmap="RdBu_r", vmin=-1.6, vmax=1.6, aspect="auto")
    ax.set_yticks(range(10))
    ax.set_yticklabels([f"{KANNADA[c]} ({c})" for c in range(10)], fontsize=9)
    ax.set_xticks(range(29))
    ax.set_xticklabels(FEATURE_NAMES, rotation=90, fontsize=7.5)
    fig.colorbar(im, ax=ax, shrink=0.85, label="Mean z-scored value")
    ax.set_title("Per-class KTSD signature: mean standardized value of every feature (training split)")
    _save(fig, out, "fig_digit_profiles")


def fig_loop_distribution(out):
    """Distribution of detected loop counts per class."""
    d = np.load(os.path.join(FEAT, "train.npz")); F, y = d["X"], d["y"]
    loops = F[:, 0].astype(int)
    counts = np.zeros((10, 4))
    for c in range(10):
        v = loops[y == c]
        for k in range(4):
            counts[c, k] = (v == k).mean() * 100 if k < 3 else (v >= 3).mean() * 100
    fig, ax = plt.subplots(figsize=(9.5, 3.8))
    bottom = np.zeros(10)
    labels = ["0 loops", "1 loop", "2 loops", ">=3 loops"]
    colors = ["#d9d9d9", "#9ecae1", "#4c78a8", "#20425c"]
    for k in range(4):
        ax.bar(range(10), counts[:, k], bottom=bottom, color=colors[k], label=labels[k])
        bottom += counts[:, k]
    ax.set_xticks(range(10))
    ax.set_xticklabels([f"{KANNADA[c]}\n{c}" for c in range(10)])
    ax.set_ylabel("Share of training images (%)")
    ax.set_title("Detected loop-count distribution per class (training split)")
    ax.legend(ncol=4, fontsize=8, loc="lower center", bbox_to_anchor=(0.5, -0.42))
    _save(fig, out, "fig_loop_distribution")


def fig_cv_folds(out):
    cv = json.load(open(os.path.join(RESULTS, "cv_final.json")))
    accs = [cv[str(k)] * 100 for k in range(5)]
    fig, ax = plt.subplots(figsize=(7.5, 3.4))
    bars = ax.bar(range(1, 6), accs, color="#4c78a8")
    m = np.mean(accs)
    ax.axhline(m, ls="--", c="red", lw=1.2, label=f"mean {m:.2f}%")
    for b, a in zip(bars, accs):
        ax.annotate(f"{a:.2f}", (b.get_x() + b.get_width() / 2, a), ha="center",
                    va="bottom", fontsize=9)
    ax.set_ylim(90, 96); ax.set_xlabel("Fold"); ax.set_ylabel("Validation accuracy (%)")
    ax.set_title("Five-fold cross-validation on the training split (per-fold standardization)")
    ax.legend()
    _save(fig, out, "fig_cv_folds")


def fig_grid_heatmap(out):
    cells = json.load(open(os.path.join(RESULTS, "gridsearch_cells.json")))
    Cs = [0.1, 1, 10, 100]
    Gs = ["scale", 0.01, 0.03, 0.1, 0.3]
    M = np.zeros((len(Cs), len(Gs)))
    for i, C in enumerate(Cs):
        for j, g in enumerate(Gs):
            M[i, j] = np.mean([cells[f"C={C}|gamma={g}|fold={f}"] for f in range(5)]) * 100
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    im = ax.imshow(M, cmap="viridis")
    for i in range(len(Cs)):
        for j in range(len(Gs)):
            ax.text(j, i, f"{M[i, j]:.1f}", ha="center", va="center",
                    color="white" if M[i, j] < 88 else "black", fontsize=9)
    ax.set_xticks(range(len(Gs))); ax.set_xticklabels([str(g) for g in Gs])
    ax.set_yticks(range(len(Cs))); ax.set_yticklabels([str(c) for c in Cs])
    ax.set_xlabel(r"$\gamma$"); ax.set_ylabel("C")
    ax.set_title("Grid search: 5-fold CV accuracy (%) per configuration")
    fig.colorbar(im, ax=ax, shrink=0.85)
    _save(fig, out, "fig_grid_heatmap")


def fig_top2_scatter(out):
    d = np.load(os.path.join(FEAT, "train.npz")); F, y = d["X"], d["y"]
    rng = np.random.RandomState(0)
    idx = np.concatenate([rng.choice(np.where(y == c)[0], 200, replace=False) for c in range(10)])
    fn = {n: i for i, n in enumerate(FEATURE_NAMES)}
    pairs = [("num_loops", "num_endpoints"), ("num_endpoints", "num_junctions"),
             ("num_loops", "circularity"), ("loop_area_ratio", "circularity")]
    db = json.load(open(os.path.join(RESULTS, "davies_bouldin.json")))
    fig, axes = plt.subplots(2, 2, figsize=(10, 7.6))
    cmap = plt.cm.tab10
    for ax, (a, b) in zip(axes.ravel(), pairs):
        for c in range(10):
            m = y[idx] == c
            ja = F[idx[m], fn[a]] + rng.normal(0, 0.05, m.sum())
            jb = F[idx[m], fn[b]] + rng.normal(0, 0.05 * max(1, F[:, fn[b]].std()), m.sum())
            ax.scatter(ja, jb, s=6, color=cmap(c), alpha=0.55, label=str(c) if a == "num_loops" and b == "num_endpoints" else None)
        ax.set_xlabel(a); ax.set_ylabel(b)
        ax.set_title(f"DB index = {db[f'{a}|{b}']:.2f}", fontsize=10)
    handles = [plt.Line2D([], [], marker="o", ls="", color=cmap(c), label=f"{KANNADA[c]} ({c})") for c in range(10)]
    fig.legend(handles=handles, loc="lower center", ncol=10, fontsize=8, frameon=False)
    fig.suptitle("Pairwise feature projections and Davies-Bouldin separability (lower is better)")
    fig.tight_layout(rect=[0, 0.05, 1, 1])
    _save(fig, out, "fig_top2_scatter")


def fig_tsne(out):
    d = np.load(os.path.join(RESULTS, "tsne_embedding.npz"))
    emb, y = d["emb"], d["y"]
    fig, ax = plt.subplots(figsize=(7.8, 6.2))
    cmap = plt.cm.tab10
    for c in range(10):
        m = y == c
        ax.scatter(emb[m, 0], emb[m, 1], s=8, color=cmap(c), alpha=0.7,
                   label=f"{KANNADA[c]} ({c})")
    ax.legend(ncol=2, fontsize=8, markerscale=1.6)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title("t-SNE embedding of standardized KTSD features (3,000 training samples)")
    _save(fig, out, "fig_tsne")


def fig_feature_vs_accuracy(out):
    st = json.load(open(os.path.join(RESULTS, "feature_count_curve.json")))
    ks = sorted(int(k) for k in st)
    accs = [st[str(k)] * 100 for k in ks]
    fig, ax = plt.subplots(figsize=(7.5, 3.6))
    ax.plot(ks, accs, "o-", color="#4c78a8", lw=2)
    for k, a in zip(ks, accs):
        ax.annotate(f"{a:.1f}", (k, a), textcoords="offset points", xytext=(0, 7),
                    ha="center", fontsize=8)
    ax.set_xlabel("Number of features (ranked by mutual information)")
    ax.set_ylabel("Official test accuracy (%)")
    ax.set_title("Accuracy as features are added in mutual-information order (20k training protocol)")
    ax.grid(alpha=0.3)
    _save(fig, out, "fig_feature_vs_accuracy")


def fig_augmentation(out):
    import cv2
    X, y = _load("train")
    rng = np.random.RandomState(5)
    fig, axes = plt.subplots(2, 5, figsize=(10, 4.4))
    idx = [rng.choice(np.where(y == c)[0]) for c in [0, 2, 5, 7, 8]]
    for k, i in enumerate(idx):
        axes[0, k].imshow(X[i], cmap="gray")
        axes[0, k].set_title(f"{KANNADA[y[i]]} ({y[i]}) original", fontsize=9)
        M = cv2.getRotationMatrix2D((14, 14), -10, 1.0)
        rot = cv2.warpAffine(X[i], M, (28, 28), borderValue=0)
        axes[1, k].imshow(rot, cmap="gray")
        axes[1, k].set_title("rotated $-10^\\circ$", fontsize=9)
    for ax in axes.ravel():
        ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle("Training augmentation: each image contributes itself and one $-10^\\circ$ rotation")
    _save(fig, out, "fig_augmentation")


def fig_loop_sensitivity(out):
    sens = json.load(open(os.path.join(RESULTS, "loop_area_sensitivity.json")))
    fig, ax = plt.subplots(figsize=(7.5, 3.8))
    thresholds = sorted(int(k) for k in sens)
    for c in range(10):
        vals = [sens[str(t)][str(c)] for t in thresholds]
        ax.plot(thresholds, vals, "o-", ms=3, label=f"{c}")
    ax.set_xlabel(r"Minimum hole area $A_{\min}$ (px at $100\times100$)")
    ax.set_ylabel("Mean detected holes")
    ax.set_title("Loop-detection sensitivity to the area threshold, per class")
    ax.legend(ncol=5, fontsize=7.5, title="class", title_fontsize=8)
    ax.grid(alpha=0.3)
    _save(fig, out, "fig_loop_sensitivity")


def fig_confusion_dig_wrap(out):
    pass  # generated by make_figures.py (fig_confusion) as fig_confusion_dig


FIGS2 = {f.__name__: f for f in [
    fig_pipeline, fig_digit_profiles, fig_loop_distribution, fig_cv_folds,
    fig_grid_heatmap, fig_top2_scatter, fig_tsne, fig_feature_vs_accuracy,
    fig_augmentation, fig_loop_sensitivity]}

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="figures")
    ap.add_argument("--which", default="all")
    a = ap.parse_args()
    todo = FIGS2 if a.which == "all" else {a.which: FIGS2[a.which]}
    for n, f in todo.items():
        f(a.out)
