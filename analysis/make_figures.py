"""Regenerate every manuscript figure from real experiment artifacts.

All figures are produced from the datasets under ``data_local/``, the feature
archives under ``features*/``, and the machine-readable results under
``repo/results/`` — nothing is drawn by hand and no external image tools are
used. Output: 300-dpi PNG + vector PDF per figure.

Usage:
    python analysis/make_figures.py --out figures --which fig_samples
    python analysis/make_figures.py --out figures --which all
"""

import argparse
import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from ktsd import preprocess_mnist  # noqa: E402
from ktsd.features import FEATURE_NAMES, FEATURE_GROUPS  # noqa: E402

plt.rcParams.update({
    "font.size": 11, "axes.titlesize": 12, "axes.labelsize": 11,
    "xtick.labelsize": 10, "ytick.labelsize": 10, "legend.fontsize": 10,
    "figure.dpi": 100, "savefig.dpi": 300, "savefig.bbox": "tight",
})

# Register a Kannada-capable font if available; matplotlib >= 3.7 falls back
# per-glyph across the font.family list, so Latin text keeps DejaVu Sans.
from matplotlib import font_manager  # noqa: E402
for _cand in [
        os.path.join(os.path.dirname(__file__), "..", "..", "fonts",
                     "NotoSansKannada.ttf"),
        os.path.join(os.path.dirname(__file__), "..", "fonts",
                     "NotoSansKannada.ttf")]:
    if os.path.exists(_cand):
        font_manager.fontManager.addfont(_cand)
        plt.rcParams["font.family"] = ["DejaVu Sans", "Noto Sans Kannada"]
        break

KANNADA = ["೦", "೧", "೨", "೩", "೪",
           "೫", "೬", "೭", "೮", "೯"]
# resolve data/prediction dirs relative to the experiment workspace (parent
# of the repo) so the script runs from either the repo root or the workspace
_HERE = os.path.dirname(os.path.abspath(__file__))
_WORK = os.path.abspath(os.path.join(_HERE, "..", ".."))
DATA = os.path.join(_WORK, "data_local")
FEAT = os.path.join(_WORK, "features")
PRED = os.path.join(_WORK, "ovo_aug")
RESULTS = os.path.join(_HERE, "..", "results")


def _save(fig, out, name):
    os.makedirs(out, exist_ok=True)
    fig.savefig(os.path.join(out, f"{name}.png"))
    fig.savefig(os.path.join(out, f"{name}.pdf"))
    plt.close(fig)
    print(f"saved {name}")


def _load(name):
    d = np.load(os.path.join(DATA, f"X_{name}.npz"))["arr_0"]
    y = np.load(os.path.join(DATA, f"y_{name}.npz"))["arr_0"]
    return d, y


def fig_samples(out):
    """Sample grid: one row per class, raw 28x28 images."""
    X, y = _load("train")
    rng = np.random.RandomState(3)
    fig, axes = plt.subplots(10, 8, figsize=(7.5, 10))
    for d in range(10):
        idx = rng.choice(np.where(y == d)[0], 8, replace=False)
        for k, i in enumerate(idx):
            ax = axes[d, k]
            ax.imshow(X[i], cmap="gray")
            ax.set_xticks([]); ax.set_yticks([])
            if k == 0:
                ax.set_ylabel(f"{KANNADA[d]}  ({d})", rotation=0, labelpad=22,
                              fontsize=13, va="center")
    fig.suptitle("Kannada-MNIST training samples (one row per numeral class)", y=0.995)
    _save(fig, out, "fig_samples")


def fig_dataset_stats(out):
    """Class balance and mean intensity per split (fixes truncated labels)."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    names = ["train", "test", "dig"]
    labels = ["Official train (60,000)", "Official test (10,000)", "Dig-MNIST (10,240)"]
    width = 0.27
    for k, (n, lab) in enumerate(zip(names, labels)):
        _, y = _load(n)
        counts = np.bincount(y, minlength=10)
        axes[0].bar(np.arange(10) + (k - 1) * width, counts, width, label=lab)
    axes[0].set_xlabel("Numeral class"); axes[0].set_ylabel("Number of samples")
    axes[0].set_xticks(range(10)); axes[0].legend(fontsize=8)
    axes[0].set_title("Class distribution per split")

    for n, lab in zip(names, labels):
        X, _ = _load(n)
        m = X.reshape(len(X), -1).mean(1)
        axes[1].hist(m, bins=50, alpha=0.55, label=lab, density=True)
    axes[1].set_xlabel("Mean pixel intensity per image")
    axes[1].set_ylabel("Density"); axes[1].legend(fontsize=8)
    axes[1].set_title("Foreground intensity distribution")
    _save(fig, out, "fig_dataset_stats")


def fig_preprocessing(out):
    """Preprocessing stages for three digits (correct row labels)."""
    import cv2
    from skimage.morphology import skeletonize
    X, y = _load("train")
    rng = np.random.RandomState(7)
    digits = [2, 5, 6]
    fig, axes = plt.subplots(3, 4, figsize=(8.6, 6.4))
    for r, d in enumerate(digits):
        i = rng.choice(np.where(y == d)[0])
        raw = X[i]
        up = cv2.resize(raw, (100, 100), interpolation=cv2.INTER_CUBIC)
        binary = preprocess_mnist(raw)
        skel = skeletonize(binary > 0)
        for c, (im, t) in enumerate([(raw, "Input 28x28"),
                                     (up, "Upsampled 100x100"),
                                     (binary, "Binarized"),
                                     (skel, "Skeleton")]):
            ax = axes[r, c]
            ax.imshow(im, cmap="gray")
            ax.set_xticks([]); ax.set_yticks([])
            if r == 0:
                ax.set_title(t)
            if c == 0:
                ax.set_ylabel(f"Digit {d}  ({KANNADA[d]})", rotation=90, fontsize=11)
    _save(fig, out, "fig_preprocessing")


def fig_topology_overlay(out):
    """Interpretability overlay: loops, endpoints, junctions on skeletons."""
    import cv2
    from skimage.morphology import skeletonize
    from ktsd.features import _neighbor_counts, MIN_LOOP_AREA
    X, y = _load("train")
    rng = np.random.RandomState(11)
    fig, axes = plt.subplots(2, 5, figsize=(11, 4.8))
    for d in range(10):
        ax = axes[d // 5, d % 5]
        # pick a representative sample: prefer loop detections when the class
        # has loops, and among those the cleanest skeleton (fewest junctions)
        cand = rng.choice(np.where(y == d)[0], 60, replace=False)
        scored = []
        for i in cand:
            b = preprocess_mnist(X[i])
            cs, h = cv2.findContours(b, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            hl = ([c_ for c_, hh in zip(cs, h[0]) if hh[3] != -1
                   and cv2.contourArea(c_) > MIN_LOOP_AREA] if h is not None else [])
            sk_ = skeletonize(b > 0).astype(np.uint8)
            nj = int(((sk_ > 0) & (_neighbor_counts(sk_) >= 3)).sum())
            scored.append((len(hl) > 0, -nj, i, hl))
        scored.sort(key=lambda t: (t[0], t[1]), reverse=True)
        chosen, holes = scored[0][2], scored[0][3]
        b = preprocess_mnist(X[chosen])
        sk = skeletonize(b > 0).astype(np.uint8)
        counts = _neighbor_counts(sk)
        ey, ex = np.where((sk > 0) & (counts == 1))
        jy, jx = np.where((sk > 0) & (counts >= 3))
        ax.imshow(b, cmap="gray")
        ys_, xs_ = np.where(sk > 0)
        ax.scatter(xs_, ys_, s=0.4, c="tab:blue", alpha=0.7)
        for c_ in holes:
            c_ = c_.squeeze()
            if c_.ndim == 2:
                ax.plot(np.append(c_[:, 0], c_[0, 0]),
                        np.append(c_[:, 1], c_[0, 1]), c="tab:green", lw=2)
        ax.scatter(ex, ey, s=45, marker="o", facecolors="none",
                   edgecolors="tab:red", linewidths=1.6, label="endpoint")
        ax.scatter(jx, jy, s=45, marker="s", facecolors="none",
                   edgecolors="orange", linewidths=1.6, label="junction")
        ax.set_title(f"{KANNADA[d]}  ({d})", fontsize=12)
        ax.set_xticks([]); ax.set_yticks([])
    handles = [plt.Line2D([], [], color="tab:blue", lw=2, label="skeleton"),
               plt.Line2D([], [], color="tab:green", lw=2, label="loop (enclosed hole)"),
               plt.Line2D([], [], marker="o", mfc="none", mec="tab:red", ls="", label="endpoint"),
               plt.Line2D([], [], marker="s", mfc="none", mec="orange", ls="", label="junction")]
    fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False)
    fig.suptitle("KTSD structural elements detected on Kannada numerals")
    fig.tight_layout(rect=[0, 0.05, 1, 1])
    _save(fig, out, "fig_topology_overlay")


def fig_feature_distributions(out):
    """Distributions of key interpretable features per class."""
    d = np.load(os.path.join(FEAT, "train.npz")); F, y = d["X"], d["y"]
    feats = [("num_loops", 0), ("num_endpoints", 8),
             ("num_junctions", 13), ("circularity", 28)]
    fig, axes = plt.subplots(2, 2, figsize=(10, 6.4))
    for ax, (name, col) in zip(axes.ravel(), feats):
        data = [F[y == c, col] for c in range(10)]
        bp = ax.boxplot(data, positions=range(10), widths=0.65,
                        showfliers=False, patch_artist=True)
        for p in bp["boxes"]:
            p.set_facecolor("#9ecae1")
        ax.set_xticks(range(10))
        ax.set_xticklabels([f"{KANNADA[c]}\n{c}" for c in range(10)], fontsize=9)
        ax.set_title(name)
        ax.set_xlabel("Numeral class"); ax.set_ylabel("Feature value")
    fig.suptitle("Per-class distribution of interpretable KTSD features (training split)")
    fig.tight_layout()
    _save(fig, out, "fig_feature_distributions")


def fig_correlation(out):
    C = np.load(os.path.join(RESULTS, "feature_correlation.npy"))
    fig, ax = plt.subplots(figsize=(9.5, 8.2))
    im = ax.imshow(C, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(29)); ax.set_yticks(range(29))
    ax.set_xticklabels(FEATURE_NAMES, rotation=90, fontsize=8)
    ax.set_yticklabels(FEATURE_NAMES, fontsize=8)
    fig.colorbar(im, ax=ax, shrink=0.8, label="Pearson correlation")
    ax.set_title("KTSD feature correlation (training split, standardized)")
    _save(fig, out, "fig_correlation")


def fig_importance(out):
    mi = json.load(open(os.path.join(RESULTS, "fi_mi.json")))
    lw = json.load(open(os.path.join(RESULTS, "fi_logreg.json")))
    rf = json.load(open(os.path.join(RESULTS, "fi_rf.json")))
    fig, axes = plt.subplots(1, 3, figsize=(13, 6.8), sharey=True)
    order = np.argsort([mi[n] for n in FEATURE_NAMES])
    names_sorted = [FEATURE_NAMES[i] for i in order]
    for ax, (title, src) in zip(
            axes, [("Mutual information", mi),
                   ("Multinomial logistic |w| (standardized)", lw),
                   ("Random-forest impurity", rf)]):
        vals = [src[n] for n in names_sorted]
        ax.barh(range(29), vals, color="#4c78a8")
        ax.set_yticks(range(29)); ax.set_yticklabels(names_sorted, fontsize=8)
        ax.set_title(title, fontsize=10)
    fig.suptitle("Feature importance, three train-only estimators "
                 "(rows ordered by mutual information)")
    fig.tight_layout()
    _save(fig, out, "fig_importance")


def fig_ablation(out):
    ab = json.load(open(os.path.join(RESULTS, "ablation.json")))
    drop = {k[5:]: v for k, v in ab.items() if k.startswith("drop_")}
    only = {k[5:]: v for k, v in ab.items() if k.startswith("only_")}
    full = ab["all"]["test_acc"] * 100
    groups = list(FEATURE_GROUPS)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=False)
    x = np.arange(len(groups))
    axes[0].bar(x, [drop[g]["test_acc"] * 100 for g in groups], color="#e45756")
    axes[0].axhline(full, ls="--", c="k", lw=1, label=f"all 29 features ({full:.2f}%)")
    axes[0].set_xticks(x); axes[0].set_xticklabels(groups, rotation=20, ha="right", fontsize=9)
    axes[0].set_ylabel("Official test accuracy (%)")
    axes[0].set_title("Drop one feature group"); axes[0].legend(fontsize=8)
    axes[0].set_ylim(70, 87)
    axes[1].bar(x, [only[g]["test_acc"] * 100 for g in groups], color="#4c78a8")
    axes[1].axhline(full, ls="--", c="k", lw=1)
    axes[1].set_xticks(x); axes[1].set_xticklabels(groups, rotation=20, ha="right", fontsize=9)
    axes[1].set_title("Single feature group alone")
    for ax in axes:
        for p in ax.patches:
            ax.annotate(f"{p.get_height():.1f}", (p.get_x() + p.get_width() / 2,
                        p.get_height()), ha="center", va="bottom", fontsize=8)
    fig.suptitle("Feature-group ablation (identical 20k training protocol)")
    fig.tight_layout()
    _save(fig, out, "fig_ablation")


def fig_confusion(out):
    from sklearn.metrics import confusion_matrix
    for name, path in [("test", os.path.join(PRED, "pred_test.npz")),
                       ("dig", os.path.join(PRED, "pred_dig.npz"))]:
        d = np.load(path)
        cm = confusion_matrix(d["y"], d["pred"])
        cmn = cm / cm.sum(1, keepdims=True) * 100
        fig, ax = plt.subplots(figsize=(7.6, 6.6))
        im = ax.imshow(cmn, cmap="Greens", vmin=0, vmax=100)
        for i in range(10):
            for j in range(10):
                ax.text(j, i, f"{cmn[i, j]:.1f}" if cmn[i, j] >= 0.05 else "",
                        ha="center", va="center", fontsize=7.5,
                        color="white" if cmn[i, j] > 55 else "black")
        ticks = [f"{KANNADA[c]}\n{c}" for c in range(10)]
        ax.set_xticks(range(10)); ax.set_xticklabels(ticks, fontsize=9)
        ax.set_yticks(range(10)); ax.set_yticklabels(ticks, fontsize=9)
        ax.set_xlabel("Predicted class"); ax.set_ylabel("True class")
        title = ("Official test split" if name == "test"
                 else "Dig-MNIST (out-of-distribution)")
        acc = (d["pred"] == d["y"]).mean() * 100
        ax.set_title(f"KTSD confusion matrix, {title} — accuracy {acc:.2f}%")
        fig.colorbar(im, ax=ax, shrink=0.8, label="Row-normalized %")
        _save(fig, out, f"fig_confusion_{name}")


def fig_per_class(out):
    d = np.load(os.path.join(PRED, "pred_test.npz"))
    accs = [(d["pred"][d["y"] == c] == c).mean() * 100 for c in range(10)]
    fig, ax = plt.subplots(figsize=(8.4, 3.8))
    bars = ax.bar(range(10), accs, color="#4c78a8")
    for b, a in zip(bars, accs):
        ax.annotate(f"{a:.1f}", (b.get_x() + b.get_width() / 2, a),
                    ha="center", va="bottom", fontsize=9)
    ax.set_xticks(range(10))
    ax.set_xticklabels([f"{KANNADA[c]}\n{c}" for c in range(10)])
    ax.set_ylim(70, 100)
    ax.set_xlabel("Numeral class"); ax.set_ylabel("Recall (%)")
    ax.set_title("Per-class recall on the official test split (KTSD, augmented)")
    _save(fig, out, "fig_per_class")


def fig_classifier_comparison(out):
    s = json.load(open(os.path.join(RESULTS, "significance_test.json")))
    sd = json.load(open(os.path.join(RESULTS, "significance_dig.json")))
    order = sorted(s["accuracy"], key=lambda k: -s["accuracy"][k]["acc"])
    labels = {"rbf_svm": "RBF-SVM", "random_forest": "Random forest",
              "knn": "k-NN (k=5)", "logreg": "Logistic regression",
              "linear_svm": "Linear SVM", "naive_bayes": "Gaussian NB"}
    fig, ax = plt.subplots(figsize=(9, 4.2))
    x = np.arange(len(order)); w = 0.38
    for k, (src, lab, col) in enumerate([(s, "Official test", "#4c78a8"),
                                         (sd, "Dig-MNIST (OOD)", "#f58518")]):
        acc = [src["accuracy"][n]["acc"] * 100 for n in order]
        lo = [src["accuracy"][n]["wilson95"][0] * 100 for n in order]
        hi = [src["accuracy"][n]["wilson95"][1] * 100 for n in order]
        err = [np.array(acc) - lo, np.array(hi) - np.array(acc)]
        ax.bar(x + (k - 0.5) * w, acc, w, yerr=err, capsize=3, label=lab, color=col)
    ax.set_xticks(x); ax.set_xticklabels([labels[n] for n in order], fontsize=9)
    ax.set_ylabel("Accuracy (%)"); ax.legend()
    ax.set_title("Classifier comparison on KTSD features "
                 "(official 60k training, 95% Wilson intervals)")
    _save(fig, out, "fig_classifier_comparison")


def fig_descriptor_comparison(out):
    s = json.load(open(os.path.join(RESULTS, "significance_descriptors_test.json")))
    sd = json.load(open(os.path.join(RESULTS, "significance_descriptors_dig.json")))
    order = ["hybrid_hog_ktsd", "hog_linear", "sift_bovw", "zernike", "ktsd_aug"]
    labels = {"ktsd_aug": "KTSD\n29-d", "zernike": "Zernike\n36-d",
              "sift_bovw": "SIFT-BoVW\n64-d", "hog_linear": "HOG\n2916-d",
              "hybrid_hog_ktsd": "HOG+KTSD\n2945-d"}
    fig, ax = plt.subplots(figsize=(9, 4.2))
    x = np.arange(len(order)); w = 0.38
    for k, (src, lab, col) in enumerate([(s, "Official test", "#4c78a8"),
                                         (sd, "Dig-MNIST (OOD)", "#f58518")]):
        acc = [src["accuracy"][n]["acc"] * 100 for n in order]
        lo = [src["accuracy"][n]["wilson95"][0] * 100 for n in order]
        hi = [src["accuracy"][n]["wilson95"][1] * 100 for n in order]
        err = [np.array(acc) - lo, np.array(hi) - np.array(acc)]
        ax.bar(x + (k - 0.5) * w, acc, w, yerr=err, capsize=3, label=lab, color=col)
    ax.set_xticks(x); ax.set_xticklabels([labels[n] for n in order], fontsize=9)
    ax.set_ylabel("Accuracy (%)"); ax.legend()
    ax.set_ylim(40, 100)
    ax.set_title("Descriptor comparison, identical protocol "
                 "(95% Wilson intervals)")
    _save(fig, out, "fig_descriptor_comparison")


def fig_shap(out):
    ex = json.load(open(os.path.join(RESULTS, "shap_examples.json")))
    X, _ = _load("test")
    show = [e for e in ex if e["true"] == e["pred"]][:2] + \
           [e for e in ex if e["true"] != e["pred"]][:2]
    fig = plt.figure(figsize=(11.5, 6.6))
    gs = fig.add_gridspec(2, 4, width_ratios=[1, 2.4, 1, 2.4])
    for k, e in enumerate(show):
        r, c = divmod(k, 2)
        ax_img = fig.add_subplot(gs[r, c * 2])
        ax_img.imshow(X[e["test_idx"]], cmap="gray")
        ok = e["true"] == e["pred"]
        ax_img.set_title(f"true {KANNADA[e['true']]} ({e['true']}) → "
                         f"pred {KANNADA[e['pred']]} ({e['pred']})",
                         fontsize=10, color="black" if ok else "#b30000")
        ax_img.set_xticks([]); ax_img.set_yticks([])
        contrib = sorted(e["shap_for_pred"].items(), key=lambda kv: -abs(kv[1]))[:7]
        names = [n for n, _ in contrib][::-1]
        vals = [v for _, v in contrib][::-1]
        ax_b = fig.add_subplot(gs[r, c * 2 + 1])
        ax_b.barh(range(len(vals)), vals,
                  color=["#4c78a8" if v > 0 else "#e45756" for v in vals])
        ax_b.set_yticks(range(len(names))); ax_b.set_yticklabels(names, fontsize=8)
        ax_b.axvline(0, c="k", lw=0.8)
        ax_b.set_xlabel("SHAP value toward predicted class", fontsize=8)
    fig.suptitle("Feature-level explanations (KernelSHAP on the RBF-SVM decision function)")
    fig.tight_layout()
    _save(fig, out, "fig_shap_examples")


FIGS = {f.__name__: f for f in [
    fig_samples, fig_dataset_stats, fig_preprocessing, fig_topology_overlay,
    fig_feature_distributions, fig_correlation, fig_importance, fig_ablation,
    fig_confusion, fig_per_class, fig_classifier_comparison,
    fig_descriptor_comparison, fig_shap]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="figures")
    ap.add_argument("--which", default="all")
    args = ap.parse_args()
    todo = FIGS if args.which == "all" else {args.which: FIGS[args.which]}
    for name, fn in todo.items():
        fn(args.out)


if __name__ == "__main__":
    main()
