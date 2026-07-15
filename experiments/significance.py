"""Statistical validation of classifier comparisons.

* McNemar's test (with continuity correction) between every pair of
  classifiers on the same test set — the appropriate paired test for
  comparing two classifiers on identical samples.
* 95% Wilson score intervals for each accuracy.
* Bootstrap (10,000 resamples) confidence intervals for pairwise accuracy
  differences.

Usage:
    python experiments/significance.py --pred-dir clf_comp --set test \
        --out results/significance_test.json
"""

import argparse
import glob
import json
import os

import numpy as np
from scipy import stats


def wilson_ci(k, n, z=1.959963984540054):
    p = k / n
    denom = 1 + z ** 2 / n
    center = (p + z ** 2 / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2)) / denom
    return center - half, center + half


def mcnemar(y, pa, pb):
    """Continuity-corrected McNemar chi-square test."""
    a_right = pa == y
    b_right = pb == y
    b01 = int(np.sum(a_right & ~b_right))
    b10 = int(np.sum(~a_right & b_right))
    if b01 + b10 == 0:
        return b01, b10, 0.0, 1.0
    chi2 = (abs(b01 - b10) - 1) ** 2 / (b01 + b10)
    p = float(stats.chi2.sf(chi2, df=1))
    return b01, b10, float(chi2), p


def bootstrap_diff(y, pa, pb, n_boot=10000, seed=42, block=1000):
    rng = np.random.RandomState(seed)
    n = len(y)
    # bootstrap the per-sample correctness *difference* (vectorized in blocks)
    delta = (pa == y).astype(np.float32) - (pb == y).astype(np.float32)
    diffs = np.empty(n_boot, dtype=np.float64)
    for start in range(0, n_boot, block):
        b = min(block, n_boot - start)
        idx = rng.randint(0, n, size=(b, n))
        diffs[start:start + b] = delta[idx].mean(axis=1)
    return float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred-dir", required=True)
    ap.add_argument("--set", default="test", choices=["test", "dig"])
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    preds = {}
    y = None
    for f in sorted(glob.glob(os.path.join(args.pred_dir, "*.npz"))):
        name = os.path.splitext(os.path.basename(f))[0]
        d = np.load(f)
        if f"pred_{args.set}" not in d:
            continue
        preds[name] = d[f"pred_{args.set}"]
        y = d[f"y_{args.set}"]

    out = {"set": args.set, "n": int(len(y)), "accuracy": {}, "pairwise": {}}
    for name, p in preds.items():
        k = int((p == y).sum())
        lo, hi = wilson_ci(k, len(y))
        out["accuracy"][name] = {"acc": k / len(y),
                                 "wilson95": [lo, hi]}
        print(f"{name:15s} acc={k/len(y)*100:6.2f}%  "
              f"95% CI [{lo*100:.2f}, {hi*100:.2f}]")

    names = sorted(preds)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            b01, b10, chi2, p = mcnemar(y, preds[a], preds[b])
            dlo, dhi = bootstrap_diff(y, preds[a], preds[b])
            out["pairwise"][f"{a}|{b}"] = {
                "only_a_correct": b01, "only_b_correct": b10,
                "mcnemar_chi2": chi2, "p": p,
                "acc_diff_boot95": [dlo, dhi],
            }
            print(f"{a} vs {b}: chi2={chi2:.1f} p={p:.2e} "
                  f"diff95=[{dlo*100:+.2f}, {dhi*100:+.2f}]")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=1)


if __name__ == "__main__":
    main()
