"""Feature-group ablation study (resumable; one configuration per call).

Two families, all trained on the official 60k split (no augmentation, so the
comparison isolates the feature groups) and evaluated on the official test
split:

* drop-one: KTSD minus one feature group;
* only-one: a single feature group alone.

Usage:
    python experiments/ablation.py --features-dir features \
        --state results/ablation.json
"""

import argparse
import json
import os
import sys
import time

import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from ktsd.features import FEATURE_NAMES, FEATURE_GROUPS  # noqa: E402

SEED = 42


def configs():
    yield "all", list(range(len(FEATURE_NAMES)))
    for g, names in FEATURE_GROUPS.items():
        keep = [i for i, n in enumerate(FEATURE_NAMES) if n not in names]
        yield f"drop_{g}", keep
    for g, names in FEATURE_GROUPS.items():
        keep = [i for i, n in enumerate(FEATURE_NAMES) if n in names]
        yield f"only_{g}", keep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features-dir", default="features")
    ap.add_argument("--state", required=True)
    ap.add_argument("--subsample", type=int, default=0)
    args = ap.parse_args()

    state = {}
    if os.path.exists(args.state):
        state = json.load(open(args.state))

    d = np.load(os.path.join(args.features_dir, "train.npz"))
    X, y = d["X"], d["y"]
    dt = np.load(os.path.join(args.features_dir, "test.npz"))
    Xt, yt = dt["X"], dt["y"]

    if args.subsample and args.subsample < len(X):
        # identical stratified subsample for every configuration, so the
        # ablation deltas are directly comparable
        rng = np.random.RandomState(SEED)
        idx = np.sort(np.concatenate(
            [rng.choice(np.where(y == c)[0], args.subsample // 10,
                        replace=False) for c in range(10)]))
        X, y = X[idx], y[idx]

    for name, cols in configs():
        if name in state:
            continue
        t0 = time.time()
        sc = StandardScaler().fit(X[:, cols])
        # tol relaxed uniformly across all configurations: some reduced
        # feature sets converge extremely slowly at default tol; the same
        # tolerance is used for every config so deltas stay comparable
        svm = SVC(kernel="rbf", C=10, gamma="scale", cache_size=700,
                  tol=0.01, max_iter=200000, random_state=SEED)
        svm.fit(sc.transform(X[:, cols]), y)
        acc = float((svm.predict(sc.transform(Xt[:, cols])) == yt).mean())
        state[name] = {"n_features": len(cols), "test_acc": acc}
        json.dump(state, open(args.state, "w"), indent=1)
        print(f"{name}: {len(cols)} dims, test {acc*100:.2f}%  "
              f"({time.time()-t0:.0f}s)")
        break  # one config per invocation

    done = len(state)
    total = 1 + 2 * len(FEATURE_GROUPS)
    print(f"{done}/{total} configs done")
    if done == total:
        for k, v in sorted(state.items(), key=lambda kv: -kv[1]["test_acc"]):
            print(f"{k:28s} {v['n_features']:3d} dims  {v['test_acc']*100:.2f}%")


if __name__ == "__main__":
    main()
