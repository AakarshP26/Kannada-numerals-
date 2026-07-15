"""Resumable 5-fold CV of the final configuration on the full training split.

Per-fold standardization via Pipeline (no leakage). One fold per invocation
(each fold fits ~48k samples); state accumulates in a JSON file.

Usage:
    python experiments/cv_final.py --features features/train.npz \
        --state results/cv_final.json [--C 10] [--gamma scale]
"""

import argparse
import json
import os
import time

import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

SEED = 42


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", required=True)
    ap.add_argument("--state", required=True)
    ap.add_argument("--C", type=float, default=10)
    ap.add_argument("--gamma", default="scale")
    args = ap.parse_args()

    state = {}
    if os.path.exists(args.state):
        state = json.load(open(args.state))

    d = np.load(args.features)
    X, y = d["X"], d["y"]
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    folds = list(skf.split(X, y))

    for k, (tr, va) in enumerate(folds):
        if str(k) in state:
            continue
        gamma = args.gamma if args.gamma == "scale" else float(args.gamma)
        pipe = make_pipeline(StandardScaler(),
                             SVC(kernel="rbf", C=args.C, gamma=gamma,
                                 cache_size=800, random_state=SEED))
        t0 = time.time()
        pipe.fit(X[tr], y[tr])
        acc = float(pipe.score(X[va], y[va]))
        state[str(k)] = acc
        json.dump(state, open(args.state, "w"))
        print(f"fold {k}: {acc*100:.2f}%  ({time.time()-t0:.0f}s)")
        break  # one fold per invocation

    if len(state) == 5:
        accs = [state[str(k)] for k in range(5)]
        print(f"5-fold CV: {np.mean(accs)*100:.2f} +- {np.std(accs)*100:.2f}")


if __name__ == "__main__":
    main()
