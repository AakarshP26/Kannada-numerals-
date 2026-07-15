"""Resumable RBF-SVM grid search with leakage-free per-fold standardization.

Hyperparameters are selected by stratified 5-fold cross-validation on a
stratified 20,000-sample subset of the official Kannada-MNIST training split
(subsampling keeps the search tractable on CPU; the winning configuration is
refit on the full training split). Standardization statistics are computed
inside each fold on the training portion only, via a Pipeline, so no
information from validation folds leaks into scaling.

State is kept in a JSON file; each invocation completes as many (params, fold)
cells as fit in ``--budget`` seconds and exits, so the search can resume.

Usage:
    python experiments/gridsearch.py --features F/train.npz --state S.json \
        [--budget 30] [--subsample 20000]
    python experiments/gridsearch.py --state S.json --report
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

C_GRID = [0.1, 1, 10, 100]
GAMMA_GRID = ["scale", 0.01, 0.03, 0.1, 0.3]
N_FOLDS = 5
SEED = 42


def cells():
    for C in C_GRID:
        for gamma in GAMMA_GRID:
            for fold in range(N_FOLDS):
                yield C, gamma, fold


def cell_key(C, gamma, fold):
    return f"C={C}|gamma={gamma}|fold={fold}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features")
    ap.add_argument("--state", required=True)
    ap.add_argument("--budget", type=float, default=30.0)
    ap.add_argument("--subsample", type=int, default=20000)
    ap.add_argument("--report", action="store_true")
    args = ap.parse_args()

    state = {}
    if os.path.exists(args.state):
        with open(args.state) as f:
            state = json.load(f)

    if args.report:
        agg = {}
        for C in C_GRID:
            for gamma in GAMMA_GRID:
                scores = [state.get(cell_key(C, gamma, f)) for f in range(N_FOLDS)]
                if all(s is not None for s in scores):
                    agg[(C, gamma)] = (float(np.mean(scores)), float(np.std(scores)))
        for (C, gamma), (m, s) in sorted(agg.items(), key=lambda kv: -kv[1][0]):
            print(f"C={C:<6} gamma={str(gamma):<6} cv_acc={m*100:.2f} +- {s*100:.2f}")
        done = len([k for k in state if k.startswith("C=")])
        total = len(C_GRID) * len(GAMMA_GRID) * N_FOLDS
        print(f"cells done: {done}/{total}")
        return

    d = np.load(args.features)
    X, y = d["X"], d["y"]

    rng = np.random.RandomState(SEED)
    if args.subsample and args.subsample < len(X):
        idx = []
        per_class = args.subsample // len(np.unique(y))
        for c in np.unique(y):
            cls = np.where(y == c)[0]
            idx.extend(rng.choice(cls, per_class, replace=False))
        idx = np.array(sorted(idx))
        X, y = X[idx], y[idx]

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    folds = list(skf.split(X, y))

    def eval_cell(C, gamma, fold):
        tr, va = folds[fold]
        pipe = make_pipeline(
            StandardScaler(),
            SVC(kernel="rbf", C=C, gamma=gamma, cache_size=400, random_state=SEED),
        )
        pipe.fit(X[tr], y[tr])
        return float(pipe.score(X[va], y[va]))

    from joblib import Parallel, delayed

    pending = [(C, g, f) for C, g, f in cells() if cell_key(C, g, f) not in state]
    t_start = time.time()
    n_done = 0
    while pending and time.time() - t_start < args.budget:
        batch, pending = pending[:4], pending[4:]
        scores = Parallel(n_jobs=len(batch))(
            delayed(eval_cell)(C, g, f) for C, g, f in batch
        )
        for (C, g, f), acc in zip(batch, scores):
            state[cell_key(C, g, f)] = acc
        n_done += len(batch)
        with open(args.state, "w") as f:
            json.dump(state, f)

    done = len([k for k in state if k.startswith("C=")])
    total = len(C_GRID) * len(GAMMA_GRID) * N_FOLDS
    print(f"+{n_done} cells this run; {done}/{total} total")


if __name__ == "__main__":
    main()
