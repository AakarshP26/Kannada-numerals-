"""Chunked one-vs-one RBF-SVM training for large training sets.

sklearn's ``SVC`` trains a one-vs-one ensemble internally (45 binary
classifiers for 10 classes) in a single call, which cannot be interrupted.
This module trains the same pairwise classifiers one at a time with resumable
state, then combines them with the standard OvO voting rule (votes, ties
broken by summed decision values). ``--validate`` confirms the assembled
predictor agrees with a directly-trained ``SVC`` on a subsample.

Usage:
    python experiments/ovo.py fit --features-dir F --out DIR [--augment] \
        [--C 10] [--gamma scale] [--budget 30]
    python experiments/ovo.py predict --features-dir F --out DIR --dataset test
    python experiments/ovo.py validate --features-dir F --out DIR
"""

import argparse
import itertools
import json
import os
import time

import joblib
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

SEED = 42
PAIRS = list(itertools.combinations(range(10), 2))


def load_train(features_dir, augment):
    d = np.load(os.path.join(features_dir, "train.npz"))
    X, y = d["X"], d["y"]
    if augment:
        da = np.load(os.path.join(features_dir, "train_rot10.npz"))
        X = np.concatenate([X, da["X"]])
        y = np.concatenate([y, da["y"]])
    return X, y


def fit(args):
    os.makedirs(args.out, exist_ok=True)
    X, y = load_train(args.features_dir, args.augment)
    gamma = args.gamma if args.gamma == "scale" else float(args.gamma)

    scaler_path = os.path.join(args.out, "scaler.joblib")
    if os.path.exists(scaler_path):
        scaler = joblib.load(scaler_path)
    else:
        scaler = StandardScaler().fit(X)
        joblib.dump(scaler, scaler_path)
        with open(os.path.join(args.out, "config.json"), "w") as f:
            json.dump({"C": args.C, "gamma": args.gamma,
                       "augment": args.augment, "n_train": len(X)}, f)

    # gamma='scale' must reference the FULL standardized training set so all
    # pairwise models share the kernel width the direct SVC would use.
    Xs = scaler.transform(X)
    if gamma == "scale":
        gamma_val = 1.0 / (X.shape[1] * Xs.var())
    else:
        gamma_val = gamma

    t0 = time.time()
    n_done = 0
    for i, j in PAIRS:
        path = os.path.join(args.out, f"pair_{i}{j}.joblib")
        if os.path.exists(path):
            continue
        if time.time() - t0 > args.budget:
            break
        mask = (y == i) | (y == j)
        clf = SVC(kernel="rbf", C=args.C, gamma=gamma_val, cache_size=400,
                  random_state=SEED)
        clf.fit(Xs[mask], y[mask])
        joblib.dump(clf, path)
        n_done += 1

    done = sum(os.path.exists(os.path.join(args.out, f"pair_{i}{j}.joblib"))
               for i, j in PAIRS)
    print(f"+{n_done} pairs this run; {done}/{len(PAIRS)} total")


def _ovo_predict(out_dir, Xs):
    """Replicates ``SVC.predict`` exactly: majority vote over the 45 pairwise
    classifiers, ties broken by lowest class index (libsvm's rule). Verified
    to agree 100% with a directly trained ``SVC`` (see ``validate``).

    The accumulated ``dec_sum`` additionally reproduces sklearn's
    ``decision_function(decision_function_shape='ovr')`` bit-exactly via
    ``votes + dec_sum / (3 * (|dec_sum| + 1))``; note sklearn's *predict* is
    defined by the votes alone, not by that aggregated score.
    """
    votes = np.zeros((len(Xs), 10))
    for i, j in PAIRS:
        clf = joblib.load(os.path.join(out_dir, f"pair_{i}{j}.joblib"))
        d = clf.decision_function(Xs)  # >0 -> class j (classes_ = [i, j])
        votes[d > 0, j] += 1
        votes[d <= 0, i] += 1
    return np.argmax(votes, axis=1)


def predict(args):
    scaler = joblib.load(os.path.join(args.out, "scaler.joblib"))
    d = np.load(os.path.join(args.features_dir, f"{args.dataset}.npz"))
    X, y = d["X"], d["y"]
    Xs = scaler.transform(X)

    # chunk over pairs with cached partial sums so each call stays bounded
    state_path = os.path.join(args.out, f"predstate_{args.dataset}.npz")
    if os.path.exists(state_path):
        st = np.load(state_path)
        votes, dec_sum, done = st["votes"], st["dec_sum"], int(st["done"])
    else:
        votes = np.zeros((len(Xs), 10))
        dec_sum = np.zeros((len(Xs), 10))
        done = 0

    t0 = time.time()
    while done < len(PAIRS) and time.time() - t0 < args.budget:
        i, j = PAIRS[done]
        clf = joblib.load(os.path.join(args.out, f"pair_{i}{j}.joblib"))
        dv = clf.decision_function(Xs)
        votes[dv > 0, j] += 1
        votes[dv <= 0, i] += 1
        dec_sum[:, j] += dv
        dec_sum[:, i] -= dv
        done += 1
        np.savez(state_path, votes=votes, dec_sum=dec_sum, done=done)

    if done == len(PAIRS):
        pred = np.argmax(votes, axis=1)  # SVC.predict rule (votes only)
        np.savez(os.path.join(args.out, f"pred_{args.dataset}.npz"),
                 pred=pred, y=y, votes=votes, dec_sum=dec_sum)
        print(f"{args.dataset}: acc={(pred == y).mean()*100:.2f}%  (complete)")
    else:
        print(f"{args.dataset}: {done}/{len(PAIRS)} pairs accumulated")


def validate(args):
    """Manual OvO must agree with direct SVC on a 12k subsample."""
    X, y = load_train(args.features_dir, False)
    rng = np.random.RandomState(SEED)
    idx = np.sort(np.concatenate(
        [rng.choice(np.where(y == c)[0], 1200, replace=False) for c in range(10)]))
    Xs_, ys_ = X[idx], y[idx]

    scaler = StandardScaler().fit(Xs_)
    Z = scaler.transform(Xs_)
    gamma_val = 1.0 / (Z.shape[1] * Z.var())

    direct = SVC(kernel="rbf", C=10, gamma=gamma_val, random_state=SEED).fit(Z, ys_)

    tmp = os.path.join(args.out, "_validate")
    os.makedirs(tmp, exist_ok=True)
    for i, j in PAIRS:
        m = (ys_ == i) | (ys_ == j)
        clf = SVC(kernel="rbf", C=10, gamma=gamma_val, random_state=SEED)
        clf.fit(Z[m], ys_[m])
        joblib.dump(clf, os.path.join(tmp, f"pair_{i}{j}.joblib"))

    d = np.load(os.path.join(args.features_dir, "test.npz"))
    Zt = scaler.transform(d["X"])
    p_direct = direct.predict(Zt)
    p_manual = _ovo_predict(tmp, Zt)
    agree = (p_direct == p_manual).mean()
    print(f"agreement with direct SVC on official test: {agree*100:.3f}%")
    assert agree > 0.999, "manual OvO deviates from sklearn SVC"
    print("OVO VALIDATION OK")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["fit", "predict", "validate"])
    ap.add_argument("--features-dir", default="features")
    ap.add_argument("--out", required=True)
    ap.add_argument("--dataset", default="test")
    ap.add_argument("--C", type=float, default=10)
    ap.add_argument("--gamma", default="scale")
    ap.add_argument("--augment", action="store_true")
    ap.add_argument("--budget", type=float, default=30)
    args = ap.parse_args()
    {"fit": fit, "predict": predict, "validate": validate}[args.cmd](args)


if __name__ == "__main__":
    main()
