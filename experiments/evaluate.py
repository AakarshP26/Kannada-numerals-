"""Train the final KTSD model and evaluate on all test sets.

Protocol:
* train on the official Kannada-MNIST training split (60,000), optionally
  with the x2 rotation augmentation (each image + its -10 degree copy);
* standardization fit on training data only;
* evaluate on the official 10,000-image test split, the out-of-distribution
  Dig-MNIST split (10,240), and (if available) the custom scan set.

Outputs: model, per-set predictions, accuracy/precision/recall/F1 per class,
confusion matrices — all as machine-readable artifacts under --out.

Usage:
    python experiments/evaluate.py --features-dir F --out results/ktsd \
        --C 10 --gamma scale [--augment] [--stage fit|predict|report]
"""

import argparse
import json
import os
import time

import joblib
import numpy as np
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix)
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

SEED = 42


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features-dir", default="features")
    ap.add_argument("--out", required=True)
    ap.add_argument("--C", type=float, default=10)
    ap.add_argument("--gamma", default="scale")
    ap.add_argument("--augment", action="store_true")
    ap.add_argument("--stage", default="all",
                    choices=["fit", "predict", "report", "all"])
    ap.add_argument("--predict-set", default=None, help="test|dig (stage=predict)")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    gamma = args.gamma if args.gamma == "scale" else float(args.gamma)
    model_path = os.path.join(args.out, "model.joblib")

    if args.stage in ("fit", "all"):
        d = np.load(os.path.join(args.features_dir, "train.npz"))
        X, y = d["X"], d["y"]
        if args.augment:
            da = np.load(os.path.join(args.features_dir, "train_rot10.npz"))
            X = np.concatenate([X, da["X"]])
            y = np.concatenate([y, da["y"]])
        scaler = StandardScaler().fit(X)
        t0 = time.time()
        svm = SVC(kernel="rbf", C=args.C, gamma=gamma, cache_size=900,
                  random_state=SEED)
        svm.fit(scaler.transform(X), y)
        joblib.dump({"svm": svm, "scaler": scaler,
                     "config": {"C": args.C, "gamma": args.gamma,
                                "augment": args.augment, "n_train": len(X)}},
                    model_path)
        print(f"fit n={len(X)}: {time.time()-t0:.1f}s, "
              f"SVs={svm.n_support_.sum()}")

    if args.stage in ("predict", "all"):
        m = joblib.load(model_path)
        svm, scaler = m["svm"], m["scaler"]
        sets = [args.predict_set] if args.predict_set else ["test", "dig"]
        for name in sets:
            d = np.load(os.path.join(args.features_dir, f"{name}.npz"))
            X, y = d["X"], d["y"]
            t0 = time.time()
            pred = svm.predict(scaler.transform(X))
            np.savez(os.path.join(args.out, f"pred_{name}.npz"),
                     pred=pred, y=y)
            print(f"{name}: acc={accuracy_score(y, pred)*100:.2f}%  "
                  f"({time.time()-t0:.1f}s)")

    if args.stage in ("report", "all"):
        report = {}
        for name in ["test", "dig"]:
            p = os.path.join(args.out, f"pred_{name}.npz")
            if not os.path.exists(p):
                continue
            d = np.load(p)
            pred, y = d["pred"], d["y"]
            report[name] = {
                "accuracy": float(accuracy_score(y, pred)),
                "per_class": classification_report(y, pred, output_dict=True),
                "confusion_matrix": confusion_matrix(y, pred).tolist(),
            }
        with open(os.path.join(args.out, "report.json"), "w") as f:
            json.dump(report, f, indent=1)
        for name, r in report.items():
            print(f"{name}: acc={r['accuracy']*100:.2f}%  "
                  f"macroF1={r['per_class']['macro avg']['f1-score']*100:.2f}%")


if __name__ == "__main__":
    main()
