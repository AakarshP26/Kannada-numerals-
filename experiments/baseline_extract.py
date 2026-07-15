"""Chunked extraction of baseline descriptors (HOG, Zernike, SIFT-BoVW).

All descriptors operate on the same preprocessed 100x100 binary images as
KTSD, so comparisons isolate the descriptor itself.

SIFT-BoVW is two-phase:
  1. ``--descriptor sift --codebook``: learn a k=64 MiniBatchKMeans codebook
     from dense SIFT descriptors of a 10,000-image stratified subset of the
     TRAINING split only (no test leakage).
  2. ``--descriptor sift``: encode any split against the saved codebook.

Usage mirrors extract.py:
    python experiments/baseline_extract.py --descriptor hog --dataset train \
        --start 0 --end 30000 [--merge]
"""

import argparse
import glob
import os
import sys
import time
from multiprocessing import Pool

import joblib
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from ktsd import preprocess_mnist  # noqa: E402
from ktsd import baselines  # noqa: E402

_KMEANS = None
_DESC = None


def _init(descriptor, codebook_path):
    global _KMEANS, _DESC
    _DESC = descriptor
    if descriptor == "sift" and codebook_path:
        _KMEANS = joblib.load(codebook_path)


def _extract_one(img):
    b = preprocess_mnist(img)
    if _DESC == "hog":
        f = baselines.extract_hog(b)
    elif _DESC == "zernike":
        f = baselines.extract_zernike(b)
    else:
        f = baselines.bovw_encode(baselines.sift_descriptors(b), _KMEANS)
    return np.nan_to_num(np.asarray(f, dtype=np.float32))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data_local")
    ap.add_argument("--out-dir", default="features_baseline")
    ap.add_argument("--descriptor", required=True,
                    choices=["hog", "zernike", "sift"])
    ap.add_argument("--dataset", default="train", choices=["train", "test", "dig"])
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--end", type=int, default=None)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--merge", action="store_true")
    ap.add_argument("--codebook", action="store_true",
                    help="learn the SIFT codebook (phase 1)")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    tag = f"{args.descriptor}_{args.dataset}"
    codebook_path = os.path.join(args.out_dir, "sift_codebook.joblib")

    if args.codebook:
        from sklearn.cluster import MiniBatchKMeans
        X = np.load(os.path.join(args.data_dir, "X_train.npz"))["arr_0"]
        y = np.load(os.path.join(args.data_dir, "y_train.npz"))["arr_0"]
        rng = np.random.RandomState(42)
        idx = np.sort(np.concatenate(
            [rng.choice(np.where(y == c)[0], 1000, replace=False)
             for c in range(10)]))
        t0 = time.time()
        descs = []
        with Pool(args.workers, initializer=_init, initargs=("siftdesc", None)) as p:
            descs = p.map(_sift_desc_one, X[idx], chunksize=64)
        D = np.concatenate([d for d in descs if len(d)])
        km = MiniBatchKMeans(n_clusters=64, random_state=42, batch_size=4096,
                             n_init=3).fit(D.astype(np.float64))
        joblib.dump(km, codebook_path)
        print(f"codebook: {D.shape[0]} descriptors -> k=64 "
              f"({time.time()-t0:.0f}s)")
        return

    if args.merge:
        chunks = sorted(glob.glob(os.path.join(args.out_dir, f"{tag}_chunk_*.npz")),
                        key=lambda p: int(p.rsplit("_", 1)[1].split(".")[0]))
        Xs, ys = [], []
        expected = 0
        for c in chunks:
            d = np.load(c)
            assert d["start"] == expected, f"missing chunk before {c}"
            Xs.append(d["X"]); ys.append(d["y"])
            expected = int(d["end"])
        X = np.concatenate(Xs); y = np.concatenate(ys)
        np.savez_compressed(os.path.join(args.out_dir, f"{tag}.npz"), X=X, y=y)
        print(f"merged -> {tag}.npz {X.shape}")
        return

    X = np.load(os.path.join(args.data_dir, f"X_{args.dataset}.npz"))["arr_0"]
    y = np.load(os.path.join(args.data_dir, f"y_{args.dataset}.npz"))["arr_0"]
    end = min(args.end if args.end is not None else len(X), len(X))
    imgs, labels = X[args.start:end], y[args.start:end]

    t0 = time.time()
    if args.workers == 1:
        # cv2.SIFT inside forked workers can deadlock on some builds;
        # workers=1 runs inline.
        _init(args.descriptor, codebook_path)
        feats = [_extract_one(im) for im in imgs]
    else:
        with Pool(args.workers, initializer=_init,
                  initargs=(args.descriptor, codebook_path)) as pool:
            feats = pool.map(_extract_one, imgs, chunksize=64)
    feats = np.asarray(feats)

    out = os.path.join(args.out_dir, f"{tag}_chunk_{args.start}.npz")
    np.savez_compressed(out, X=feats, y=labels, start=args.start, end=end)
    print(f"{tag}[{args.start}:{end}] {feats.shape} {time.time()-t0:.0f}s")


def _sift_desc_one(img):
    return baselines.sift_descriptors(preprocess_mnist(img))


if __name__ == "__main__":
    main()
