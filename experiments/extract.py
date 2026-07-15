"""Chunked, resumable KTSD feature extraction.

Extracts KTSD features for a slice of a dataset and stores the chunk as an
``.npz``; chunks are later merged with ``--merge``. This makes long
extractions restartable and lets them run in bounded time slices.

Usage:
    python experiments/extract.py --data-dir DATA --out-dir OUT \
        --dataset train --start 0 --end 20000 [--augment rot-10]
    python experiments/extract.py --out-dir OUT --dataset train --merge

Datasets: train | test | dig  (official Kannada-MNIST tensors)
Augment:  none (default) | rot-10  (the x2 training augmentation used in the
          paper: each image additionally contributes a -10 degree rotation;
          pass --augment rot-10 to extract features of the rotated copies)
"""

import argparse
import glob
import os
import sys
import time
from multiprocessing import Pool

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from ktsd import extract_ktsd_features, preprocess_mnist  # noqa: E402


def rotate_minus10(img):
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), -10, 1.0)
    return cv2.warpAffine(img, M, (w, h), borderValue=0)


def _extract_one(img):
    feats = extract_ktsd_features(preprocess_mnist(img))
    return np.nan_to_num(feats, nan=0.0, posinf=0.0, neginf=0.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data_local")
    ap.add_argument("--out-dir", default="features")
    ap.add_argument("--dataset", required=True, choices=["train", "test", "dig"])
    ap.add_argument("--augment", default="none", choices=["none", "rot-10"])
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--end", type=int, default=None)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--merge", action="store_true")
    args = ap.parse_args()

    tag = args.dataset + ("_rot10" if args.augment == "rot-10" else "")
    os.makedirs(args.out_dir, exist_ok=True)

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
        out = os.path.join(args.out_dir, f"{tag}.npz")
        np.savez_compressed(out, X=X, y=y)
        print(f"merged {len(chunks)} chunks -> {out}  X={X.shape}")
        return

    X = np.load(os.path.join(args.data_dir, f"X_{args.dataset}.npz"))["arr_0"]
    y = np.load(os.path.join(args.data_dir, f"y_{args.dataset}.npz"))["arr_0"]
    end = min(args.end if args.end is not None else len(X), len(X))
    imgs = X[args.start:end]
    labels = y[args.start:end]

    if args.augment == "rot-10":
        imgs = np.stack([rotate_minus10(im) for im in imgs])

    t0 = time.time()
    with Pool(args.workers) as pool:
        feats = pool.map(_extract_one, imgs, chunksize=64)
    feats = np.asarray(feats)
    dt = time.time() - t0

    out = os.path.join(args.out_dir, f"{tag}_chunk_{args.start}.npz")
    np.savez_compressed(out, X=feats, y=labels, start=args.start, end=end)
    print(f"{tag}[{args.start}:{end}] -> {out}  {feats.shape}  "
          f"{dt:.1f}s  ({dt/len(imgs)*1000:.2f} ms/img wall)")


if __name__ == "__main__":
    main()
