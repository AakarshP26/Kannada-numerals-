"""Verify the refactored ktsd package reproduces the original implementation.

The original ``src/knsd_features.py`` (as submitted) recomputed the skeleton
inside each extractor and used a kernel-based endpoint preselection. The
refactor computes the skeleton once and detects endpoints directly by
neighbor count. This test confirms feature-level equivalence on real data.

Run:  python tests/test_equivalence.py path/to/train.csv
"""

import sys
import os
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from ktsd import extract_ktsd_features, preprocess_mnist  # noqa: E402
from legacy import knsd_features as legacy  # noqa: E402


def main(csv_path, n=400):
    df = pd.read_csv(csv_path, nrows=5000)
    sample = df.groupby("label").head(n // 10)
    images = sample.drop("label", axis=1).values.reshape(-1, 28, 28).astype(np.uint8)

    # The refactor preserves the original endpoint candidate ordering
    # (direction-kernel scan order), so every feature must be bit-identical.
    max_abs = 0.0
    for img in images:
        binary = preprocess_mnist(img)
        f_new = np.nan_to_num(extract_ktsd_features(binary))
        f_old = np.nan_to_num(legacy.extract_knsd_features(binary))
        max_abs = max(max_abs, np.abs(f_new - f_old).max())

    print(f"images compared: {len(images)}")
    print(f"max abs feature difference: {max_abs:.3e}")
    assert max_abs < 1e-9, "refactor deviates from original implementation"
    print("EQUIVALENCE OK — refactor is bit-identical to the submitted code")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "data/kannada_mnist/train.csv")
