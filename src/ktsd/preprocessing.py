"""Preprocessing for KTSD feature extraction.

Two entry points are provided:

* :func:`preprocess_image` — for camera/scanner images with unknown lighting
  (adaptive thresholding, largest-component cropping). Used by the demo app.
* :func:`preprocess_mnist` — for 28x28 Kannada-MNIST tensors, which are already
  clean, centered, dark-background images (global threshold after upsampling).

All images are upsampled to ``IMAGE_SIZE`` x ``IMAGE_SIZE`` (100x100) **before**
binarization and skeletonization. At the native 28x28 resolution, strokes are
only 1-3 px wide and skeletonization produces unstable topology (spurious
endpoints/junctions, broken loops); operating at 100x100 makes the skeleton and
contour hierarchy substantially more stable. See the paper (Sec. IV-A) for the
supporting experiment.
"""

import cv2
import numpy as np

#: Working resolution for all structural analysis.
IMAGE_SIZE = 100

#: Global threshold applied to upsampled Kannada-MNIST images.
MNIST_BINARY_THRESHOLD = 30

#: Adaptive threshold parameters for scanned/camera images.
ADAPTIVE_BLOCK_SIZE = 21
ADAPTIVE_C = 10


def preprocess_mnist(img):
    """Binarize a 28x28 Kannada-MNIST image at working resolution.

    Parameters
    ----------
    img : (28, 28) uint8 array, dark background, bright stroke.

    Returns
    -------
    (IMAGE_SIZE, IMAGE_SIZE) uint8 binary image with values {0, 255}.
    """
    resized = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_CUBIC)
    _, binary = cv2.threshold(resized, MNIST_BINARY_THRESHOLD, 255, cv2.THRESH_BINARY)
    return binary


def preprocess_image(img):
    """Full preprocessing for real-world images (demo app / custom scans).

    Steps: grayscale -> Gaussian adaptive threshold (inverted) -> 3x3
    morphological open+close -> crop to the largest external contour with
    10 px padding -> pad to square -> resize to ``IMAGE_SIZE``.

    Returns ``None`` when no foreground component is found.
    """
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img.copy()

    binary = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, ADAPTIVE_BLOCK_SIZE, ADAPTIVE_C,
    )

    kernel = np.ones((3, 3), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=1)

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    main_contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(main_contour)

    padding = 10
    x = max(0, x - padding)
    y = max(0, y - padding)
    w = min(binary.shape[1] - x, w + 2 * padding)
    h = min(binary.shape[0] - y, h + 2 * padding)

    cropped = binary[y:y + h, x:x + w]
    max_dim = max(w, h)
    square = np.zeros((max_dim, max_dim), dtype=np.uint8)
    x_offset = (max_dim - w) // 2
    y_offset = (max_dim - h) // 2
    square[y_offset:y_offset + h, x_offset:x_offset + w] = cropped

    return cv2.resize(square, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_AREA)
