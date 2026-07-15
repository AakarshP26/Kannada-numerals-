"""Classical descriptor baselines: HOG, Zernike moments, SIFT bag-of-visual-words.

All baselines consume the same preprocessed binary images as KTSD
(``preprocess_mnist``, 100x100), so comparisons isolate the descriptor.

* HOG: 9 orientations, 10x10-px cells, 2x2-cell blocks, L2-Hys — 2916 dims.
* Zernike: magnitudes of Zernike moments up to radial order 10 over the
  smallest disk enclosing the image — 36 dims (implemented in numpy; no
  external dependency).
* SIFT-BoVW: dense-keypoint SIFT descriptors quantized against a k=64
  MiniBatchKMeans codebook learned on training images only — 64 dims.
"""

import math

import numpy as np
import cv2
from skimage.feature import hog


# ---------------------------------------------------------------------------
# HOG
# ---------------------------------------------------------------------------

def extract_hog(binary_img):
    return hog(
        binary_img,
        orientations=9,
        pixels_per_cell=(10, 10),
        cells_per_block=(2, 2),
        block_norm="L2-Hys",
        feature_vector=True,
    )


# ---------------------------------------------------------------------------
# Zernike moments (numpy implementation)
# ---------------------------------------------------------------------------

def _zernike_R(n, m, rho):
    """Radial Zernike polynomial R_n^m evaluated on array rho."""
    R = np.zeros_like(rho)
    for s in range((n - m) // 2 + 1):
        num = ((-1) ** s) * math.factorial(n - s)
        den = (math.factorial(s)
               * math.factorial((n + m) // 2 - s)
               * math.factorial((n - m) // 2 - s))
        R += (num / den) * rho ** (n - 2 * s)
    return R


def zernike_indices(max_order=10):
    return [(n, m) for n in range(max_order + 1)
            for m in range(n % 2, n + 1, 2)]


def extract_zernike(binary_img, max_order=10):
    """Magnitudes |Z_nm| of Zernike moments up to ``max_order``.

    The image is mapped onto the unit disk centered at the image center;
    pixels outside the disk are ignored (standard formulation).
    """
    h, w = binary_img.shape
    yy, xx = np.mgrid[:h, :w].astype(float)
    cx, cy = (w - 1) / 2.0, (h - 1) / 2.0
    r = min(cx, cy)
    x = (xx - cx) / r
    y = (yy - cy) / r
    rho = np.sqrt(x ** 2 + y ** 2)
    theta = np.arctan2(y, x)
    mask = (rho <= 1.0) & (binary_img > 0)

    rho_m = rho[mask]
    theta_m = theta[mask]
    npix = mask.sum()
    feats = []
    for n, m in zernike_indices(max_order):
        if npix == 0:
            feats.append(0.0)
            continue
        R = _zernike_R(n, m, rho_m)
        V = R * np.exp(-1j * m * theta_m)
        Z = (n + 1) / np.pi * V.sum() / max(npix, 1)
        feats.append(np.abs(Z))
    return np.asarray(feats)


# ---------------------------------------------------------------------------
# SIFT bag of visual words
# ---------------------------------------------------------------------------

_SIFT = None


def _sift():
    global _SIFT
    if _SIFT is None:
        _SIFT = cv2.SIFT_create()
    return _SIFT


def _dense_keypoints(size=100, step=12, kp_size=16):
    pts = []
    for y in range(step // 2, size, step):
        for x in range(step // 2, size, step):
            pts.append(cv2.KeyPoint(float(x), float(y), kp_size))
    return pts


_DENSE_KPS = None


def sift_descriptors(binary_img):
    """Dense SIFT descriptors on the binary image (128-d each)."""
    global _DENSE_KPS
    if _DENSE_KPS is None:
        _DENSE_KPS = _dense_keypoints(binary_img.shape[0])
    _, desc = _sift().compute(binary_img, _DENSE_KPS)
    if desc is None:
        desc = np.zeros((0, 128), dtype=np.float32)
    return desc


def bovw_encode(descriptors, kmeans):
    """Normalized histogram of codebook assignments.

    Assignment is computed directly against ``kmeans.cluster_centers_``
    (argmin of squared euclidean distance), which is equivalent to
    ``kmeans.predict`` but avoids per-call validation overhead.
    """
    centers = kmeans.cluster_centers_
    k = len(centers)
    if len(descriptors) == 0:
        return np.zeros(k)
    D = descriptors.astype(np.float64)
    d2 = (D * D).sum(1)[:, None] - 2 * D @ centers.T + (centers * centers).sum(1)[None]
    words = np.argmin(d2, axis=1)
    hist = np.bincount(words, minlength=k).astype(float)
    return hist / hist.sum()
