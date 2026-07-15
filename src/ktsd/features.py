"""KTSD feature extraction.

The 29-dimensional KTSD descriptor combines five groups of interpretable
structural measurements. Feature provenance is deliberately two-branch:

* **Region-based features** (loops, stroke statistics) are computed on the
  cleaned *binary image*, where enclosed holes and region geometry are
  well-defined and robust to stroke width.
* **Skeleton-based features** (endpoints, junctions, direction histogram)
  are computed on the *morphological skeleton*, where stroke topology
  (terminations, crossings, local orientation) is well-defined.

The skeleton is computed exactly once per image and shared by all
skeleton-based extractors.

Feature vector layout (29 dims):

====  ========================  =========================================
idx   name                      definition
====  ========================  =========================================
0     num_loops                 number of enclosed holes (area > 50 px)
1     total_loop_area           sum of hole areas (px)
2     avg_loop_area             mean hole area (px)
3     loop_area_ratio           total hole area / foreground area
4     loop_cx                   x-centroid of largest hole / IMAGE_SIZE
5     loop_cy                   y-centroid of largest hole / IMAGE_SIZE
6     avg_loop_cy               mean y-centroid over holes / IMAGE_SIZE
7     loop_area_cv              coefficient of variation of hole areas,
                                sigma_A / mu_A (dimensionless)
8     num_endpoints             skeleton pixels with exactly 1 neighbor
9-10  endpoint_avg_x/_y         mean endpoint position (normalized)
11-12 endpoint_x/_y_spread      endpoint bounding extent (normalized)
13    num_junctions             skeleton pixels with >= 3 neighbors
14-15 junction_avg_x/_y         mean junction position (normalized)
16-23 curv_0 .. curv_7          8-bin normalized histogram of local stroke
                                tangent orientation along the skeleton
24    aspect_ratio              bbox width / height
25    solidity                  contour area / convex hull area
26    extent                    contour area / bounding box area
27    perimeter                 contour perimeter / IMAGE_SIZE
28    circularity               4*pi*area / perimeter^2 (isoperimetric)
====  ========================  =========================================
"""

import cv2
import numpy as np
from skimage.morphology import skeletonize

from .preprocessing import IMAGE_SIZE

#: Minimum hole area (px at 100x100) counted as a loop; suppresses
#: binarization pinholes.
MIN_LOOP_AREA = 50

#: Minimum normalized distance between two distinct endpoints.
ENDPOINT_MERGE_RADIUS = 0.05

#: Number of orientation histogram bins.
N_CURVATURE_BINS = 8


# ---------------------------------------------------------------------------
# Region-based features (computed on the binary image)
# ---------------------------------------------------------------------------

def extract_loop_features(binary_img):
    """Loop (enclosed hole) features from the contour hierarchy.

    Loops are detected as internal contours (contours with a parent in the
    ``RETR_TREE`` hierarchy), i.e. background regions fully enclosed by the
    stroke. This is the region-topology notion of a "hole" (2D Betti number
    b1 of the foreground region) and is intentionally computed on the binary
    image rather than the skeleton: hole detection is robust to stroke width
    and thinning artifacts, whereas skeleton cycles can be broken by a single
    missing pixel. See ``ktsd.topology`` for the empirical comparison.

    Returns 8 features:
    ``[num_loops, total_loop_area, avg_loop_area, loop_area_ratio,
    loop_cx, loop_cy, avg_loop_cy, loop_area_cv]``

    ``loop_area_cv`` is the coefficient of variation of hole areas,
    CV = sigma_A / mu_A (population std over mean), a dimensionless measure
    of how unequal in size the holes are; 0 when fewer than two holes.
    """
    contours, hierarchy = cv2.findContours(
        binary_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
    )

    if hierarchy is None:
        return [0, 0, 0, 0, 0.5, 0.5, 0, 0]

    hierarchy = hierarchy[0]

    loops = []
    for cnt, h in zip(contours, hierarchy):
        if h[3] != -1:  # has a parent contour => enclosed hole
            area = cv2.contourArea(cnt)
            if area > MIN_LOOP_AREA:
                M = cv2.moments(cnt)
                if M["m00"] > 0:
                    cx = M["m10"] / M["m00"]
                    cy = M["m01"] / M["m00"]
                    loops.append(
                        {"area": area, "centroid": (cx / IMAGE_SIZE, cy / IMAGE_SIZE)}
                    )

    num_loops = len(loops)
    total_area = binary_img.sum() / 255

    if num_loops == 0:
        return [0, 0, 0, 0, 0.5, 0.5, 0, 0]

    total_loop_area = sum(l["area"] for l in loops)
    avg_loop_area = total_loop_area / num_loops
    loop_area_ratio = total_loop_area / total_area if total_area > 0 else 0

    largest = max(loops, key=lambda l: l["area"])
    loop_cx, loop_cy = largest["centroid"]
    avg_cy = sum(l["centroid"][1] for l in loops) / num_loops

    if num_loops > 1:
        areas = np.array([l["area"] for l in loops], dtype=float)
        loop_area_cv = float(np.std(areas) / (np.mean(areas) + 1e-6))
    else:
        loop_area_cv = 0.0

    return [
        num_loops, total_loop_area, avg_loop_area, loop_area_ratio,
        loop_cx, loop_cy, avg_cy, loop_area_cv,
    ]


def extract_stroke_statistics(binary_img):
    """Global shape statistics of the largest foreground contour.

    Returns 5 features ``[aspect_ratio, solidity, extent, perimeter,
    circularity]`` with the standard definitions:

    * solidity  = A_contour / A_convex_hull
    * extent    = A_contour / A_bounding_box
    * circularity = 4 * pi * A / P^2 (isoperimetric quotient, 1 for a disk)
    """
    contours, _ = cv2.findContours(binary_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return [0, 0, 0, 0, 0]

    main_cnt = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(main_cnt)

    aspect_ratio = w / (h + 1e-6)

    hull = cv2.convexHull(main_cnt)
    hull_area = cv2.contourArea(hull)
    cnt_area = cv2.contourArea(main_cnt)
    solidity = cnt_area / (hull_area + 1e-6)
    extent = cnt_area / (w * h + 1e-6)

    perimeter = cv2.arcLength(main_cnt, True)
    circularity = 4 * np.pi * cnt_area / (perimeter ** 2 + 1e-6)

    return [aspect_ratio, solidity, extent, perimeter / IMAGE_SIZE, circularity]


# ---------------------------------------------------------------------------
# Skeleton-based features
# ---------------------------------------------------------------------------

def _neighbor_counts(skeleton):
    """8-neighborhood neighbor count for every skeleton pixel."""
    kernel = np.ones((3, 3), np.uint8)
    counts = cv2.filter2D(skeleton, cv2.CV_16S, kernel, borderType=cv2.BORDER_CONSTANT)
    return counts - skeleton  # exclude the center pixel itself


#: Neighbor directions in the candidate-ordering used by endpoint merging
#: (S, SE, E, NE, N, NW, W, SW — the original kernel scan order, kept for
#: bit-exact reproducibility of the submitted implementation).
_ENDPOINT_DIRS = ((1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0), (-1, -1), (0, -1), (1, -1))


def extract_endpoint_features(skeleton):
    """Stroke termination features.

    An endpoint is a skeleton pixel with exactly one 8-connected neighbor.
    Endpoints closer than ``ENDPOINT_MERGE_RADIUS`` (normalized) are merged
    to suppress duplicate detections from staircase pixels; merging keeps the
    first candidate in a fixed direction-then-raster order.

    Returns 5 features: ``[count, avg_x, avg_y, x_spread, y_spread]``.
    """
    counts = _neighbor_counts(skeleton)
    endpoint_mask = (skeleton > 0) & (counts == 1)

    padded = np.pad(skeleton, 1)
    candidates = []
    for dy, dx in _ENDPOINT_DIRS:
        neighbor = padded[1 + dy: padded.shape[0] - 1 + dy,
                          1 + dx: padded.shape[1] - 1 + dx]
        ys, xs = np.where(endpoint_mask & (neighbor > 0))
        candidates.extend(
            (x / IMAGE_SIZE, y / IMAGE_SIZE) for y, x in zip(ys, xs)
        )

    unique = []
    for x, y in candidates:
        if not any(np.sqrt((x - u[0]) ** 2 + (y - u[1]) ** 2) < ENDPOINT_MERGE_RADIUS
                   for u in unique):
            unique.append((x, y))

    count = len(unique)
    if count == 0:
        return [0, 0.5, 0.5, 0, 0]

    axs = [p[0] for p in unique]
    ays = [p[1] for p in unique]
    x_spread = max(axs) - min(axs) if count > 1 else 0
    y_spread = max(ays) - min(ays) if count > 1 else 0

    return [count, float(np.mean(axs)), float(np.mean(ays)), x_spread, y_spread]


def extract_junction_features(skeleton):
    """Stroke intersection features.

    A junction is a skeleton pixel with three or more 8-connected neighbors
    (a branch point of the stroke graph).

    Returns 3 features: ``[count, avg_x, avg_y]``.
    """
    counts = _neighbor_counts(skeleton)
    ys, xs = np.where((skeleton > 0) & (counts >= 3))

    count = len(xs)
    if count == 0:
        return [0, 0.5, 0.5]

    return [count, float(xs.mean() / IMAGE_SIZE), float(ys.mean() / IMAGE_SIZE)]


def extract_curvature_histogram(skeleton, n_bins=N_CURVATURE_BINS):
    """Normalized histogram of local stroke tangent orientation.

    The skeleton is traced as contours; at each interior contour point the
    local tangent angle theta_i = atan2(y_{i+1} - y_{i-1}, x_{i+1} - x_{i-1})
    is computed by central differences, and angles are histogrammed into
    ``n_bins`` equal bins over [-pi, pi], then normalized to sum to 1.
    The histogram captures the distribution of stroke directions (and hence
    how curved / how anisotropic the numeral's strokes are).

    Returns ``n_bins`` features.
    """
    skel_u8 = (skeleton * 255).astype(np.uint8)
    contours, _ = cv2.findContours(skel_u8, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return np.zeros(n_bins).tolist()

    all_angles = []
    for cnt in contours:
        if len(cnt) < 5:
            continue
        cnt = cnt.squeeze()
        if len(cnt.shape) == 1:
            continue
        dx = cnt[2:, 0].astype(float) - cnt[:-2, 0]
        dy = cnt[2:, 1].astype(float) - cnt[:-2, 1]
        all_angles.append(np.arctan2(dy, dx))

    if not all_angles:
        return np.zeros(n_bins).tolist()

    angles = np.concatenate(all_angles)
    hist, _ = np.histogram(angles, bins=n_bins, range=(-np.pi, np.pi))
    hist = hist.astype(float)
    hist /= (hist.sum() + 1e-6)
    return hist.tolist()


# ---------------------------------------------------------------------------
# Full descriptor
# ---------------------------------------------------------------------------

def extract_ktsd_features(binary_img):
    """Extract the complete 29-dimensional KTSD descriptor.

    Parameters
    ----------
    binary_img : (IMAGE_SIZE, IMAGE_SIZE) uint8 binary image ({0, 255}).

    Returns
    -------
    (29,) float array. See module docstring for the layout.
    """
    skeleton = skeletonize(binary_img > 0).astype(np.uint8)

    loop_feats = extract_loop_features(binary_img)          # binary image
    stroke_feats = extract_stroke_statistics(binary_img)    # binary image
    endpoint_feats = extract_endpoint_features(skeleton)    # skeleton
    junction_feats = extract_junction_features(skeleton)    # skeleton
    curvature_feats = extract_curvature_histogram(skeleton) # skeleton

    features = (loop_feats + endpoint_feats + junction_feats
                + curvature_feats + stroke_feats)
    return np.array(features)


FEATURE_NAMES = [
    "num_loops", "total_loop_area", "avg_loop_area", "loop_area_ratio",
    "loop_cx", "loop_cy", "avg_loop_cy", "loop_area_cv",
    "num_endpoints", "endpoint_avg_x", "endpoint_avg_y",
    "endpoint_x_spread", "endpoint_y_spread",
    "num_junctions", "junction_avg_x", "junction_avg_y",
    "curv_0", "curv_1", "curv_2", "curv_3", "curv_4", "curv_5", "curv_6", "curv_7",
    "aspect_ratio", "solidity", "extent", "perimeter", "circularity",
]

FEATURE_GROUPS = {
    "loop": FEATURE_NAMES[0:8],
    "endpoint": FEATURE_NAMES[8:13],
    "junction": FEATURE_NAMES[13:16],
    "direction_histogram": FEATURE_NAMES[16:24],
    "stroke_statistics": FEATURE_NAMES[24:29],
}
