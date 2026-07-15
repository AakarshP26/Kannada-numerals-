"""Topology utilities: comparing hole-based and skeleton-based loop counts.

KTSD detects loops as enclosed background regions (contour-hierarchy holes).
An alternative is to count independent cycles of the skeleton graph. The two
notions coincide for ideal shapes but can diverge on real handwriting:

* a hairline break in a drawn loop destroys the skeleton cycle *and* the
  enclosed hole (both methods agree);
* a loop pinched to zero interior area (filled blob) keeps a skeleton cycle
  only if thinning happens to preserve it — the hole disappears first;
* skeleton spurs and thinning artifacts can create or destroy cycles without
  any perceptual change to the numeral, whereas holes are unaffected.

:func:`skeleton_cycle_count` computes the cycle rank (first Betti number) of
the skeleton graph, b1 = E - V + C, where V is the number of skeleton pixels,
E the number of 8-adjacent pixel pairs, and C the number of connected
components. ``experiments/loop_justification.py`` uses it to quantify how
often the two loop notions agree on Kannada-MNIST, supporting the design
choice discussed in the paper.
"""

import numpy as np
from scipy import ndimage


def skeleton_cycle_count(skeleton):
    """Cycle rank b1 = E - V + C of the 8-connected skeleton graph.

    Parameters
    ----------
    skeleton : 2D {0,1} array.

    Returns
    -------
    int — number of independent cycles in the skeleton.
    """
    sk = skeleton.astype(bool)
    V = int(sk.sum())
    if V == 0:
        return 0

    # Count 8-adjacent pixel pairs without double counting: for each pixel,
    # look only at 4 "forward" neighbors (E, SE, S, SW).
    E = 0
    padded = np.pad(sk, 1)
    core = padded[1:-1, 1:-1]
    for dy, dx in ((0, 1), (1, 1), (1, 0), (1, -1)):
        shifted = padded[1 + dy: padded.shape[0] - 1 + dy,
                         1 + dx: padded.shape[1] - 1 + dx]
        E += int(np.logical_and(core, shifted).sum())

    _, C = ndimage.label(sk, structure=np.ones((3, 3)))
    return E - V + C


def hole_count(binary_img, min_area=50):
    """Number of enclosed holes (same rule as KTSD loop detection)."""
    import cv2

    contours, hierarchy = cv2.findContours(
        binary_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
    )
    if hierarchy is None:
        return 0
    hierarchy = hierarchy[0]
    n = 0
    for cnt, h in zip(contours, hierarchy):
        if h[3] != -1 and cv2.contourArea(cnt) > min_area:
            n += 1
    return n
