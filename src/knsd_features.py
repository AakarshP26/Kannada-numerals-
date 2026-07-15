"""
Kannada Numeral Shape Descriptor (KNSD)
Novel Loop-Based Feature Extractor for Kannada Handwritten Numeral Recognition

This module implements the KNSD feature extraction pipeline:
1. Loop Detection - Count, area, and position of closed loops
2. Endpoint Detection - Stroke termination points
3. Junction Detection - Stroke intersection points
4. Curvature Histogram - Distribution of curve directions
5. Stroke Statistics - Circularity, solidity, extent


"""

import cv2
import numpy as np
from scipy import ndimage
from skimage.morphology import skeletonize

# Configuration
IMAGE_SIZE = 100


def preprocess_image(img):
    """
    Preprocess image for KNSD feature extraction.
    
    Args:
        img: Input image (BGR or grayscale)
        
    Returns:
        Binary image normalized to IMAGE_SIZE x IMAGE_SIZE
    """
    # Convert to grayscale if needed
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img.copy()
    
    # Adaptive thresholding for varying lighting
    binary = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY_INV, 21, 10
    )
    
    # Morphological noise removal
    kernel = np.ones((3, 3), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=1)
    
    # Find largest contour (the character)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return None
    
    main_contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(main_contour)
    
    # Add padding
    padding = 10
    x = max(0, x - padding)
    y = max(0, y - padding)
    w = min(binary.shape[1] - x, w + 2 * padding)
    h = min(binary.shape[0] - y, h + 2 * padding)
    
    # Crop and make square
    cropped = binary[y:y+h, x:x+w]
    max_dim = max(w, h)
    square = np.zeros((max_dim, max_dim), dtype=np.uint8)
    x_offset = (max_dim - w) // 2
    y_offset = (max_dim - h) // 2
    square[y_offset:y_offset+h, x_offset:x_offset+w] = cropped
    
    # Resize to standard size
    normalized = cv2.resize(square, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_AREA)
    
    return normalized


def extract_loop_features(binary_img):
    """
    Extract loop-based features specific to Kannada numerals.
    
    Kannada numerals have characteristic loops:
    - ೦ (0): One large circular loop
    - ೬ (6): One loop at bottom
    - ೮ (8): Two connected loops
    - ೯ (9): One loop at top
    
    Returns:
        8 features: [num_loops, total_area, avg_area, ratio, 
                     loop_cx, loop_cy, avg_cy, variance]
    """
    # Find contours with hierarchy
    contours, hierarchy = cv2.findContours(
        binary_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
    )
    
    if hierarchy is None:
        return [0, 0, 0, 0, 0.5, 0.5, 0, 0]
    
    hierarchy = hierarchy[0]
    
    # Find loops (contours with parent = holes inside the character)
    loops = []
    for i, (cnt, h) in enumerate(zip(contours, hierarchy)):
        if h[3] != -1:  # Has parent (is a hole)
            area = cv2.contourArea(cnt)
            if area > 50:  # Filter noise
                M = cv2.moments(cnt)
                if M["m00"] > 0:
                    cx = M["m10"] / M["m00"]
                    cy = M["m01"] / M["m00"]
                    loops.append({
                        'area': area,
                        'centroid': (cx / IMAGE_SIZE, cy / IMAGE_SIZE),
                    })
    
    num_loops = len(loops)
    total_area = binary_img.sum() / 255
    
    if num_loops == 0:
        return [0, 0, 0, 0, 0.5, 0.5, 0, 0]
    
    total_loop_area = sum(l['area'] for l in loops)
    avg_loop_area = total_loop_area / num_loops
    loop_area_ratio = total_loop_area / total_area if total_area > 0 else 0
    
    largest_loop = max(loops, key=lambda x: x['area'])
    loop_cx, loop_cy = largest_loop['centroid']
    avg_cy = sum(l['centroid'][1] for l in loops) / num_loops
    
    if num_loops > 1:
        areas = [l['area'] for l in loops]
        loop_variance = np.std(areas) / (np.mean(areas) + 1e-6)
    else:
        loop_variance = 0
    
    return [
        num_loops, total_loop_area, avg_loop_area, loop_area_ratio,
        loop_cx, loop_cy, avg_cy, loop_variance
    ]


def extract_endpoint_features(binary_img):
    """
    NOVEL: Extract stroke endpoint features.
    
    Returns:
        5 features: [count, avg_x, avg_y, x_spread, y_spread]
    """
    skeleton = skeletonize(binary_img > 0).astype(np.uint8)
    
    endpoint_kernels = [
        np.array([[0, 0, 0], [0, 1, 0], [0, 1, 0]]),
        np.array([[0, 0, 0], [0, 1, 0], [0, 0, 1]]),
        np.array([[0, 0, 0], [0, 1, 1], [0, 0, 0]]),
        np.array([[0, 0, 1], [0, 1, 0], [0, 0, 0]]),
        np.array([[0, 1, 0], [0, 1, 0], [0, 0, 0]]),
        np.array([[1, 0, 0], [0, 1, 0], [0, 0, 0]]),
        np.array([[0, 0, 0], [1, 1, 0], [0, 0, 0]]),
        np.array([[0, 0, 0], [0, 1, 0], [1, 0, 0]]),
    ]
    
    endpoint_positions = []
    
    for kernel in endpoint_kernels:
        result = cv2.filter2D(skeleton, -1, kernel)
        matches = np.where((result >= 2) & (skeleton > 0))
        for y, x in zip(matches[0], matches[1]):
            neighbors = skeleton[max(0,y-1):y+2, max(0,x-1):x+2].sum() - skeleton[y,x]
            if neighbors == 1:
                endpoint_positions.append((x / IMAGE_SIZE, y / IMAGE_SIZE))
    
    # Remove duplicates
    unique_endpoints = []
    for ep in endpoint_positions:
        is_dup = any(np.sqrt((ep[0]-u[0])**2 + (ep[1]-u[1])**2) < 0.05 for u in unique_endpoints)
        if not is_dup:
            unique_endpoints.append(ep)
    
    count = len(unique_endpoints)
    
    if count == 0:
        return [0, 0.5, 0.5, 0, 0]
    
    avg_x = sum(p[0] for p in unique_endpoints) / count
    avg_y = sum(p[1] for p in unique_endpoints) / count
    
    if count > 1:
        x_spread = max(p[0] for p in unique_endpoints) - min(p[0] for p in unique_endpoints)
        y_spread = max(p[1] for p in unique_endpoints) - min(p[1] for p in unique_endpoints)
    else:
        x_spread, y_spread = 0, 0
    
    return [count, avg_x, avg_y, x_spread, y_spread]


def extract_junction_features(binary_img):
    """
    NOVEL: Extract stroke junction/intersection features.
    
    Returns:
        3 features: [count, avg_x, avg_y]
    """
    skeleton = skeletonize(binary_img > 0).astype(np.uint8)
    
    junction_positions = []
    h, w = skeleton.shape
    
    for y in range(1, h-1):
        for x in range(1, w-1):
            if skeleton[y, x]:
                neighbors = skeleton[y-1:y+2, x-1:x+2].sum() - 1
                if neighbors >= 3:
                    junction_positions.append((x / IMAGE_SIZE, y / IMAGE_SIZE))
    
    count = len(junction_positions)
    
    if count == 0:
        return [0, 0.5, 0.5]
    
    avg_x = sum(p[0] for p in junction_positions) / count
    avg_y = sum(p[1] for p in junction_positions) / count
    
    return [count, avg_x, avg_y]


def extract_curvature_histogram(binary_img, n_bins=8):
    """
    Extract curvature histogram of strokes.
    
    Returns:
        n_bins features (normalized histogram)
    """
    skeleton = skeletonize(binary_img > 0).astype(np.uint8) * 255
    contours, _ = cv2.findContours(skeleton, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    
    if not contours:
        return np.zeros(n_bins).tolist()
    
    all_angles = []
    
    for cnt in contours:
        if len(cnt) < 5:
            continue
        cnt = cnt.squeeze()
        if len(cnt.shape) == 1:
            continue
        for i in range(1, len(cnt) - 1):
            dx = cnt[i+1, 0] - cnt[i-1, 0]
            dy = cnt[i+1, 1] - cnt[i-1, 1]
            all_angles.append(np.arctan2(dy, dx))
    
    if not all_angles:
        return np.zeros(n_bins).tolist()
    
    hist, _ = np.histogram(all_angles, bins=n_bins, range=(-np.pi, np.pi))
    hist = hist.astype(float)
    hist /= (hist.sum() + 1e-6)
    
    return hist.tolist()


def extract_stroke_statistics(binary_img):
    """
    Extract general stroke statistics.
    
    Returns:
        5 features: [aspect_ratio, solidity, extent, perimeter, circularity]
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


def extract_knsd_features(binary_img):
    """
    Extract complete KNSD feature vector.
    
    Total: 29 features
    - Loop features: 8
    - Endpoint features: 5
    - Junction features: 3
    - Curvature histogram: 8
    - Stroke statistics: 5
    """
    loop_feats = extract_loop_features(binary_img)
    endpoint_feats = extract_endpoint_features(binary_img)
    junction_feats = extract_junction_features(binary_img)
    curvature_feats = extract_curvature_histogram(binary_img, n_bins=8)
    stroke_feats = extract_stroke_statistics(binary_img)
    
    features = loop_feats + endpoint_feats + junction_feats + curvature_feats + stroke_feats
    
    return np.array(features)


# Feature names for reference
FEATURE_NAMES = [
    'num_loops', 'total_loop_area', 'avg_loop_area', 'loop_area_ratio',
    'loop_cx', 'loop_cy', 'avg_loop_cy', 'loop_variance',
    'num_endpoints', 'endpoint_avg_x', 'endpoint_avg_y', 'endpoint_x_spread', 'endpoint_y_spread',
    'num_junctions', 'junction_avg_x', 'junction_avg_y',
    'curv_0', 'curv_1', 'curv_2', 'curv_3', 'curv_4', 'curv_5', 'curv_6', 'curv_7',
    'aspect_ratio', 'solidity', 'extent', 'perimeter', 'circularity'
]
