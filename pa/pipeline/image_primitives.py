"""
Layer 0 — Image Primitives.

Stateless functions that operate on numpy arrays (BGR or grayscale).
No analysis logic lives here — only reusable image processing operations.

All functions accept and return numpy arrays.
No function here imports from any other pa.pipeline module.
"""

from __future__ import annotations

import functools
import math
import cv2
import numpy as np
from typing import List, Optional, Tuple


# ---------------------------------------------------------------------------
# Conversion
# ---------------------------------------------------------------------------

def to_grayscale(img: np.ndarray) -> np.ndarray:
    """Convert a BGR image to grayscale. No-op if already single-channel."""
    if img.ndim == 2:
        return img
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


# ---------------------------------------------------------------------------
# Blurring
# ---------------------------------------------------------------------------

def blur_gaussian(img: np.ndarray, kernel_size: int = 5) -> np.ndarray:
    """Gaussian blur. kernel_size must be odd."""
    k = kernel_size if kernel_size % 2 == 1 else kernel_size + 1
    return cv2.GaussianBlur(img, (k, k), 0)


def blur_bilateral(
    img: np.ndarray,
    d: int = 9,
    sigma_color: float = 75.0,
    sigma_space: float = 75.0,
) -> np.ndarray:
    """Bilateral blur. Preserves edges better than Gaussian."""
    return cv2.bilateralFilter(img, d, sigma_color, sigma_space)


def blur(
    img: np.ndarray,
    method: str = "gaussian",
    kernel_size: int = 5,
    d: int = 9,
    sigma_color: float = 75.0,
    sigma_space: float = 75.0,
) -> np.ndarray:
    """
    Dispatch to blur_gaussian or blur_bilateral by name, or pass through unchanged.

    method: "none" | "gaussian" | "bilateral"
    kernel_size: Gaussian kernel size (must be odd).
    d, sigma_color, sigma_space: bilateral filter parameters.
    """
    if method == "none":
        return img.copy()
    if method == "gaussian":
        return blur_gaussian(img, kernel_size)
    if method == "bilateral":
        return blur_bilateral(img, d, sigma_color, sigma_space)
    raise ValueError(f"Unknown blur method: {method!r}. Use 'none', 'gaussian', or 'bilateral'.")


# ---------------------------------------------------------------------------
# Gradient detection (Sobel)
# ---------------------------------------------------------------------------

def gradient(
    img: np.ndarray,
    direction: str = "xy",
    ksize: int = 3,
) -> np.ndarray:
    """
    Compute Sobel gradient magnitude.

    direction:
        "none" — pass through unchanged (returns float32 of input).
        "xy"   — combined magnitude √(Gx² + Gy²).
        "x"    — |Gx| (highlights vertical edges / horizontal transitions).
        "y"    — |Gy| (highlights horizontal edges / vertical transitions).
    Returns float32 array normalised to [0, 255].
    """
    gray = to_grayscale(img).astype(np.float32)
    if direction == "none":
        max_val = gray.max()
        if max_val > 0:
            gray = gray / max_val * 255.0
        return gray

    ksize = ksize if ksize % 2 == 1 else ksize + 1   # enforce odd
    if direction in ("xy", "x"):
        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=ksize)
    if direction in ("xy", "y"):
        gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=ksize)

    if direction == "xy":
        mag = np.sqrt(gx ** 2 + gy ** 2)
    elif direction == "x":
        mag = np.abs(gx)
    elif direction == "y":
        mag = np.abs(gy)
    else:
        raise ValueError(
            f"Unknown gradient direction: {direction!r}. "
            "Use 'none', 'xy', 'x', or 'y'."
        )

    # Normalise to [0, 255]
    max_val = mag.max()
    if max_val > 0:
        mag = (mag / max_val) * 255.0
    return mag.astype(np.float32)


def apply_gradient_step(
    img: np.ndarray,
    method: str,
    ksize: int,
) -> np.ndarray:
    """
    Apply an optional Sobel gradient pre-processing step and return uint8.

    method == "none" → return ``img`` unchanged.
    Otherwise calls ``gradient(img, direction=method, ksize=ksize)`` and
    converts the result to uint8.  Used as a cacheable pipeline step inserted
    between blur and the mode-specific algorithm.
    """
    if method == "none":
        return img
    mag = gradient(img, direction=method, ksize=ksize)
    return np.clip(mag, 0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# A:B frame differencing
# ---------------------------------------------------------------------------

def contrast_ab(img_a: np.ndarray, img_b: np.ndarray) -> np.ndarray:
    """
    Compute absolute per-pixel difference between two frames (grayscale).
    Returns uint8 array.
    """
    a = to_grayscale(img_a).astype(np.int16)
    b = to_grayscale(img_b).astype(np.int16)
    diff = np.abs(a - b)
    return np.clip(diff, 0, 255).astype(np.uint8)


def accumulate_ab(frames: list[np.ndarray]) -> np.ndarray:
    """
    Given N frames, compute the sum of all consecutive A:B differences.
    Returns float32 accumulation array (not normalised).
    """
    if len(frames) < 2:
        raise ValueError("accumulate_ab requires at least 2 frames.")
    acc = np.zeros_like(to_grayscale(frames[0]), dtype=np.float32)
    for i in range(len(frames) - 1):
        acc += contrast_ab(frames[i], frames[i + 1]).astype(np.float32)
    return acc


# ---------------------------------------------------------------------------
# Reference-contrast differencing
# ---------------------------------------------------------------------------

def compute_contrast_image(
    sample_gray: np.ndarray,
    reference_gray: np.ndarray,
    mode: str = "absolute",
    contrast_min: float = 0.0,
    contrast_max: float = 100.0,
) -> np.ndarray:
    """
    Compute a per-pixel contrast image from two pre-blurred grayscale frames.

    Parameters
    ----------
    sample_gray    : uint8 grayscale — the image with the feature of interest
                     (e.g. tip with liquid).
    reference_gray : uint8 grayscale — the baseline image (e.g. tip without
                     liquid), matched by position.
    mode           : "absolute" | "relative"
        "absolute"  |sample - reference| in [0, 255].
                    Values < contrast_min → 0 (suppressed).
                    Values ≥ contrast_max → 255 (saturated).
                    Values in between are linearly stretched to [0, 255].
        "relative"  Signed (sample - reference) centred at 127.
                    sample > reference → output > 127 (bright).
                    sample < reference → output < 127 (dark).
                    |diff| < contrast_min → 127 (neutral grey).
                    |diff| ≥ contrast_max → 0 or 255 (saturated).
    contrast_min   : Minimum magnitude threshold (0–255). Differences smaller
                     than this are treated as noise and suppressed.
    contrast_max   : Maximum magnitude cap (0–255). Differences at or above
                     this are fully saturated.  Must be > contrast_min.

    Returns
    -------
    uint8 grayscale array with the same shape as the inputs.
    """
    s = sample_gray.astype(np.float32)
    r = reference_gray.astype(np.float32)

    c_min = float(contrast_min)
    c_max = float(contrast_max)
    span = max(c_max - c_min, 1e-6)

    if mode == "relative":
        diff = s - r  # signed, [-255, +255]
        abs_diff = np.abs(diff)
        # Scale magnitude from [c_min, c_max] → [0, 1]
        scaled = np.clip((abs_diff - c_min) / span, 0.0, 1.0)
        # Map: positive diff → [127, 255], negative → [0, 127], zero → 127
        out = 127.0 + np.sign(diff) * scaled * 127.0
    else:  # "absolute"
        diff = np.abs(s - r)
        out = np.clip((diff - c_min) / span, 0.0, 1.0) * 255.0

    return np.clip(out, 0, 255).astype(np.uint8)


def build_contrast_working_frame(
    sample_bgr: np.ndarray,
    reference_bgr: np.ndarray,
    params: dict,
) -> np.ndarray:
    """
    Build a BGR uint8 contrast frame from sample and reference images.

    Reads contrast params from *params*:
        contrast_blur_method          ("gaussian")
        contrast_blur_kernel          (5)
        contrast_bilateral_d          (9)
        contrast_bilateral_sigma_color (75.0)
        contrast_bilateral_sigma_space (75.0)
        contrast_mode                 ("absolute")
        contrast_min                  (0.0)
        contrast_max                  (100.0)

    The blur is applied to both images individually before subtraction so that
    noise does not amplify the difference signal.  The resulting grayscale
    contrast image is converted to BGR so it can replace a normal working frame
    in any mode.
    """
    blur_method = params.get("contrast_blur_method", "gaussian")
    blur_kernel = int(params.get("contrast_blur_kernel", 5))
    bil_d       = int(params.get("contrast_bilateral_d", 9))
    bil_sc      = float(params.get("contrast_bilateral_sigma_color", 75.0))
    bil_ss      = float(params.get("contrast_bilateral_sigma_space", 75.0))

    sample_gray = to_grayscale(sample_bgr)
    ref_gray    = to_grayscale(reference_bgr)

    sample_blurred = blur(sample_gray, blur_method, blur_kernel, bil_d, bil_sc, bil_ss)
    ref_blurred    = blur(ref_gray,    blur_method, blur_kernel, bil_d, bil_sc, bil_ss)

    contrast_gray = compute_contrast_image(
        sample_blurred,
        ref_blurred,
        mode=params.get("contrast_mode", "absolute"),
        contrast_min=float(params.get("contrast_min", 0.0)),
        contrast_max=float(params.get("contrast_max", 100.0)),
    )
    return cv2.cvtColor(contrast_gray, cv2.COLOR_GRAY2BGR)



def normalize(img: np.ndarray, out_min: float = 0.0, out_max: float = 255.0) -> np.ndarray:
    """Min-max normalise to [out_min, out_max]. Returns float32."""
    arr = img.astype(np.float32)
    lo, hi = arr.min(), arr.max()
    if hi == lo:
        return np.full_like(arr, out_min)
    return (arr - lo) / (hi - lo) * (out_max - out_min) + out_min


# ---------------------------------------------------------------------------
# HDR merge
# ---------------------------------------------------------------------------

def hdr_merge(images: list[np.ndarray]) -> np.ndarray:
    """
    Merge a bracketed exposure sequence into an HDR image using Mertens fusion.
    Returns uint8 BGR image.
    """
    merge = cv2.createMergeMertens()
    fused = merge.process(images)
    return np.clip(fused * 255, 0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# ROI extraction
# ---------------------------------------------------------------------------

def extract_roi(img: np.ndarray, roi: Tuple[int, int, int, int]) -> np.ndarray:
    """
    Extract a rectangular ROI from an image.
    roi = (x, y, width, height) in pixel coordinates.
    Returns a view (not a copy) where possible.
    """
    x, y, w, h = roi
    return img[y: y + h, x: x + w]


@functools.lru_cache(maxsize=64)
def roi_polygon_mask(
    roi_points_key: tuple,   # tuple of (int_x, int_y) — pre-rounded polygon vertices
    bbox_x: int,
    bbox_y: int,
    bbox_w: int,
    bbox_h: int,
) -> np.ndarray:
    """Return a bool mask of shape (bbox_h, bbox_w), True = inside polygon.

    All arguments are hashable so ``lru_cache`` works across calls.
    Polygon vertices should be pre-rounded to integers for a stable key.
    Cached for the process lifetime; a sweep over many images sharing the
    same polygon pays the rasterisation cost only once.
    """
    mask = np.zeros((bbox_h, bbox_w), dtype=np.uint8)
    pts = np.array(
        [(x - bbox_x, y - bbox_y) for x, y in roi_points_key],
        dtype=np.int32,
    )
    cv2.fillPoly(mask, [pts], 1)
    return mask.astype(bool)


def expand_roi_points(
    roi_points: List[Tuple],
    expansion_px: int,
    img_h: int,
    img_w: int,
) -> List[Tuple[float, float]]:
    """
    Expand (or contract) a convex polygon by ``expansion_px`` pixels
    perpendicular to each edge.

    Each edge is offset outward along its outward normal; new vertices are the
    intersections of adjacent offset edges.  This gives uniform expansion in
    every direction regardless of the polygon's aspect ratio — unlike a
    centroid-based approach which is skewed by the aspect ratio.

    A negative value contracts the polygon.  Vertices are clamped to the image
    boundary.  Handles both CW and CCW vertex orderings.
    """
    if not roi_points or expansion_px == 0:
        return roi_points

    n = len(roi_points)
    pts = [(float(p[0]), float(p[1])) for p in roi_points]

    # Signed area (screen coords, y-down). Positive = CW on screen.
    area2 = sum(
        pts[i][0] * pts[(i + 1) % n][1] - pts[(i + 1) % n][0] * pts[i][1]
        for i in range(n)
    )
    sign = 1.0 if area2 >= 0 else -1.0

    # Outward unit normal for each edge i → (i+1)
    normals: List[Tuple[float, float]] = []
    for i in range(n):
        x0, y0 = pts[i]
        x1, y1 = pts[(i + 1) % n]
        dx, dy = x1 - x0, y1 - y0
        length = math.hypot(dx, dy)
        if length < 1e-9:
            normals.append((0.0, 0.0))
        else:
            # CW winding (sign=1): outward normal of (dx,dy) is (dy,-dx)/length
            normals.append((sign * dy / length, sign * (-dx) / length))

    # New vertex i = move along bisector of the outward normals of the two
    # edges that meet there, by the amount required to achieve expansion_px
    # perpendicular to each edge.
    expanded: List[Tuple[float, float]] = []
    for i in range(n):
        n_prev = normals[(i - 1) % n]
        n_curr = normals[i]
        bx = n_prev[0] + n_curr[0]
        by = n_prev[1] + n_curr[1]
        b_len = math.hypot(bx, by)
        if b_len < 1e-9:
            # Anti-parallel normals (degenerate 180° corner) — use edge normal
            bx, by = n_curr
            offset = float(expansion_px)
        else:
            bx /= b_len
            by /= b_len
            # cos of the half-angle between adjacent edge normals
            cos_half = bx * n_curr[0] + by * n_curr[1]
            # Clamp to prevent extreme offsets at near-flat corners (< ~6°)
            cos_half = max(cos_half, 0.1)
            offset = float(expansion_px) / cos_half

        x = pts[i][0] + offset * bx
        y = pts[i][1] + offset * by
        x = max(0.0, min(float(img_w - 1), x))
        y = max(0.0, min(float(img_h - 1), y))
        expanded.append((x, y))

    return expanded
