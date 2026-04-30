"""
Layer 2 — Window Contrast Alignment mode (tip).

Rather than scanning point-by-point for individual edge pixels (like InwardEdgeScan),
this evaluates the entire ROI boundary by measuring the intensity contrast between
a thin strip just *inside* and a thin strip just *outside* each boundary edge.
It then slides each edge across ±scan_range pixels to find the position that
maximises contrast, producing a single refined TipCandidate.

Algorithm origin: derived from the TA2 horizontal/vertical edge analysis in
  VS_Code_Test/VideoDeltaHeatMap1.py (_ta2_analyze_tip_position), simplified and
  generalised for the PA pipeline architecture.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import numpy as np

from pa.pipeline.types import ImageSet, TipRegion, TipCandidate
from pa.pipeline.cache import ProcessedImageCache
from pa.pipeline.modes.base import BaseTipMode
from pa.pipeline import image_primitives as ip


class WindowContrastAlign(BaseTipMode):
    """
    Params (analysis_config.json 'full_tip' pipeline):
        window_width (int, default 10):   strip half-width (px) sampled on each side of boundary
        scan_range (int, default 20):     max pixels to slide each edge in either direction
        blur_method (str, default "gaussian")
        blur_kernel (int, default 5)
        use_ab (bool, default False)
        min_confidence (float, default 0.3)
        max_candidates (int, default 3):  currently always 1, kept for API consistency
    """

    def run(self, image_set: ImageSet, cache: ProcessedImageCache) -> TipRegion:
        p = self.params
        use_ab = p.get("use_ab", False)

        # ── 1. Build working image ────────────────────────────────────────────
        if p.get("use_contrast", False) and image_set.reference_frames:
            ref = image_set.reference_frames[0]
            gray8 = ip.to_grayscale(
                ip.build_contrast_working_frame(image_set.frames[0], ref, p)
            )
        elif use_ab and len(image_set.frames) >= 2:
            working = cache.get_or_compute(
                image_set.pipette_index, "accumulate_ab", {},
                lambda: ip.accumulate_ab(image_set.frames),
            )
            gray8 = np.clip(working, 0, 255).astype(np.uint8)
        else:
            gray8 = cache.get_or_compute(
                image_set.pipette_index, "to_grayscale_frame0", {},
                lambda: ip.to_grayscale(image_set.frames[0]),
            )

        # ── 2. Blur ───────────────────────────────────────────────────────────
        blur_params = {
            "method":      p.get("blur_method", "gaussian"),
            "kernel_size": p.get("blur_kernel", 5),
            "d":           p.get("bilateral_d", 9),
            "sigma_color": p.get("bilateral_sigma_color", 75.0),
            "sigma_space": p.get("bilateral_sigma_space", 75.0),
        }
        blurred = cache.get_or_compute(
            image_set.pipette_index, "blur", blur_params,
            lambda: ip.blur(gray8, **blur_params),
        )

        # ── 3. Optional gradient step ─────────────────────────────────────────
        grad_method = p.get("gradient_method", "none")
        grad_ksize  = p.get("gradient_ksize", 3)
        if grad_method != "none":
            proc = cache.get_or_compute(
                image_set.pipette_index, "gradient",
                {"method": grad_method, "ksize": grad_ksize},
                lambda: ip.apply_gradient_step(blurred, grad_method, grad_ksize),
            )
        else:
            proc = blurred

        # ── 4. Resolve ROI bounding box ───────────────────────────────────────
        img_h, img_w = proc.shape
        if image_set.roi:
            x0, y0, rw, rh = image_set.roi
            x1, y1 = x0 + rw, y0 + rh
        elif image_set.roi_points and len(image_set.roi_points) >= 4:
            xs = [pt[0] for pt in image_set.roi_points]
            ys = [pt[1] for pt in image_set.roi_points]
            x0, y0, x1, y1 = int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))
        else:
            x0, y0, x1, y1 = 0, 0, img_w, img_h

        # Clamp to image
        x0 = max(0, x0);  y0 = max(0, y0)
        x1 = min(img_w, x1);  y1 = min(img_h, y1)

        # ── 5. Expand ROI if requested ────────────────────────────────────────
        expand = int(p.get("roi_expansion_px", 0))
        if expand > 0:
            x0 = max(0, x0 - expand)
            y0 = max(0, y0 - expand)
            x1 = min(img_w, x1 + expand)
            y1 = min(img_h, y1 + expand)

        window_w   = max(1, p.get("window_width", 10))
        scan_range = max(0, p.get("scan_range", 20))

        # ── 6. Refine each edge ───────────────────────────────────────────────
        left_x,  left_score  = _scan_edge_horizontal(proc, x0, y0, y1, scan_range, window_w, direction="left")
        right_x, right_score = _scan_edge_horizontal(proc, x1, y0, y1, scan_range, window_w, direction="right")
        bottom_y, bottom_score = _scan_edge_vertical(proc, y1, x0, x1, scan_range, window_w, direction="bottom")

        # ── 5. Build candidate ────────────────────────────────────────────────
        scores = [s for s in (left_score, right_score, bottom_score) if s > 0]
        confidence = float(np.mean(scores)) if scores else 0.0

        min_conf = p.get("min_confidence", 0.3)
        if confidence < min_conf or left_x is None or right_x is None or bottom_y is None:
            return TipRegion(
                mode="WindowContrastAlign",
                pipette_index=image_set.pipette_index,
                candidates=[],
            )

        # Derive region: (y_px, z_px, width_px, height_px) in global image coords
        tip_y    = float(left_x)
        tip_z    = float(y0)
        width_px = float(right_x - left_x)
        height_px = float(bottom_y - y0)

        # Angle: if left/right edge refined positions imply a tilt
        angle_deg = _estimate_angle(left_x, right_x, bottom_y, y0)

        candidate = TipCandidate(
            region=(tip_y, tip_z, width_px, height_px),
            angle_deg=angle_deg,
            confidence=min(confidence, 1.0),
            source_mode="WindowContrastAlign",
        )
        return TipRegion(
            mode="WindowContrastAlign",
            pipette_index=image_set.pipette_index,
            candidates=[candidate],
        )


# ---------------------------------------------------------------------------
# Edge-scanning helpers
# ---------------------------------------------------------------------------

def _contrast(inner: np.ndarray, outer: np.ndarray) -> float:
    """Normalised contrast between two pixel arrays. Returns 0 if trivial."""
    i_mean = float(inner.mean()) if inner.size > 0 else 0.0
    o_mean = float(outer.mean()) if outer.size > 0 else 0.0
    denom = i_mean + o_mean
    if denom < 1e-3:
        return 0.0
    return abs(i_mean - o_mean) / denom


def _scan_edge_horizontal(
    img: np.ndarray,
    edge_x: int,
    y0: int,
    y1: int,
    scan_range: int,
    window_w: int,
    direction: str,          # "left" (inward = right) | "right" (inward = left)
) -> Tuple[Optional[int], float]:
    """
    Slide the vertical edge boundary ±scan_range pixels and return the position
    and contrast score that maximise |inner_strip_mean − outer_strip_mean|.

    For the left edge:  inner strip = [x, x+w),  outer strip = [x-w, x)
    For the right edge: inner strip = [x-w, x),  outer strip = [x, x+w)
    """
    h, w = img.shape
    best_x     = edge_x
    best_score = 0.0

    for delta in range(-scan_range, scan_range + 1):
        x = edge_x + delta
        if direction == "left":
            inner_x0 = x
            inner_x1 = min(w, x + window_w)
            outer_x0 = max(0, x - window_w)
            outer_x1 = x
        else:  # right
            inner_x0 = max(0, x - window_w)
            inner_x1 = x
            outer_x0 = x
            outer_x1 = min(w, x + window_w)

        if inner_x1 <= inner_x0 or outer_x1 <= outer_x0:
            continue

        inner = img[y0:y1, inner_x0:inner_x1]
        outer = img[y0:y1, outer_x0:outer_x1]
        score = _contrast(inner, outer)
        if score > best_score:
            best_score = score
            best_x = x

    return best_x, best_score


def _scan_edge_vertical(
    img: np.ndarray,
    edge_y: int,
    x0: int,
    x1: int,
    scan_range: int,
    window_w: int,
    direction: str,          # "bottom" (inward = upward)
) -> Tuple[Optional[int], float]:
    """
    Slide the horizontal (bottom) boundary ±scan_range pixels.
    inner strip = rows [y-w, y),  outer strip = rows [y, y+w).
    """
    h, w = img.shape
    best_y     = edge_y
    best_score = 0.0

    for delta in range(-scan_range, scan_range + 1):
        y = edge_y + delta
        inner_y0 = max(0, y - window_w)
        inner_y1 = y
        outer_y0 = y
        outer_y1 = min(h, y + window_w)

        if inner_y1 <= inner_y0 or outer_y1 <= outer_y0:
            continue

        inner = img[inner_y0:inner_y1, x0:x1]
        outer = img[outer_y0:outer_y1, x0:x1]
        score = _contrast(inner, outer)
        if score > best_score:
            best_score = score
            best_y = y

    return best_y, best_score


def _estimate_angle(left_x: int, right_x: int, bottom_y: int, top_y: int) -> float:
    """
    Estimate tilt angle from the width:height aspect of the refined bounding box.
    Returns 0° for symmetric/upright tips.
    """
    width  = abs(right_x - left_x)
    height = abs(bottom_y - top_y)
    if height == 0 or width == 0:
        return 0.0
    # Simple heuristic: deviation of centre from midpoint is not measurable here;
    # return 0 unless a future version tracks per-row centre positions.
    return 0.0
