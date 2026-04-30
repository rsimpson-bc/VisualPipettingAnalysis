"""
Layer 2 — ROI Edge Fit mode (tip).

Slides the calibrated ROI polygon template over the image and finds the
position where its boundary best aligns with detected edge pixels.

Algorithm
---------
1.  Pre-process the image (blur + Sobel gradient → binary edge map).
2.  Build a *soft* boundary template: every pixel in the template canvas is
    weighted by a falloff function of its L2 distance to the nearest polygon
    boundary pixel.
    Two falloff types are supported:
      'gaussian'  w = exp(−d² / (2·σ²))        σ = score_falloff_px
      'inverse'   w = σ / (σ + d)              σ = score_falloff_px  (1/x shape)
    When score_cutoff_px > 0, any pixel farther than that distance
    contributes 0, short-circuiting computation for distant pixels.
3.  FFT cross-correlation (cv2.matchTemplate with TM_CCORR) gives the
    sum of (edge_map × template) for every integer (dx, dy) within the
    search window simultaneously.
4.  The top-k positions (non-maximum suppression) are extracted from the
    coarse score map and each is refined with a small fine-grid search
    (±refine_range_px at refine_step_px).
5.  The best overall position defines the fitted ROI polygon.  The output
    is the polygon's center-top coordinate (cx, ty) and the translated
    polygon itself (stored in TipCandidate.metadata for downstream
    visualization).

Output
------
TipCandidate where:
  • region      = axis-aligned bbox of the fitted polygon (x, y, w, h)
  • confidence  = normalised boundary-alignment score  (0–1)
  • metadata    = {
        center_top          : (cx, ty) — fitted tip top-centre in image coords
        offset_dx / offset_dy : int   — best translation from nominal ROI
        fit_score           : float   — raw score (sum of template × edge map)
        fitted_roi_polygon  : [(x,y), …] — translated polygon points
    }
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from pa.pipeline.types import ImageSet, TipRegion, TipCandidate
from pa.pipeline.cache import ProcessedImageCache
from pa.pipeline.modes.base import BaseTipMode
from pa.pipeline import image_primitives as ip


class ROIEdgeFit(BaseTipMode):
    """
    Params
    ------
    use_ab (bool)
    blur_method (str)          : "none" | "gaussian" | "bilateral"
    blur_kernel (int)          : Gaussian kernel size
    bilateral_d (int)          : bilateral filter diameter
    bilateral_sigma_color (float)
    bilateral_sigma_space (float)
    gradient_ksize (int)       : Sobel kernel size (3, 5, or 7)
    gradient_threshold (float) : binary edge-map threshold
    roi_expansion_px (int)     : ROI2 expands ROI1 by this many px on every
                                 side. The template (ROI1) is allowed to slide
                                 within ROI2; this is the entire search budget.
                                 If 0, the algorithm returns ROI1 unchanged.
    n_coarse_peaks (int)       : number of FFT peaks to refine
    refine_range_px (int)      : ±px radius for fine-grid refinement
    refine_step_px (int)       : step size for fine-grid search
    score_falloff_type (str)   : "gaussian" | "inverse"
    score_falloff_px (float)   : σ / half-value distance for the falloff
    score_cutoff_px (float)    : pixels beyond this distance score 0; 0 = no cutoff
    """

    def run(self, image_set: ImageSet, cache: ProcessedImageCache) -> TipRegion:
        tip_region, _ = self._run_internal(image_set, cache)
        return tip_region

    def run_debug(self, image_set: ImageSet, cache: ProcessedImageCache):
        """Return (TipRegion, engine_result_dict) for debug inspection."""
        return self._run_internal(image_set, cache)

    def _run_internal(self, image_set: ImageSet, cache: ProcessedImageCache):
        params = self.params
        use_ab = params.get("use_ab", False)

        # ── 1. Build working image ────────────────────────────────────────────
        if params.get("use_contrast", False) and image_set.reference_frames:
            ref = image_set.reference_frames[0]
            bgr = ip.build_contrast_working_frame(image_set.frames[0], ref, params)
        elif use_ab and len(image_set.frames) >= 2:
            working = cache.get_or_compute(
                image_set.pipette_index, "accumulate_ab", {},
                lambda: ip.accumulate_ab(image_set.frames),
            )
            bgr = cv2.cvtColor(
                np.clip(working, 0, 255).astype(np.uint8), cv2.COLOR_GRAY2BGR
            )
        else:
            bgr = image_set.frames[0]

        # ── 2. Resolve ROI polygon ────────────────────────────────────────────
        h, w = bgr.shape[:2]
        if image_set.roi_points and len(image_set.roi_points) >= 4:
            roi_points = image_set.roi_points
        else:
            roi_points = [(0, 0), (w - 1, 0), (w - 1, h - 1), (0, h - 1)]

        # roi_expansion_px is the search budget in this mode (sr is read from
        # the engine). It widens the patch (ROI2) so the template (ROI1) can
        # slide within it; it does NOT change the template shape or the output
        # polygon size.

        # ── 3. Run the engine ─────────────────────────────────────────────────
        engine = _ROIEdgeFitEngine(params)
        engine_result = engine.fit(bgr, roi_points, cache)

        # ── 4. Convert to TipRegion ───────────────────────────────────────────
        candidates = _result_to_candidates(engine_result, "ROIEdgeFit")
        tip_region = TipRegion(
            mode="ROIEdgeFit",
            pipette_index=image_set.pipette_index,
            candidates=candidates,
        )
        return tip_region, engine_result


# ---------------------------------------------------------------------------
# Result conversion
# ---------------------------------------------------------------------------

def _result_to_candidates(
    result: Optional[Dict[str, Any]], source_mode: str
) -> List[TipCandidate]:
    if not result or not result.get("fitted_roi_polygon"):
        return []
    norm_score = result.get("normalised_score", 0.0)
    if norm_score <= 0.0:
        return []
    fitted = result["fitted_roi_polygon"]
    xs = [p[0] for p in fitted]
    ys = [p[1] for p in fitted]
    region = (
        float(min(xs)),
        float(min(ys)),
        float(max(xs) - min(xs)),
        float(max(ys) - min(ys)),
    )
    return [TipCandidate(
        region=region,
        angle_deg=0.0,
        confidence=float(np.clip(norm_score, 0.0, 1.0)),
        source_mode=source_mode,
        metadata={
            "center_top":         result.get("center_top"),
            "offset_dx":          result.get("best_dx", 0),
            "offset_dy":          result.get("best_dy", 0),
            "fit_score":          result.get("best_score", 0.0),
            "fitted_roi_polygon": fitted,
        },
    )]


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------

class _ROIEdgeFitEngine:

    def __init__(self, params: dict):
        self._params        = params          # forwarded to edge detector when needed
        self.blur_method    = params.get("blur_method", "gaussian")
        self.blur_kernel    = params.get("blur_kernel", 5)
        self.bilateral_d    = params.get("bilateral_d", 9)
        self.bilateral_sc   = params.get("bilateral_sigma_color", 75)
        self.bilateral_ss   = params.get("bilateral_sigma_space", 75)
        self.gradient_ksize = params.get("gradient_ksize", 3)
        self.grad_threshold = float(params.get("gradient_threshold", 30.0))
        self.scoring_signal = params.get("scoring_signal", "edge_map")  # "edge_map"|"gradient_magnitude"|"raw_intensity"
        # Search budget = roi_expansion_px (no separate search_range_px).
        # ROI1 = calibrated polygon; ROI2 = ROI1 + roi_expansion_px on every side.
        # Template (ROI1) slides within ROI2.
        self.search_range   = max(0, int(params.get("roi_expansion_px", 0)))
        self.n_coarse_peaks = int(params.get("n_coarse_peaks", 3))
        self.refine_range   = int(params.get("refine_range_px", 5))
        self.refine_step    = max(1, int(params.get("refine_step_px", 1)))
        self.falloff_type   = params.get("score_falloff_type", "gaussian")
        self.falloff_px     = float(params.get("score_falloff_px", 10.0))
        self.cutoff_px      = float(params.get("score_cutoff_px", 0.0))
        self.use_edge_det   = bool(params.get("use_edge_detection", False))
        self.edge_outward_offset = int(params.get("edge_outward_offset_px", 0))

    def fit(self, bgr: np.ndarray, roi_points: list, cache) -> Dict[str, Any]:
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY) if bgr.ndim == 3 else bgr
        blurred = ip.blur(
            gray,
            method=self.blur_method,
            kernel_size=self.blur_kernel,
            d=self.bilateral_d,
            sigma_color=self.bilateral_sc,
            sigma_space=self.bilateral_ss,
        )

        gmag: Optional[np.ndarray] = None
        k = self.gradient_ksize if self.gradient_ksize % 2 == 1 else self.gradient_ksize + 1

        # ROI3 = ROI1 expanded by the search budget — used as the scan region
        # for InwardEdgeScan so it sees edges that may lie outside ROI1.
        img_h, img_w = bgr.shape[:2]
        roi3_points = (
            ip.expand_roi_points(roi_points, self.search_range, img_h, img_w)
            if self.search_range > 0 else roi_points
        )

        if self.use_edge_det:
            # Use N-of-M confirmed InwardEdgeScan points as the fitting signal
            edge_map = build_edge_point_map(bgr, roi3_points, self._params)
        elif self.scoring_signal == "raw_intensity":
            # Use blurred grayscale directly — no Sobel at all
            edge_map = (blurred.astype(np.float32) / 255.0)
        elif self.scoring_signal == "gradient_magnitude":
            # Continuous Sobel magnitude — no threshold
            gx = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=k)
            gy = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=k)
            gmag = np.sqrt(gx ** 2 + gy ** 2)
            edge_map = (gmag / (gmag.max() + 1e-6)).astype(np.float32)
        else:  # "edge_map" (default) — Sobel + threshold → binary
            gx = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=k)
            gy = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=k)
            gmag = np.sqrt(gx ** 2 + gy ** 2)
            edge_map = (gmag >= self.grad_threshold).astype(np.float32)

        return self._fit_to_edge_map(edge_map, bgr.shape[:2], roi_points, gmag)

    def _fit_to_edge_map(
        self,
        edge_map: np.ndarray,
        img_shape: Tuple[int, int],
        roi_points: list,
        gradient_map: np.ndarray,
    ) -> Dict[str, Any]:
        img_h, img_w = img_shape
        xs = [p[0] for p in roi_points]
        ys = [p[1] for p in roi_points]
        ox, oy = int(min(xs)), int(min(ys))
        roi_w = max(1, int(max(xs)) - ox)
        roi_h = max(1, int(max(ys)) - oy)
        sr = self.search_range

        # Extract search patch — centred on the nominal ROI position
        px0 = max(0, ox - sr)
        py0 = max(0, oy - sr)
        px1 = min(img_w, ox + roi_w + sr)
        py1 = min(img_h, oy + roi_h + sr)
        if px1 - px0 < roi_w or py1 - py0 < roi_h:
            return _empty_result(roi_points)
        patch = edge_map[py0:py1, px0:px1]

        # Contract the matching template to compensate for inward edge-detection
        # bias (detected edges land slightly inside the true boundary).
        # The output 'fitted' polygon still uses the original roi_points.
        if self.edge_outward_offset:
            template_pts = ip.expand_roi_points(
                roi_points, -self.edge_outward_offset, img_h, img_w
            )
        else:
            template_pts = roi_points

        # Build soft boundary template
        template = build_boundary_template(
            template_pts, ox, oy, roi_w, roi_h,
            self.falloff_type, self.falloff_px, self.cutoff_px,
        )
        if patch.shape[0] < template.shape[0] or patch.shape[1] < template.shape[1]:
            return _empty_result(roi_points)

        # Coarse FFT cross-correlation
        score_map = cv2.matchTemplate(patch, template, cv2.TM_CCORR)
        # Normalise against the template's boundary-pixel count (pixels at
        # distance 0 from the polygon edge, where template ≈ 1.0).  Using
        # sum(template) as the ceiling is incorrect: it assumes every pixel
        # in the window is an edge, giving near-zero scores for good real fits.
        boundary_pixels = float(np.sum(template >= (1.0 - 1e-5)))
        max_possible = boundary_pixels if boundary_pixels > 0 else float(np.sum(template))

        # Fine-grid refinement around top-k FFT peaks
        peaks = _find_top_k_peaks(
            score_map, self.n_coarse_peaks, min_distance=max(4, self.refine_range)
        )
        best_score = -1.0
        best_dy = 0
        best_dx = 0
        for (cr, cc, _) in peaks:
            for ddy in range(-self.refine_range, self.refine_range + 1, self.refine_step):
                for ddx in range(-self.refine_range, self.refine_range + 1, self.refine_step):
                    r = cr + ddy
                    c = cc + ddx
                    if (r < 0 or c < 0
                            or r + roi_h > patch.shape[0]
                            or c + roi_w > patch.shape[1]):
                        continue
                    region = patch[r:r + roi_h, c:c + roi_w]
                    score = float(np.dot(region.ravel(), template.ravel()))
                    if score > best_score:
                        best_score = score
                        best_dy = (py0 + r) - oy
                        best_dx = (px0 + c) - ox

        normalised_score = best_score / max_possible if max_possible > 0 else 0.0
        fitted = [(p[0] + best_dx, p[1] + best_dy) for p in roi_points]
        center_top = _polygon_center_top(fitted)

        # ROI3 = ROI1 expanded by the full search budget using the same
        # perpendicular-offset method, clamped to image bounds.
        roi3_points = (
            ip.expand_roi_points(roi_points, sr, img_h, img_w)
            if sr > 0 else list(roi_points)
        )

        return {
            "fitted_roi_polygon": fitted,
            "roi_points":        roi_points,
            "roi3_points":       roi3_points,
            "best_dx":           best_dx,
            "best_dy":           best_dy,
            "best_score":        best_score,
            "normalised_score":  normalised_score,
            "center_top":        center_top,
            "score_map":         score_map,
            "edge_map":          edge_map,
            "gradient_map":      gradient_map,
            "roi_bbox":          (ox, oy, roi_w, roi_h),
            "patch_origin":      (px0, py0),
            "patch_bbox":        (px0, py0, px1 - px0, py1 - py0),
        }


# ---------------------------------------------------------------------------
# Module-level helpers (reused by roi_contrast_fit)
# ---------------------------------------------------------------------------

def build_boundary_template(
    roi_points: list,
    ox: int,
    oy: int,
    roi_w: int,
    roi_h: int,
    falloff_type: str,
    falloff_px: float,
    cutoff_px: float,
) -> np.ndarray:
    """
    Float32 template where each pixel's weight = falloff(dist to nearest polygon edge).

    falloff_type "gaussian" : exp(-d² / (2·σ²))   σ = falloff_px
    falloff_type "inverse"  : σ / (σ + d)          σ = falloff_px (1/x relationship)
    cutoff_px > 0           : pixels beyond this distance → 0 (speeds up scoring)
    """
    canvas = np.zeros((roi_h, roi_w), dtype=np.uint8)
    local_pts = np.array(
        [[int(p[0]) - ox, int(p[1]) - oy] for p in roi_points], dtype=np.int32
    )
    cv2.polylines(canvas, [local_pts], isClosed=True, color=255, thickness=1)

    # Distance of every pixel from the nearest boundary pixel
    inv = (255 - canvas).astype(np.uint8)
    dist = cv2.distanceTransform(inv, cv2.DIST_L2, 5).astype(np.float32)

    # Apply cutoff: mark distant pixels as "infinite" so they get 0 weight
    if cutoff_px > 0:
        dist = np.where(dist > cutoff_px, np.inf, dist)

    sigma = max(float(falloff_px), 1e-6)
    if falloff_type == "inverse":
        weights = np.where(np.isfinite(dist), sigma / (sigma + dist), 0.0)
    else:  # gaussian
        weights = np.where(np.isfinite(dist), np.exp(-(dist ** 2) / (2.0 * sigma ** 2)), 0.0)

    return weights.astype(np.float32)


def build_interior_template(
    roi_points: list,
    ox: int,
    oy: int,
    roi_w: int,
    roi_h: int,
) -> np.ndarray:
    """Binary float32 mask — 1.0 inside the ROI polygon, 0.0 outside."""
    canvas = np.zeros((roi_h, roi_w), dtype=np.uint8)
    local_pts = np.array(
        [[int(p[0]) - ox, int(p[1]) - oy] for p in roi_points], dtype=np.int32
    )
    cv2.fillPoly(canvas, [local_pts], 255)
    return (canvas.astype(np.float32) / 255.0)


def _find_top_k_peaks(
    score_map: np.ndarray, k: int, min_distance: int = 5
) -> List[Tuple[int, int, float]]:
    """Extract up to k peaks with non-maximum suppression."""
    peaks: List[Tuple[int, int, float]] = []
    sm = score_map.copy()
    for _ in range(k):
        idx = int(np.argmax(sm))
        r, c = divmod(idx, sm.shape[1])
        val = float(sm[r, c])
        if val <= 0:
            break
        peaks.append((r, c, val))
        r0, r1 = max(0, r - min_distance), min(sm.shape[0], r + min_distance + 1)
        c0, c1 = max(0, c - min_distance), min(sm.shape[1], c + min_distance + 1)
        sm[r0:r1, c0:c1] = 0.0
    return peaks


def _polygon_center_top(
    polygon: List[Tuple[float, float]]
) -> Tuple[float, float]:
    """Center-top of a polygon: (mean-x of topmost vertices, min-y)."""
    min_y = min(p[1] for p in polygon)
    top_pts = [p for p in polygon if abs(p[1] - min_y) < 2]
    cx = float(np.mean([p[0] for p in top_pts])) if top_pts else float(
        np.mean([p[0] for p in polygon])
    )
    return (cx, float(min_y))


def _empty_result(roi_points: list) -> Dict[str, Any]:
    return {
        "fitted_roi_polygon": None,
        "best_dx": 0, "best_dy": 0,
        "best_score": 0.0, "normalised_score": 0.0,
        "center_top": None,
        "score_map": None, "edge_map": None, "gradient_map": None,
    }


def build_edge_point_map(
    bgr: np.ndarray,
    roi_points: list,
    params: dict,
) -> np.ndarray:
    """
    Run the InwardEdgeScan N-of-M confirmed edge detector and return a sparse
    float32 map the same size as ``bgr`` where each confirmed edge point = 1.0
    and everything else = 0.0.

    This lets ROIEdgeFit / ROIContrastFit use the higher-quality, noise-filtered
    edge points as their fitting signal instead of the raw Sobel threshold map.
    All InwardEdgeScan detection params (confirmation_n/m, intensity_delta, etc.)
    are read from ``params`` by the engine, so they can be tuned independently in
    the GUI just like any other parameter.

    Called only when use_edge_detection=True.
    """
    from pa.pipeline.modes.inward_edge_scan import _TipEdgeScanEngine

    h, w = bgr.shape[:2]
    edge_map = np.zeros((h, w), dtype=np.float32)

    engine = _TipEdgeScanEngine(params)
    result = engine.detect_edges_in_roi(bgr, roi_points, tip_idx=0, cache=None)

    for key in ("left_edge", "right_edge", "bottom_edge"):
        edge = result.get(key) or {}
        for pt in edge.get("points", []):
            px, py = int(round(pt[0])), int(round(pt[1]))
            if 0 <= px < w and 0 <= py < h:
                edge_map[py, px] = 1.0

    return edge_map
