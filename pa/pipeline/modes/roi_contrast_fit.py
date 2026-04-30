"""
Layer 2 — ROI Contrast Fit mode (tip).

Slides the calibrated ROI polygon over the image and finds the position
that maximises the total gradient magnitude *inside* the ROI.

Design intent
-------------
The contrast fit rewards positions where the ROI interior encloses the
most gradient energy.  Unlike ROIEdgeFit (which scores alignment of the
ROI *boundary* against detected edges), this mode rewards the tip region
as a whole being "full of contrast" — useful when the tip edge doesn't
quite trigger edge detection but produces a strong gradient signal a few
pixels inward, or when working with A:B accumulated contrast images.

NOTE: This mode is designed for use with contrast/accumulated images (e.g.
A:B difference frames via use_ab=True).  Raw single frames are accepted but
may produce noisier results.

Algorithm
---------
1.  Pre-process the image (blur + Sobel gradient magnitude, NOT thresholded).
2.  Build a binary interior mask matching the ROI polygon shape.
3.  FFT cross-correlation (cv2.matchTemplate + TM_CCORR) gives the sum of
    gradient magnitude inside the translated mask at every (dx, dy) offset.
    Positions where the translated ROI would extend outside the image are
    zeroed out before peak selection.
4.  Top-k peaks → fine-grid refinement (same as ROIEdgeFit).
5.  Best position → fitted polygon + center-top coordinate.

Output
------
TipCandidate where:
  • region      = axis-aligned bbox of the fitted polygon (x, y, w, h)
  • confidence  = normalised contrast score (0–1)
  • metadata    = {
        center_top          : (cx, ty) — fitted tip top-centre in image coords
        offset_dx / offset_dy : int   — best translation from nominal ROI
        contrast_score      : float   — raw summed gradient inside fitted ROI
        fitted_roi_polygon  : [(x,y), …] — translated polygon points
    }
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from pa.pipeline.types import ImageSet, TipRegion, TipCandidate
from pa.pipeline.cache import ProcessedImageCache
from pa.pipeline.modes.base import BaseTipMode
from pa.pipeline import image_primitives as ip
from pa.pipeline.modes.roi_edge_fit import (
    build_interior_template,
    build_edge_point_map,
    _find_top_k_peaks,
    _polygon_center_top,
    _empty_result,
)


class ROIContrastFit(BaseTipMode):
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
    roi_expansion_px (int)     : ROI2 expands ROI1 by this many px on every
                                 side. The template (ROI1) is allowed to slide
                                 within ROI2; this is the entire search budget.
                                 If 0, the algorithm returns ROI1 unchanged.
    n_coarse_peaks (int)       : number of FFT peaks to refine
    refine_range_px (int)      : ±px radius for fine-grid refinement
    refine_step_px (int)       : step size for fine-grid search
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
        engine = _ROIContrastFitEngine(params)
        engine_result = engine.fit(bgr, roi_points, cache)

        # ── 4. Convert to TipRegion ───────────────────────────────────────────
        candidates = _result_to_candidates(engine_result)
        tip_region = TipRegion(
            mode="ROIContrastFit",
            pipette_index=image_set.pipette_index,
            candidates=candidates,
        )
        return tip_region, engine_result


# ---------------------------------------------------------------------------
# Result conversion
# ---------------------------------------------------------------------------

def _result_to_candidates(
    result: Optional[Dict[str, Any]],
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
        source_mode="ROIContrastFit",
        metadata={
            "center_top":         result.get("center_top"),
            "offset_dx":          result.get("best_dx", 0),
            "offset_dy":          result.get("best_dy", 0),
            "contrast_score":     result.get("best_score", 0.0),
            "fitted_roi_polygon": fitted,
        },
    )]


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------

class _ROIContrastFitEngine:

    def __init__(self, params: dict):
        self._params        = params          # forwarded to edge detector when needed
        self.blur_method    = params.get("blur_method", "gaussian")
        self.blur_kernel    = params.get("blur_kernel", 5)
        self.bilateral_d    = params.get("bilateral_d", 9)
        self.bilateral_sc   = params.get("bilateral_sigma_color", 75)
        self.bilateral_ss   = params.get("bilateral_sigma_space", 75)
        self.gradient_ksize = params.get("gradient_ksize", 3)
        # Search budget = roi_expansion_px (no separate search_range_px).
        # ROI1 = calibrated polygon; ROI2 = ROI1 + roi_expansion_px on every side.
        # Template (ROI1) slides within ROI2.
        self.search_range   = max(0, int(params.get("roi_expansion_px", 0)))
        self.n_coarse_peaks = int(params.get("n_coarse_peaks", 3))
        self.refine_range   = int(params.get("refine_range_px", 5))
        self.refine_step    = max(1, int(params.get("refine_step_px", 1)))
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
        k = self.gradient_ksize if self.gradient_ksize % 2 == 1 else self.gradient_ksize + 1
        gx = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=k)
        gy = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=k)
        gmag = np.sqrt(gx ** 2 + gy ** 2).astype(np.float32)

        # ROI3 = ROI1 expanded by the search budget — used as the scan region
        # for InwardEdgeScan so it sees edges that may lie outside ROI1.
        img_h, img_w = bgr.shape[:2]
        roi3_points = (
            ip.expand_roi_points(roi_points, self.search_range, img_h, img_w)
            if self.search_range > 0 else roi_points
        )

        if self.use_edge_det:
            # Use N-of-M confirmed InwardEdgeScan points as the scoring signal.
            # Each confirmed point contributes 1.0; contrast inside the ROI
            # then = count of confirmed edge points enclosed at each offset.
            scoring_map = build_edge_point_map(bgr, roi3_points, self._params)
        else:
            # Sum the continuous gradient magnitude inside the ROI (default)
            scoring_map = gmag

        return self._fit_to_gradient(scoring_map, bgr.shape[:2], roi_points)

    def _fit_to_gradient(
        self,
        gmag: np.ndarray,
        img_shape: Tuple[int, int],
        roi_points: list,
    ) -> Dict[str, Any]:
        img_h, img_w = img_shape
        xs = [p[0] for p in roi_points]
        ys = [p[1] for p in roi_points]
        ox, oy = int(min(xs)), int(min(ys))
        roi_w = max(1, int(max(xs)) - ox)
        roi_h = max(1, int(max(ys)) - oy)
        sr = self.search_range

        # Extract search patch
        px0 = max(0, ox - sr)
        py0 = max(0, oy - sr)
        px1 = min(img_w, ox + roi_w + sr)
        py1 = min(img_h, oy + roi_h + sr)
        if px1 - px0 < roi_w or py1 - py0 < roi_h:
            return _empty_result(roi_points)
        patch = gmag[py0:py1, px0:px1]

        # Contract the matching template to compensate for inward edge-detection
        # bias (gradient peak is slightly inside the true boundary).
        # The output 'fitted' polygon still uses the original roi_points.
        if self.edge_outward_offset:
            template_pts = ip.expand_roi_points(
                roi_points, -self.edge_outward_offset, img_h, img_w
            )
        else:
            template_pts = roi_points

        template = build_interior_template(template_pts, ox, oy, roi_w, roi_h)
        if patch.shape[0] < template.shape[0] or patch.shape[1] < template.shape[1]:
            return _empty_result(roi_points)

        # Coarse FFT cross-correlation
        score_map = cv2.matchTemplate(patch, template, cv2.TM_CCORR)

        # Zero out positions where the translated ROI would exit the image.
        # score_map[r,c] → template top-left at full-image coords (py0+r, px0+c).
        smh, smw = score_map.shape
        r_idx = np.arange(smh, dtype=np.int32)[:, None]
        c_idx = np.arange(smw, dtype=np.int32)[None, :]
        ty = py0 + r_idx
        tx = px0 + c_idx
        out_of_bounds = (ty < 0) | (tx < 0) | (ty + roi_h > img_h) | (tx + roi_w > img_w)
        score_map[out_of_bounds] = 0.0

        # Normalise: max possible = interior_pixels × max_gradient_value
        # (i.e. what score would be achieved if every interior pixel were at
        # the maximum observed gradient magnitude).
        interior_pixels = float(np.sum(template))   # constant (same ROI every time)
        max_patch_val = float(gmag.max()) if gmag.max() > 0 else 1.0
        max_possible = max_patch_val * interior_pixels

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
                    # Skip positions where full ROI leaves the image
                    if (py0 + r < 0 or px0 + c < 0
                            or py0 + r + roi_h > img_h
                            or px0 + c + roi_w > img_w):
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

        # ROI3 = ROI1 expanded by the full search budget using perpendicular
        # edge offsets, clamped to image bounds.
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
            "gradient_map":      gmag,
            "roi_bbox":          (ox, oy, roi_w, roi_h),
            "patch_origin":      (px0, py0),
            "patch_bbox":        (px0, py0, px1 - px0, py1 - py0),
        }
