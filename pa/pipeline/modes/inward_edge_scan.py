"""
Layer 2 — Inward Edge Scan mode (tip).

Scans inward from the known ROI boundary toward the tip centre using anchor-first
N-of-M confirmation + intensity verification + post-processing filters.

Algorithm origin: TipEdgeDetector in VS_Code_Test/VideoDeltaHeatMap1.py.
Migrated to fit the PA pipeline architecture (Layer 2 mode → TipRegion output).

Key design changes from the original:
  - Pre-processing (blur) and gradient computation use the ProcessedImageCache
    so results are shared if another mode runs on the same step.
  - All GUI / Tkinter code removed.
  - Output is TipRegion(candidates=[TipCandidate, ...]) instead of a raw dict.
  - process_name / PID logging removed (pool_manager handles parallelism).
  - The polygon ROI comes from ImageSet.roi_points; if absent the full image
    boundary is used as a 4-corner fallback.
"""

from __future__ import annotations

import math
import time
from typing import Dict, List, Optional, Tuple, Any

import cv2
import numpy as np

from pa.pipeline.types import ImageSet, TipRegion, TipCandidate
from pa.pipeline.cache import ProcessedImageCache
from pa.pipeline.modes.base import BaseTipMode
from pa.pipeline import image_primitives as ip


class InwardEdgeScan(BaseTipMode):
    """
    Params (all match analysis_config.json 'full_tip' pipeline):
        gradient_threshold (float)
        confirmation_n (int)
        confirmation_m (int)
        intensity_delta (float)
        intensity_avg_pixels (int)
        confirmation_brightness_threshold (int)
        edge_detection_offset (int)
        min_edge_length_enabled (bool)
        min_edge_length (int)
        consistency_check_enabled (bool)
        consistency_window (int)
        consistency_sigma (float)
        outlier_removal_enabled (bool)
        outlier_threshold (float)
        section_offset (int)
        bottom_search_depth (int)
        bottom_scan_half_width (int)
        detect_left (bool)
        detect_right (bool)
        detect_bottom (bool)
        blur_method (str)
        blur_kernel (int)
        use_ab (bool)
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

        expansion = params.get("roi_expansion_px", 0)
        if expansion:
            roi_points = ip.expand_roi_points(roi_points, expansion, h, w)

        # ── 3. Run the core engine ────────────────────────────────────────────
        engine = _TipEdgeScanEngine(params)
        result = engine.detect_edges_in_roi(bgr, roi_points, image_set.pipette_index,
                                            cache)

        # ── 4. Convert to TipRegion ───────────────────────────────────────────
        candidates = _result_to_candidates(result, image_set.pipette_index)
        tip_region = TipRegion(
            mode="InwardEdgeScan",
            pipette_index=image_set.pipette_index,
            candidates=candidates,
        )
        return tip_region, result


# ---------------------------------------------------------------------------
# ROI expansion helper
# ---------------------------------------------------------------------------

def _expand_roi_points(
    roi_points: List[Tuple], expansion_px: int, img_h: int, img_w: int
) -> List[Tuple[float, float]]:
    """
    Expand a polygon outward by ``expansion_px`` pixels in every direction.

    Each vertex is shifted away from the polygon centroid by ``expansion_px``
    pixels, then clamped to the image boundary.  Works for any convex polygon
    (the calibrated ROI trapezoids are always convex).
    """
    if not roi_points or expansion_px <= 0:
        return roi_points
    pts = [(float(p[0]), float(p[1])) for p in roi_points]
    cx = sum(p[0] for p in pts) / len(pts)
    cy = sum(p[1] for p in pts) / len(pts)
    expanded = []
    for px, py in pts:
        dx, dy = px - cx, py - cy
        dist = math.hypot(dx, dy)
        if dist > 0:
            px += expansion_px * dx / dist
            py += expansion_px * dy / dist
        # Clamp to image bounds
        px = max(0.0, min(float(img_w - 1), px))
        py = max(0.0, min(float(img_h - 1), py))
        expanded.append((px, py))
    return expanded


# ---------------------------------------------------------------------------
# Result conversion helpers
# ---------------------------------------------------------------------------

def _result_to_candidates(result: Dict[str, Any], pipette_index: int) -> List[TipCandidate]:
    quality_score = result.get("quality_score", 0.0)
    confidence = min(quality_score / 100.0, 1.0)
    region = _derive_region(result)
    angle  = _derive_angle(result)

    if region is None or confidence == 0.0:
        return []

    # NOTE: currently emits only one candidate (the bounding box of all detected
    # edge points).  Multi-candidate support is tracked as a future expansion —
    # see TipCandidate.metadata TODO comment in pa/pipeline/types.py.
    return [TipCandidate(
        region=region,
        angle_deg=angle,
        confidence=confidence,
        source_mode="InwardEdgeScan",
    )]


def _derive_region(result: Dict[str, Any]) -> Optional[Tuple[float, float, float, float]]:
    all_points: List[Tuple[float, float]] = []
    for key in ("left_edge", "right_edge", "bottom_edge"):
        edge = result.get(key)
        if edge:
            all_points.extend(edge.get("points", []))
    if not all_points:
        return None
    xs = [p[0] for p in all_points]
    ys = [p[1] for p in all_points]
    return (float(min(xs)), float(min(ys)),
            float(max(xs) - min(xs)), float(max(ys) - min(ys)))


def _derive_angle(result: Dict[str, Any]) -> float:
    for key in ("left_edge", "right_edge"):
        edge = result.get(key)
        if edge and len(edge.get("points", [])) >= 2:
            pts = np.array(edge["points"])
            try:
                coeffs = np.polyfit(pts[:, 1], pts[:, 0], 1)
                return float(math.degrees(math.atan(coeffs[0])))
            except (np.linalg.LinAlgError, ValueError):
                pass
    return 0.0


# ---------------------------------------------------------------------------
# Core engine — migrated from TipEdgeDetector in VideoDeltaHeatMap1.py
# ---------------------------------------------------------------------------

class _TipEdgeScanEngine:
    """
    Pure-algorithmic engine. No Layer 0/1 imports — receives already-prepared
    arrays from InwardEdgeScan.run(). Unit-testable independently.
    """

    def __init__(self, params: dict):
        self.blur_method = params.get("blur_method", "gaussian")
        self.gaussian_kernel = params.get("blur_kernel", params.get("gaussian_kernel", 5))
        self.bilateral_d = params.get("bilateral_d", 9)
        self.bilateral_sigma_color = params.get("bilateral_sigma_color", 75)
        self.bilateral_sigma_space = params.get("bilateral_sigma_space", 75)

        self.gradient_threshold = params.get("gradient_threshold", 30)
        self.gradient_ksize = params.get("gradient_ksize", 3)
        self.confirmation_n = params.get("confirmation_n", 3)
        self.confirmation_m = params.get("confirmation_m", 5)
        self.intensity_delta = params.get("intensity_delta", 10)
        self.intensity_avg_pixels = params.get("intensity_avg_pixels", 3)
        self.confirmation_brightness_threshold = params.get("confirmation_brightness_threshold", 128)
        self.edge_detection_offset = params.get("edge_detection_offset", 0)

        self.min_edge_length_enabled = params.get("min_edge_length_enabled", True)
        self.min_edge_length = params.get("min_edge_length", 5)
        self.consistency_check_enabled = params.get("consistency_check_enabled", True)
        self.consistency_window = params.get("consistency_window", 3)
        self.consistency_sigma = params.get("consistency_sigma", 3.0)
        self.outlier_removal_enabled = params.get("outlier_removal_enabled", True)
        self.outlier_threshold = params.get("outlier_threshold", 2.0)

        self.section_offset = params.get("section_offset", 2)
        self.bottom_search_depth = params.get("bottom_search_depth", 20)
        self.bottom_scan_half_width = params.get("bottom_scan_half_width", 50)

        self.detect_left = params.get("detect_left", True)
        self.detect_right = params.get("detect_right", True)
        self.detect_bottom = params.get("detect_bottom", True)

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def detect_edges_in_roi(self, image, roi_points, tip_idx, cache=None):
        start_time = time.time()

        roi_image, roi_offset, roi_mask = self._extract_roi(image, roi_points)
        if roi_image is None:
            return self._empty_result(tip_idx, roi_points, time.time() - start_time)

        gray = (cv2.cvtColor(roi_image, cv2.COLOR_BGR2GRAY)
                if len(roi_image.shape) == 3 else roi_image.copy())
        processed = self._apply_preprocessing(gray)
        gradient_x, gradient_y, gradient_mag = self._compute_gradients(processed)

        left_edge_data = right_edge_data = bottom_edge_data = None

        if self.detect_left:
            left_edge_data = self._detect_left_edge_sections(
                processed, gradient_x, gradient_mag, roi_points, roi_offset, roi_mask)
        if self.detect_right:
            right_edge_data = self._detect_right_edge_sections(
                processed, gradient_x, gradient_mag, roi_points, roi_offset, roi_mask)
        if self.detect_bottom:
            start_x = self._estimate_bottom_start_x(
                left_edge_data, right_edge_data,
                roi_offset[0] + processed.shape[1] // 2)
            bottom_edge_data = self._detect_bottom_edge(
                processed, gradient_y, gradient_mag, roi_offset, start_x, roi_mask,
                search_depth=self.bottom_search_depth)

        quality_score, quality_rating = self._compute_quality_score(
            left_edge_data, right_edge_data, bottom_edge_data)

        return {
            "tip_idx": tip_idx,
            "roi": roi_points,
            "roi_offset": roi_offset,
            "left_edge": left_edge_data,
            "right_edge": right_edge_data,
            "bottom_edge": bottom_edge_data,
            "quality_score": quality_score,
            "quality_rating": quality_rating,
            "processing_time": time.time() - start_time,
            "gradient_maps": {
                "gradient_x": gradient_x,
                "gradient_y": gradient_y,
                "gradient_mag": gradient_mag,
            },
        }

    # ------------------------------------------------------------------
    # Section / pair helpers
    # ------------------------------------------------------------------

    def _extract_pairs_from_roi(self, roi_points):
        n = len(roi_points)
        if n < 4 or n % 2 != 0:
            return []
        P = n // 2
        pairs = [{"left": roi_points[0], "right": roi_points[1]}]
        for k in range(1, P - 1):
            pairs.append({"left": roi_points[n - k], "right": roi_points[k + 1]})
        if P >= 2:
            pairs.append({"left": roi_points[P + 1], "right": roi_points[P]})
        pairs.sort(key=lambda p: (p["left"][1] + p["right"][1]) / 2)
        return pairs

    def _detect_left_edge_sections(self, processed, gradient_x, gradient_mag,
                                    roi_points, roi_offset, roi_mask):
        return self._detect_edge_sections(
            processed, gradient_x, gradient_mag,
            roi_points, roi_offset, roi_mask, direction="left")

    def _detect_right_edge_sections(self, processed, gradient_x, gradient_mag,
                                     roi_points, roi_offset, roi_mask):
        return self._detect_edge_sections(
            processed, gradient_x, gradient_mag,
            roi_points, roi_offset, roi_mask, direction="right")

    def _detect_edge_sections(self, processed, gradient_x_or_y, gradient_mag,
                               roi_points, roi_offset, roi_mask, direction):
        pairs = self._extract_pairs_from_roi(roi_points)
        P = len(pairs)
        h_total = processed.shape[0]

        if P < 2:
            if direction == "left":
                return self._detect_left_edge(
                    processed, gradient_x_or_y, gradient_mag, roi_offset, roi_mask)
            return self._detect_right_edge(
                processed, gradient_x_or_y, gradient_mag, roi_offset, roi_mask)

        all_points, all_gradients, all_rejected, sections = [], [], [], []

        for i in range(P - 1):
            p0, p1 = pairs[i], pairs[i + 1]
            y_top_global = (min(p0["left"][1], p0["right"][1],
                                p1["left"][1], p1["right"][1]) - self.section_offset)
            y_bot_global = (max(p0["left"][1], p0["right"][1],
                                p1["left"][1], p1["right"][1]) + self.section_offset)

            y_local_top = max(0, int(y_top_global) - roi_offset[1])
            y_local_bot = min(h_total, int(y_bot_global) - roi_offset[1] + 1)

            if y_local_top >= y_local_bot:
                sections.append(None)
                continue

            sub_img   = processed[y_local_top:y_local_bot, :]
            sub_gxy   = gradient_x_or_y[y_local_top:y_local_bot, :]
            sub_gmag  = gradient_mag[y_local_top:y_local_bot, :]
            sub_mask  = (roi_mask[y_local_top:y_local_bot, :]
                         if roi_mask is not None else None)
            sub_off   = (roi_offset[0], roi_offset[1] + y_local_top)

            fn = self._detect_left_edge if direction == "left" else self._detect_right_edge
            sec_result = fn(sub_img, sub_gxy, sub_gmag, sub_off, sub_mask)
            sec_result["section_y_range"] = (y_local_top, y_local_bot)
            sections.append(sec_result)
            all_points.extend(sec_result["points"])
            all_gradients.extend(sec_result["gradients"])
            all_rejected.extend(sec_result["rejected_candidates"])

        merged = self._compute_edge_metrics(all_points, all_gradients, h_total, all_rejected)
        return {
            "points": all_points,
            "gradients": all_gradients,
            "rejected_candidates": all_rejected,
            "sections": sections,
            **merged,
        }

    # ------------------------------------------------------------------
    # ROI extraction
    # ------------------------------------------------------------------

    def _extract_roi(self, image, roi_points):
        if len(roi_points) < 4:
            return None, (0, 0), None
        xs = [p[0] for p in roi_points]
        ys = [p[1] for p in roi_points]
        h, w = image.shape[:2]
        x_min = max(0, int(min(xs)))
        y_min = max(0, int(min(ys)))
        x_max = min(w, int(max(xs)))
        y_max = min(h, int(max(ys)))
        if x_max <= x_min or y_max <= y_min:
            return None, (0, 0), None

        roi     = image[y_min:y_max, x_min:x_max]
        roi_h   = y_max - y_min
        roi_w   = x_max - x_min
        poly_mask = np.zeros((roi_h, roi_w), dtype=np.uint8)
        local_pts = np.array(
            [[int(p[0]) - x_min, int(p[1]) - y_min] for p in roi_points],
            dtype=np.int32,
        )
        cv2.fillPoly(poly_mask, [local_pts], 255)
        return roi, (x_min, y_min), poly_mask

    # ------------------------------------------------------------------
    # Pre-processing and gradients
    # ------------------------------------------------------------------

    def _apply_preprocessing(self, gray_image):
        return ip.blur(
            gray_image,
            method=self.blur_method,
            kernel_size=self.gaussian_kernel,
            d=self.bilateral_d,
            sigma_color=self.bilateral_sigma_color,
            sigma_space=self.bilateral_sigma_space,
        )

    def _compute_gradients(self, image):
        k = self.gradient_ksize if self.gradient_ksize % 2 == 1 else self.gradient_ksize + 1
        gradient_x   = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=k)
        gradient_y   = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=k)
        gradient_mag = np.sqrt(gradient_x ** 2 + gradient_y ** 2)
        return gradient_x, gradient_y, gradient_mag

    # ------------------------------------------------------------------
    # Per-edge scanners
    # ------------------------------------------------------------------

    def _detect_left_edge(self, image, gradient_x, gradient_mag, roi_offset, roi_mask=None):
        h, w = image.shape
        edge_points, gradients, rejected = [], [], []
        for y in range(h):
            col_start, col_end = self._row_bounds(roi_mask, y, w, rejected)
            if col_start is None:
                continue
            edge_x = self._scan_row_for_edge(
                image[y, col_start:col_end],
                gradient_x[y, col_start:col_end],
                gradient_mag[y, col_start:col_end],
                direction="left_to_right",
            )
            if edge_x is not None:
                edge_x += col_start
                edge_points.append((edge_x + roi_offset[0], y + roi_offset[1]))
                gradients.append(float(gradient_mag[y, edge_x]))
            else:
                rejected.append((y, "no_edge_found"))

        if edge_points:
            edge_points, gradients = self._apply_postprocessing(edge_points, gradients, "vertical")
        return {"points": edge_points, "gradients": gradients, "rejected_candidates": rejected,
                **self._compute_edge_metrics(edge_points, gradients, h, rejected)}

    def _detect_right_edge(self, image, gradient_x, gradient_mag, roi_offset, roi_mask=None):
        h, w = image.shape
        edge_points, gradients, rejected = [], [], []
        for y in range(h):
            col_start, col_end = self._row_bounds(roi_mask, y, w, rejected)
            if col_start is None:
                continue
            edge_x = self._scan_row_for_edge(
                image[y, col_start:col_end],
                gradient_x[y, col_start:col_end],
                gradient_mag[y, col_start:col_end],
                direction="right_to_left",
            )
            if edge_x is not None:
                edge_x += col_start
                edge_points.append((edge_x + roi_offset[0], y + roi_offset[1]))
                gradients.append(float(gradient_mag[y, edge_x]))
            else:
                rejected.append((y, "no_edge_found"))

        if edge_points:
            edge_points, gradients = self._apply_postprocessing(edge_points, gradients, "vertical")
        return {"points": edge_points, "gradients": gradients, "rejected_candidates": rejected,
                **self._compute_edge_metrics(edge_points, gradients, h, rejected)}

    def _row_bounds(self, roi_mask, y, w, rejected):
        """Return (col_start, col_end) for a row, or (None, None) if row is empty."""
        if roi_mask is not None:
            valid_cols = np.where(roi_mask[y, :] > 0)[0]
            if len(valid_cols) == 0:
                rejected.append((y, "outside_roi"))
                return None, None
            return int(valid_cols[0]), int(valid_cols[-1]) + 1
        return 0, w

    def _detect_bottom_edge(self, image, gradient_y, gradient_mag, roi_offset, start_x,
                             roi_mask=None, search_depth=0):
        h, w = image.shape
        edge_points, gradients, rejected = [], [], []

        local_cx  = int(round(start_x - roi_offset[0]))
        hw        = self.bottom_scan_half_width
        col_left  = max(0, local_cx - hw)
        col_right = min(w - 1, local_cx + hw)

        if roi_mask is not None:
            row_has_pixels = roi_mask.any(axis=1)
            if not row_has_pixels.any():
                return self._empty_edge_result()
            roi_bottom = int(np.max(np.where(row_has_pixels)[0]))
            row_top    = max(0, roi_bottom - search_depth + 1) if search_depth > 0 else 0
            rect_mask  = np.zeros((h, w), dtype=np.uint8)
            rect_mask[row_top:roi_bottom + 1, col_left:col_right + 1] = 255
            search_mask = np.minimum(roi_mask, rect_mask)
        else:
            row_top     = max(0, h - search_depth) if search_depth > 0 else 0
            search_mask = np.zeros((h, w), dtype=np.uint8)
            search_mask[row_top:h, col_left:col_right + 1] = 255

        x_scan_range = np.where(search_mask.any(axis=0))[0]
        if len(x_scan_range) == 0:
            return self._empty_edge_result()

        for x in x_scan_range:
            valid_rows = np.where(search_mask[:, x] > 0)[0]
            if len(valid_rows) == 0:
                rejected.append((x, "outside_roi"))
                continue
            row_start, row_end = int(valid_rows[0]), int(valid_rows[-1]) + 1
            edge_y = self._scan_column_for_edge(
                image[row_start:row_end, x],
                gradient_y[row_start:row_end, x],
                gradient_mag[row_start:row_end, x],
                direction="bottom_to_top",
            )
            if edge_y is not None:
                edge_y += row_start
                edge_points.append((x + roi_offset[0], edge_y + roi_offset[1]))
                gradients.append(float(gradient_mag[edge_y, x]))
            else:
                rejected.append((x, "no_edge_found"))

        if edge_points:
            edge_points, gradients = self._apply_postprocessing(edge_points, gradients, "horizontal")
        return {"points": edge_points, "gradients": gradients, "rejected_candidates": rejected,
                **self._compute_edge_metrics(edge_points, gradients, len(x_scan_range), rejected)}

    # ------------------------------------------------------------------
    # Row / column scanners (anchor-first N-of-M + intensity verification)
    # ------------------------------------------------------------------

    def _scan_row_for_edge(self, row_pixels, row_gradient_x, row_gradient_mag,
                            direction="left_to_right"):
        N = self.confirmation_n
        M = self.confirmation_m
        K = self.intensity_avg_pixels
        B = self.confirmation_brightness_threshold
        n = len(row_pixels)

        if direction == "left_to_right":
            if n < M + 2 * K:
                return None
            for x in range(K, n - M - K + 1):
                if row_gradient_mag[x] < self.gradient_threshold:
                    continue
                if np.sum(row_pixels[x + 1:x + M] >= B) < N - 1:
                    continue
                if K > 0:
                    if abs(float(np.mean(row_pixels[x + M:x + M + K])) -
                           float(np.mean(row_pixels[x - K:x]))) < self.intensity_delta:
                        continue
                return max(0, min(n - 1, x + self.edge_detection_offset))
        else:
            if n < M + 2 * K:
                return None
            for x in range(n - K - 1, M + K - 2, -1):
                if row_gradient_mag[x] < self.gradient_threshold:
                    continue
                if np.sum(row_pixels[x - M + 1:x] >= B) < N - 1:
                    continue
                if K > 0:
                    if abs(float(np.mean(row_pixels[x + 1:x + 1 + K])) -
                           float(np.mean(row_pixels[x - M + 1 - K:x - M + 1]))) < self.intensity_delta:
                        continue
                return max(0, min(n - 1, x - self.edge_detection_offset))
        return None

    def _scan_column_for_edge(self, col_pixels, col_gradient_y, col_gradient_mag,
                               direction="bottom_to_top"):
        N = self.confirmation_n
        M = self.confirmation_m
        K = self.intensity_avg_pixels
        B = self.confirmation_brightness_threshold
        n = len(col_pixels)

        if direction == "bottom_to_top":
            if n < M + 2 * K:
                return None
            for y in range(n - K - 1, M + K - 2, -1):
                if col_gradient_mag[y] < self.gradient_threshold:
                    continue
                if np.sum(col_pixels[y - M + 1:y] >= B) < N - 1:
                    continue
                if K > 0:
                    if abs(float(np.mean(col_pixels[y + 1:y + 1 + K])) -
                           float(np.mean(col_pixels[y - M + 1 - K:y - M + 1]))) < self.intensity_delta:
                        continue
                return max(0, min(n - 1, y - self.edge_detection_offset))
        else:
            if n < M + 2 * K:
                return None
            for y in range(K, n - M - K + 1):
                if col_gradient_mag[y] < self.gradient_threshold:
                    continue
                if np.sum(col_pixels[y + 1:y + M] >= B) < N - 1:
                    continue
                if K > 0:
                    if abs(float(np.mean(col_pixels[y + M:y + M + K])) -
                           float(np.mean(col_pixels[y - K:y]))) < self.intensity_delta:
                        continue
                return max(0, min(n - 1, y + self.edge_detection_offset))
        return None

    # ------------------------------------------------------------------
    # Post-processing
    # ------------------------------------------------------------------

    def _apply_postprocessing(self, edge_points, gradients, edge_type):
        if len(edge_points) < 3:
            return edge_points, gradients
        points = np.array(edge_points)
        grads  = np.array(gradients)
        if self.min_edge_length_enabled and len(points) < self.min_edge_length:
            return [], []
        if self.consistency_check_enabled and len(points) >= self.consistency_window:
            points, grads = self._apply_consistency_filter(points, grads, edge_type)
        if self.outlier_removal_enabled and len(points) > 5:
            points, grads = self._remove_outliers(points, grads, edge_type)
        return points.tolist(), grads.tolist()

    def _apply_consistency_filter(self, points, gradients, edge_type):
        if len(points) < self.consistency_window:
            return points, gradients
        coord_idx = 0 if edge_type == "vertical" else 1
        mask = np.ones(len(points), dtype=bool)
        for i in range(len(points)):
            ws = max(0, i - self.consistency_window // 2)
            we = min(len(points), i + self.consistency_window // 2 + 1)
            if we - ws < 3:
                continue
            window = points[ws:we, coord_idx]
            std_c  = np.std(window)
            if std_c > 0:
                if (abs(points[i, coord_idx] - np.median(window)) /
                        max(std_c, 1.0)) > self.consistency_sigma:
                    mask[i] = False
        return points[mask], gradients[mask]

    def _remove_outliers(self, points, gradients, edge_type):
        if len(points) < 3:
            return points, gradients
        indep_idx, dep_idx = (1, 0) if edge_type == "vertical" else (0, 1)
        indep = points[:, indep_idx].astype(float)
        dep   = points[:, dep_idx].astype(float)
        if np.std(dep) < 1e-6:
            return points, gradients
        try:
            predicted = np.polyval(np.polyfit(indep, dep, 1), indep)
        except (np.linalg.LinAlgError, ValueError):
            return points, gradients
        residuals = np.abs(dep - predicted)
        mask = residuals <= self.outlier_threshold * max(np.std(residuals), 1.0)
        return points[mask], gradients[mask]

    # ------------------------------------------------------------------
    # Metrics and helpers
    # ------------------------------------------------------------------

    def _compute_edge_metrics(self, edge_points, gradients, total_possible, rejected):
        if not edge_points:
            return {"coverage": 0.0, "mean_gradient": 0.0, "gradient_std": 0.0,
                    "position_std": 0.0, "outliers_removed": 0, "confirmation_rate": 0.0}
        pts  = np.array(edge_points)
        grds = np.array(gradients)
        return {
            "coverage": float(len(edge_points) / max(1, total_possible)),
            "mean_gradient": float(np.mean(grds)),
            "gradient_std": float(np.std(grds)),
            "position_std": float(np.std(pts[:, 0]) if len(pts) > 1 else 0.0),
            "outliers_removed": len(rejected),
            "confirmation_rate": float(
                len(edge_points) / max(1, len(edge_points) + len(rejected))),
        }

    def _estimate_bottom_start_x(self, left_edge_data, right_edge_data, fallback_x):
        if left_edge_data and right_edge_data:
            lpts = left_edge_data.get("points", [])
            rpts = right_edge_data.get("points", [])
            if lpts and rpts:
                return int(
                    (np.median([p[0] for p in lpts]) + np.median([p[0] for p in rpts])) / 2)
        return fallback_x

    def _compute_quality_score(self, left_edge, right_edge, bottom_edge):
        scores = []
        for edge in (left_edge, right_edge, bottom_edge):
            if edge and edge.get("coverage", 0) > 0:
                scores.append(
                    edge["coverage"] * 40
                    + min(40, edge["mean_gradient"] / 2.55)
                    + max(0, 20 - edge["position_std"])
                )
        avg = float(np.mean(scores)) if scores else 0.0
        rating = ("Excellent" if avg >= 90 else "Good" if avg >= 75
                  else "Fair" if avg >= 50 else "Poor" if avg >= 25 else "Failed")
        return avg, rating

    def _empty_result(self, tip_idx, roi_points, processing_time):
        return {"tip_idx": tip_idx, "roi": roi_points, "roi_offset": (0, 0),
                "left_edge": None, "right_edge": None, "bottom_edge": None,
                "quality_score": 0.0, "quality_rating": "Failed",
                "processing_time": processing_time, "gradient_maps": None}

    def _empty_edge_result(self):
        return {"points": [], "gradients": [], "rejected_candidates": [],
                "coverage": 0.0, "mean_gradient": 0.0, "gradient_std": 0.0,
                "position_std": 0.0, "outliers_removed": 0, "confirmation_rate": 0.0}
