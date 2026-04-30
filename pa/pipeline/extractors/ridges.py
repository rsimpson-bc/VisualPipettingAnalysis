"""
Layer 1 — Ridge Extractor.

Detects near-vertical structural ridges (lines) in a liquid-column ROI.
Produces four independent 1-D signals that the LineContinuity Layer-2 modes
each read independently:

  ridge_count_signal      — per-row count of ridge-feature starts (0→255 transitions)
  termination_signal      — row-to-row change in ridge count (optionally window-smoothed)
  ridge_spacing_signal    — normalised pixel-count difference between adjacent bands
                            (captures pattern changes at the liquid interface)
  correlation_signal      — 1 − band correlation (low = structural discontinuity)
  line_density_signal     — normalised vertical-line-count difference between bands
  edge_map                — 2-D binary ridge mask (H×W, uint8 0/255)

Algorithm origin: _analyze_tip_line_continuity() in
  VS_Code_Test/VideoDeltaHeatMap1.py, steps 1–6.
Extracted here so all four LineContinuity modes share a single cached computation.
"""

from __future__ import annotations

import cv2
import numpy as np

from pa.pipeline.types import ImageSet, RawFeatures
from pa.pipeline.cache import ProcessedImageCache
from pa.pipeline.extractors.base import BaseExtractor
from pa.pipeline import image_primitives as ip


class RidgeExtractor(BaseExtractor):
    """
    Params (analysis_config.json):
        blur_kernel (int, default 5):           Gaussian blur kernel size
        gradient_direction (str, default "xy"): accepted for API compat; always XY
        angle_tolerance (float, default 30):    ±degrees from horizontal accepted as
                                                vertical-line gradient
        ridge_threshold (float, default 30):    minimum gradient magnitude to flag ridge
        morph_kernel_height (int, default 5):   structuring element height (rows)
        morph_kernel_width  (int, default 1):   structuring element width  (cols)
        morph_iterations (int, default 1):      iterations for close/open
        morph_close_enabled (bool, default True)
        morph_open_enabled  (bool, default True)
        band_height (int, default 10):          rows in each comparison band for the
                                                pattern/correlation/density signals;
                                                falls back to `correlation_window` if set
        termination_window (int, default 0):    ±window rows for aggregating terminations
        use_ab (bool, default False)
    """

    def _extract(self, image_set: ImageSet, cache: ProcessedImageCache) -> RawFeatures:
        p = self.params

        # ── 1. Working image ──────────────────────────────────────────────────
        use_ab = p.get("use_ab", False)
        if use_ab and len(image_set.frames) >= 2:
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

        # ── 2. ROI crop ───────────────────────────────────────────────────────
        gray8 = ip.extract_roi(gray8, image_set.roi) if image_set.roi else gray8

        # ── 3. Blur ───────────────────────────────────────────────────────────
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

        # ── 4. Gradients ──────────────────────────────────────────────────────
        grad_method = p.get("gradient_method", p.get("gradient_direction", "xy"))
        grad_ksize  = p.get("gradient_ksize", 3)
        grad_ksize  = grad_ksize if grad_ksize % 2 == 1 else grad_ksize + 1

        grad_x = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=grad_ksize)
        grad_y = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=grad_ksize)

        if grad_method == "xy":
            magnitude = np.sqrt(grad_x ** 2 + grad_y ** 2)
        elif grad_method == "x":
            magnitude = np.abs(grad_x)
        elif grad_method == "y":
            magnitude = np.abs(grad_y)
        else:  # "none" — use intensity directly
            magnitude = blurred.astype(np.float64)
        gradient_angle = np.arctan2(grad_y, grad_x) * 180.0 / np.pi  # -180 .. +180

        # ── 5. Orientation filter — keep near-vertical lines ──────────────────
        # Vertical lines have *horizontal* gradients (≈ 0° or ±180°).
        angle_tol = p.get("angle_tolerance", 30.0)
        if grad_method == "xy":
            gradient_angle = np.arctan2(grad_y, grad_x) * 180.0 / np.pi  # -180 .. +180
            h_mask = (
                (np.abs(gradient_angle)           <= angle_tol) |
                (np.abs(gradient_angle - 180.0)   <= angle_tol) |
                (np.abs(gradient_angle + 180.0)   <= angle_tol)
            )
            vertical_magnitude = magnitude * h_mask
        else:
            # For single-direction gradients or intensity, skip angle filtering
            vertical_magnitude = magnitude

        # ── 6. Threshold → binary ridge mask ─────────────────────────────────
        ridge_threshold = p.get("ridge_threshold", 30.0)
        ridge_mask_raw = (vertical_magnitude > ridge_threshold).astype(np.uint8) * 255

        # ── 7. Morphological operations ───────────────────────────────────────
        ridge_mask = ridge_mask_raw.copy()
        kh = max(1, p.get("morph_kernel_height", 5))
        kw = max(1, p.get("morph_kernel_width", 1))
        n_iter = p.get("morph_iterations", 1)
        do_close = p.get("morph_close_enabled", True)
        do_open  = p.get("morph_open_enabled", True)

        if n_iter > 0 and (do_close or do_open):
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kw, kh))
            if do_close:
                ridge_mask = cv2.morphologyEx(ridge_mask, cv2.MORPH_CLOSE, kernel,
                                              iterations=n_iter)
            if do_open:
                ridge_mask = cv2.morphologyEx(ridge_mask, cv2.MORPH_OPEN, kernel,
                                              iterations=n_iter)

        # ── 8. Compute all four 1-D signals ───────────────────────────────────
        height = ridge_mask.shape[0]
        band_h = p.get("band_height", p.get("correlation_window", 10))
        term_win = p.get("termination_window", 0)

        z_axis             = np.arange(height)
        ridge_count_signal = _compute_feature_counts(ridge_mask)
        termination_signal = _compute_termination_scores(ridge_count_signal, term_win)
        ridge_spacing_signal = _compute_pattern_scores(ridge_mask, height, band_h)
        correlation_signal = _compute_correlation_scores(ridge_mask, height, band_h)
        line_density_signal = _compute_density_scores(ridge_mask, height, band_h)

        return RawFeatures(
            mode="RidgeExtractor",
            pipette_index=image_set.pipette_index,
            z_axis_px=z_axis,
            ridge_count_signal=ridge_count_signal,
            termination_signal=termination_signal,
            ridge_spacing_signal=ridge_spacing_signal,   # pattern change signal
            correlation_signal=correlation_signal,
            line_density_signal=line_density_signal,
            edge_map=ridge_mask,
        )


# ---------------------------------------------------------------------------
# Signal computation helpers
# ---------------------------------------------------------------------------

def _compute_feature_counts(ridge_mask: np.ndarray) -> np.ndarray:
    """Per-row count of 0→255 transitions (ridge feature starts)."""
    height = ridge_mask.shape[0]
    counts = np.zeros(height, dtype=np.float32)
    for y in range(height):
        row = ridge_mask[y, :]
        counts[y] = float(np.sum((row[1:] == 255) & (row[:-1] == 0)))
    return counts


def _compute_termination_scores(feature_counts: np.ndarray, window: int) -> np.ndarray:
    """
    Row-to-row absolute change in feature count, optionally aggregated over ±window.
    Normalised to [0, 1].
    """
    height = len(feature_counts)
    raw = np.zeros(height, dtype=np.float32)
    for y in range(1, height):
        raw[y] = abs(feature_counts[y] - feature_counts[y - 1])

    if window > 0:
        kern = np.ones(2 * window + 1, dtype=np.float32)
        raw = np.convolve(raw, kern, mode="same").astype(np.float32)

    mx = raw.max()
    return raw / mx if mx > 0 else raw


def _compute_pattern_scores(ridge_mask: np.ndarray, height: int, band_h: int) -> np.ndarray:
    """
    Normalised pixel-count difference between the upper and lower band around each row.
    Measures pattern changes (density of ridges above vs below).
    """
    scores = np.zeros(height, dtype=np.float32)
    for y in range(band_h, height):
        upper_px = float(np.sum(ridge_mask[y - band_h:y] > 0))
        lower_px = float(np.sum(ridge_mask[y:min(y + band_h, height)] > 0))
        total = upper_px + lower_px
        if total > 0:
            scores[y] = abs(upper_px - lower_px) / total
    mx = scores.max()
    return scores / mx if mx > 0 else scores


def _compute_correlation_scores(ridge_mask: np.ndarray, height: int,
                                 band_h: int) -> np.ndarray:
    """
    1 − Pearson correlation between flattened upper and lower bands.
    High value (low correlation) indicates a structural discontinuity.
    """
    scores = np.zeros(height, dtype=np.float32)
    for y in range(band_h, height - band_h):
        upper = ridge_mask[y - band_h:y].flatten().astype(np.float64)
        lower = ridge_mask[y:y + band_h].flatten().astype(np.float64)
        if upper.std() > 0 and lower.std() > 0:
            corr = float(np.corrcoef(upper, lower)[0, 1])
            scores[y] = 0.0 if np.isnan(corr) else max(0.0, 1.0 - corr)
    mx = scores.max()
    return (scores / mx).astype(np.float32) if mx > 0 else scores


def _compute_density_scores(ridge_mask: np.ndarray, height: int,
                             band_h: int) -> np.ndarray:
    """
    Normalised difference in vertical-line-segment count between adjacent bands.
    A vertical line start is a 0→255 transition going *down* a column.
    """
    scores = np.zeros(height, dtype=np.float32)

    def _count_vertical_starts(band: np.ndarray) -> float:
        count = 0
        for x in range(band.shape[1]):
            col = band[:, x]
            count += int(np.sum((col[1:] == 255) & (col[:-1] == 0)))
        return float(count)

    for y in range(band_h, height - band_h):
        upper_n = _count_vertical_starts(ridge_mask[y - band_h:y])
        lower_n = _count_vertical_starts(ridge_mask[y:y + band_h])
        total = upper_n + lower_n
        if total > 0:
            scores[y] = abs(upper_n - lower_n) / total
    mx = scores.max()
    return (scores / mx).astype(np.float32) if mx > 0 else scores
