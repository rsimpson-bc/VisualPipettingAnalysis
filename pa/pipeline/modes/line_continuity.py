"""
Layer 2 — Line Continuity modes (liquid).

Four independent modes, each drawing one signal from RidgeExtractor output.
They share the same extractor call (cached) but produce separate ZProfiles
so they can be weighted independently by the integrator.
"""

from __future__ import annotations
from typing import Optional, Tuple
import numpy as np
from pa.pipeline.types import ImageSet, ZProfile, RawFeatures
from pa.pipeline.cache import ProcessedImageCache
from pa.pipeline.modes.base import BaseLiquidMode
from pa.pipeline.extractors.ridges import RidgeExtractor
from pa.pipeline import image_primitives as ip
from pa.pipeline import signal_primitives as sp


def _apply_contrast(image_set: ImageSet, params: dict) -> ImageSet:
    """Return a new ImageSet with frames[0] replaced by the contrast image."""
    if not params.get("use_contrast", False) or not image_set.reference_frames:
        return image_set
    ref = image_set.reference_frames[0]
    contrast_bgr = ip.build_contrast_working_frame(image_set.frames[0], ref, params)
    return ImageSet(
        pipette_index=image_set.pipette_index,
        frames=[contrast_bgr] + list(image_set.frames[1:]),
        source_paths=image_set.source_paths,
        roi=image_set.roi,
        roi_points=image_set.roi_points,
    )


def _apply_roi_expansion(image_set: ImageSet, params: dict) -> ImageSet:
    """Expand the ROI polygon outward by ``roi_expansion_px`` pixels.

    Returns the input image_set unchanged when expansion == 0 or roi_points
    is missing.  Otherwise returns a new ImageSet with both ``roi`` (the
    expanded bounding box) and ``roi_points`` (the expanded polygon) updated.
    """
    expansion = int(params.get("roi_expansion_px", 0) or 0)
    if expansion == 0 or not image_set.roi_points or not image_set.frames:
        return image_set

    img_h, img_w = image_set.frames[0].shape[:2]
    expanded_pts = ip.expand_roi_points(image_set.roi_points, expansion, img_h, img_w)
    xs = [p[0] for p in expanded_pts]
    ys = [p[1] for p in expanded_pts]
    x0 = max(0, int(min(xs)))
    y0 = max(0, int(min(ys)))
    x1 = min(img_w, int(max(xs)) + 1)
    y1 = min(img_h, int(max(ys)) + 1)
    expanded_bbox = (x0, y0, max(1, x1 - x0), max(1, y1 - y0))

    return ImageSet(
        pipette_index=image_set.pipette_index,
        frames=image_set.frames,
        source_paths=image_set.source_paths,
        roi=expanded_bbox,
        roi_points=expanded_pts,
        reference_frames=image_set.reference_frames,
        reference_source_paths=image_set.reference_source_paths,
    )


def _get_ridge_features(params, image_set, cache):
    """Shared helper — runs RidgeExtractor once per step (cache hit on repeat calls)."""
    prepared = _apply_roi_expansion(_apply_contrast(image_set, params), params)
    return RidgeExtractor(params).extract(prepared, cache)


def _mask(signal: np.ndarray, params: dict) -> np.ndarray:
    """Zero out ignore_top_rows / ignore_bottom_rows entries of a signal."""
    return sp.mask_signal_edges(
        signal,
        ignore_top=int(params.get("ignore_top_rows", 0) or 0),
        ignore_bottom=int(params.get("ignore_bottom_rows", 0) or 0),
    )


class LineContinuityTerminations(BaseLiquidMode):
    """
    Signal: per-row count of ridge terminations.
    A sudden increase in terminations signals a liquid transition.
    """
    def run(self, image_set: ImageSet, cache: ProcessedImageCache) -> ZProfile:
        f = _get_ridge_features(self.params, image_set, cache)
        return ZProfile(
            mode="LineContinuity_Terminations",
            pipette_index=image_set.pipette_index,
            z_axis_px=f.z_axis_px,
            signal=_mask(f.termination_signal, self.params),
        )

    def run_debug(self, image_set, cache):
        f = _get_ridge_features(self.params, image_set, cache)
        profile = ZProfile(mode="LineContinuity_Terminations",
                           pipette_index=image_set.pipette_index,
                           z_axis_px=f.z_axis_px,
                           signal=_mask(f.termination_signal, self.params))
        return profile, f


class LineContinuityPatternChange(BaseLiquidMode):
    """
    Signal: per-row ridge count and average spacing combined.
    Pattern discontinuities signal a liquid transition.
    """
    def run(self, image_set: ImageSet, cache: ProcessedImageCache) -> ZProfile:
        f = _get_ridge_features(self.params, image_set, cache)
        return ZProfile(
            mode="LineContinuity_PatternChange",
            pipette_index=image_set.pipette_index,
            z_axis_px=f.z_axis_px,
            signal=_mask(f.ridge_spacing_signal, self.params),
        )

    def run_debug(self, image_set, cache):
        f = _get_ridge_features(self.params, image_set, cache)
        profile = ZProfile(mode="LineContinuity_PatternChange",
                           pipette_index=image_set.pipette_index,
                           z_axis_px=f.z_axis_px,
                           signal=_mask(f.ridge_spacing_signal, self.params))
        return profile, f


class LineContinuityCorrelation(BaseLiquidMode):
    """
    Signal: cross-row correlation between adjacent ridges.
    Low correlation across a gap signals a liquid transition.
    """
    def run(self, image_set: ImageSet, cache: ProcessedImageCache) -> ZProfile:
        f = _get_ridge_features(self.params, image_set, cache)
        return ZProfile(
            mode="LineContinuity_Correlation",
            pipette_index=image_set.pipette_index,
            z_axis_px=f.z_axis_px,
            signal=_mask(f.correlation_signal, self.params),
        )

    def run_debug(self, image_set, cache):
        f = _get_ridge_features(self.params, image_set, cache)
        profile = ZProfile(mode="LineContinuity_Correlation",
                           pipette_index=image_set.pipette_index,
                           z_axis_px=f.z_axis_px,
                           signal=_mask(f.correlation_signal, self.params))
        return profile, f


class LineContinuityDensity(BaseLiquidMode):
    """
    Signal: per-row ridge density (ridges per unit height).
    Low density regions indicate potential liquid boundaries.
    """
    def run(self, image_set: ImageSet, cache: ProcessedImageCache) -> ZProfile:
        f = _get_ridge_features(self.params, image_set, cache)
        return ZProfile(
            mode="LineContinuity_Density",
            pipette_index=image_set.pipette_index,
            z_axis_px=f.z_axis_px,
            signal=_mask(f.line_density_signal, self.params),
        )

    def run_debug(self, image_set, cache):
        f = _get_ridge_features(self.params, image_set, cache)
        profile = ZProfile(mode="LineContinuity_Density",
                           pipette_index=image_set.pipette_index,
                           z_axis_px=f.z_axis_px,
                           signal=_mask(f.line_density_signal, self.params))
        return profile, f
