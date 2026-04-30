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


def _get_ridge_features(params, image_set, cache):
    """Shared helper — runs RidgeExtractor once per step (cache hit on repeat calls)."""
    return RidgeExtractor(params).extract(_apply_contrast(image_set, params), cache)


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
            signal=f.termination_signal,
        )

    def run_debug(self, image_set, cache):
        f = _get_ridge_features(self.params, image_set, cache)
        profile = ZProfile(mode="LineContinuity_Terminations",
                           pipette_index=image_set.pipette_index,
                           z_axis_px=f.z_axis_px, signal=f.termination_signal)
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
            signal=f.ridge_spacing_signal,
        )

    def run_debug(self, image_set, cache):
        f = _get_ridge_features(self.params, image_set, cache)
        profile = ZProfile(mode="LineContinuity_PatternChange",
                           pipette_index=image_set.pipette_index,
                           z_axis_px=f.z_axis_px, signal=f.ridge_spacing_signal)
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
            signal=f.correlation_signal,
        )

    def run_debug(self, image_set, cache):
        f = _get_ridge_features(self.params, image_set, cache)
        profile = ZProfile(mode="LineContinuity_Correlation",
                           pipette_index=image_set.pipette_index,
                           z_axis_px=f.z_axis_px, signal=f.correlation_signal)
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
            signal=f.line_density_signal,
        )

    def run_debug(self, image_set, cache):
        f = _get_ridge_features(self.params, image_set, cache)
        profile = ZProfile(mode="LineContinuity_Density",
                           pipette_index=image_set.pipette_index,
                           z_axis_px=f.z_axis_px, signal=f.line_density_signal)
        return profile, f
