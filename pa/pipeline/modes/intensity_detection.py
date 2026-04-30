"""
Layer 2 — Intensity Detection mode (liquid).

Uses accumulated A:B frame differences to build a per-row intensity signal.
High-intensity rows indicate regions of change (liquid presence/transitions).
"""

from __future__ import annotations
from typing import Optional, Tuple
import numpy as np
from pa.pipeline.types import ImageSet, ZProfile, RawFeatures
from pa.pipeline.cache import ProcessedImageCache
from pa.pipeline.modes.base import BaseLiquidMode
from pa.pipeline.extractors.intensity import IntensityExtractor
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


def _apply_roi_expansion(image_set: ImageSet, params: dict) -> ImageSet:
    """Return a new ImageSet whose roi/roi_points are expanded outward by
    ``roi_expansion_px``.  Original points stay in roi_points only when no
    expansion is configured; otherwise the expanded polygon replaces them
    so the extractor analyses the wider region.

    If no expansion is requested, or if roi_points is missing, the input
    image_set is returned unchanged.
    """
    expansion = int(params.get("roi_expansion_px", 0) or 0)
    if expansion <= 0 or not image_set.roi_points or not image_set.frames:
        return image_set

    img_h, img_w = image_set.frames[0].shape[:2]
    expanded_pts = ip.expand_roi_points(
        image_set.roi_points, expansion, img_h, img_w,
    )
    # Derive the bounding box of the expanded polygon, clamped to the image.
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


class IntensityDetection(BaseLiquidMode):
    """
    Params (passed to IntensityExtractor + local):
        blur_method (str):       default "gaussian"
        blur_kernel (int):       default 5
        use_ab (bool):           default True
        roi_expansion_px (int):  default 0 — when > 0, ROI1 is expanded
                                 outward by this many pixels on every side
                                 (ROI3) and the per-row signal is computed
                                 over the expanded region.
    """

    def run(self, image_set: ImageSet, cache: ProcessedImageCache) -> ZProfile:
        prepared = _apply_roi_expansion(
            _apply_contrast(image_set, self.params), self.params
        )
        features = IntensityExtractor(self.params).extract(prepared, cache)
        return ZProfile(
            mode="IntensityDetection",
            pipette_index=image_set.pipette_index,
            z_axis_px=features.z_axis_px,
            signal=features.intensity_signal,
        )

    def run_debug(self, image_set, cache):
        prepared = _apply_roi_expansion(
            _apply_contrast(image_set, self.params), self.params
        )
        features = IntensityExtractor(self.params).extract(prepared, cache)
        profile = ZProfile(mode="IntensityDetection",
                           pipette_index=image_set.pipette_index,
                           z_axis_px=features.z_axis_px, signal=features.intensity_signal)
        return profile, features
