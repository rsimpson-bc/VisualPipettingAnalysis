"""
Layer 2 — Row Contrast Detection mode (liquid).

Computes a per-row standard deviation (or variance) signal within the ROI.
Rows with high cross-sectional contrast — such as a liquid meniscus or tip
edge transition — produce a strong peak in the signal.

This is complementary to IntensityDetection (which uses per-row *mean*):
- IntensityDetection highlights rows that are globally bright/dark.
- RowContrastDetection highlights rows where there is a sharp local gradient
  across the width, even when absolute brightness is unremarkable.
"""

from __future__ import annotations
from typing import Optional, Tuple
import numpy as np
from pa.pipeline.types import ImageSet, ZProfile, RawFeatures
from pa.pipeline.cache import ProcessedImageCache
from pa.pipeline.modes.base import BaseLiquidMode
from pa.pipeline.extractors.row_contrast import RowContrastExtractor
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
    """Expand the ROI polygon outward by ``roi_expansion_px``."""
    expansion = int(params.get("roi_expansion_px", 0) or 0)
    if expansion == 0 or not image_set.roi_points or not image_set.frames:
        return image_set

    img_h, img_w = image_set.frames[0].shape[:2]
    expanded_pts = ip.expand_roi_points(
        image_set.roi_points, expansion, img_h, img_w,
    )
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


class RowContrastDetection(BaseLiquidMode):
    """Liquid detection using per-row cross-sectional std/variance."""

    def run(self, image_set: ImageSet, cache: ProcessedImageCache) -> ZProfile:
        prepared = _apply_roi_expansion(
            _apply_contrast(image_set, self.params), self.params
        )
        features = RowContrastExtractor(self.params).extract(prepared, cache)
        _sig = sp.mask_signal_edges(
            features.intensity_signal,
            ignore_top=int(self.params.get("ignore_top_rows", 0) or 0),
            ignore_bottom=int(self.params.get("ignore_bottom_rows", 0) or 0),
        )
        return ZProfile(
            mode="RowContrastDetection",
            pipette_index=image_set.pipette_index,
            z_axis_px=features.z_axis_px,
            signal=_sig,
        )

    def run_debug(self, image_set: ImageSet, cache: ProcessedImageCache):
        prepared = _apply_roi_expansion(
            _apply_contrast(image_set, self.params), self.params
        )
        features = RowContrastExtractor(self.params).extract(prepared, cache)
        _sig = sp.mask_signal_edges(
            features.intensity_signal,
            ignore_top=int(self.params.get("ignore_top_rows", 0) or 0),
            ignore_bottom=int(self.params.get("ignore_bottom_rows", 0) or 0),
        )
        profile = ZProfile(
            mode="RowContrastDetection",
            pipette_index=image_set.pipette_index,
            z_axis_px=features.z_axis_px,
            signal=_sig,
        )
        return profile, features
