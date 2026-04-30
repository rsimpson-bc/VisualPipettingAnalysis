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


class IntensityDetection(BaseLiquidMode):
    """
    Params (passed to IntensityExtractor + local):
        blur_method (str):    default "gaussian"
        blur_kernel (int):    default 5
        use_ab (bool):        default True
    """

    def run(self, image_set: ImageSet, cache: ProcessedImageCache) -> ZProfile:
        features = IntensityExtractor(self.params).extract(
            _apply_contrast(image_set, self.params), cache
        )
        return ZProfile(
            mode="IntensityDetection",
            pipette_index=image_set.pipette_index,
            z_axis_px=features.z_axis_px,
            signal=features.intensity_signal,
        )

    def run_debug(self, image_set, cache):
        features = IntensityExtractor(self.params).extract(
            _apply_contrast(image_set, self.params), cache
        )
        profile = ZProfile(mode="IntensityDetection",
                           pipette_index=image_set.pipette_index,
                           z_axis_px=features.z_axis_px, signal=features.intensity_signal)
        return profile, features
