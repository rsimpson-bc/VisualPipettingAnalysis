"""
Layer 2 — Extrema Lines mode (liquid).

Stitches per-row extrema into lines, then produces two detrended signals:
  - detrended line density  (look for local minima → sudden dips)
  - detrended termination density (look for local maxima → sudden spikes)

Both are detrended to account for the natural increase in line count as Z
increases (tip gets wider).
"""

from __future__ import annotations
from typing import Tuple
import numpy as np
from pa.pipeline.types import ImageSet, ZProfile, RawFeatures
from pa.pipeline.cache import ProcessedImageCache
from pa.pipeline.modes.base import BaseLiquidMode
from pa.pipeline.extractors.extrema_lines import ExtremaLinesExtractor
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


class ExtremaLines(BaseLiquidMode):
    """
    Returns TWO ZProfiles per call — one for density, one for terminations.
    Callers should call run_all() to get both.

    Params:
        min_line_length (int)
        stitch_gap (int)
        detrend_method (str):  "linear" | "polynomial"
        detrend_degree (int):  used only for polynomial
        blur_kernel (int)
        use_ab (bool)
    """

    def run(self, image_set: ImageSet, cache: ProcessedImageCache) -> ZProfile:
        """Returns the detrended line density signal (primary output)."""
        profiles = self.run_all(image_set, cache)
        return profiles[0]

    def run_all(
        self, image_set: ImageSet, cache: ProcessedImageCache
    ) -> tuple[ZProfile, ZProfile]:
        """
        Returns (density_profile, termination_profile).
        Both are detrended. Use this when you need both signals.
        """
        f = ExtremaLinesExtractor(self.params).extract(
            _apply_contrast(image_set, self.params), cache
        )
        method = self.params.get("detrend_method", "linear")
        degree = self.params.get("detrend_degree", 2)

        density_detrended = sp.detrend(
            f.extrema_minima_signal,
            method=method,
            **({"degree": degree} if method == "polynomial" else {}),
        )
        termination_detrended = sp.detrend(
            f.extrema_termination_signal,
            method=method,
            **({"degree": degree} if method == "polynomial" else {}),
        )

        density_profile = ZProfile(
            mode="ExtremaLines_Density",
            pipette_index=image_set.pipette_index,
            z_axis_px=f.z_axis_px,
            signal=density_detrended,
        )
        termination_profile = ZProfile(
            mode="ExtremaLines_Terminations",
            pipette_index=image_set.pipette_index,
            z_axis_px=f.z_axis_px,
            signal=termination_detrended,
        )
        return density_profile, termination_profile

    def run_all_debug(
        self, image_set: ImageSet, cache: ProcessedImageCache
    ) -> Tuple[ZProfile, ZProfile, RawFeatures]:
        """
        Like run_all(), but also returns the RawFeatures for debug inspection.
        Returns (density_profile, termination_profile, features).
        """
        density_profile, termination_profile = self.run_all(image_set, cache)
        # Re-extract to get the features; the cache ensures no redundant computation.
        f = ExtremaLinesExtractor(self.params).extract(
            _apply_contrast(image_set, self.params), cache
        )
        return density_profile, termination_profile, f
