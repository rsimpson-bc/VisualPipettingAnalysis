"""
Layer 1 — Intensity Extractor.

Computes per-row intensity from an A:B accumulated difference image within the ROI.
Used by: IntensityDetection mode.
"""

from __future__ import annotations
import numpy as np
from pa.pipeline.types import ImageSet, RawFeatures
from pa.pipeline.cache import ProcessedImageCache
from pa.pipeline.extractors.base import BaseExtractor
from pa.pipeline import image_primitives as ip


class IntensityExtractor(BaseExtractor):
    """
    Params:
        blur_method (str):   "gaussian" | "bilateral"
        blur_kernel (int):   kernel size for gaussian blur
        use_ab (bool):       True = accumulate A:B diffs; False = single frame
    """

    def _extract(self, image_set: ImageSet, cache: ProcessedImageCache) -> RawFeatures:
        params = self.params

        # Step 1: build working image (single frame or A:B accumulation)
        use_ab = params.get("use_ab", True)
        if use_ab and len(image_set.frames) >= 2:
            working = cache.get_or_compute(
                image_set.pipette_index, "accumulate_ab", {},
                lambda: ip.accumulate_ab(image_set.frames),
            )
        else:
            working = cache.get_or_compute(
                image_set.pipette_index, "to_grayscale_frame0", {},
                lambda: ip.to_grayscale(image_set.frames[0]),
            )

        # Step 2: blur
        blur_params = {
            "method":      params.get("blur_method", "gaussian"),
            "kernel_size": params.get("blur_kernel", 5),
            "d":           params.get("bilateral_d", 9),
            "sigma_color": params.get("bilateral_sigma_color", 75.0),
            "sigma_space": params.get("bilateral_sigma_space", 75.0),
        }
        blurred = cache.get_or_compute(
            image_set.pipette_index, "blur", blur_params,
            lambda: ip.blur(working.astype(np.uint8), **blur_params),
        )

        # Step 3: optional gradient step
        grad_method = params.get("gradient_method", "none")
        grad_ksize  = params.get("gradient_ksize", 3)
        grad_params = {"method": grad_method, "ksize": grad_ksize}
        if grad_method != "none":
            processed = cache.get_or_compute(
                image_set.pipette_index, "gradient", grad_params,
                lambda: ip.apply_gradient_step(blurred, grad_method, grad_ksize),
            )
        else:
            processed = blurred

        # Step 4: extract ROI
        roi_img = ip.extract_roi(processed, image_set.roi) if image_set.roi else processed

        # Step 4: per-row mean intensity → 1-D signal
        intensity_signal = roi_img.astype(np.float32).mean(axis=1)
        z_axis = np.arange(len(intensity_signal))

        return RawFeatures(
            mode="IntensityExtractor",
            pipette_index=image_set.pipette_index,
            z_axis_px=z_axis,
            intensity_signal=intensity_signal,
        )
