"""
Layer 1 — Sobel Edge Map Extractor.

Produces a 2-D edge map using the Sobel operator.
Used by: InwardEdgeScan, WindowContrastAlign modes.
"""

from __future__ import annotations
import numpy as np
from pa.pipeline.types import ImageSet, RawFeatures
from pa.pipeline.cache import ProcessedImageCache
from pa.pipeline.extractors.base import BaseExtractor
from pa.pipeline import image_primitives as ip


class SobelEdgeExtractor(BaseExtractor):
    """
    Params:
        blur_method (str)
        blur_kernel (int)
        gradient_direction (str):  "xy" | "x" | "y"
        use_ab (bool)
    """

    def _extract(self, image_set: ImageSet, cache: ProcessedImageCache) -> RawFeatures:
        params = self.params
        use_ab = params.get("use_ab", False)

        # Build working image
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

        # Blur
        blur_params = {"method": params.get("blur_method", "gaussian"),
                       "kernel_size": params.get("blur_kernel", 5)}
        blurred = cache.get_or_compute(
            image_set.pipette_index, "blur", blur_params,
            lambda: ip.blur(working.astype(np.uint8), **blur_params),
        )

        # Gradient
        direction = params.get("gradient_direction", "xy")
        grad_params = {"direction": direction}
        edge_map = cache.get_or_compute(
            image_set.pipette_index, f"gradient_{direction}", grad_params,
            lambda: ip.gradient(blurred, direction=direction),
        )

        # Extract ROI
        roi_map = ip.extract_roi(edge_map, image_set.roi) if image_set.roi else edge_map

        return RawFeatures(
            mode="SobelEdgeExtractor",
            pipette_index=image_set.pipette_index,
            edge_map=roi_map,
        )
