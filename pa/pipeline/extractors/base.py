"""
Layer 1 — Base class for all feature extractors.

Each extractor takes a processed image (or images) and returns a RawFeatures
object. Extractors know nothing about analysis logic — they only produce signals.
"""

from __future__ import annotations
from abc import ABC, abstractmethod
import numpy as np
from pa.pipeline.types import ImageSet, RawFeatures
from pa.pipeline.cache import ProcessedImageCache


class BaseExtractor(ABC):
    """
    All Layer 1 extractors inherit from this.

    Subclasses implement _extract() and declare which fields of RawFeatures
    they populate. The base class handles cache lookup/storage.
    """

    def __init__(self, params: dict):
        self.params = params

    def extract(self, image_set: ImageSet, cache: ProcessedImageCache) -> RawFeatures:
        """Public entry point. Subclasses implement _extract()."""
        return self._extract(image_set, cache)

    @abstractmethod
    def _extract(self, image_set: ImageSet, cache: ProcessedImageCache) -> RawFeatures:
        raise NotImplementedError
