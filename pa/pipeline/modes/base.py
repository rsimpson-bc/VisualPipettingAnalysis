"""
Layer 2 — Base class for all analysis modes.

Each mode takes an ImageSet + ProcessedImageCache + params and returns
either a ZProfile (liquid modes) or a TipRegion (tip modes).
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Optional, Tuple
from pa.pipeline.types import ImageSet, ZProfile, TipRegion, RawFeatures
from pa.pipeline.cache import ProcessedImageCache


class BaseLiquidMode(ABC):
    """Base for all liquid analysis modes. Returns a ZProfile."""

    def __init__(self, params: dict):
        self.params = params

    @abstractmethod
    def run(self, image_set: ImageSet, cache: ProcessedImageCache) -> ZProfile:
        raise NotImplementedError

    def run_debug(
        self, image_set: ImageSet, cache: ProcessedImageCache
    ) -> Tuple[ZProfile, Optional[RawFeatures]]:
        """
        Like run(), but also returns the underlying RawFeatures for debug inspection.
        The default implementation calls run() and returns (profile, None).
        Subclasses that have meaningful feature data should override this.
        """
        return self.run(image_set, cache), None


class BaseTipMode(ABC):
    """Base for all tip analysis modes. Returns a TipRegion."""

    def __init__(self, params: dict):
        self.params = params

    @abstractmethod
    def run(self, image_set: ImageSet, cache: ProcessedImageCache) -> TipRegion:
        raise NotImplementedError

    def run_debug(self, image_set: ImageSet, cache: ProcessedImageCache):
        """
        Like run(), but also returns any extra engine data for debug inspection.
        Returns (TipRegion, extras_dict_or_None).
        InwardEdgeScan overrides this to return the full engine result dict.
        """
        return self.run(image_set, cache), None
