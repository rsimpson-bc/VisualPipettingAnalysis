"""
Layer 3 — Base class for all interpreters.

Interpreters convert raw mode output (ZProfile or TipRegion) into
Points of Interest or ranked TipCandidates. They apply thresholds,
detrending, and noise rejection — mode-specific logic lives here.
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from typing import List
from pa.pipeline.types import ZProfile, TipRegion, PointOfInterest, TipCandidate


class BaseLiquidInterpreter(ABC):
    """Converts a ZProfile → List[PointOfInterest]."""

    def __init__(self, params: dict):
        self.params = params

    @abstractmethod
    def interpret(self, profile: ZProfile) -> List[PointOfInterest]:
        raise NotImplementedError


class BaseTipInterpreter(ABC):
    """Converts a TipRegion → List[TipCandidate] (re-ranked / filtered)."""

    def __init__(self, params: dict):
        self.params = params

    @abstractmethod
    def interpret(self, region: TipRegion) -> List[TipCandidate]:
        raise NotImplementedError
