"""
Layer 3 — Tip interpreters.

Convert TipRegion (raw candidates from a mode) into a filtered, re-ranked
List[TipCandidate]. Low-confidence candidates are dropped; the list is
sorted best-first.
"""

from __future__ import annotations

from typing import List
from pa.pipeline.types import TipRegion, TipCandidate
from pa.pipeline.interpreters.base import BaseTipInterpreter


class ConfidenceFilterInterpreter(BaseTipInterpreter):
    """
    Drops candidates below a minimum confidence threshold and re-sorts.

    Params:
        min_confidence (float):   0–1, candidates below this are dropped
        max_candidates (int):     keep only the top N after filtering
    """

    def interpret(self, region: TipRegion) -> List[TipCandidate]:
        min_conf = self.params.get("min_confidence", 0.3)
        max_candidates = self.params.get("max_candidates", 3)
        filtered = [c for c in region.candidates if c.confidence >= min_conf]
        filtered.sort(key=lambda c: c.confidence, reverse=True)
        return filtered[:max_candidates]
