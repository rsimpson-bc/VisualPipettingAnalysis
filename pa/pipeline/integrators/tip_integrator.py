"""
Layer 4 — Tip Integrator.

Receives interpreted TipCandidates from all active tip modes for one pipette,
cross-validates them using cluster-then-pick, and produces a
TipIdentificationResult.

Cross-validation:
  - Candidates from different modes that agree within ±consensus_window_px
    vote for the same location.
  - The consensus location is the weighted centroid of the cluster.
  - If two modes both rank a location highly, that location wins.
  - The result also carries the tip_bottom_z_px for use by LiquidIntegrator.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional
import numpy as np

from pa.pipeline.types import TipCandidate, TipIdentificationResult


class TipIntegrator:
    """
    Params (from integrator_params in PipelineConfig):
        consensus_window_px (int):      candidates within this window are considered the same location
        min_modes_for_consensus (int):  minimum number of modes that must agree
        pixel_to_mm_z (float):          calibration
    """

    def __init__(self, params: dict, instrument_config: dict):
        self.params = params
        self.instrument_config = instrument_config

    def integrate(
        self,
        step_id: str,
        pipette_index: int,
        candidate_lists: List[List[TipCandidate]],
        expected_tip_type: Optional[str] = None,
    ) -> TipIdentificationResult:
        """
        candidate_lists: one list per active tip mode, best-first within each.
        """
        all_candidates: List[TipCandidate] = [c for clist in candidate_lists for c in clist]

        if not all_candidates:
            return TipIdentificationResult(
                step_id=step_id,
                pipette_index=pipette_index,
                tip_present=None,
                perceived_tip_type=None,
                tip_type_match=None,
                seating_depth_px=None,
                tip_angle_deg=None,
                seating_status="low_confidence",
                confidence=0.0,
                comments="No candidates produced by any mode.",
                _candidates=all_candidates,
            )

        clusters = self._cluster(all_candidates)
        clusters.sort(key=lambda c: c["score"], reverse=True)

        top = clusters[0]
        n_modes = len({c.source_mode for c in top["members"]})
        min_modes = self.params.get("min_modes_for_consensus", 1)
        confident = n_modes >= min_modes

        best = top["members"][0]   # highest-confidence candidate in winning cluster
        avg_angle = float(np.mean([c.angle_deg for c in top["members"]]))
        avg_conf = float(np.mean([c.confidence for c in top["members"]]))

        # TODO: derive tip_type from candidate region geometry or model
        perceived_type = None
        type_match = (
            (perceived_type == expected_tip_type)
            if (perceived_type and expected_tip_type)
            else None
        )

        seating_status = "ok" if confident else "low_confidence"

        # Extract tip bottom Z from best candidate region (y, z, w, h)
        _, z_px, _, _ = best.region

        return TipIdentificationResult(
            step_id=step_id,
            pipette_index=pipette_index,
            tip_present=True if confident else None,
            perceived_tip_type=perceived_type,
            tip_type_match=type_match,
            seating_depth_px=float(z_px),
            tip_angle_deg=avg_angle,
            seating_status=seating_status,
            confidence=avg_conf * 100.0,  # schema uses 0–100
            _candidates=all_candidates,
        )

    def _cluster(self, candidates: List[TipCandidate]) -> List[Dict[str, Any]]:
        """Group candidates by Z position of their region."""
        window = self.params.get("consensus_window_px", 10)
        clusters: List[Dict[str, Any]] = []

        for cand in sorted(candidates, key=lambda c: c.confidence, reverse=True):
            _, z_px, _, _ = cand.region
            placed = False
            for cluster in clusters:
                if abs(z_px - cluster["centroid_z"]) <= window:
                    cluster["members"].append(cand)
                    cluster["score"] += cand.confidence
                    cluster["centroid_z"] = np.mean([
                        c.region[1] for c in cluster["members"]
                    ])
                    placed = True
                    break
            if not placed:
                clusters.append({
                    "centroid_z": float(z_px),
                    "score": cand.confidence,
                    "members": [cand],
                })

        return clusters
