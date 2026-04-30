"""
Layer 4 — Liquid Integrator.

Receives interpreted PointsOfInterest from all active liquid modes for one
pipette, merges them using cluster-then-pick, and produces a
LiquidMeasurementResult.

Cluster-then-pick algorithm:
  1. Collect all POIs from all modes (each already weighted).
  2. Cluster POIs that fall within ±cluster_window_px of each other.
  3. Score each cluster = sum of member weights.
  4. If the highest-scoring cluster exceeds a quorum threshold → confident result.
  5. If multiple clusters have similar scores (within spread_flag_threshold of
     each other) → flag as low_confidence and record all clusters.
  6. Volume conversion from pixel locations uses the pixel_to_mm calibration.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional
import numpy as np

from pa.pipeline.types import PointOfInterest, LiquidMeasurementResult


class LiquidIntegrator:
    """
    Params (from integrator_params in PipelineConfig):
        cluster_window_px (int):        POIs within this window are in the same cluster
        quorum_threshold (float):       minimum cluster score to be considered confident
        spread_flag_threshold (float):  if top-2 cluster scores are within this ratio, flag
        pixel_to_mm_z (float):          calibration — mm per pixel on Z axis
        tip_bottom_z_px (float|None):   from TipIntegrator; used as lower bound if available
    """

    def __init__(self, params: dict, instrument_config: dict):
        self.params = params
        self.instrument_config = instrument_config

    def integrate(
        self,
        step_id: str,
        pipette_index: int,
        poi_lists: List[List[PointOfInterest]],
    ) -> LiquidMeasurementResult:
        """
        poi_lists: one list per active mode, in any order.
        """
        all_pois: List[PointOfInterest] = [p for plist in poi_lists for p in plist]

        if not all_pois:
            return LiquidMeasurementResult(
                step_id=step_id,
                pipette_index=pipette_index,
                total_volume_ul=None,
                confidence=0.0,
                comments="No points of interest produced by any mode.",
                _poi_list=all_pois,
            )

        clusters = self._cluster(all_pois)
        clusters.sort(key=lambda c: c["score"], reverse=True)

        top = clusters[0]
        second_score = clusters[1]["score"] if len(clusters) > 1 else 0.0
        spread_flag = (
            len(clusters) > 1
            and (top["score"] - second_score) / (top["score"] + 1e-6)
            < self.params.get("spread_flag_threshold", 0.2)
        )

        quorum = self.params.get("quorum_threshold", 1.0)
        confident = top["score"] >= quorum and not spread_flag

        if not confident:
            return LiquidMeasurementResult(
                step_id=step_id,
                pipette_index=pipette_index,
                total_volume_ul=None,
                confidence=float(top["score"]),
                comments=(
                    "Mode disagreement — multiple candidate clusters."
                    if spread_flag else "Insufficient quorum score."
                ),
                _poi_list=all_pois,
            )

        # TODO: convert top cluster Z position to volume (requires slug geometry)
        transition_z_px = top["centroid_z"]
        confidence = min(float(top["score"]) / quorum, 1.0)

        return LiquidMeasurementResult(
            step_id=step_id,
            pipette_index=pipette_index,
            total_volume_ul=None,   # TODO: implement px→volume conversion
            confidence=confidence * 100.0,  # schema uses 0–100
            comments=f"Transition at Z={transition_z_px:.1f}px (volume conversion TODO).",
            _poi_list=all_pois,
        )

    def _cluster(self, pois: List[PointOfInterest]) -> List[Dict[str, Any]]:
        """
        Simple 1-D greedy clustering by Z position.
        Returns list of dicts: {centroid_z, score, members}.
        """
        window = self.params.get("cluster_window_px", 15)
        sorted_pois = sorted(pois, key=lambda p: p.z_px)
        clusters: List[Dict[str, Any]] = []

        for poi in sorted_pois:
            placed = False
            for cluster in clusters:
                if abs(poi.z_px - cluster["centroid_z"]) <= window:
                    cluster["members"].append(poi)
                    # Update centroid as weighted mean
                    total_weight = sum(m.weight for m in cluster["members"])
                    cluster["centroid_z"] = (
                        sum(m.z_px * m.weight for m in cluster["members"]) / total_weight
                    )
                    cluster["score"] = total_weight
                    placed = True
                    break
            if not placed:
                clusters.append({
                    "centroid_z": poi.z_px,
                    "score": poi.weight,
                    "members": [poi],
                })

        return clusters
