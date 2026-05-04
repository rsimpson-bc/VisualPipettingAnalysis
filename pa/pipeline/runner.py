"""
Pipeline runner — the single entry point that executes a full named pipeline
for one pipette and returns a typed result.

Called from:
  - pa/api/_dispatch.py  (live mode, parallel per-pipette)
  - pa/optimization/step_runner.py  (optimization mode, single-threaded)
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np

from pa.pipeline.types import (
    ImageSet, PipelineConfig, ZProfile, TipRegion,
    LiquidMeasurementResult, TipIdentificationResult,
    DebugData,
)
from pa.pipeline.cache import ProcessedImageCache
from pa.pipeline.integrators.liquid_integrator import LiquidIntegrator
from pa.pipeline.integrators.tip_integrator import TipIntegrator

# Mode registry — maps mode_name string → class
# New modes are registered here; no other file needs to change.
_LIQUID_MODES: Dict[str, type] = {}
_TIP_MODES: Dict[str, type] = {}


def _register_modes():
    """Lazy import to avoid circular imports at module load time."""
    global _LIQUID_MODES, _TIP_MODES
    if _LIQUID_MODES:
        return  # already registered

    from pa.pipeline.modes.intensity_detection import IntensityDetection
    from pa.pipeline.modes.row_contrast_detection import RowContrastDetection
    from pa.pipeline.modes.line_continuity import (
        LineContinuityTerminations,
        LineContinuityPatternChange,
        LineContinuityCorrelation,
        LineContinuityDensity,
    )
    from pa.pipeline.modes.extrema_lines import ExtremaLines
    from pa.pipeline.modes.inward_edge_scan import InwardEdgeScan
    from pa.pipeline.modes.window_contrast_align import WindowContrastAlign
    from pa.pipeline.modes.roi_edge_fit import ROIEdgeFit
    from pa.pipeline.modes.roi_contrast_fit import ROIContrastFit

    _LIQUID_MODES.update({
        "IntensityDetection":           IntensityDetection,
        "RowContrastDetection":          RowContrastDetection,
        "LineContinuity_Terminations":  LineContinuityTerminations,
        "LineContinuity_PatternChange": LineContinuityPatternChange,
        "LineContinuity_Correlation":   LineContinuityCorrelation,
        "LineContinuity_Density":       LineContinuityDensity,
        "ExtremaLines":                 ExtremaLines,
    })
    _TIP_MODES.update({
        "InwardEdgeScan":           InwardEdgeScan,
        "WindowContrastAlign":      WindowContrastAlign,
        "ROIEdgeFit":               ROIEdgeFit,
        "ROIContrastFit":           ROIContrastFit,
    })


# Interpreter registry
_LIQUID_INTERPRETERS: Dict[str, type] = {}
_TIP_INTERPRETERS: Dict[str, type] = {}


def _register_interpreters():
    global _LIQUID_INTERPRETERS, _TIP_INTERPRETERS
    if _LIQUID_INTERPRETERS:
        return

    from pa.pipeline.interpreters.liquid_interpreters import (
        IntensityInterpreter,
        RowContrastInterpreter,
        LineContinuityThresholdInterpreter,
        LineContinuityCorrelationInterpreter,
        ExtremaLinesDensityInterpreter,
        ExtremaLinesTerminationInterpreter,
    )
    from pa.pipeline.interpreters.tip_interpreters import ConfidenceFilterInterpreter

    _LIQUID_INTERPRETERS.update({
        "IntensityDetection":              IntensityInterpreter,
        "RowContrastDetection":            RowContrastInterpreter,
        "LineContinuity_Terminations":     LineContinuityThresholdInterpreter,
        "LineContinuity_PatternChange":    LineContinuityThresholdInterpreter,
        "LineContinuity_Correlation":      LineContinuityCorrelationInterpreter,
        "LineContinuity_Density":          LineContinuityThresholdInterpreter,
        "ExtremaLines_Density":            ExtremaLinesDensityInterpreter,
        "ExtremaLines_Terminations":       ExtremaLinesTerminationInterpreter,
    })
    _TIP_INTERPRETERS.update({
        "InwardEdgeScan":       ConfidenceFilterInterpreter,
        "WindowContrastAlign":  ConfidenceFilterInterpreter,
        "ROIEdgeFit":           ConfidenceFilterInterpreter,
        "ROIContrastFit":       ConfidenceFilterInterpreter,
    })


# ---------------------------------------------------------------------------
# Public entry points
# ---------------------------------------------------------------------------

def run_liquid_pipeline(
    step_id: str,
    image_set: ImageSet,
    pipeline_config: PipelineConfig,
    instrument_config: Dict[str, Any],
    cache: ProcessedImageCache,
    debug: bool = False,
) -> Union[LiquidMeasurementResult, Tuple[LiquidMeasurementResult, DebugData]]:
    _register_modes()
    _register_interpreters()

    poi_lists = []
    debug_data = DebugData(step_id=step_id, pipette_index=image_set.pipette_index,
                           source_image=image_set.frames[0].copy() if image_set.frames else None) \
        if debug else None

    for mode_cfg in pipeline_config.modes:
        if not mode_cfg.enabled:
            continue
        mode_cls = _LIQUID_MODES.get(mode_cfg.mode_name)
        if mode_cls is None:
            continue

        mode = mode_cls(mode_cfg.params)

        # ExtremaLines produces two ZProfiles — handle specially
        if mode_cfg.mode_name == "ExtremaLines":
            if debug:
                density_profile, term_profile, features = mode.run_all_debug(image_set, cache)
            else:
                density_profile, term_profile = mode.run_all(image_set, cache)
                features = None

            density_pois, term_pois = [], []
            for profile, pois_out in [(density_profile, density_pois),
                                      (term_profile, term_pois)]:
                interpreter_cls = _LIQUID_INTERPRETERS.get(profile.mode)
                if interpreter_cls:
                    interp_params = dict(mode_cfg.params)
                    interp_params.setdefault("mode_weight", mode_cfg.weight)
                    pois = interpreter_cls(interp_params).interpret(profile)
                    pois_out.extend(pois)
                    poi_lists.append(pois)

            if debug and debug_data is not None:
                from pa.pipeline.debug_collector import collect_extrema_stages
                if features is None:
                    from pa.pipeline.extractors.extrema_lines import ExtremaLinesExtractor
                    features = ExtremaLinesExtractor(mode_cfg.params).extract(image_set, cache)
                for stage in collect_extrema_stages(
                    image_set.pipette_index, features,
                    density_profile, term_profile,
                    density_pois, term_pois, cache, mode_cfg.params,
                ):
                    debug_data.add(stage)
        else:
            if debug:
                profile, features = mode.run_debug(image_set, cache)
            else:
                profile: ZProfile = mode.run(image_set, cache)
                features = None

            interpreter_cls = _LIQUID_INTERPRETERS.get(mode_cfg.mode_name)
            pois = []
            if interpreter_cls:
                interp_params = dict(mode_cfg.params)
                interp_params.setdefault("mode_weight", mode_cfg.weight)
                pois = interpreter_cls(interp_params).interpret(profile)
                poi_lists.append(pois)

            if debug and debug_data is not None:
                from pa.pipeline.debug_collector import (
                    collect_intensity_stages, collect_ridge_stages,
                )
                if mode_cfg.mode_name in ("IntensityDetection", "RowContrastDetection"):
                    for stage in collect_intensity_stages(
                        image_set.pipette_index, profile, pois, cache, mode_cfg.params,
                        roi=image_set.roi,
                        roi_points=image_set.roi_points,
                        image_size=(image_set.frames[0].shape[0],
                                    image_set.frames[0].shape[1]) if image_set.frames else None,
                        mode_name=mode_cfg.mode_name,
                        features=features,
                    ):
                        debug_data.add(stage)
                elif mode_cfg.mode_name.startswith("LineContinuity"):
                    for stage in collect_ridge_stages(
                        image_set.pipette_index, mode_cfg.mode_name,
                        features, profile, pois, cache, mode_cfg.params,
                        roi=image_set.roi,
                        roi_points=image_set.roi_points,
                        image_size=(image_set.frames[0].shape[0],
                                    image_set.frames[0].shape[1]) if image_set.frames else None,
                    ):
                        debug_data.add(stage)

    integrator = LiquidIntegrator(pipeline_config.integrator_params, instrument_config)
    result = integrator.integrate(step_id, image_set.pipette_index, poi_lists)

    if debug and debug_data is not None:
        from pa.pipeline.debug_collector import build_liquid_result_overlay
        if debug_data.source_image is not None:
            flat_pois = [p for lst in poi_lists for p in lst]
            debug_data.result_overlay = build_liquid_result_overlay(
                debug_data.source_image, poi_lists, image_set.roi,
            )
        return result, debug_data

    return result


def run_tip_pipeline(
    step_id: str,
    image_set: ImageSet,
    pipeline_config: PipelineConfig,
    instrument_config: Dict[str, Any],
    cache: ProcessedImageCache,
    expected_tip_type: str = None,
    debug: bool = False,
) -> Union[TipIdentificationResult, Tuple[TipIdentificationResult, DebugData]]:
    _register_modes()
    _register_interpreters()

    candidate_lists = []
    debug_data = DebugData(step_id=step_id, pipette_index=image_set.pipette_index,
                           source_image=image_set.frames[0].copy() if image_set.frames else None) \
        if debug else None

    for mode_cfg in pipeline_config.modes:
        if not mode_cfg.enabled:
            continue
        mode_cls = _TIP_MODES.get(mode_cfg.mode_name)
        if mode_cls is None:
            continue

        mode = mode_cls(mode_cfg.params)
        if debug:
            tip_region, engine_extras = mode.run_debug(image_set, cache)
        else:
            tip_region: TipRegion = mode.run(image_set, cache)
            engine_extras = None

        interpreter_cls = _TIP_INTERPRETERS.get(mode_cfg.mode_name)
        candidates = []
        if interpreter_cls:
            candidates = interpreter_cls(mode_cfg.params).interpret(tip_region)
            candidate_lists.append(candidates)

        if debug and debug_data is not None:
            from pa.pipeline.debug_collector import (
                collect_inward_edge_stages, collect_window_contrast_stages,
                collect_roi_fit_stages,
            )
            best = candidates[0] if candidates else None
            if mode_cfg.mode_name == "InwardEdgeScan":
                for stage in collect_inward_edge_stages(
                    image_set.pipette_index, engine_extras, cache, mode_cfg.params, candidates,
                ):
                    debug_data.add(stage)
            elif mode_cfg.mode_name == "WindowContrastAlign":
                for stage in collect_window_contrast_stages(
                    image_set.pipette_index, cache, mode_cfg.params, best,
                ):
                    debug_data.add(stage)
            elif mode_cfg.mode_name in ("ROIEdgeFit", "ROIContrastFit"):
                for stage in collect_roi_fit_stages(
                    image_set.pipette_index, mode_cfg.mode_name, engine_extras, candidates,
                ):
                    debug_data.add(stage)

    integrator = TipIntegrator(pipeline_config.integrator_params, instrument_config)
    result = integrator.integrate(
        step_id, image_set.pipette_index, candidate_lists, expected_tip_type
    )

    if debug and debug_data is not None:
        from pa.pipeline.debug_collector import build_tip_result_overlay
        if debug_data.source_image is not None:
            # Collect engine_extras from last InwardEdgeScan for the overlay
            flat_candidates = [c for lst in candidate_lists for c in lst]
            debug_data.result_overlay = build_tip_result_overlay(
                debug_data.source_image, None, flat_candidates, image_set.roi_points,
            )
        return result, debug_data

    return result
