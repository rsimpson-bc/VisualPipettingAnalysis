"""
Liquid measurement analysis (Tier 3 — LiquidMeasurement steps).
Delegates to pa.pipeline.runner, which dispatches through the full 5-layer pipeline.
"""

from __future__ import annotations
from typing import Any, Dict
from pa.api.schemas_http import AnalyzeStepRequest
from pa.state.session_state import SessionContext
from pa.state import session_state
from pa.pipeline.runner import run_liquid_pipeline
from pa.pipeline.types import ImageSet, PipelineConfig, ModeConfig
from pa.pipeline.cache import ProcessedImageCache
from pa.io import image_loader
from pa.analysis.shared_geometry import lookup_roi


def analyze(req: AnalyzeStepRequest, ctx: SessionContext) -> Dict[str, Any]:
    """
    Run liquid measurement for all pipettes in the step.
    Returns a dict matching liquid_measurement_result.schema.json.
    """
    analysis_cfg = session_state.get_analysis_config()
    pipeline_name = getattr(req, 'pipeline', 'full_liquid')
    raw_pipeline = analysis_cfg["pipelines"].get(pipeline_name, analysis_cfg["pipelines"]["full_liquid"])

    pipeline_config = PipelineConfig(
        name=raw_pipeline["name"],
        analysis_type="LiquidMeasurement",
        modes=[ModeConfig(**m) for m in raw_pipeline["modes"]],
        integrator_params=raw_pipeline.get("integrator_params", {}),
    )

    pipette_results = []
    for pip in req.pipettes:
        frames, paths = image_loader.load_step_frames(
            req.subfolders, ctx.run_folder,
            transform=session_state.get_frame_transform(),
        )
        roi_bbox, _roi_points = lookup_roi(
            session_state.get_instrument_config(), pip.index,
            getattr(pip, 'expected_tip_type', None))
        image_set = ImageSet(
            pipette_index=pip.index,
            frames=frames,
            source_paths=paths,
            roi=roi_bbox,
        )
        cache = ProcessedImageCache()
        result = run_liquid_pipeline(
            step_id=req.step_id,
            image_set=image_set,
            pipeline_config=pipeline_config,
            instrument_config=session_state.get_instrument_config(),
            cache=cache,
        )
        pipette_results.append({
            "index": result.pipette_index,
            "total_volume_ul": result.total_volume_ul,
            "slugs": result.slugs,
            "bubbles": result.bubbles,
            "droplets": result.droplets,
            "confidence": result.confidence,
            "comments": result.comments,
        })

    overall = _overall_status(pipette_results)
    return {
        "step_id": req.step_id,
        "analysis_type": "LiquidMeasurement",
        "command": req.model_dump(),
        "pipettes": pipette_results,
        "overall_status": overall,
    }


def _overall_status(results):
    confidences = [r["confidence"] for r in results]
    if not confidences:
        return "low_confidence"
    avg = sum(confidences) / len(confidences)
    if avg >= 80:
        return "ok"
    if avg >= 50:
        return "warning"
    return "low_confidence"
