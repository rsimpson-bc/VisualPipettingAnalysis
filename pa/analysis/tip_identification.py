"""
Tip identification analysis (Tier 3 — TipIdentification steps).
Delegates to pa.pipeline.runner, which dispatches through the full 5-layer pipeline.
"""

from __future__ import annotations
from typing import Any, Dict
from pa.api.schemas_http import AnalyzeStepRequest
from pa.state.session_state import SessionContext
from pa.state import session_state
from pa.pipeline.runner import run_tip_pipeline
from pa.pipeline.types import ImageSet, PipelineConfig, ModeConfig
from pa.pipeline.cache import ProcessedImageCache
from pa.io import image_loader
from pa.analysis.shared_geometry import lookup_roi


def analyze(req: AnalyzeStepRequest, ctx: SessionContext) -> Dict[str, Any]:
    """
    Run tip identification for all pipettes in the step.
    Returns a dict matching tip_identification_result.schema.json.
    """
    analysis_cfg = session_state.get_analysis_config()
    pipeline_name = getattr(req, 'pipeline', 'full_tip')
    raw_pipeline = analysis_cfg["pipelines"].get(pipeline_name, analysis_cfg["pipelines"]["full_tip"])

    pipeline_config = PipelineConfig(
        name=raw_pipeline["name"],
        analysis_type="TipIdentification",
        modes=[ModeConfig(**m) for m in raw_pipeline["modes"]],
        integrator_params=raw_pipeline.get("integrator_params", {}),
    )

    pipette_results = []
    for pip in req.pipettes:
        frames, paths = image_loader.load_step_frames(
            req.subfolders, ctx.run_folder,
            transform=session_state.get_frame_transform(),
        )
        roi_bbox, roi_points = lookup_roi(
            session_state.get_instrument_config(), pip.index, pip.expected_tip_type)
        image_set = ImageSet(
            pipette_index=pip.index,
            frames=frames,
            source_paths=paths,
            roi=roi_bbox,
            roi_points=roi_points,
        )
        cache = ProcessedImageCache()
        result = run_tip_pipeline(
            step_id=req.step_id,
            image_set=image_set,
            pipeline_config=pipeline_config,
            instrument_config=session_state.get_instrument_config(),
            cache=cache,
            expected_tip_type=pip.expected_tip_type,
        )
        pipette_results.append({
            "index": result.pipette_index,
            "tip_present": result.tip_present,
            "perceived_tip_type": result.perceived_tip_type,
            "tip_type_match": result.tip_type_match,
            "seating_depth_px": result.seating_depth_px,
            "tip_angle_deg": result.tip_angle_deg,
            "seating_status": result.seating_status,
            "confidence": result.confidence,
            "comments": result.comments,
        })

    overall = _overall_status([r["seating_status"] for r in pipette_results])
    return {
        "step_id": req.step_id,
        "analysis_type": "TipIdentification",
        "command": req.model_dump(),
        "pipettes": pipette_results,
        "overall_status": overall,
    }


def _overall_status(statuses):
    if any(s in ("missing", "unexpected", "wrong_type") for s in statuses):
        return "error"
    if any(s in ("warning_angle", "warning_height", "low_confidence") for s in statuses):
        return "warning"
    return "ok"
