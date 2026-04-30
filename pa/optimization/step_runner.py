"""
Optimization harness — per-step runner.

Loads images + expected values for one step from a stored run folder,
builds ImageSets, and runs the pipeline without any HTTP machinery.
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from pa.pipeline.types import ImageSet, PipelineConfig, ModeConfig
from pa.pipeline.cache import ProcessedImageCache
from pa.pipeline import pool_manager


def run_step(
    step_id: str,
    run_folder: str,
    instrument_config: Dict[str, Any],
    pipeline_config: Dict[str, Any],
) -> Tuple[List[Any], Optional[Dict[str, Any]]]:
    """
    Load images + expected values for step_id, run the pipeline, return results.

    Returns:
        (pa_results, expected_values)
        pa_results: list of LiquidMeasurementResult or TipIdentificationResult
                    (one per pipette)
        expected_values: dict loaded from step expected file, or None
    """
    # Load step command (mirrors what Piomek2 would have sent via HTTP)
    step_command = _load_step_command(step_id, run_folder)
    expected = _load_expected(step_id, run_folder)

    # Build ImageSets
    image_sets = _build_image_sets(step_command, run_folder, instrument_config)

    # Build PipelineConfig from the raw dict
    p_cfg = _build_pipeline_config(pipeline_config, step_command["analysis_type"])

    # Run per-pipette (single-threaded here; parallelism via pool_manager in live mode)
    cache = ProcessedImageCache()
    results = [_run_one_pipette(img_set, p_cfg, instrument_config, cache, step_id)
               for img_set in image_sets]

    return results, expected


def _run_one_pipette(
    image_set: ImageSet,
    pipeline_config: PipelineConfig,
    instrument_config: Dict[str, Any],
    cache: ProcessedImageCache,
    step_id: str,
) -> Any:
    """Dispatch to LiquidPipeline or TipPipeline based on analysis_type."""
    if pipeline_config.analysis_type == "LiquidMeasurement":
        from pa.pipeline.runner import run_liquid_pipeline
        return run_liquid_pipeline(step_id, image_set, pipeline_config, instrument_config, cache)
    elif pipeline_config.analysis_type == "TipIdentification":
        from pa.pipeline.runner import run_tip_pipeline
        return run_tip_pipeline(step_id, image_set, pipeline_config, instrument_config, cache)
    else:
        raise ValueError(f"Unknown analysis_type: {pipeline_config.analysis_type}")


def _load_step_command(step_id: str, run_folder: str) -> Dict[str, Any]:
    """Load step_meta.json for this step. Falls back to searching subfolders."""
    # Try Analysis/<step_id>_result.json to get the echoed command
    result_path = os.path.join(run_folder, "Analysis", f"{step_id}_result.json")
    if os.path.exists(result_path):
        with open(result_path, encoding="utf-8") as f:
            result = json.load(f)
        if "command" in result:
            return result["command"]

    raise FileNotFoundError(
        f"Cannot find step command for {step_id} in {run_folder}. "
        "Run an analysis first to produce a result file with an echoed command."
    )


def _load_expected(step_id: str, run_folder: str) -> Optional[Dict[str, Any]]:
    """Load expected_<step_id>.json if it exists."""
    expected_path = os.path.join(run_folder, "Analysis", f"expected_{step_id}.json")
    if os.path.exists(expected_path):
        with open(expected_path, encoding="utf-8") as f:
            return json.load(f)
    return None


def _build_image_sets(
    step_command: Dict[str, Any],
    run_folder: str,
    instrument_config: Dict[str, Any],
) -> List[ImageSet]:
    """Build one ImageSet per pipette from the step command subfolders."""
    image_sets = []
    for pip in step_command.get("pipettes", []):
        frames, paths = [], []
        for subfolder in step_command.get("subfolders", []):
            folder = os.path.join(run_folder, subfolder)
            if not os.path.isdir(folder):
                continue
            for fname in sorted(os.listdir(folder)):
                if fname.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):
                    fpath = os.path.join(folder, fname)
                    img = cv2.imread(fpath)
                    if img is not None:
                        frames.append(img)
                        paths.append(fpath)

        # TODO: look up ROI from instrument_config per pipette index + tip type
        roi = None

        image_sets.append(ImageSet(
            pipette_index=pip["index"],
            frames=frames,
            source_paths=paths,
            roi=roi,
        ))
    return image_sets


def _build_pipeline_config(raw: Dict[str, Any], analysis_type: str) -> PipelineConfig:
    """Convert raw dict (from analysis_config.json) to a PipelineConfig."""
    modes = [
        ModeConfig(
            mode_name=m["mode_name"],
            enabled=m.get("enabled", True),
            weight=m.get("weight", 1.0),
            params=m.get("params", {}),
        )
        for m in raw.get("modes", [])
    ]
    return PipelineConfig(
        name=raw.get("name", "unnamed"),
        analysis_type=analysis_type,
        modes=modes,
        integrator_params=raw.get("integrator_params", {}),
    )
