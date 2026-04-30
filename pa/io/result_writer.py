"""
Result writer — serialises analysis result dicts to the Analysis/ subfolder.
"""

from __future__ import annotations
import json
import os
from typing import Any, Dict


def write(result: Dict[str, Any], run_folder: str) -> str:
    """
    Write a result dict as JSON to <run_folder>/Analysis/<step_id>_result.json.

    Returns the absolute path of the written file.
    """
    analysis_dir = os.path.join(run_folder, "Analysis")
    os.makedirs(analysis_dir, exist_ok=True)

    step_id = result.get("step_id", "unknown_step")
    filename = f"{step_id}_result.json"
    out_path = os.path.join(analysis_dir, filename)

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    return out_path
