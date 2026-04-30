"""
Loaders for Tier 1 (instrument config) and Tier 2 (method context) data.
Both are read once (Tier 1 at process startup, Tier 2 at session start)
and cached in session_state.
"""

import json
import os
from typing import Any, Dict


def load_instrument_config(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_analysis_config(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_method_context(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def derive_run_folder(method_context_path: str) -> str:
    """
    The method_context.json lives in the run root folder created by Piomek2,
    so its parent directory is the run root.
    """
    return os.path.dirname(os.path.abspath(method_context_path))
