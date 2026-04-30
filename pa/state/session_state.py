"""
In-memory state for the PA process.

Two tiers of state live here:

  Startup state  — loaded once when the process starts (Tier 1: instrument config).
                   Never cleared while the process is alive.

  Method state   — loaded per method run (Tier 2: method context + run folder).
                   Set by POST /session/start, cleared by POST /session/stop.
                   The process remains alive between method runs.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional
import numpy as np
from pa.state import loaders


# ---------------------------------------------------------------------------
# Startup state (Tier 1)
# ---------------------------------------------------------------------------

@dataclass
class StartupState:
    instrument_config: Dict[str, Any]
    instrument_config_path: str
    analysis_config: Dict[str, Any]
    analysis_config_path: str
    # Cached per-frame transform built from camera_calibration in instrument_config.
    # None when no calibration has been saved.
    frame_transform: Optional[Callable[[np.ndarray], np.ndarray]] = field(
        default=None, compare=False
    )


_startup: Optional[StartupState] = None


def _build_frame_transform(
    instrument_config: Dict[str, Any],
) -> Optional[Callable[[np.ndarray], np.ndarray]]:
    """Build a composed undistort+rotate callable from instrument_config.

    Returns None when no camera_calibration key is present.
    """
    cal = instrument_config.get("camera_calibration")
    if cal is None:
        return None

    from pa.io.undistortion import apply_maps, apply_rotation, compute_maps

    camera_matrix = np.array(cal["camera_matrix"], dtype=np.float64)
    dist_coeffs = np.array(cal["dist_coeffs"], dtype=np.float64)
    image_size: tuple[int, int] = tuple(cal["image_size"])  # (w, h)
    rotation_deg: float = cal.get("rotation_deg", 0.0)

    map1, map2 = compute_maps(camera_matrix, dist_coeffs, image_size)

    def _transform(frame: np.ndarray) -> np.ndarray:
        out = apply_maps(frame, map1, map2)
        if rotation_deg != 0.0:
            out = apply_rotation(out, rotation_deg)
        return out

    return _transform


def startup(instrument_config_path: str, analysis_config_path: str) -> None:
    """Called once at process start. Loads Tier 1 data."""
    global _startup
    instrument_config = loaders.load_instrument_config(instrument_config_path)
    analysis_config = loaders.load_analysis_config(analysis_config_path)
    _startup = StartupState(
        instrument_config=instrument_config,
        instrument_config_path=instrument_config_path,
        analysis_config=analysis_config,
        analysis_config_path=analysis_config_path,
        frame_transform=_build_frame_transform(instrument_config),
    )


def get_instrument_config() -> Dict[str, Any]:
    if _startup is None:
        raise RuntimeError("PA process not yet initialized. Was startup() called?")
    return _startup.instrument_config


def get_analysis_config() -> Dict[str, Any]:
    if _startup is None:
        raise RuntimeError("PA process not yet initialized. Was startup() called?")
    return _startup.analysis_config


def get_frame_transform() -> Optional[Callable[[np.ndarray], np.ndarray]]:
    """Return the cached per-frame undistort+rotate callable, or None."""
    if _startup is None:
        return None
    return _startup.frame_transform


# ---------------------------------------------------------------------------
# Method state (Tier 2)
# ---------------------------------------------------------------------------

@dataclass
class SessionContext:
    """Active while a method is running."""
    method_context: Dict[str, Any]
    run_folder: str


_ctx: Optional[SessionContext] = None


def initialize(method_context_path: str) -> None:
    """Called at POST /session/start. Loads Tier 2 data for this method run."""
    global _ctx
    if _startup is None:
        raise RuntimeError("PA process not yet initialized. Was startup() called?")
    method_context = loaders.load_method_context(method_context_path)
    run_folder = loaders.derive_run_folder(method_context_path)
    _ctx = SessionContext(
        method_context=method_context,
        run_folder=run_folder,
    )


def get() -> SessionContext:
    if _ctx is None:
        raise RuntimeError("No active method session. Call POST /session/start first.")
    return _ctx


def is_initialized() -> bool:
    return _ctx is not None


def clear() -> None:
    """Called at POST /session/stop. Clears Tier 2 state; process stays alive."""
    global _ctx
    _ctx = None
