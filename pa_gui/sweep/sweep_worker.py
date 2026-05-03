"""Sweep worker thread for the Parameter Sweep tab.

Iterates over a cartesian product of parameter values, runs IntensityDetection
on each image for each combo, scores the result against GT annotations, and
emits signals for the UI.

Caching strategy
----------------
The IntensityDetection pipeline is decomposed into two stages:

  Stage A — "signal params": everything that affects the intensity signal array.
    Keys: use_contrast, roi_expansion_px, use_ab, blur_method, blur_kernel,
          bilateral_d, bilateral_sigma_color, bilateral_sigma_space,
          gradient_method, gradient_ksize,
          contrast_mode, contrast_blur_method, contrast_blur_kernel,
          contrast_bilateral_d, contrast_bilateral_sigma_color,
          contrast_bilateral_sigma_space, contrast_min, contrast_max

  Stage B — "score params": everything downstream (interpreter thresholds etc.)
    Keys: intensity_threshold, min_distance
    (These don't affect the signal — only how peaks are detected for final
    LiquidMeasurementResult.  The scorer reads the raw signal directly, so
    Stage B params never require recomputation.)

Before the main loop the worker groups all combos by their Stage-A key.  Each
unique Stage-A group is computed *once* per image.  All combos in that group
re-use the cached signal for scoring.

This means:
  - Sweeping only ``min_distance`` over 20 values × 5 images = 5 signal runs
    (not 100).
  - Sweeping ``blur_kernel`` over 5 values × ``min_distance`` over 4 values
    × 5 images = 25 signal runs (not 100).
"""

from __future__ import annotations

import itertools
import os
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from PySide6.QtCore import QThread, Signal

from pa_gui.sweep.gt_data import ImageGT
from pa_gui.sweep.sweep_scorer import ScoreResult, score_signal, aggregate_scores

# Params that determine the signal array.  All others are "score-only".
_SIGNAL_PARAM_KEYS = frozenset({
    "use_contrast",
    "roi_expansion_px",
    "use_ab",
    "blur_method",
    "blur_kernel",
    "bilateral_d",
    "bilateral_sigma_color",
    "bilateral_sigma_space",
    "gradient_method",
    "gradient_ksize",
    "stat",                     # RowContrastDetection: std vs variance vs std_v vs variance_v
    "vertical_window_px",       # RowContrastDetection vertical window
    "peak_min_distance",        # RowContrastDetection peak_count_h / peak_prominence_sum_h
    "peak_min_prominence",      # RowContrastDetection peak_count_h / peak_prominence_sum_h
    "peak_width_exponent",      # RowContrastDetection peak_count_h / peak_prominence_sum_h
    "contrast_mode",
    "contrast_blur_method",
    "contrast_blur_kernel",
    "contrast_bilateral_d",
    "contrast_bilateral_sigma_color",
    "contrast_bilateral_sigma_space",
    "contrast_min",
    "contrast_max",
})


def _signal_cache_key(params: Dict[str, Any]) -> frozenset:
    """Hashable key capturing only the signal-affecting params."""
    return frozenset(
        (k, v) for k, v in params.items() if k in _SIGNAL_PARAM_KEYS
    )


class SweepWorker(QThread):
    """Run a parameter sweep and emit results.

    Signals
    -------
    progress(combo_index, total_combos, current_params_dict)
        Emitted before each combo runs.  ``combo_index`` is 1-based.

    result_ready(combo_index, params_dict, per_image_scores, mean_score, type_means, per_image_type_scores)
        Emitted after each combo completes.
        ``per_image_scores`` is a list of floats, one per image.
        ``per_image_type_scores`` is a list (same length) of ``Dict[str, float]`` or ``None`` for skipped images.

    finished_sweep()
        Emitted once after all combos complete (or after interruption).

    error(message)
        Emitted on an unexpected exception; sweep continues with remaining combos.
    """

    progress      = Signal(int, int, dict)              # (combo_idx, total, params)
    result_ready  = Signal(int, dict, list, float, dict, list)  # (combo_idx, params, per_img, mean, type_means, per_img_types)
    finished_sweep = Signal()
    error         = Signal(str)

    def __init__(
        self,
        combos: List[Dict[str, Any]],
        image_paths: List[str],
        reference_folders: List[str],
        image_gts: Dict[str, "ImageGT"],
        tolerance_px: int,
        instrument_config_path: str,
        pipette_index: int,
        tip_type: str,
        mode_name: str = "IntensityDetection",
        parent=None,
    ) -> None:
        super().__init__(parent)
        self._mode_name   = mode_name
        self._combos      = combos
        self._image_paths = image_paths
        # Pre-scan each reference folder into a sorted file list
        _exts = {".jpg", ".jpeg", ".png", ".bmp"}
        self._reference_path_groups: List[List[str]] = []
        for folder in reference_folders:
            if os.path.isdir(folder):
                files = sorted(
                    os.path.join(folder, f)
                    for f in os.listdir(folder)
                    if os.path.splitext(f)[1].lower() in _exts
                )
                if files:
                    self._reference_path_groups.append(files)
        self._image_gts   = image_gts
        self._tolerance_px        = tolerance_px
        self._instrument_config_path = instrument_config_path
        self._pipette_index       = pipette_index
        self._tip_type            = tip_type

    # ── Public helper ─────────────────────────────────────────────────────

    @staticmethod
    def estimate_seconds(
        n_combos: int,
        n_images: int,
        combos: List[Dict[str, Any]],
    ) -> float:
        """Rough time estimate in seconds."""
        unique_signal_keys = {_signal_cache_key(c) for c in combos}
        n_signal_runs = len(unique_signal_keys) * n_images
        # ~0.25 s per signal run; scoring is negligible
        return n_signal_runs * 0.25

    # ── QThread entry point ───────────────────────────────────────────────

    def run(self) -> None:
        try:
            self._run_sweep()
        finally:
            self.finished_sweep.emit()

    def _run_sweep(self) -> None:
        combos = self._combos
        total  = len(combos)

        # Build a per-image signal cache: (image_path, signal_cache_key) → (signal, z_axis, roi_y0)
        signal_cache: Dict[Tuple[str, frozenset], Tuple[np.ndarray, np.ndarray, int]] = {}

        instrument_config = self._load_instrument_config()

        for i, params in enumerate(combos, start=1):
            if self.isInterruptionRequested():
                break

            self.progress.emit(i, total, dict(params))

            per_image_results: List[ScoreResult] = []

            for img_path in self._image_paths:
                if self.isInterruptionRequested():
                    break

                try:
                    sig_key = (img_path, _signal_cache_key(params))

                    if sig_key in signal_cache:
                        signal, z_axis, roi_y0 = signal_cache[sig_key]
                    else:
                        signal, z_axis, roi_y0 = self._compute_signal(
                            img_path, params, instrument_config
                        )
                        signal_cache[sig_key] = (signal, z_axis, roi_y0)

                    gt = self._image_gts.get(img_path)
                    annotations = gt.annotations if gt else []

                    score_result = score_signal(
                        signal, z_axis, annotations, roi_y0, self._tolerance_px,
                        intensity_threshold=float(params.get("intensity_threshold", 0.0)),
                    )
                    per_image_results.append(score_result)

                except Exception as exc:
                    import traceback
                    self.error.emit(
                        f"[combo {i}, image {os.path.basename(img_path)}] "
                        f"{exc}\n{traceback.format_exc()}"
                    )
                    per_image_results.append(ScoreResult())

            mean, per_scores, type_means, per_img_types = aggregate_scores(per_image_results)
            self.result_ready.emit(i, dict(params), per_scores, mean, type_means, per_img_types)

    # ── Internal helpers ──────────────────────────────────────────────────

    def _load_instrument_config(self) -> Dict[str, Any]:
        if not self._instrument_config_path:
            return {}
        try:
            import json
            with open(self._instrument_config_path, "r", encoding="utf-8") as fh:
                return json.load(fh)
        except Exception:
            return {}

    def _build_image_set_for(
        self,
        img_path: str,
        instrument_config: Dict[str, Any],
    ):
        """Load one sample image (+ paired reference if available) and build ImageSet."""
        from pa.pipeline.types import ImageSet
        from pa.analysis.shared_geometry import lookup_roi

        img = cv2.imread(img_path)
        if img is None:
            raise RuntimeError(f"Could not read image: {img_path}")

        # Paired reference by sort order; average across all reference folders
        img_idx = self._image_paths.index(img_path) if img_path in self._image_paths else 0
        ref_frames: List[np.ndarray] = []
        ref_paths:  List[str]        = []
        if self._reference_path_groups:
            raw_refs: List[np.ndarray] = []
            for group in self._reference_path_groups:
                if img_idx < len(group):
                    r = cv2.imread(group[img_idx])
                    if r is not None:
                        raw_refs.append(r.astype(np.float32))
                        ref_paths.append(group[img_idx])
            if raw_refs:
                if len(raw_refs) == 1:
                    averaged = raw_refs[0].astype(np.uint8)
                else:
                    averaged = np.mean(np.stack(raw_refs, axis=0), axis=0).astype(np.uint8)
                ref_frames = [averaged]

        # ROI from instrument config
        roi_bbox   = None
        roi_points = None
        if instrument_config:
            try:
                roi_bbox, roi_points = lookup_roi(
                    instrument_config, self._pipette_index, self._tip_type or None
                )
            except Exception:
                pass

        return ImageSet(
            pipette_index=self._pipette_index,
            frames=[img],
            source_paths=[img_path],
            roi=roi_bbox,
            roi_points=roi_points,
            reference_frames=ref_frames,
            reference_source_paths=ref_paths,
        )

    def _compute_signal(
        self,
        img_path: str,
        params: Dict[str, Any],
        instrument_config: Dict[str, Any],
    ) -> Tuple[np.ndarray, np.ndarray, int]:
        """Run the selected mode's run_debug() and return (signal, z_axis, roi_y0)."""
        from pa.pipeline.cache import ProcessedImageCache
        if self._mode_name == "RowContrastDetection":
            from pa.pipeline.modes.row_contrast_detection import RowContrastDetection as _Mode
        else:
            from pa.pipeline.modes.intensity_detection import IntensityDetection as _Mode

        image_set = self._build_image_set_for(img_path, instrument_config)

        cache = ProcessedImageCache()
        mode  = _Mode(params)
        profile, _features = mode.run_debug(image_set, cache)

        # Normalize the signal exactly as the interpreter does so that
        # intensity_threshold comparisons are valid in the scorer.
        from pa.pipeline.signal_primitives import normalize_signal
        normalized = normalize_signal(profile.signal)

        # roi_y0 after any expansion the mode applied: use roi bbox y
        roi_y0 = int(image_set.roi[1]) if image_set.roi else 0

        return normalized, profile.z_axis_px, roi_y0
