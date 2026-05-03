"""
Background worker that runs a single pipeline mode with two different param
sets (A and B) and returns the resulting debug stages for each.

Used by ABCompareDialog.
"""

from __future__ import annotations

import copy
import traceback
from typing import Any, Dict, List

from PySide6.QtCore import QThread, Signal


class _ABWorkerBase(QThread):
    """Shared image-loading and pipeline-running logic for A/B workers."""

    progress = Signal(str)
    error    = Signal(str)

    def __init__(
        self,
        mode_name: str,
        params_a: Dict[str, Any],
        params_b: Dict[str, Any],
        image_paths: List[str],
        reference_paths: List[str] = (),
        frame_index: int = 0,
        instrument_config_path: str = "",
        tip_type: str = "",
        parent=None,
    ) -> None:
        super().__init__(parent)
        self._mode_name    = mode_name
        self._params_a     = copy.deepcopy(params_a)
        self._params_b     = copy.deepcopy(params_b)
        self._image_paths  = list(image_paths)
        self._reference_paths = list(reference_paths)
        self._frame_index  = frame_index
        self._instrument_config_path = instrument_config_path
        self._tip_type = tip_type or None

    # ------------------------------------------------------------------
    def _load_frames(self):
        """Load images and reference frames; return (frames, reference_frames, source_image).
        Returns None on failure (error signal already emitted)."""
        import cv2

        all_frames: list = []
        for p in self._image_paths:
            img = cv2.imread(p)
            if img is not None:
                all_frames.append(img)
        if not all_frames:
            self.error.emit("No images could be loaded from the selected path.")
            return None

        fi = min(self._frame_index, len(all_frames) - 1)
        source_image = all_frames[fi]
        frames = all_frames if len(all_frames) == 2 else [all_frames[fi]]

        all_ref: list = []
        for p in self._reference_paths:
            import cv2 as _cv2
            img = _cv2.imread(p)
            if img is not None:
                all_ref.append(img)

        if all_ref:
            ref_fi = min(fi, len(all_ref) - 1)
            reference_frames = all_ref if len(all_ref) == 2 else [all_ref[ref_fi]]
            self.progress.emit(
                f"Reference: frame {ref_fi + 1}/{len(all_ref)} loaded"
            )
        else:
            reference_frames = []

        return frames, reference_frames, source_image

    # ------------------------------------------------------------------
    def _resolve_roi(self, ic, pipette_index: int):
        """Resolve ROI bbox + points for a single pipette from loaded IC dict."""
        roi_bbox = None
        roi_points = None
        try:
            from pa.analysis.shared_geometry import lookup_roi
            roi_bbox, roi_points = lookup_roi(ic, pipette_index, self._tip_type)
            if roi_points:
                xs = [p[0] for p in roi_points]
                ys = [p[1] for p in roi_points]
                self.progress.emit(
                    f"ROI: pipette {pipette_index} \u2014 "
                    f"x=[{min(xs):.0f}\u2013{max(xs):.0f}] "
                    f"y=[{min(ys):.0f}\u2013{max(ys):.0f}] px"
                )
            else:
                self.progress.emit(
                    f"ROI: not found for pipette {pipette_index} \u2014 using full image"
                )
        except Exception as exc:
            self.progress.emit(f"ROI warning: {exc}")
        return roi_bbox, roi_points

    # ------------------------------------------------------------------
    def _run_one(self, image_set, params: Dict[str, Any]):
        """Run the configured mode once and return a list of DebugStages."""
        from pa.pipeline.cache import ProcessedImageCache
        from pa.pipeline.runner import (
            _register_modes, _register_interpreters,
            _LIQUID_MODES, _TIP_MODES,
            _LIQUID_INTERPRETERS, _TIP_INTERPRETERS,
        )
        from pa.pipeline.types import DebugData

        _register_modes()
        _register_interpreters()

        cache = ProcessedImageCache()
        debug_data = DebugData(
            step_id="ab_compare",
            pipette_index=image_set.pipette_index,
            source_image=image_set.frames[0].copy() if image_set.frames else None,
        )

        # ── Liquid modes ─────────────────────────────────────────────────
        mode_cls = _LIQUID_MODES.get(self._mode_name)
        if mode_cls is not None:
            mode = mode_cls(params)

            if self._mode_name == "ExtremaLines":
                density_profile, term_profile, features = mode.run_all_debug(
                    image_set, cache
                )
                p = dict(params)
                d_cls = _LIQUID_INTERPRETERS.get("ExtremaLines_Density")
                t_cls = _LIQUID_INTERPRETERS.get("ExtremaLines_Terminations")
                density_pois = d_cls(p).interpret(density_profile) if d_cls else []
                term_pois    = t_cls(p).interpret(term_profile)    if t_cls else []
                from pa.pipeline.debug_collector import collect_extrema_stages
                for stage in collect_extrema_stages(
                    image_set.pipette_index, features,
                    density_profile, term_profile,
                    density_pois, term_pois, cache, params,
                ):
                    debug_data.add(stage)

            else:
                profile, features = mode.run_debug(image_set, cache)
                interp_cls = _LIQUID_INTERPRETERS.get(self._mode_name)
                pois = interp_cls(dict(params)).interpret(profile) if interp_cls else []

                if self._mode_name in ("IntensityDetection", "RowContrastDetection"):
                    from pa.pipeline.debug_collector import collect_intensity_stages
                    for stage in collect_intensity_stages(
                        image_set.pipette_index, profile, pois, cache, params,
                        roi=image_set.roi,
                        roi_points=image_set.roi_points,
                        image_size=(image_set.frames[0].shape[0],
                                    image_set.frames[0].shape[1]) if image_set.frames else None,
                        mode_name=self._mode_name,
                    ):
                        debug_data.add(stage)
                elif self._mode_name.startswith("LineContinuity"):
                    from pa.pipeline.debug_collector import collect_ridge_stages
                    for stage in collect_ridge_stages(
                        image_set.pipette_index, self._mode_name,
                        features, profile, pois, cache, params,
                    ):
                        debug_data.add(stage)

            return debug_data.stages

        # ── Tip modes ──────────────────────────────────────────────────
        mode_cls = _TIP_MODES.get(self._mode_name)
        if mode_cls is not None:
            mode = mode_cls(params)
            tip_region, engine_extras = mode.run_debug(image_set, cache)
            interp_cls = _TIP_INTERPRETERS.get(self._mode_name)
            candidates = (
                interp_cls(dict(params)).interpret(tip_region)
                if interp_cls else []
            )

            if self._mode_name == "InwardEdgeScan":
                from pa.pipeline.debug_collector import collect_inward_edge_stages
                for stage in collect_inward_edge_stages(
                    image_set.pipette_index, engine_extras, cache, params, candidates,
                ):
                    debug_data.add(stage)
            elif self._mode_name == "WindowContrastAlign":
                from pa.pipeline.debug_collector import collect_window_contrast_stages
                best = candidates[0] if candidates else None
                for stage in collect_window_contrast_stages(
                    image_set.pipette_index, cache, params, best,
                ):
                    debug_data.add(stage)
            elif self._mode_name in ("ROIEdgeFit", "ROIContrastFit"):
                from pa.pipeline.debug_collector import collect_roi_fit_stages
                for stage in collect_roi_fit_stages(
                    image_set.pipette_index, self._mode_name, engine_extras, candidates,
                ):
                    debug_data.add(stage)

            return debug_data.stages

        raise ValueError(f"Unknown pipeline mode: {self._mode_name!r}")


class ABCompareWorker(_ABWorkerBase):
    """Runs one pipeline mode with params_a and params_b on the same image (single pipette)."""

    # stages_a, stages_b, source_image (BGR ndarray),
    # contrast_image_a, contrast_image_b (BGR ndarray or None), roi_bbox (tuple|None)
    result_ready = Signal(list, list, object, object, object, object)

    def __init__(
        self,
        mode_name: str,
        params_a: Dict[str, Any],
        params_b: Dict[str, Any],
        image_paths: List[str],
        reference_paths: List[str] = (),
        frame_index: int = 0,
        pipette_index: int = 0,
        instrument_config_path: str = "",
        tip_type: str = "",
        parent=None,
    ) -> None:
        super().__init__(
            mode_name=mode_name,
            params_a=params_a,
            params_b=params_b,
            image_paths=image_paths,
            reference_paths=reference_paths,
            frame_index=frame_index,
            instrument_config_path=instrument_config_path,
            tip_type=tip_type,
            parent=parent,
        )
        self._pipette_index = pipette_index

    # ------------------------------------------------------------------
    def run(self) -> None:
        try:
            import json
            import os
            from pa.pipeline.types import ImageSet

            loaded = self._load_frames()
            if loaded is None:
                return
            frames, reference_frames, source_image = loaded

            # Resolve ROI
            roi_bbox   = None
            roi_points = None
            if self._instrument_config_path and os.path.isfile(self._instrument_config_path):
                try:
                    with open(self._instrument_config_path, encoding="utf-8") as f:
                        ic = json.load(f)
                    roi_bbox, roi_points = self._resolve_roi(ic, self._pipette_index)
                except Exception as exc:
                    self.progress.emit(f"ROI warning: {exc}")
            else:
                self.progress.emit("ROI: no instrument config \u2014 using full image")

            image_set = ImageSet(
                pipette_index=self._pipette_index,
                frames=frames,
                source_paths=list(self._image_paths),
                roi=roi_bbox,
                roi_points=roi_points,
                reference_frames=reference_frames,
                reference_source_paths=list(self._reference_paths),
            )

            self.progress.emit("Running A\u2026")
            stages_a = self._run_one(image_set, self._params_a)
            self.progress.emit("Running B\u2026")
            stages_b = self._run_one(image_set, self._params_b)

            contrast_image_a = None
            contrast_image_b = None
            if reference_frames:
                from pa.pipeline import image_primitives as ip
                ref_display = reference_frames[0]
                sample = frames[0]
                if self._params_a.get("use_contrast", False):
                    contrast_image_a = ip.build_contrast_working_frame(
                        sample, ref_display, self._params_a
                    )
                if self._params_b.get("use_contrast", False):
                    contrast_image_b = ip.build_contrast_working_frame(
                        sample, ref_display, self._params_b
                    )

            self.result_ready.emit(stages_a, stages_b, source_image,
                                   contrast_image_a, contrast_image_b, roi_bbox)

        except Exception:
            self.error.emit(traceback.format_exc())


# ---------------------------------------------------------------------------
# Multi-tip worker
# ---------------------------------------------------------------------------

class ABMultiTipWorker(_ABWorkerBase):
    """Runs A/B comparison for all pipettes in the instrument config.

    Emits ``multi_result_ready`` once with a list of per-pipette result dicts:
        {
            "pipette_index": int,           # 0-based
            "stages_a":      List[DebugStage],
            "stages_b":      List[DebugStage],
        }
    """

    multi_result_ready = Signal(list)   # List[dict]

    def __init__(
        self,
        mode_name: str,
        params_a: Dict[str, Any],
        params_b: Dict[str, Any],
        image_paths: List[str],
        reference_paths: List[str] = (),
        frame_index: int = 0,
        pipette_count: int = 8,
        instrument_config_path: str = "",
        tip_type: str = "",
        parent=None,
    ) -> None:
        super().__init__(
            mode_name=mode_name,
            params_a=params_a,
            params_b=params_b,
            image_paths=image_paths,
            reference_paths=reference_paths,
            frame_index=frame_index,
            instrument_config_path=instrument_config_path,
            tip_type=tip_type,
            parent=parent,
        )
        self._pipette_count = pipette_count

    # ------------------------------------------------------------------
    def run(self) -> None:
        try:
            import json
            import os
            from pa.pipeline.types import ImageSet

            loaded = self._load_frames()
            if loaded is None:
                return
            frames, reference_frames, _source_image = loaded

            # Load IC once for all ROI lookups
            ic = None
            if self._instrument_config_path and os.path.isfile(self._instrument_config_path):
                try:
                    with open(self._instrument_config_path, encoding="utf-8") as f:
                        ic = json.load(f)
                except Exception as exc:
                    self.progress.emit(f"IC load warning: {exc}")

            results: List[dict] = []
            n = self._pipette_count

            for i in range(n):
                if self.isInterruptionRequested():
                    break

                self.progress.emit(f"Running tip {i + 1}/{n}\u2026")

                roi_bbox, roi_points = (
                    self._resolve_roi(ic, i) if ic is not None
                    else (None, None)
                )

                image_set = ImageSet(
                    pipette_index=i,
                    frames=frames,
                    source_paths=list(self._image_paths),
                    roi=roi_bbox,
                    roi_points=roi_points,
                    reference_frames=reference_frames,
                    reference_source_paths=list(self._reference_paths),
                )

                stages_a = self._run_one(image_set, self._params_a)
                stages_b = self._run_one(image_set, self._params_b)

                results.append({
                    "pipette_index": i,
                    "stages_a": stages_a,
                    "stages_b": stages_b,
                })

            self.multi_result_ready.emit(results)

        except Exception:
            self.error.emit(traceback.format_exc())
