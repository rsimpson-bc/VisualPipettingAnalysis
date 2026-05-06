"""
Background worker for Mode Compare dialog.

Runs a single pipeline mode once per selected preset (sequentially) on the
same image / pipette, emitting a result signal after each preset so the UI
can populate incrementally.
"""

from __future__ import annotations

import copy
import traceback
from typing import Any, Dict, List, Optional

from PySide6.QtCore import Signal

from pa_gui.analysis.ab_compare_worker import _ABWorkerBase


class ModeCompareWorker(_ABWorkerBase):
    """
    Runs one pipeline mode with N different preset param sets on the same image.

    Signals
    -------
    preset_done(int, list, object, object, object)
        Emitted after each preset completes:
          - preset_idx  : int  — index into the preset_entries list
          - stages      : List[DebugStage]
          - source_image: np.ndarray (BGR)
          - contrast_image: np.ndarray | None  — contrast frame if use_contrast=True
          - roi_bbox    : tuple | None  — (x, y, w, h)

    all_done()
        Emitted once after all presets have been processed.

    progress(str)   — inherited
    error(str)      — inherited
    """

    preset_done = Signal(int, list, object, object, object)
    all_done    = Signal()

    def __init__(
        self,
        mode_name: str,
        preset_entries: List[Dict[str, Any]],
        image_paths: List[str],
        reference_paths: List[str] = (),
        frame_index: int = 0,
        pipette_index: int = 0,
        instrument_config_path: str = "",
        tip_type: str = "",
        parent=None,
    ) -> None:
        # _ABWorkerBase expects params_a and params_b; supply dummies —
        # we override run() completely.
        super().__init__(
            mode_name=mode_name,
            params_a={},
            params_b={},
            image_paths=image_paths,
            reference_paths=reference_paths,
            frame_index=frame_index,
            instrument_config_path=instrument_config_path,
            tip_type=tip_type,
            parent=parent,
        )
        self._preset_entries = list(preset_entries)
        self._pipette_index  = pipette_index

    # ------------------------------------------------------------------
    def run(self) -> None:
        try:
            import json
            import os
            from pa.pipeline.types import ImageSet
            from pa.pipeline import image_primitives as ip

            loaded = self._load_frames()
            if loaded is None:
                return
            frames, reference_frames, source_image = loaded

            # Resolve ROI once — shared across all presets
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
                self.progress.emit("ROI: no instrument config — using full image")

            image_set = ImageSet(
                pipette_index=self._pipette_index,
                frames=frames,
                source_paths=list(self._image_paths),
                roi=roi_bbox,
                roi_points=roi_points,
                reference_frames=reference_frames,
                reference_source_paths=list(self._reference_paths),
            )

            for idx, entry in enumerate(self._preset_entries):
                if self.isInterruptionRequested():
                    break
                desc = entry.get("description", f"Preset {idx + 1}")
                self.progress.emit(
                    f"Running preset {idx + 1}/{len(self._preset_entries)}: "
                    f"{desc[:40]}…" if len(desc) > 40 else
                    f"Running preset {idx + 1}/{len(self._preset_entries)}: {desc}"
                )

                params = copy.deepcopy(entry.get("params_b", {}))

                # Override mode_name per preset so _run_one dispatches correctly
                self._mode_name = entry.get("mode", self._mode_name)

                try:
                    stages = self._run_one(image_set, params)
                except Exception as exc:
                    self.progress.emit(
                        f"Preset {idx + 1} failed: {exc}"
                    )
                    stages = []

                # Build per-preset contrast image if use_contrast is enabled
                contrast_image: Optional[object] = None
                if reference_frames and params.get("use_contrast", False):
                    try:
                        contrast_image = ip.build_contrast_working_frame(
                            frames[0], reference_frames[0], params
                        )
                    except Exception:
                        contrast_image = None

                self.preset_done.emit(idx, stages, source_image, contrast_image, roi_bbox)

            self.all_done.emit()

        except Exception:
            self.error.emit(traceback.format_exc())
