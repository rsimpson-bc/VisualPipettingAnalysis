"""
Background worker thread for running PA analysis steps directly in-process.

The runner calls pa.state.session_state.startup() once (loading Tier 1 data),
constructs a SessionContext for the supplied run folder, then iterates the
step list calling the appropriate analysis function for each step.

All signals are emitted from the worker thread and delivered to the main thread
via Qt's queued connection mechanism.
"""

from __future__ import annotations

from typing import List

import traceback

from PySide6.QtCore import QThread, Signal

from pa_gui.analysis.models import StepDef


class AnalysisRunner(QThread):
    step_started  = Signal(str)          # step_id
    step_finished = Signal(str, object)  # step_id, result dict
    step_failed   = Signal(str, str)     # step_id, error message
    all_finished  = Signal()

    def __init__(
        self,
        steps: List[StepDef],
        run_folder: str,
        ic_path: str,
        ac_path: str,
        parent=None,
    ) -> None:
        super().__init__(parent)
        # Capture a snapshot of the step list so the runner is not affected
        # by UI edits that happen after the run starts.
        self._steps: List[StepDef] = list(steps)
        self._run_folder = run_folder
        self._ic_path    = ic_path
        self._ac_path    = ac_path
        self._abort      = False

    def abort(self) -> None:
        """Request a graceful stop after the current step completes."""
        self._abort = True

    # ------------------------------------------------------------------
    def run(self) -> None:
        try:
            # Import lazily so startup cost is paid only when the runner
            # actually starts, not when the GUI is loaded.
            from pa.state import session_state
            from pa.state.session_state import SessionContext
            from pa.api.schemas_http import AnalyzeStepRequest, PipetteCommand
            import pa.analysis.liquid_measurement as lm
            import pa.analysis.tip_identification as ti

            # Tier 1: load instrument config + analysis config into the
            # session_state global (idempotent; safe to call again).
            session_state.startup(self._ic_path, self._ac_path)

            # Tier 2: create a minimal session context from the run folder.
            # We skip method_context.json because the GUI builds steps manually.
            ctx = SessionContext(method_context={}, run_folder=self._run_folder)

            for step in self._steps:
                if self._abort:
                    break

                self.step_started.emit(step.step_id)
                try:
                    req = AnalyzeStepRequest(
                        step_id=step.step_id,
                        analysis_type=step.analysis_type,
                        subfolders=step.subfolders,
                        pipettes=[
                            PipetteCommand(
                                index=p.index,
                                expected_tip_type=p.tip_type,
                                expected_liquid_ul=p.expected_liquid_ul,
                            )
                            for p in step.pipettes
                        ],
                    )
                    if step.analysis_type == "TipIdentification":
                        result, debug_data_list = ti.analyze(req, ctx, debug=True)
                    else:
                        result, debug_data_list = lm.analyze(req, ctx, debug=True)

                    self.step_finished.emit(step.step_id, (result, debug_data_list))

                except Exception as exc:
                    tb = traceback.format_exc()
                    self.step_failed.emit(step.step_id, tb)

        except Exception as exc:
            # Startup failure — mark every step as failed.
            tb = traceback.format_exc()
            msg = f"Startup error: {exc}\n\n{tb}"
            for step in self._steps:
                self.step_failed.emit(step.step_id, msg)

        finally:
            self.all_finished.emit()
