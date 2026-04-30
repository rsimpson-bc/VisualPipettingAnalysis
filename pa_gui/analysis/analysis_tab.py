"""
AnalysisTab — top-level widget for the "Run Analysis" tab.

Layout:

    ┌─ Config toolbar ────────────────────────────────────────────────────────┐
    │ Run Folder:        [__________________________________]  [Browse]        │
    │ Instrument Config: [__________________]  [Browse]                       │
    │ Analysis Config:   [__________________]  [Browse]                       │
    └─────────────────────────────────────────────────────────────────────────┘
    ┌─ Left (1) ─────────────┬─ Right (3) ─────────────────────────────────┐
    │ Steps:                 │  [Step Editor tab]  [Results tab]           │
    │  step_001  [TipID]     │                                             │
    │  step_002  [LiqMeas] ✓ │  < StepEditorWidget or ResultsPanel >      │
    │                        │                                             │
    │  [+] [−] [↑] [↓]       │                                             │
    │  ──────────────────    │                                             │
    │  [▶ Run All]           │                                             │
    │  [■ Stop]              │                                             │
    │  [progress bar]        │                                             │
    └────────────────────────┴─────────────────────────────────────────────┘
    ┌─ Status bar ────────────────────────────────────────────────────────────┐
    │ Ready                                                                   │
    └─────────────────────────────────────────────────────────────────────────┘
"""

from __future__ import annotations

import json
from typing import List, Optional

from PySide6.QtCore import Qt, QSettings, QTimer
from PySide6.QtWidgets import (
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QSplitter,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from pa_gui.analysis.models import StepDef
from pa_gui.analysis.runner import AnalysisRunner
from pa_gui.analysis.results_panel import ResultsPanel
from pa_gui.analysis.step_editor import StepEditorWidget
from pa_gui.params.pipeline_config_editor import PipelineConfigEditor

_S_ORG = "rsimpson-bc"
_S_APP = "PA-GUI"


class AnalysisTab(QWidget):
    """Main widget for the Run Analysis tab."""

    def __init__(
        self,
        instrument_config_path: Optional[str] = None,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self._steps: List[StepDef] = []
        self._runner: Optional[AnalysisRunner] = None

        self._build_ui()
        self._restore_settings()

        if instrument_config_path:
            self._ic_edit.setText(instrument_config_path)

    # ── UI construction ────────────────────────────────────────────────────────

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(6, 6, 6, 6)
        root.setSpacing(4)

        root.addWidget(self._build_config_toolbar())

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(self._build_left_panel())
        splitter.addWidget(self._build_right_panel())
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 3)
        root.addWidget(splitter, 1)

        root.addWidget(self._build_status_bar())

    def _build_config_toolbar(self) -> QWidget:
        bar = QWidget()
        layout = QVBoxLayout(bar)
        layout.setContentsMargins(0, 0, 0, 4)
        layout.setSpacing(3)

        # Row 1 — Run Folder
        r1 = QHBoxLayout()
        r1.addWidget(QLabel("Run Folder:"))
        self._run_folder_edit = QLineEdit()
        self._run_folder_edit.setPlaceholderText("Select run folder containing image subfolders…")
        r1.addWidget(self._run_folder_edit, 1)
        btn_rf = QPushButton("Browse…")
        btn_rf.clicked.connect(self._on_browse_run_folder)
        r1.addWidget(btn_rf)
        layout.addLayout(r1)

        # Row 2 — Instrument Config + Analysis Config
        r2 = QHBoxLayout()
        r2.addWidget(QLabel("Instrument Config:"))
        self._ic_edit = QLineEdit()
        self._ic_edit.setPlaceholderText("instrument_config.json")
        r2.addWidget(self._ic_edit, 1)
        btn_ic = QPushButton("Browse…")
        btn_ic.clicked.connect(lambda: self._browse_json(self._ic_edit, "instrument_config_path"))
        r2.addWidget(btn_ic)
        r2.addSpacing(16)
        r2.addWidget(QLabel("Analysis Config:"))
        self._ac_edit = QLineEdit()
        self._ac_edit.setPlaceholderText("analysis_config.json")
        r2.addWidget(self._ac_edit, 1)
        btn_ac = QPushButton("Browse…")
        btn_ac.clicked.connect(lambda: self._browse_json(self._ac_edit, "analysis/analysis_config_path"))
        r2.addWidget(btn_ac)
        btn_edit_params = QPushButton("Edit Params…")
        btn_edit_params.setToolTip("Open the pipeline parameter editor for the current analysis config.")
        btn_edit_params.clicked.connect(self._on_edit_params)
        r2.addWidget(btn_edit_params)
        layout.addLayout(r2)

        return bar

    def _build_left_panel(self) -> QWidget:
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 4, 0)
        layout.setSpacing(4)

        layout.addWidget(QLabel("<b>Steps</b>"))

        # Save/load row
        file_row = QHBoxLayout()
        self._btn_save = QPushButton("💾 Save Steps…")
        self._btn_save.setToolTip("Save current step list to a JSON file")
        self._btn_save.clicked.connect(self._on_save_steps)
        self._btn_load = QPushButton("📂 Load Steps…")
        self._btn_load.setToolTip("Load a previously saved step list from a JSON file")
        self._btn_load.clicked.connect(self._on_load_steps)
        file_row.addWidget(self._btn_save)
        file_row.addWidget(self._btn_load)
        file_row.addStretch()
        layout.addLayout(file_row)

        self._step_list = QListWidget()
        self._step_list.currentRowChanged.connect(self._on_step_selected)
        layout.addWidget(self._step_list, 1)

        # Step management buttons
        btn_row = QHBoxLayout()
        for symbol, tip, slot in [
            ("+",  "Add new step",       self._on_add_step),
            ("−",  "Remove step",        self._on_remove_step),
            ("↑",  "Move step up",       self._on_move_up),
            ("↓",  "Move step down",     self._on_move_down),
        ]:
            b = QPushButton(symbol)
            b.setFixedWidth(30)
            b.setToolTip(tip)
            b.clicked.connect(slot)
            btn_row.addWidget(b)
        btn_row.addStretch()
        layout.addLayout(btn_row)

        # Divider
        sep = QWidget()
        sep.setFixedHeight(1)
        sep.setStyleSheet("background:#555;")
        layout.addWidget(sep)

        # Run / Stop
        self._run_btn = QPushButton("▶  Run All")
        self._run_btn.setFixedHeight(36)
        self._run_btn.setStyleSheet(
            "QPushButton { background:#2a7a2a; color:white; font-weight:bold; }"
            "QPushButton:disabled { background:#444; color:#888; }"
        )
        self._run_btn.clicked.connect(self._on_run_all)
        layout.addWidget(self._run_btn)

        self._stop_btn = QPushButton("■  Stop")
        self._stop_btn.setFixedHeight(26)
        self._stop_btn.setEnabled(False)
        self._stop_btn.setStyleSheet(
            "QPushButton:enabled { color:#e74c3c; font-weight:bold; }"
        )
        self._stop_btn.clicked.connect(self._on_stop)
        layout.addWidget(self._stop_btn)

        self._progress = QProgressBar()
        self._progress.setVisible(False)
        self._progress.setTextVisible(True)
        layout.addWidget(self._progress)

        return panel

    def _build_right_panel(self) -> QWidget:
        self._right_tabs = QTabWidget()

        self._editor = StepEditorWidget()
        self._editor.step_saved.connect(self._on_step_saved)
        self._right_tabs.addTab(self._editor, "Step Editor")

        self._results = ResultsPanel()
        self._results.step_selected.connect(self._on_results_step_selected)
        self._right_tabs.addTab(self._results, "Results")

        from pa_gui.analysis.debug_inspector import DebugInspectorWidget
        self._inspector = DebugInspectorWidget()
        self._inspector.inspect_requested.connect(self._on_inspect_requested)
        self._right_tabs.addTab(self._inspector, "Inspect")

        return self._right_tabs

    def _build_status_bar(self) -> QLabel:
        self._status = QLabel("Ready")
        self._status.setWordWrap(True)
        self._status.setStyleSheet(
            "QLabel { background:#f0f0f0; border-top:1px solid #ccc; padding:2px 6px; }"
        )
        return self._status

    # ── Settings ───────────────────────────────────────────────────────────────

    def _restore_settings(self) -> None:
        s = QSettings(_S_ORG, _S_APP)
        for edit, key in [
            (self._run_folder_edit, "analysis/run_folder"),
            (self._ac_edit,         "analysis/analysis_config_path"),
            (self._ic_edit,         "instrument_config_path"),
        ]:
            val = s.value(key, "")
            if val and not edit.text():
                edit.setText(str(val))
        # Push run folder to editor
        rf = self._run_folder_edit.text()
        if rf:
            self._editor.set_run_folder(rf)

    # ── Toolbar slots ──────────────────────────────────────────────────────────

    def _on_save_steps(self) -> None:
        """Save the current step list to a JSON file."""
        s = QSettings(_S_ORG, _S_APP)
        last = s.value("analysis/steps_file", "")
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Step List", last,
            "Step list (*.steps.json);;JSON files (*.json);;All files (*)"
        )
        if not path:
            return
        if not path.endswith(".json"):
            path += ".steps.json"
        payload = {
            "run_folder":        self._run_folder_edit.text().strip(),
            "instrument_config": self._ic_edit.text().strip(),
            "analysis_config":   self._ac_edit.text().strip(),
            "steps": [step.to_dict() for step in self._steps],
        }
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2)
            s.setValue("analysis/steps_file", path)
            self._status.setText(f"Steps saved to {path}")
            self._flash_button(self._btn_save, "✓ Saved", "💾 Save Steps…")
        except OSError as exc:
            QMessageBox.critical(self, "Save failed", str(exc))

    def _on_load_steps(self) -> None:
        """Load a step list from a JSON file."""
        s = QSettings(_S_ORG, _S_APP)
        last = s.value("analysis/steps_file", "")
        path, _ = QFileDialog.getOpenFileName(
            self, "Load Step List", last,
            "Step list (*.steps.json);;JSON files (*.json);;All files (*)"
        )
        if not path:
            return
        try:
            with open(path, "r", encoding="utf-8") as f:
                payload = json.load(f)
        except (OSError, json.JSONDecodeError) as exc:
            QMessageBox.critical(self, "Load failed", str(exc))
            return

        from pa_gui.analysis.models import StepDef as _StepDef
        try:
            steps = [_StepDef.from_dict(d) for d in payload.get("steps", [])]
        except (KeyError, ValueError, TypeError) as exc:
            QMessageBox.critical(self, "Invalid step file", str(exc))
            return

        # Apply paths from file only if the fields are currently empty
        if payload.get("run_folder") and not self._run_folder_edit.text().strip():
            self._run_folder_edit.setText(payload["run_folder"])
        if payload.get("instrument_config") and not self._ic_edit.text().strip():
            self._ic_edit.setText(payload["instrument_config"])
        if payload.get("analysis_config") and not self._ac_edit.text().strip():
            self._ac_edit.setText(payload["analysis_config"])

        self._steps = steps
        self._refresh_step_list()
        self._results.populate(self._steps)
        s.setValue("analysis/steps_file", path)
        self._status.setText(f"Loaded {len(steps)} step(s) from {path}")
        self._flash_button(self._btn_load, "✓ Loaded", "📂 Load Steps…")

    def _flash_button(self, btn: QPushButton, flash_text: str, original_text: str, ms: int = 1500) -> None:
        """Briefly change a button's label to confirm an action, then restore it."""
        btn.setText(flash_text)
        btn.setEnabled(False)
        QTimer.singleShot(ms, lambda: (btn.setText(original_text), btn.setEnabled(True)))

    def _on_browse_run_folder(self) -> None:
        folder = QFileDialog.getExistingDirectory(
            self, "Select Run Folder", self._run_folder_edit.text()
        )
        if folder:
            self._run_folder_edit.setText(folder)
            self._editor.set_run_folder(folder)
            QSettings(_S_ORG, _S_APP).setValue("analysis/run_folder", folder)

    def _browse_json(self, edit: QLineEdit, settings_key: str) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Open JSON", edit.text(),
            "JSON files (*.json);;All files (*)"
        )
        if path:
            edit.setText(path)
            QSettings(_S_ORG, _S_APP).setValue(settings_key, path)

    def _on_edit_params(self) -> None:
        """Open the pipeline parameter editor in a dialog."""
        from PySide6.QtWidgets import QDialog, QDialogButtonBox
        dlg = QDialog(self)
        dlg.setWindowTitle("Pipeline Parameter Editor")
        dlg.resize(900, 650)
        layout = QVBoxLayout(dlg)
        layout.setContentsMargins(6, 6, 6, 6)
        editor = PipelineConfigEditor(
            self._ac_edit.text() or None,
            instrument_config_path=self._ic_edit.text() or None,
            parent=dlg,
        )
        editor.config_saved.connect(
            lambda path: self._ac_edit.setText(path)
        )
        layout.addWidget(editor, 1)
        btn_box = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        btn_box.rejected.connect(dlg.reject)
        layout.addWidget(btn_box)
        dlg.exec()

    # ── Step list management ───────────────────────────────────────────────────

    def _on_add_step(self) -> None:
        n = len(self._steps) + 1
        step = StepDef(step_id=f"step_{n:03d}", analysis_type="TipIdentification")
        self._steps.append(step)
        self._step_list.addItem(self._step_label(step))
        self._step_list.setCurrentRow(len(self._steps) - 1)

    def _on_remove_step(self) -> None:
        row = self._step_list.currentRow()
        if 0 <= row < len(self._steps):
            self._steps.pop(row)
            self._step_list.takeItem(row)

    def _on_move_up(self) -> None:
        row = self._step_list.currentRow()
        if row > 0:
            self._steps[row], self._steps[row - 1] = self._steps[row - 1], self._steps[row]
            self._refresh_step_list()
            self._step_list.setCurrentRow(row - 1)

    def _on_move_down(self) -> None:
        row = self._step_list.currentRow()
        if 0 <= row < len(self._steps) - 1:
            self._steps[row], self._steps[row + 1] = self._steps[row + 1], self._steps[row]
            self._refresh_step_list()
            self._step_list.setCurrentRow(row + 1)

    def _on_step_selected(self, row: int) -> None:
        if 0 <= row < len(self._steps):
            self._editor.load_step(self._steps[row])
            self._right_tabs.setCurrentWidget(self._editor)

    def _on_step_saved(self, step: StepDef) -> None:
        row = self._step_list.currentRow()
        if 0 <= row < len(self._steps):
            # Preserve existing run result.
            existing = self._steps[row]
            step.result = existing.result
            step.error  = existing.error
            self._steps[row] = step
        else:
            self._steps.append(step)
        self._refresh_step_list()
        self._status.setText(f"Step saved: {step.step_id}")

    def _refresh_step_list(self) -> None:
        current = self._step_list.currentRow()
        self._step_list.clear()
        for s in self._steps:
            self._step_list.addItem(self._step_label(s))
        self._step_list.setCurrentRow(current)

    @staticmethod
    def _step_label(step: StepDef) -> str:
        _icons = {
            "pending": "", "ok": " ✓", "warning": " ⚠",
            "low_confidence": " ⚠", "error": " ✗", "failed": " ✗",
        }
        short = "TipID" if step.analysis_type == "TipIdentification" else "LiqMeas"
        icon  = _icons.get(step.status, "")
        return f"{step.step_id}  [{short}]{icon}"

    # ── Runner ─────────────────────────────────────────────────────────────────

    def _on_run_all(self) -> None:
        if not self._steps:
            QMessageBox.information(
                self, "No steps", "Add at least one step before running."
            )
            return

        rf = self._run_folder_edit.text().strip()
        ic = self._ic_edit.text().strip()
        ac = self._ac_edit.text().strip()
        missing = [(lbl, v) for lbl, v in [
            ("Run folder",        rf),
            ("Instrument config", ic),
            ("Analysis config",   ac),
        ] if not v]
        if missing:
            labels = ", ".join(lbl for lbl, _ in missing)
            QMessageBox.warning(self, "Missing configuration",
                                f"Please set: {labels}")
            return

        # Validate: every step must have at least one pipette
        steps_no_pipettes = [s.step_id for s in self._steps if not s.pipettes]
        if steps_no_pipettes:
            QMessageBox.warning(
                self, "Steps missing pipettes",
                "The following steps have no pipettes configured:\n  "
                + "\n  ".join(steps_no_pipettes)
                + "\n\nAdd at least one pipette in the Step Editor before running."
            )
            return

        # Reset previous results
        for step in self._steps:
            step.result = None
            step.error  = None
            step.debug_data = None
        self._refresh_step_list()
        self._results.populate(self._steps)
        self._right_tabs.setCurrentWidget(self._results)

        self._run_btn.setEnabled(False)
        self._stop_btn.setEnabled(True)
        self._progress.setVisible(True)
        self._progress.setMaximum(len(self._steps))
        self._progress.setValue(0)
        self._status.setText("Running…")

        self._runner = AnalysisRunner(
            steps=self._steps,
            run_folder=rf,
            ic_path=ic,
            ac_path=ac,
            parent=self,
        )
        self._runner.step_started.connect(self._on_step_started)
        self._runner.step_finished.connect(self._on_step_finished)
        self._runner.step_failed.connect(self._on_step_failed)
        self._runner.all_finished.connect(self._on_all_finished)
        self._runner.start()

    def _on_stop(self) -> None:
        if self._runner:
            self._runner.abort()
        self._status.setText("Stopping after current step…")

    def _on_results_step_selected(self, step: StepDef) -> None:
        """Load the debug inspector when the user clicks a step in the results tree."""
        self._inspector.load(step)
        self._right_tabs.setCurrentWidget(self._inspector)

    def _on_inspect_requested(self, debug_data, initial_stage: int) -> None:
        """Open the StageDetailDialog for the given DebugData."""
        from pa_gui.analysis.stage_detail_dialog import StageDetailDialog
        dlg = StageDetailDialog(debug_data, initial_stage, parent=self)
        dlg.exec()

    def _on_step_started(self, step_id: str) -> None:
        self._status.setText(f"Running: {step_id}…")

    def _on_step_finished(self, step_id: str, payload: object) -> None:
        step = self._find_step(step_id)
        if step:
            # payload is (result_dict, debug_data_list) from the runner
            if isinstance(payload, tuple) and len(payload) == 2:
                result, debug_data_list = payload
                step.result = result  # type: ignore[assignment]
                step.debug_data = debug_data_list
            else:
                step.result = payload  # type: ignore[assignment]
        self._progress.setValue(self._progress.value() + 1)
        self._refresh_step_list()
        if step:
            self._results.update_step(step)

    def _on_step_failed(self, step_id: str, error: str) -> None:
        step = self._find_step(step_id)
        if step:
            step.error = error
        self._progress.setValue(self._progress.value() + 1)
        self._refresh_step_list()
        if step:
            self._results.update_step(step)

    def _on_all_finished(self) -> None:
        self._run_btn.setEnabled(True)
        self._stop_btn.setEnabled(False)
        self._progress.setVisible(False)
        ok   = sum(1 for s in self._steps if s.status == "ok")
        warn = sum(1 for s in self._steps if s.status in ("warning", "low_confidence"))
        fail = sum(1 for s in self._steps if s.status in ("error", "failed"))
        total = len(self._steps)
        self._status.setText(
            f"Done — {ok}/{total} OK,  {warn} warning,  {fail} failed"
        )
        self._results.populate(self._steps)

    def _find_step(self, step_id: str) -> Optional[StepDef]:
        for s in self._steps:
            if s.step_id == step_id:
                return s
        return None
