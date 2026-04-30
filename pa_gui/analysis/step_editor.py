"""
StepEditorWidget — form for creating or editing a single StepDef.

Layout:

    Step ID: [___________]   Type: [TipIdentification ▼]
    ┌─ Image subfolders ─────────────────────────────────┐
    │  subfolder_A                                       │
    │  subfolder_B                                       │
    │  [Add…]  [Remove]                                  │
    └────────────────────────────────────────────────────┘
    ┌─ Pipettes ─────────────────────────────────────────┐
    │  Tip #  │  Tip Type  │  Expected µL               │
    │  [Add Pipette]  [Remove]                           │
    └────────────────────────────────────────────────────┘
                                           [Save Step]
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QListWidget,
    QPushButton,
    QSpinBox,
    QTableWidget,
    QVBoxLayout,
    QWidget,
)

from pa_gui.analysis.models import ANALYSIS_TYPES, PipetteDef, StepDef
from pa_gui.roi_calibration.models import TIP_TYPES


class StepEditorWidget(QWidget):
    """Form for creating / editing a single StepDef."""

    step_saved = Signal(object)  # StepDef

    def __init__(self, run_folder: str = "", parent=None) -> None:
        super().__init__(parent)
        self._run_folder = run_folder
        self._build_ui()

    # ── Public API ──────────────────────────────────────────────────────────

    def set_run_folder(self, path: str) -> None:
        self._run_folder = path

    def load_step(self, step: StepDef) -> None:
        """Populate the form from an existing StepDef."""
        self._id_edit.setText(step.step_id)
        self._type_combo.setCurrentText(step.analysis_type)
        self._sub_list.clear()
        for s in step.subfolders:
            self._sub_list.addItem(s)
        self._pip_table.setRowCount(0)
        for p in step.pipettes:
            self._add_pipette_row(p)

    def clear(self) -> None:
        self._id_edit.clear()
        self._type_combo.setCurrentIndex(0)
        self._sub_list.clear()
        self._pip_table.setRowCount(0)

    def current_step(self) -> Optional[StepDef]:
        """Build and return a StepDef from the current form values, or None."""
        step_id = self._id_edit.text().strip()
        if not step_id:
            return None

        subfolders = [
            self._sub_list.item(i).text()
            for i in range(self._sub_list.count())
        ]

        is_liquid = self._type_combo.currentText() == "LiquidMeasurement"
        pipettes = []
        for row in range(self._pip_table.rowCount()):
            idx_w  = self._pip_table.cellWidget(row, 0)
            type_w = self._pip_table.cellWidget(row, 1)
            ul_w   = self._pip_table.cellWidget(row, 2)
            if idx_w and type_w:
                ul = ul_w.value() if (ul_w and is_liquid and ul_w.value() > 0) else None
                pipettes.append(PipetteDef(
                    index=idx_w.value() - 1,      # store 0-based
                    tip_type=type_w.currentText(),
                    expected_liquid_ul=ul,
                ))

        return StepDef(
            step_id=step_id,
            analysis_type=self._type_combo.currentText(),
            subfolders=subfolders,
            pipettes=pipettes,
        )

    # ── UI construction ──────────────────────────────────────────────────────

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)

        # ── Step ID + analysis type ──────────────────────────────────
        id_row = QHBoxLayout()
        id_row.addWidget(QLabel("Step ID:"))
        self._id_edit = QLineEdit()
        self._id_edit.setPlaceholderText("e.g. step_001")
        id_row.addWidget(self._id_edit, 1)
        id_row.addSpacing(12)
        id_row.addWidget(QLabel("Type:"))
        self._type_combo = QComboBox()
        for t in ANALYSIS_TYPES:
            self._type_combo.addItem(t)
        self._type_combo.currentTextChanged.connect(self._on_type_changed)
        id_row.addWidget(self._type_combo)
        layout.addLayout(id_row)

        # ── Image subfolders ─────────────────────────────────────────
        sub_group = QGroupBox("Image subfolders  (relative to run folder)")
        sub_layout = QVBoxLayout(sub_group)
        sub_layout.setSpacing(4)
        self._sub_list = QListWidget()
        self._sub_list.setMaximumHeight(90)
        sub_layout.addWidget(self._sub_list)
        sub_btns = QHBoxLayout()
        btn_add_sub = QPushButton("Add…")
        btn_add_sub.clicked.connect(self._on_add_subfolder)
        btn_rem_sub = QPushButton("Remove")
        btn_rem_sub.clicked.connect(self._on_remove_subfolder)
        sub_btns.addWidget(btn_add_sub)
        sub_btns.addWidget(btn_rem_sub)
        sub_btns.addStretch()
        sub_layout.addLayout(sub_btns)
        layout.addWidget(sub_group)

        # ── Pipettes ─────────────────────────────────────────────────
        pip_group = QGroupBox("Pipettes")
        pip_layout = QVBoxLayout(pip_group)
        pip_layout.setSpacing(4)
        self._pip_table = QTableWidget(0, 3)
        self._pip_table.setHorizontalHeaderLabels(["Tip #", "Tip Type", "Expected µL"])
        hdr = self._pip_table.horizontalHeader()
        hdr.setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        hdr.setMinimumSectionSize(40)
        self._pip_table.setMinimumWidth(0)
        self._pip_table.verticalHeader().setVisible(False)
        self._pip_table.setMaximumHeight(170)
        pip_layout.addWidget(self._pip_table)
        pip_btns = QHBoxLayout()
        btn_add_pip = QPushButton("Add Pipette")
        btn_add_pip.clicked.connect(lambda: self._add_pipette_row())
        btn_rem_pip = QPushButton("Remove")
        btn_rem_pip.clicked.connect(self._on_remove_pipette)
        pip_btns.addWidget(btn_add_pip)
        pip_btns.addWidget(btn_rem_pip)
        pip_btns.addStretch()
        pip_layout.addLayout(pip_btns)
        layout.addWidget(pip_group)

        layout.addStretch()

        # ── Save button ──────────────────────────────────────────────
        save_row = QHBoxLayout()
        save_row.addStretch()
        self._save_btn = QPushButton("Save Step")
        self._save_btn.setFixedHeight(30)
        self._save_btn.setStyleSheet(
            "QPushButton { background:#3a6ea5; color:white; font-weight:bold; }"
        )
        self._save_btn.clicked.connect(self._on_save)
        save_row.addWidget(self._save_btn)
        layout.addLayout(save_row)

    # ── Private helpers ──────────────────────────────────────────────────────

    def _add_pipette_row(self, pip: Optional[PipetteDef] = None) -> None:
        row = self._pip_table.rowCount()
        self._pip_table.setRowCount(row + 1)

        idx_spin = QSpinBox()
        idx_spin.setRange(1, 16)
        idx_spin.setValue((pip.index + 1) if pip else (row + 1))
        self._pip_table.setCellWidget(row, 0, idx_spin)

        type_combo = QComboBox()
        for t in TIP_TYPES:
            type_combo.addItem(t)
        if pip:
            type_combo.setCurrentText(pip.tip_type)
        self._pip_table.setCellWidget(row, 1, type_combo)

        ul_spin = QDoubleSpinBox()
        ul_spin.setRange(0, 5000)
        ul_spin.setDecimals(1)
        ul_spin.setSuffix(" µL")
        ul_spin.setToolTip("Leave at 0 to omit (optional for TipIdentification).")
        if pip and pip.expected_liquid_ul is not None:
            ul_spin.setValue(pip.expected_liquid_ul)
        is_liquid = self._type_combo.currentText() == "LiquidMeasurement"
        ul_spin.setEnabled(is_liquid)
        self._pip_table.setCellWidget(row, 2, ul_spin)

    def _on_type_changed(self, text: str) -> None:
        is_liquid = text == "LiquidMeasurement"
        for row in range(self._pip_table.rowCount()):
            w = self._pip_table.cellWidget(row, 2)
            if w:
                w.setEnabled(is_liquid)

    def _on_add_subfolder(self) -> None:
        start = self._run_folder if self._run_folder else ""
        folder = QFileDialog.getExistingDirectory(self, "Select subfolder", start)
        if not folder:
            return
        # Store relative to run_folder when possible.
        if self._run_folder:
            try:
                text = str(Path(folder).relative_to(Path(self._run_folder)))
            except ValueError:
                text = folder
        else:
            text = folder
        existing = [self._sub_list.item(i).text() for i in range(self._sub_list.count())]
        if text not in existing:
            self._sub_list.addItem(text)

    def _on_remove_subfolder(self) -> None:
        for item in self._sub_list.selectedItems():
            self._sub_list.takeItem(self._sub_list.row(item))

    def _on_remove_pipette(self) -> None:
        rows = sorted(
            {i.row() for i in self._pip_table.selectedItems()},
            reverse=True,
        )
        for row in rows:
            self._pip_table.removeRow(row)

    def _on_save(self) -> None:
        step = self.current_step()
        if step:
            self.step_saved.emit(step)
            self._save_btn.setText("✓ Saved")
            self._save_btn.setEnabled(False)
            from PySide6.QtCore import QTimer
            QTimer.singleShot(1500, lambda: (
                self._save_btn.setText("Save Step"),
                self._save_btn.setEnabled(True),
            ))
