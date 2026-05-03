"""Parameter Sweep tab — top-level widget.

Layout::

    ┌─────────────────────────────────────────────────────────────────────────┐
    │ Config toolbar: [Primary folder] [Ref folder] [IC] [Pipette] [Tip type] │
    ├──────────────────────────────┬──────────────────────────────────────────┤
    │ GTAnnotatorWidget (left 40%) │ ParamGridWidget (right 60%)              │
    │  - image list                │  - param table with sweep specs          │
    │  - line annotator view       │  - total combos + estimate               │
    │  - annotation table          │                                          │
    ├──────────────────────────────┴──────────────────────────────────────────┤
    │ [▶ Run Sweep]  [■ Stop]   Progress: ████░░░░ 12/100  roi_exp=5, blur=3  │
    ├─────────────────────────────────────────────────────────────────────────┤
    │ Rank │ Mean │ img_001 │ img_002 │ blur_kernel │ min_distance │  [A/B]   │
    │   1  │ 8.42 │  9.1    │  7.7    │     3       │      5       │ [Open]   │
    │   2  │ 7.91 │  8.3    │  7.5    │     5       │      7       │ [Open]   │
    │  …                                                      [Export CSV]    │
    └─────────────────────────────────────────────────────────────────────────┘
"""

from __future__ import annotations

import csv
import itertools
import json
import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from PySide6.QtCore import Qt, QSettings
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QAbstractItemView, QButtonGroup, QComboBox, QFileDialog, QFrame, QGroupBox,
    QHBoxLayout, QHeaderView, QLabel, QLineEdit,
    QListWidget, QMessageBox, QProgressBar, QPushButton,
    QRadioButton, QSizePolicy, QSpinBox, QSplitter, QTableWidget,
    QTableWidgetItem, QVBoxLayout, QWidget,
)

from pa_gui.sweep.gt_annotator import GTAnnotatorWidget
from pa_gui.sweep.param_grid_widget import ParamGridWidget
from pa_gui.sweep.sweep_worker import SweepWorker, _signal_cache_key

_S_ORG = "rsimpson-bc"
_S_APP = "PA-GUI"
_S_KEY = "sweep_tab"


_TYPE_ORDER = ["tip_bottom", "liquid", "bubble"]
_IMG_COL_OFFSET = 2 + len(_TYPE_ORDER)  # Rank + Mean + 3 type-mean cols


class SweepResultsWidget(QWidget):
    """Scrollable results table populated as sweep results arrive."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._image_paths: List[str] = []
        self._swept_params: List[str] = []
        self._rows: List[Tuple[float, List[float], Dict[str, Any], Dict[str, float], List[Optional[Dict[str, float]]]]] = []
        self._sort_col: int = 1          # default: Mean Score
        self._sort_asc: bool = False     # default: descending
        self._build_ui()

    def prepare(self, image_paths: List[str], swept_params: List[str]) -> None:
        """Call before a sweep starts to set up column headers."""
        self._image_paths  = image_paths
        self._swept_params = swept_params
        self._rows.clear()
        self._build_columns()

    def add_result(
        self,
        params: Dict[str, Any],
        per_image_scores: List[float],
        mean_score: float,
        type_means: Dict[str, float],
        per_img_types: List[Optional[Dict[str, float]]],
    ) -> None:
        self._rows.append((mean_score, per_image_scores, params, type_means, per_img_types))
        self._sort_rows()
        self._repopulate()

    def _build_ui(self) -> None:
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)

        hdr = QHBoxLayout()
        hdr.addWidget(QLabel("<b>Results</b>"))
        hdr.addStretch()
        export_btn = QPushButton("Export CSV…")
        export_btn.clicked.connect(self._on_export)
        hdr.addWidget(export_btn)
        lay.addLayout(hdr)

        self._table = QTableWidget(0, 2)
        self._table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self._table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self._table.verticalHeader().setVisible(False)
        self._table.setSortingEnabled(False)   # we handle sorting ourselves
        self._table.setWordWrap(True)
        hh = self._table.horizontalHeader()
        hh.setSectionsClickable(True)
        hh.setSortIndicatorShown(True)
        hh.sectionClicked.connect(self._on_header_clicked)
        self._table.cellDoubleClicked.connect(self._on_cell_double_clicked)
        lay.addWidget(self._table, 1)

    def _build_columns(self) -> None:
        img_names = [os.path.basename(p) for p in self._image_paths]
        type_cols = [f"Mean ({t})" for t in _TYPE_ORDER]
        cols = (
            ["Rank", "Mean Score"]
            + type_cols
            + img_names
            + self._swept_params
            + ["Open in A/B"]
        )
        self._table.setColumnCount(len(cols))
        self._table.setHorizontalHeaderLabels(cols)
        hh = self._table.horizontalHeader()
        for i in range(len(cols) - 1):
            hh.setSectionResizeMode(i, QHeaderView.ResizeMode.ResizeToContents)
        hh.setSectionResizeMode(len(cols) - 1, QHeaderView.ResizeMode.ResizeToContents)

    def _on_header_clicked(self, col: int) -> None:
        """Toggle sort direction if same column, else sort by new column descending."""
        n_cols = self._table.columnCount()
        # Ignore Rank (0) and Open-in-A/B (last) columns
        if col == 0 or col == n_cols - 1:
            return
        if col == self._sort_col:
            self._sort_asc = not self._sort_asc
        else:
            self._sort_col = col
            # Numeric columns default descending; param columns default ascending
            n_img = len(self._image_paths)
            param_start = _IMG_COL_OFFSET + n_img
            self._sort_asc = col >= param_start
        self._sort_rows()
        self._repopulate()

    def _row_sort_key(self, row_tuple):
        import math
        mean, per_img, params, type_means, _ = row_tuple
        col = self._sort_col
        n_img = len(self._image_paths)

        if col == 1:  # Mean Score
            return mean
        elif 2 <= col <= 1 + len(_TYPE_ORDER):
            t = _TYPE_ORDER[col - 2]
            return type_means.get(t, -1.0)
        elif _IMG_COL_OFFSET <= col < _IMG_COL_OFFSET + n_img:
            img_idx = col - _IMG_COL_OFFSET
            s = per_img[img_idx] if img_idx < len(per_img) else float("nan")
            return -1.0 if math.isnan(s) else s
        elif col >= _IMG_COL_OFFSET + n_img:
            param_idx = col - _IMG_COL_OFFSET - n_img
            if param_idx < len(self._swept_params):
                val = params.get(self._swept_params[param_idx], "")
                try:
                    return float(val)
                except (ValueError, TypeError):
                    return str(val)
        return mean  # fallback

    def _sort_rows(self) -> None:
        self._rows.sort(key=self._row_sort_key, reverse=not self._sort_asc)

    @staticmethod
    def _score_color(score: float) -> "QColor":
        """White → medium green as score goes 0 → 1 (HSV with fixed value=1)."""
        t = max(0.0, min(1.0, float(score)))
        return QColor.fromHsvF(1 / 3, t * 0.65, 1.0)

    @staticmethod
    def _fmt_cell(score: float, type_scores: Optional[Dict[str, float]]) -> str:
        """Format a per-image cell as 'score\nT:x L:x B:x' (omit missing types)."""
        import math
        if math.isnan(score):
            return "—"
        abbrevs = {"tip_bottom": "T", "liquid": "L", "bubble": "B"}
        parts = []
        if type_scores:
            for t in _TYPE_ORDER:
                v = type_scores.get(t)
                if v is not None:
                    parts.append(f"{abbrevs[t]}:{v:.2f}")
        sub = " ".join(parts)
        return f"{score:.3f}\n{sub}" if sub else f"{score:.3f}"

    def _repopulate(self) -> None:
        import math
        self._table.setRowCount(len(self._rows))
        n_img = len(self._image_paths)
        for row_idx, (mean, per_img, params, type_means, per_img_types) in enumerate(self._rows):
            col = 0
            # Rank
            self._set_cell(row_idx, col, str(row_idx + 1)); col += 1
            # Overall mean — show breakdown of type means below it
            _abbr = {"tip_bottom": "T", "liquid": "L", "bubble": "B"}
            type_sub = " ".join(
                f"{_abbr.get(t, t[0].upper())}:{type_means[t]:.2f}"
                for t in _TYPE_ORDER if t in type_means
            )
            self._set_cell(row_idx, col, f"{mean:.3f}\n{type_sub}" if type_sub else f"{mean:.3f}", score=mean); col += 1
            # Per-type means (dedicated columns)
            for t in _TYPE_ORDER:
                v = type_means.get(t)
                self._set_cell(row_idx, col, f"{v:.3f}" if v is not None else "\u2014",
                               score=v if v is not None else float("nan")); col += 1
            # Per-image scores with per-type breakdown
            for img_i, s in enumerate(per_img):
                ts = per_img_types[img_i] if img_i < len(per_img_types) else None
                self._set_cell(row_idx, col, self._fmt_cell(s, ts), score=s); col += 1
            # Pad missing image cols
            for _ in range(n_img - len(per_img)):
                self._set_cell(row_idx, col, "—"); col += 1
            # Swept params
            for key in self._swept_params:
                self._set_cell(row_idx, col, str(params.get(key, ""))); col += 1
            # Open in A/B button
            btn = QPushButton("Open")
            btn.setToolTip(
                "Open the first image with these parameters in the A/B Comparison dialog.\n"
                "Double-click any per-image score cell to open that specific image."
            )
            btn.clicked.connect(
                lambda _, p=dict(params): self._open_ab(p, 0)
            )
            self._table.setCellWidget(row_idx, col, btn)
        self._table.resizeRowsToContents()
        # Update the sort indicator arrow on the header
        order = Qt.SortOrder.AscendingOrder if self._sort_asc else Qt.SortOrder.DescendingOrder
        self._table.horizontalHeader().setSortIndicator(self._sort_col, order)

    def _set_cell(self, row: int, col: int, text: str, score: float = float("nan")) -> None:
        import math
        item = QTableWidgetItem(text)
        item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
        if not math.isnan(score):
            item.setBackground(self._score_color(score))
            item.setForeground(QColor(Qt.GlobalColor.black))
        self._table.setItem(row, col, item)

    def _open_ab(self, params: Dict[str, Any], img_idx: int = 0) -> None:
        # Walk up the widget tree to find SweepTab
        parent = self.parent()
        while parent and not isinstance(parent, SweepTab):
            parent = parent.parent()
        if parent:
            parent.open_in_ab(params, img_idx)

    def _on_cell_double_clicked(self, row: int, col: int) -> None:
        """Double-clicking a per-image score cell opens A/B for that specific image."""
        n_img = len(self._image_paths)
        # Per-image cols start at _IMG_COL_OFFSET
        if col < _IMG_COL_OFFSET or col >= _IMG_COL_OFFSET + n_img:
            return
        img_idx = col - _IMG_COL_OFFSET
        if row < len(self._rows):
            _, _, params, _, _ = self._rows[row]
            self._open_ab(dict(params), img_idx)

    def _on_export(self) -> None:
        if not self._rows:
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Export results", "", "CSV files (*.csv);;All files (*)"
        )
        if not path:
            return
        img_names = [os.path.basename(p) for p in self._image_paths]
        type_col_headers = [f"mean_{t}" for t in _TYPE_ORDER]
        headers = ["rank", "mean_score"] + type_col_headers + img_names + self._swept_params
        with open(path, "w", newline="", encoding="utf-8") as fh:
            writer = csv.writer(fh)
            writer.writerow(headers)
            import math
            for rank, (mean, per_img, params, type_means, _per_img_types) in enumerate(self._rows, start=1):
                type_vals = [round(type_means.get(t, float("nan")), 4) if type_means.get(t) is not None else "" for t in _TYPE_ORDER]
                row = [rank, round(mean, 4)] + type_vals
                row += ["" if math.isnan(s) else round(s, 4) for s in per_img]
                row += [params.get(k, "") for k in self._swept_params]
                writer.writerow(row)


class SweepTab(QWidget):
    """Top-level widget for the Parameter Sweep tab."""

    def __init__(
        self,
        instrument_config_path: Optional[str] = None,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self._worker: Optional[SweepWorker] = None
        self._schema: Optional[Dict[str, Any]] = None
        self._baseline_params: Dict[str, Any] = {}
        self._combos: List[Dict[str, Any]] = []

        self._build_ui()
        self._load_schema()
        self._restore_settings()
        self._on_roi_context_changed()  # apply ROI to annotator from restored settings

        if instrument_config_path:
            self._ic_edit.setText(instrument_config_path)

    # ── UI construction ───────────────────────────────────────────────────

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(6, 6, 6, 6)
        root.setSpacing(4)

        root.addWidget(self._build_config_bar())

        # Main splitter: annotator (left) | param grid (right)
        self._main_splitter = QSplitter(Qt.Orientation.Horizontal)

        self._gt_widget = GTAnnotatorWidget()
        self._gt_widget.gt_changed.connect(self._on_gt_changed)
        self._main_splitter.addWidget(self._gt_widget)

        self._param_grid = ParamGridWidget({}, {})
        self._param_grid.sweep_changed.connect(self._on_sweep_changed)
        self._main_splitter.addWidget(self._param_grid)

        self._main_splitter.setStretchFactor(0, 2)
        self._main_splitter.setStretchFactor(1, 3)
        root.addWidget(self._main_splitter, 3)

        root.addWidget(self._build_run_bar())

        self._results = SweepResultsWidget()
        root.addWidget(self._results, 2)

    def _build_config_bar(self) -> QWidget:
        bar = QGroupBox("Setup")
        lay = QVBoxLayout(bar)
        lay.setSpacing(3)

        # Row 0: analysis mode selector
        r0 = QHBoxLayout()
        r0.addWidget(QLabel("<b>Input mode:</b>"))
        self._mode_group = QButtonGroup(self)
        self._rb_single   = QRadioButton("Single Frame")
        self._rb_single.setToolTip(
            "Each primary image is analysed independently with no reference contrast."
        )
        self._rb_contrast = QRadioButton("Reference Contrast")
        self._rb_contrast.setToolTip(
            "Each primary image is contrasted against the matched reference image(s) "
            "before analysis.  Add one or more reference folders below."
        )
        self._rb_zstack = QRadioButton("Z-Stack  (coming soon)")
        self._rb_zstack.setEnabled(False)
        self._rb_zstack.setToolTip(
            "All primary images are treated as the same tip at different Z positions.\n"
            "The per-row std / range across frames forms the signal.\n"
            "Not yet implemented."
        )
        self._mode_group.addButton(self._rb_single,   0)
        self._mode_group.addButton(self._rb_contrast, 1)
        self._mode_group.addButton(self._rb_zstack,   2)
        self._rb_single.setChecked(True)
        for rb in (self._rb_single, self._rb_contrast, self._rb_zstack):
            r0.addWidget(rb)
        r0.addStretch()
        lay.addLayout(r0)
        self._mode_group.idToggled.connect(self._on_mode_changed)

        # Row 1: primary folder
        r1 = QHBoxLayout()
        r1.addWidget(QLabel("Primary folder:"))
        self._primary_edit = QLineEdit()
        self._primary_edit.setPlaceholderText("Folder of sample images…")
        r1.addWidget(self._primary_edit, 2)
        btn_prim = QPushButton("Browse…")
        btn_prim.clicked.connect(self._on_browse_primary)
        r1.addWidget(btn_prim)
        lay.addLayout(r1)

        # Row 1b: reference folders (multi) — hidden in Single Frame mode
        self._ref_row_widget = QWidget()
        r1b = QHBoxLayout(self._ref_row_widget)
        r1b.setContentsMargins(0, 0, 0, 0)
        r1b.addWidget(QLabel("Reference folders:"))
        self._ref_list = QListWidget()
        self._ref_list.setMaximumHeight(55)
        self._ref_list.setToolTip(
            "One or more folders of reference images (same image count as primary).\n"
            "Images are paired by sort order. If multiple folders are listed,\n"
            "same-index images are averaged into a single reference frame."
        )
        r1b.addWidget(self._ref_list, 2)
        btn_add_ref = QPushButton("Add…")
        btn_add_ref.clicked.connect(self._on_add_ref_folder)
        r1b.addWidget(btn_add_ref)
        btn_rm_ref = QPushButton("Remove")
        btn_rm_ref.clicked.connect(self._on_remove_ref_folder)
        r1b.addWidget(btn_rm_ref)
        lay.addWidget(self._ref_row_widget)
        self._ref_row_widget.setVisible(False)  # hidden until Reference Contrast selected

        # Row 2: instrument config + pipette + tip type
        r2 = QHBoxLayout()
        r2.addWidget(QLabel("Instrument Config:"))
        self._ic_edit = QLineEdit()
        self._ic_edit.setPlaceholderText("instrument_config.json")
        self._ic_edit.editingFinished.connect(self._on_roi_context_changed)
        r2.addWidget(self._ic_edit, 2)
        btn_ic = QPushButton("Browse…")
        btn_ic.clicked.connect(self._on_browse_ic)
        r2.addWidget(btn_ic)

        r2.addSpacing(12)
        r2.addWidget(QLabel("Pipette:"))
        self._pip_spin = QSpinBox()
        self._pip_spin.setRange(0, 15)
        self._pip_spin.setValue(0)
        self._pip_spin.setFixedWidth(50)
        self._pip_spin.setToolTip("0-based pipette index")
        self._pip_spin.setStyleSheet(
            "QSpinBox { border-bottom: none; }"
        )
        self._pip_spin.valueChanged.connect(self._on_roi_context_changed)
        r2.addWidget(self._pip_spin)

        r2.addSpacing(8)
        r2.addWidget(QLabel("Tip type:"))
        self._tip_edit = QLineEdit()
        self._tip_edit.setPlaceholderText("e.g. T200")
        self._tip_edit.setFixedWidth(80)
        self._tip_edit.editingFinished.connect(self._on_roi_context_changed)
        r2.addWidget(self._tip_edit)

        r2.addSpacing(12)
        r2.addWidget(QLabel("Detection Mode:"))
        self._detect_mode_combo = QComboBox()
        self._detect_mode_combo.addItems(["IntensityDetection", "RowContrastDetection"])
        self._detect_mode_combo.setToolTip(
            "IntensityDetection  — per-row mean intensity.\n"
            "RowContrastDetection — per-row cross-sectional std/variance."
        )
        self._detect_mode_combo.currentTextChanged.connect(self._on_detect_mode_changed)
        r2.addWidget(self._detect_mode_combo)
        lay.addLayout(r2)

        # Row 3: ROI status
        r3 = QHBoxLayout()
        self._roi_status_lbl = QLabel("ROI: no instrument config loaded")
        self._roi_status_lbl.setStyleSheet("color:#666; font-size:10px; font-style:italic;")
        r3.addWidget(self._roi_status_lbl)
        r3.addStretch()
        lay.addLayout(r3)

        return bar

    def _build_run_bar(self) -> QWidget:
        bar = QWidget()
        lay = QHBoxLayout(bar)
        lay.setContentsMargins(0, 0, 0, 0)

        self._run_btn = QPushButton("▶  Run Sweep")
        self._run_btn.setFixedHeight(34)
        self._run_btn.setStyleSheet(
            "QPushButton { background:#2a7a2a; color:white; font-weight:bold; }"
            "QPushButton:disabled { background:#444; color:#888; }"
        )
        self._run_btn.clicked.connect(self._on_run)
        lay.addWidget(self._run_btn)

        self._stop_btn = QPushButton("■  Stop")
        self._stop_btn.setFixedHeight(34)
        self._stop_btn.setEnabled(False)
        self._stop_btn.setStyleSheet(
            "QPushButton:enabled { color:#e74c3c; font-weight:bold; }"
        )
        self._stop_btn.clicked.connect(self._on_stop)
        lay.addWidget(self._stop_btn)

        self._progress = QProgressBar()
        self._progress.setVisible(False)
        self._progress.setTextVisible(True)
        lay.addWidget(self._progress, 1)

        self._status_lbl = QLabel("")
        self._status_lbl.setStyleSheet("color:#888; font-size:10px;")
        lay.addWidget(self._status_lbl)

        return bar

    # ── Schema / param defaults ───────────────────────────────────────────

    def _load_schema(self) -> None:
        mode_name = self._detect_mode_combo.currentText() \
            if hasattr(self, "_detect_mode_combo") else "IntensityDetection"
        schema_path = os.path.normpath(
            os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                "..", "..", "schemas", "pipeline_params", f"{mode_name}.schema.json",
            )
        )
        try:
            with open(schema_path, "r", encoding="utf-8") as fh:
                self._schema = json.load(fh)
        except Exception:
            self._schema = {"properties": {}}

        # Build baseline from schema defaults
        self._baseline_params = {
            k: prop.get("default", "")
            for k, prop in self._schema.get("properties", {}).items()
            if not k.startswith("_group_")
        }

        # Re-create param grid with real schema + baseline
        if hasattr(self, "_param_grid"):
            old = self._param_grid
            # Capture existing specs before destroying the old widget
            old_specs = old.get_specs() if hasattr(old, "get_specs") else {}
            new = ParamGridWidget(self._schema, self._baseline_params)
            new.sweep_changed.connect(self._on_sweep_changed)
            self._main_splitter.replaceWidget(
                self._main_splitter.indexOf(old), new
            )
            self._param_grid = new
            old.deleteLater()
            # Restore any specs the old widget had
            if old_specs:
                new.set_specs(old_specs)

    # ── Slots ─────────────────────────────────────────────────────────────

    def _on_browse_primary(self) -> None:
        s = QSettings(_S_ORG, _S_APP)
        start = s.value(f"{_S_KEY}/primary_dir", "")
        folder = QFileDialog.getExistingDirectory(self, "Select primary image folder", start)
        if folder:
            s.setValue(f"{_S_KEY}/primary_dir", folder)
            self._primary_edit.setText(folder)
            self._gt_widget.set_folder(folder)
            self._on_sweep_changed()

    def _on_detect_mode_changed(self, _mode: str) -> None:
        self._load_schema()
        self._save_settings()

    def _on_mode_changed(self, btn_id: int, checked: bool) -> None:
        if not checked:
            return
        # Show reference folders only for Reference Contrast mode
        self._ref_row_widget.setVisible(btn_id == 1)

    def _on_add_ref_folder(self) -> None:
        s = QSettings(_S_ORG, _S_APP)
        start = s.value(f"{_S_KEY}/ref_dir", "")
        folder = QFileDialog.getExistingDirectory(self, "Add reference image folder", start)
        if folder:
            s.setValue(f"{_S_KEY}/ref_dir", folder)
            # Avoid duplicates
            existing = [self._ref_list.item(i).text() for i in range(self._ref_list.count())]
            if folder not in existing:
                self._ref_list.addItem(folder)

    def _on_remove_ref_folder(self) -> None:
        row = self._ref_list.currentRow()
        if row >= 0:
            self._ref_list.takeItem(row)

    def _on_browse_ic(self) -> None:
        s = QSettings(_S_ORG, _S_APP)
        start = s.value(f"{_S_KEY}/ic_dir", "")
        path, _ = QFileDialog.getOpenFileName(
            self, "Select instrument config", start, "JSON (*.json);;All files (*)"
        )
        if path:
            s.setValue(f"{_S_KEY}/ic_dir", os.path.dirname(path))
            self._ic_edit.setText(path)
            self._on_roi_context_changed()

    def _on_roi_context_changed(self) -> None:
        """Propagate IC / pipette / tip-type changes to the GT annotator."""
        ic   = self._ic_edit.text().strip()
        pip  = self._pip_spin.value()
        tip  = self._tip_edit.text().strip()
        self._gt_widget.set_roi_context(ic, pip, tip)
        # Update status label
        if not ic or not os.path.isfile(ic):
            self._roi_status_lbl.setText("ROI: no instrument config loaded")
            self._roi_status_lbl.setStyleSheet("color:#666; font-size:10px; font-style:italic;")
        else:
            try:
                import json as _json
                from pa.analysis.shared_geometry import lookup_roi
                with open(ic, "r", encoding="utf-8") as fh:
                    _ic = _json.load(fh)
                roi_bbox, roi_points = lookup_roi(_ic, pip, tip or None)
                if roi_bbox:
                    x, y, w, h = roi_bbox
                    self._roi_status_lbl.setText(
                        f"ROI: pipette {pip}  x={x:.0f}–{x+w:.0f}  y={y:.0f}–{y+h:.0f} px"
                    )
                    self._roi_status_lbl.setStyleSheet("color:#6c6; font-size:10px;")
                else:
                    self._roi_status_lbl.setText(
                        f"ROI: not found for pipette {pip} — using full image"
                    )
                    self._roi_status_lbl.setStyleSheet("color:#f5a623; font-size:10px;")
            except Exception as exc:
                self._roi_status_lbl.setText(f"ROI: error — {exc}")
                self._roi_status_lbl.setStyleSheet("color:#f55; font-size:10px;")

    def _on_gt_changed(self) -> None:
        self._on_sweep_changed()

    def _on_sweep_changed(self) -> None:
        """Update the time estimate in the grid when sweep spec or image set changes."""
        combos, _ = self._param_grid.get_combos()
        if not combos:
            return
        n_images       = len(self._gt_widget.get_image_paths())
        unique_signals = len({_signal_cache_key(c) for c in combos})
        self._param_grid.update_time_estimate(n_images, unique_signals)
        self._save_settings()

    def _on_run(self) -> None:
        mode_id = self._mode_group.checkedId()  # 0=Single Frame, 1=Reference Contrast, 2=Z-Stack

        if mode_id == 2:
            QMessageBox.information(
                self, "Not yet implemented",
                "Z-Stack mode is not yet implemented.\n"
                "Please select Single Frame or Reference Contrast."
            )
            return

        primary = self._primary_edit.text().strip()
        if not primary or not os.path.isdir(primary):
            QMessageBox.warning(self, "No primary folder", "Select a primary image folder first.")
            return

        combos, err = self._param_grid.get_combos()
        if err:
            QMessageBox.warning(self, "Sweep spec error", err)
            return
        if not combos:
            QMessageBox.warning(self, "No combos", "No parameter combinations to test.")
            return

        n = len(combos)
        image_paths = self._gt_widget.get_image_paths()
        if not image_paths:
            QMessageBox.warning(self, "No images", "No images found in the primary folder.")
            return

        if n > 500:
            reply = QMessageBox.question(
                self,
                "Large sweep",
                f"{n:,} combinations × {len(image_paths)} images.\n\n"
                f"Estimated time: ~{n * len(image_paths) * 0.25:.0f} s\n\n"
                "Continue?",
            )
            if reply != QMessageBox.StandardButton.Yes:
                return

        # Determine which params actually vary (for results column headers)
        swept = [
            k for k in combos[0]
            if len({c[k] for c in combos}) > 1
        ]

        self._combos = combos

        # Prepare results table
        self._results.prepare(image_paths, swept)

        ref_folders = [
            self._ref_list.item(i).text()
            for i in range(self._ref_list.count())
        ] if mode_id == 1 else []

        # Override use_contrast in every combo to match the chosen mode
        use_contrast = (mode_id == 1)
        combos = [{**c, "use_contrast": use_contrast} for c in combos]

        self._worker = SweepWorker(
            combos              = combos,
            image_paths         = image_paths,
            reference_folders   = ref_folders,
            image_gts        = self._gt_widget.get_image_gts(),
            tolerance_px     = self._gt_widget.get_tolerance_px(),
            instrument_config_path = self._ic_edit.text().strip(),
            pipette_index    = self._pip_spin.value(),
            tip_type         = self._tip_edit.text().strip(),
            mode_name        = self._detect_mode_combo.currentText(),
            parent           = self,
        )
        self._worker.progress.connect(self._on_progress)
        self._worker.result_ready.connect(self._on_result)
        self._worker.error.connect(self._on_error)
        self._worker.finished_sweep.connect(self._on_finished)

        self._progress.setMaximum(n)
        self._progress.setValue(0)
        self._progress.setVisible(True)
        self._run_btn.setEnabled(False)
        self._stop_btn.setEnabled(True)
        self._worker.start()

    def _on_stop(self) -> None:
        if self._worker and self._worker.isRunning():
            self._worker.requestInterruption()

    def _on_progress(self, i: int, total: int, params: dict) -> None:
        self._progress.setValue(i - 1)
        short = ", ".join(
            f"{k}={v}" for k, v in list(params.items())[:4]
            if len({c.get(k) for c in self._combos}) > 1
        )
        self._status_lbl.setText(f"Running {i}/{total}  {short}")

    def _on_result(
        self,
        _combo_idx: int,
        params: dict,
        per_image_scores: list,
        mean_score: float,
        type_means: dict,
        per_img_types: list,
    ) -> None:
        self._results.add_result(params, per_image_scores, mean_score, type_means, per_img_types)

    def _on_error(self, msg: str) -> None:
        self._status_lbl.setText("Error (see console)")
        print(f"[SweepWorker] {msg}")

    def _on_finished(self) -> None:
        self._progress.setVisible(False)
        self._run_btn.setEnabled(True)
        self._stop_btn.setEnabled(False)
        self._status_lbl.setText(f"Done — {len(self._combos)} combos")

    # ── Open in A/B ───────────────────────────────────────────────────────

    def open_in_ab(self, params: Dict[str, Any], img_idx: int = 0) -> None:
        """Open ABCompareDialog with params loaded as B parameters."""
        from pa_gui.analysis.ab_compare_dialog import ABCompareDialog
        dlg = ABCompareDialog(
            mode_name=self._detect_mode_combo.currentText(),
            schema=self._schema or {},
            params_a=self._baseline_params,
            instrument_config_path=self._ic_edit.text().strip(),
            tip_type=self._tip_edit.text().strip(),
            parent=self,
        )
        dlg.show()
        # Load this combo's params as B immediately
        dlg._params_b = dict(params)
        dlg._b_form.set_params(dlg._params_b)
        # Set pipette (dialog is 1-based, sweep tab is 0-based)
        dlg._pipette_spin.setValue(self._pip_spin.value() + 1)
        # Load the specific image (not the whole folder)
        image_paths = self._gt_widget.get_image_paths()
        if not image_paths:
            return
        img_idx = max(0, min(img_idx, len(image_paths) - 1))
        img_path = image_paths[img_idx]
        dlg._load_file(img_path)

        # Load the matched reference file (if any) from the first ref folder
        if self._ref_list.count() > 0:
            ref_folder = self._ref_list.item(0).text().strip()
            if ref_folder and os.path.isdir(ref_folder):
                _exts = {".jpg", ".jpeg", ".png", ".bmp"}
                ref_files = sorted(
                    os.path.join(ref_folder, f)
                    for f in os.listdir(ref_folder)
                    if os.path.splitext(f)[1].lower() in _exts
                )
                if img_idx < len(ref_files):
                    dlg._load_ref_file(ref_files[img_idx])

    # ── State persistence ─────────────────────────────────────────────────

    def _restore_settings(self) -> None:
        s = QSettings(_S_ORG, _S_APP)
        geo = s.value(f"{_S_KEY}/geometry")
        if geo:
            pass  # widget, not dialog — can't restoreGeometry
        ic  = s.value(f"{_S_KEY}/ic_path", "")
        if ic:
            self._ic_edit.setText(ic)
        primary = s.value(f"{_S_KEY}/primary_folder", "")
        if primary and os.path.isdir(primary):
            self._primary_edit.setText(primary)
            self._gt_widget.set_folder(primary)
        ref_folders_json = s.value(f"{_S_KEY}/ref_folders", "")
        if ref_folders_json:
            try:
                for f in json.loads(ref_folders_json):
                    if isinstance(f, str) and os.path.isdir(f):
                        self._ref_list.addItem(f)
            except Exception:
                pass
        mode_id = s.value(f"{_S_KEY}/input_mode", 0)
        try:
            mode_id = int(mode_id)
        except (TypeError, ValueError):
            mode_id = 0
        btn = self._mode_group.button(mode_id)
        if btn and btn.isEnabled():
            btn.setChecked(True)
        pip = s.value(f"{_S_KEY}/pipette", 0)
        self._pip_spin.setValue(int(pip) if pip else 0)
        tip = s.value(f"{_S_KEY}/tip_type", "")
        if tip:
            self._tip_edit.setText(tip)
        # Restore sweep specs
        specs_json = s.value(f"{_S_KEY}/sweep_specs", "")
        if specs_json:
            try:
                import json as _json
                specs = _json.loads(specs_json)
                if isinstance(specs, dict):
                    self._param_grid.set_specs(specs)
            except Exception:
                pass
        # Restore detection mode (after schema is loaded, before specs)
        detect_mode = s.value(f"{_S_KEY}/detection_mode", "IntensityDetection")
        idx = self._detect_mode_combo.findText(detect_mode)
        if idx >= 0:
            self._detect_mode_combo.setCurrentIndex(idx)

    def _save_settings(self) -> None:
        s = QSettings(_S_ORG, _S_APP)
        s.setValue(f"{_S_KEY}/ic_path",        self._ic_edit.text())
        s.setValue(f"{_S_KEY}/primary_folder",  self._primary_edit.text())
        ref_folders = [self._ref_list.item(i).text() for i in range(self._ref_list.count())]
        s.setValue(f"{_S_KEY}/ref_folders",     json.dumps(ref_folders))
        s.setValue(f"{_S_KEY}/input_mode",      self._mode_group.checkedId())
        s.setValue(f"{_S_KEY}/detection_mode",  self._detect_mode_combo.currentText())
        s.setValue(f"{_S_KEY}/pipette",         self._pip_spin.value())
        s.setValue(f"{_S_KEY}/tip_type",       self._tip_edit.text())
        # Save sweep specs
        import json as _json
        s.setValue(f"{_S_KEY}/sweep_specs", _json.dumps(self._param_grid.get_specs()))

    def hideEvent(self, event):
        self._save_settings()
        super().hideEvent(event)
