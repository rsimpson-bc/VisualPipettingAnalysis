"""
ROI Calibration panel — the main widget for the "ROI Calibration" tab.

Layout (horizontal splitter, ~3:1 left:right ratio):

    ┌─ Top toolbar ───────────────────────────────────────────────────────────┐
    │ [Load File…] [Load from Run…]  │  Config: [path…] [Browse] [Load][Save] │
    └─────────────────────────────────────────────────────────────────────────┘
    ┌─ Canvas (left) ──────┬─ Right panel ──────────────────────────────────┐
    │                      │  Tip type: [50uL][200uL][1000uL]               │
    │  ✏ Draw  |  Fit      ├────────────────────────────────────────────────┤
    │                      │  Options: Ref Tip [0▼]  ☑ All tips  ☑ Anchor  │
    │  QGraphicsView       ├────────────────────────────────────────────────┤
    │  (zoomable,          │  PolygonEditor table                           │
    │   interactive)       │  (Row | Left X | Left Y | Right X | Right Y)   │
    │                      ├────────────────────────────────────────────────┤
    │                      │  [Delete Row]  [Clear All]                     │
    └──────────────────────┴────────────────────────────────────────────────┘
    ┌─ Status bar ────────────────────────────────────────────────────────────┐
    │ Ready                                                                   │
    └─────────────────────────────────────────────────────────────────────────┘
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional

from PySide6.QtCore import Qt, QSettings
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QMessageBox,
    QPushButton,
    QSizePolicy,
    QSplitter,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from pa_gui.roi_calibration.image_canvas import ImageCanvas
from pa_gui.roi_calibration.models import AbsolutePair, RoiDefinition, TIP_TYPES
from pa_gui.roi_calibration.polygon_editor import PolygonEditor
from pa_gui.roi_calibration import roi_exporter


class CalibrationPanel(QWidget):
    """Main ROI calibration panel."""

    def __init__(
        self,
        instrument_config_path: Optional[str] = None,
        parent=None,
    ) -> None:
        super().__init__(parent)

        # ── Data state ────────────────────────────────────────────────────────
        self._ic_path: Optional[str] = instrument_config_path
        self._ic_data: dict = {}
        self._mandrels: List[dict] = []
        self._ref_tip_idx: int = 0
        self._current_tip_type: str = TIP_TYPES[0]
        # One list of AbsolutePairs per tip type (edited in absolute image coords)
        self._roi_map: Dict[str, List[AbsolutePair]] = {t: [] for t in TIP_TYPES}
        # Per-tip correction offsets: {tip_type: {mandrel_index: (dx, dy)}}
        self._tip_offsets: Dict[str, Dict[int, tuple]] = {t: {} for t in TIP_TYPES}

        # ── Build UI ──────────────────────────────────────────────────────────
        root = QVBoxLayout(self)
        root.setContentsMargins(4, 4, 4, 4)
        root.setSpacing(4)

        root.addWidget(self._build_top_toolbar())

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(self._build_canvas_area())
        splitter.addWidget(self._build_right_panel())
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 1)
        root.addWidget(splitter, stretch=1)

        root.addWidget(self._build_status_bar())

        # ── Wire signals ──────────────────────────────────────────────────────
        self._canvas.pair_added.connect(self._on_pair_added)
        self._canvas.pair_moved.connect(self._on_pair_moved)
        self._canvas.pair_deleted.connect(self._on_pair_deleted)
        self._canvas.mouse_image_pos.connect(self._on_canvas_mouse_pos)
        self._canvas.mouse_left.connect(lambda: self._coord_label.setText(""))
        self._canvas.other_tip_offset_changed.connect(self._on_other_tip_offset_changed)
        self._editor.pair_changed.connect(self._on_editor_pair_changed)
        self._editor.pair_deleted.connect(self._on_editor_pair_deleted)
        self._editor.all_cleared.connect(self._on_editor_all_cleared)

        if self._ic_path:
            self._load_ic_file(self._ic_path)

        # ── Restore last-used image (after ic so overlays can be drawn) ───────
        saved_img = QSettings("rsimpson-bc", "PA-GUI").value("roi/image_path", "")
        if saved_img:
            import os
            if os.path.isfile(str(saved_img)):
                self._load_image(str(saved_img))

    # ── UI builders ────────────────────────────────────────────────────────────

    def _build_top_toolbar(self) -> QWidget:
        bar = QWidget()
        layout = QHBoxLayout(bar)
        layout.setContentsMargins(0, 0, 0, 0)

        layout.addWidget(QLabel("Image:"))
        self._img_label = QLabel("(none)")
        self._img_label.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        layout.addWidget(self._img_label)

        btn_file = QPushButton("Load File…")
        btn_file.clicked.connect(self._on_load_image_file)
        layout.addWidget(btn_file)

        btn_run = QPushButton("Load from Run…")
        btn_run.clicked.connect(self._on_load_from_run)
        layout.addWidget(btn_run)

        layout.addSpacing(16)
        layout.addWidget(QLabel("Instrument Config:"))

        self._ic_edit = QLineEdit()
        self._ic_edit.setPlaceholderText("instrument_config.json")
        if self._ic_path:
            self._ic_edit.setText(self._ic_path)
        layout.addWidget(self._ic_edit)

        btn_browse = QPushButton("Browse…")
        btn_browse.clicked.connect(self._on_browse_ic)
        layout.addWidget(btn_browse)

        btn_load = QPushButton("Load")
        btn_load.clicked.connect(self._on_load_ic)
        layout.addWidget(btn_load)

        btn_save = QPushButton("Save")
        btn_save.setStyleSheet("QPushButton { background:#4CAF50; color:white; "
                               "font-weight:bold; }")
        btn_save.clicked.connect(self._on_save_ic)
        layout.addWidget(btn_save)

        layout.addSpacing(8)
        btn_help = QPushButton("?")
        btn_help.setFixedWidth(28)
        btn_help.setToolTip("Workflow guide")
        btn_help.clicked.connect(self._on_workflow_help)
        layout.addWidget(btn_help)

        return bar

    def _build_canvas_area(self) -> QWidget:
        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)

        # Mini toolbar
        tb = QHBoxLayout()
        self._draw_btn = QPushButton("✏  Draw")
        self._draw_btn.setCheckable(True)
        self._draw_btn.setToolTip(
            "Toggle draw mode.\n"
            "Click twice per pair (zig-zag OK — left/right auto-detected).\n"
            "Escape cancels a pending first click.")
        self._draw_btn.toggled.connect(self._on_draw_mode_toggled)
        tb.addWidget(self._draw_btn)

        self._move_btn = QPushButton("↔  Move")
        self._move_btn.setCheckable(True)
        self._move_btn.setToolTip(
            "Toggle move mode.\n"
            "Drag the polygon body to reposition the entire ROI at once.\n"
            "Drag individual handles to adjust single vertices.\n"
            "Middle-drag or scroll to pan/zoom as usual.")
        self._move_btn.toggled.connect(self._on_move_mode_toggled)
        tb.addWidget(self._move_btn)

        btn_fit = QPushButton("Fit")
        btn_fit.clicked.connect(self._on_fit_view)
        tb.addWidget(btn_fit)
        tb.addStretch()

        layout.addLayout(tb)

        self._canvas = ImageCanvas()
        layout.addWidget(self._canvas, stretch=1)

        self._coord_label = QLabel("")
        self._coord_label.setAlignment(
            Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        self._coord_label.setStyleSheet(
            "QLabel { color: #888; font-size: 10px; padding: 0px 4px; }")
        layout.addWidget(self._coord_label)

        return container

    def _build_right_panel(self) -> QWidget:
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(6)

        # Tip-type tab bar
        self._tip_tabs = QTabWidget()
        for tt in TIP_TYPES:
            self._tip_tabs.addTab(QWidget(), tt)
        self._tip_tabs.currentChanged.connect(self._on_tip_type_changed)
        layout.addWidget(self._tip_tabs)

        # Options group
        opts = QGroupBox("Options")
        opts_layout = QHBoxLayout(opts)

        opts_layout.addWidget(QLabel("Ref tip:"))
        self._ref_tip_combo = QComboBox()
        for i in range(8):
            self._ref_tip_combo.addItem(f"Tip {i + 1}", i)
        self._ref_tip_combo.currentIndexChanged.connect(self._on_ref_tip_changed)
        opts_layout.addWidget(self._ref_tip_combo)

        self._show_others_cb = QCheckBox("Show all 8 tips")
        self._show_others_cb.setChecked(True)
        self._show_others_cb.toggled.connect(self._on_show_others_changed)
        opts_layout.addWidget(self._show_others_cb)

        self._show_anchor_cb = QCheckBox("Show anchor")
        self._show_anchor_cb.setChecked(True)
        self._show_anchor_cb.toggled.connect(
            lambda checked: self._canvas.set_show_anchor(checked))
        opts_layout.addWidget(self._show_anchor_cb)
        opts_layout.addStretch()

        layout.addWidget(opts)

        # Polygon table
        self._editor = PolygonEditor()
        layout.addWidget(self._editor, stretch=1)

        return panel

    def _build_status_bar(self) -> QLabel:
        self._status = QLabel("Ready")
        self._status.setWordWrap(True)
        self._status.setStyleSheet(
            "QLabel { background:#f0f0f0; border-top:1px solid #ccc; "
            "padding:2px 6px; }")
        return self._status

    # ── Instrument config I/O ──────────────────────────────────────────────────

    def _load_ic_file(self, path: str) -> bool:
        try:
            with open(path, "r", encoding="utf-8") as f:
                self._ic_data = json.load(f)
            self._ic_path = path
            self._ic_edit.setText(path)
            self._mandrels = self._ic_data.get("mandrels", [])
            # Persist for next session.
            QSettings("rsimpson-bc", "PA-GUI").setValue("instrument_config_path", path)
            # Populate ref-tip combo with real mandrel indices if available
            self._refresh_ref_tip_combo()
            self._status_msg(f"Config loaded: {Path(path).name}")
            # Load any saved ROIs and per-tip corrections
            try:
                loaded, corrections = roi_exporter.load_rois(
                    path, self._mandrels, self._ref_tip_idx)
                self._roi_map.update(loaded)
                self._tip_offsets = {t: {} for t in TIP_TYPES}
                for tip_type, corr in corrections.items():
                    self._tip_offsets[tip_type].update(corr)
                self._refresh_canvas()
                self._refresh_all_overlays()
            except Exception as e:
                self._status_msg(f"Config loaded (no ROIs: {e})")
            return True
        except Exception as e:
            self._status_msg(f"Error loading config: {e}")
            return False

    def _refresh_ref_tip_combo(self) -> None:
        """Re-populate the combo with actual mandrel indices from the config."""
        if not self._mandrels:
            return
        self._ref_tip_combo.blockSignals(True)
        self._ref_tip_combo.clear()
        for m in sorted(self._mandrels, key=lambda x: x.get("index", 0)):
            idx = m.get("index", 0)
            self._ref_tip_combo.addItem(f"Tip {idx + 1}", idx)
        # Sync _ref_tip_idx to whatever the first entry now is.
        self._ref_tip_idx = self._ref_tip_combo.itemData(0)
        self._ref_tip_combo.blockSignals(False)

    def _get_ref_tip_yz(self) -> tuple[float, float]:
        for m in self._mandrels:
            if m.get("index") == self._ref_tip_idx:
                yz = m.get("calibrated_pixel_yz", [0.0, 0.0])
                return float(yz[0]), float(yz[1])
        return 0.0, 0.0

    def _get_all_tip_yz(self) -> List[tuple[float, float]]:
        return [
            (float(m.get("calibrated_pixel_yz", [0, 0])[0]),
             float(m.get("calibrated_pixel_yz", [0, 0])[1]))
            for m in sorted(self._mandrels, key=lambda x: x.get("index", 0))
        ]

    # ── Canvas / overlay synchronisation ──────────────────────────────────────

    def _refresh_canvas(self) -> None:
        """Push current tip-type pairs into the canvas and editor."""
        pairs = self._roi_map.get(self._current_tip_type, [])
        self._canvas.set_pairs(list(pairs))
        self._editor.set_pairs(pairs)

    def _refresh_all_overlays(self) -> None:
        """Recompute the translated polygons for all other tips, applying per-tip offsets."""
        if not self._show_others_cb.isChecked():
            self._canvas.set_other_tip_overlays([])
            return

        pairs = self._roi_map.get(self._current_tip_type, [])
        if not pairs or not self._mandrels:
            self._canvas.set_other_tip_overlays([])
            return

        ref_y, ref_z = self._get_ref_tip_yz()
        roi_def = RoiDefinition.from_absolute_pairs(
            self._current_tip_type, pairs, ref_y, ref_z)

        offsets = self._tip_offsets.get(self._current_tip_type, {})
        overlays: List[List[AbsolutePair]] = []
        for m in sorted(self._mandrels, key=lambda x: x.get("index", 0)):
            tip_y = float(m.get("calibrated_pixel_yz", [0, 0])[0])
            tip_z = float(m.get("calibrated_pixel_yz", [0, 0])[1])
            tip_pairs = roi_def.to_absolute_pairs(tip_y, tip_z)
            dx, dy = offsets.get(m.get("index", 0), (0.0, 0.0))
            if abs(dx) > 1e-9 or abs(dy) > 1e-9:
                tip_pairs = [
                    AbsolutePair(p.left_x + dx, p.left_y + dy,
                                 p.right_x + dx, p.right_y + dy)
                    for p in tip_pairs
                ]
            overlays.append(tip_pairs)
        self._canvas.set_other_tip_overlays(overlays)

    # ── Slots — canvas ─────────────────────────────────────────────────────────

    def _on_pair_added(self, idx: int, pair: AbsolutePair) -> None:
        self._roi_map[self._current_tip_type] = self._canvas.get_pairs()
        self._editor.add_pair(pair)
        self._refresh_all_overlays()
        self._status_msg(f"Pair {idx} added.")

    def _on_pair_moved(self, idx: int, pair: AbsolutePair) -> None:
        self._roi_map[self._current_tip_type] = self._canvas.get_pairs()
        self._editor.update_pair(idx, pair)
        self._refresh_all_overlays()

    def _on_pair_deleted(self, idx: int) -> None:
        self._roi_map[self._current_tip_type] = self._canvas.get_pairs()
        self._editor.remove_pair(idx)
        self._refresh_all_overlays()
        self._status_msg(f"Pair {idx} deleted.")

    # ── Slots — editor ─────────────────────────────────────────────────────────

    def _on_editor_pair_changed(self, idx: int, pair: AbsolutePair) -> None:
        pairs = self._roi_map[self._current_tip_type]
        if idx < len(pairs):
            pairs[idx] = pair
        self._canvas.set_pairs(list(pairs))
        self._refresh_all_overlays()

    def _on_editor_pair_deleted(self, idx: int) -> None:
        pairs = self._roi_map[self._current_tip_type]
        if idx < len(pairs):
            pairs.pop(idx)
        self._canvas.set_pairs(list(pairs))
        self._refresh_all_overlays()

    def _on_editor_all_cleared(self) -> None:
        self._roi_map[self._current_tip_type] = []
        self._canvas.clear_pairs()
        self._refresh_all_overlays()

    # ── Slots — toolbar / options ──────────────────────────────────────────────

    def _on_other_tip_offset_changed(self, array_idx: int, dx: float, dy: float) -> None:
        """Accumulate a drag offset for a specific tip overlay and refresh."""
        sorted_mandrels = sorted(self._mandrels, key=lambda x: x.get("index", 0))
        if array_idx >= len(sorted_mandrels):
            return
        mandrel_idx = sorted_mandrels[array_idx].get("index", array_idx)
        offsets = self._tip_offsets[self._current_tip_type]
        old_dx, old_dy = offsets.get(mandrel_idx, (0.0, 0.0))
        offsets[mandrel_idx] = (old_dx + dx, old_dy + dy)
        self._refresh_all_overlays()

    def _on_draw_mode_toggled(self, checked: bool) -> None:
        if checked:
            self._move_btn.setChecked(False)
        self._canvas.set_draw_mode(checked)
        self._draw_btn.setText("✏  Drawing…" if checked else "✏  Draw")

    def _on_move_mode_toggled(self, checked: bool) -> None:
        if checked:
            self._draw_btn.setChecked(False)
        self._canvas.set_move_mode(checked)
        self._move_btn.setText("↔  Moving…" if checked else "↔  Move")

    def _on_canvas_mouse_pos(self, x: float, y: float) -> None:
        self._coord_label.setText(f"x={int(x)}  y={int(y)}")

    def _on_fit_view(self) -> None:
        self._canvas.fitInView(
            self._canvas.scene().sceneRect(),
            Qt.AspectRatioMode.KeepAspectRatio)

    def _on_tip_type_changed(self, index: int) -> None:
        # Flush canvas → roi_map before switching
        self._roi_map[self._current_tip_type] = self._canvas.get_pairs()
        self._current_tip_type = TIP_TYPES[index]
        self._refresh_canvas()
        self._refresh_all_overlays()
        self._status_msg(f"Editing: {self._current_tip_type}")

    def _on_ref_tip_changed(self, _combo_index: int) -> None:
        self._ref_tip_idx = self._ref_tip_combo.currentData()
        self._refresh_all_overlays()
        self._status_msg(f"Reference tip: Tip {self._ref_tip_idx + 1}")

    def _on_show_others_changed(self, checked: bool) -> None:
        self._canvas.set_show_other_tips(checked)
        self._refresh_all_overlays()

    # ── Image loading ──────────────────────────────────────────────────────────

    def _on_load_image_file(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Open Image", "",
            "Images (*.png *.jpg *.jpeg *.tif *.tiff *.bmp);;All files (*)")
        if path:
            self._load_image(path)

    def _on_load_from_run(self) -> None:
        dlg = _RunFolderDialog(self)
        if dlg.exec() == QDialog.DialogCode.Accepted:
            path = dlg.selected_path()
            if path:
                self._load_image(path)

    def _load_image(self, path: str) -> None:
        if self._canvas.load_image(path):
            self._img_label.setText(Path(path).name)
            # Persist for next session.
            QSettings("rsimpson-bc", "PA-GUI").setValue("roi/image_path", path)
            self._refresh_canvas()
            self._refresh_all_overlays()
            self._status_msg(f"Image: {path}")
        else:
            self._status_msg(f"Failed to load: {path}")

    # ── Config I/O slots ───────────────────────────────────────────────────────

    def _on_browse_ic(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Open Instrument Config", "",
            "JSON files (*.json);;All files (*)")
        if path:
            self._ic_edit.setText(path)

    def _on_load_ic(self) -> None:
        path = self._ic_edit.text().strip()
        if not path:
            QMessageBox.information(
                self, "Load Instrument Config",
                "No instrument config path is set.\n"
                "Click \"Browse\u2026\" to choose an instrument_config.json file, "
                "then click Load."
            )
            return
        self._load_ic_file(path)

    def _on_save_ic(self) -> None:
        path = self._ic_edit.text().strip()
        if not path:
            QMessageBox.warning(
                self, "Save ROIs",
                "No instrument config path is set.\n"
                "Browse to or type the path in the Instrument Config field, then try again."
            )
            return
        if not self._mandrels:
            QMessageBox.warning(
                self, "Save ROIs",
                "No mandrel data found in the loaded config.\n"
                "Load an instrument_config.json that contains a \"mandrels\" array first."
            )
            return
        # Flush canvas → roi_map
        self._roi_map[self._current_tip_type] = self._canvas.get_pairs()
        try:
            roi_exporter.save_rois(
                path, self._roi_map, self._mandrels, self._ref_tip_idx,
                self._tip_offsets)
            self._status_msg(f"Saved → {path}")
        except Exception as e:
            QMessageBox.critical(self, "Save Error", str(e))
            self._status_msg(f"Save failed: {e}")

    # ── Helpers ────────────────────────────────────────────────────────────────

    def _status_msg(self, msg: str) -> None:
        self._status.setText(msg)

    def _on_workflow_help(self) -> None:
        QMessageBox.information(
            self,
            "ROI Calibration — Workflow",
            "<b>ROI Calibration workflow</b><br><br>"
            "<b>1. Camera calibration first</b><br>"
            "Complete the Camera Calibration tab before drawing ROIs. "
            "ROI polygons are defined in undistorted image coordinates.<br><br>"
            "<b>2. Load an instrument config</b><br>"
            "Browse to your <tt>instrument_config.json</tt> (or <tt>pa_vision_config.json</tt>). "
            "Mandrel pixel positions must be present so the polygon can be "
            "translated to all tips automatically.<br><br>"
            "<b>3. Load a reference image</b><br>"
            "Use <i>Load File…</i> or <i>Load from Run…</i> to open an image "
            "from a real pipetting run.<br><br>"
            "<b>4. Select tip type &amp; ref tip</b><br>"
            "Choose the tip type tab (e.g. 50 µL) and pick which physical tip "
            "(Tip 1–8) was in the image — this is the reference tip.<br><br>"
            "<b>5. Draw the polygon</b><br>"
            "Click <i>✏ Draw</i> to enter draw mode. "
            "Click two points per row to define left/right boundary pairs — "
            "the order and zigzag direction don't matter. "
            "Press <i>Escape</i> to cancel a pending first click. "
            "Middle-drag or scroll to pan/zoom.<br><br>"
            "<b>6. Review overlays</b><br>"
            "Enable <i>Show all 8 tips</i> to see the polygon translated to "
            "every other tip. Adjust pairs until all overlays look correct.<br><br>"
            "<b>7. Repeat for each tip type</b><br>"
            "Switch between tip type tabs and draw a polygon for each.<br><br>"
            "<b>8. Save</b><br>"
            "Click the green <i>Save</i> button to write the ROIs back into "
            "the instrument config file.",
        )


# ── Run-folder browser dialog ──────────────────────────────────────────────────

class _RunFolderDialog(QDialog):
    """
    Two-panel dialog: left = subfolders of the run folder,
    right = image files in the selected subfolder.
    Double-click a file (or select + OK) to accept.
    """

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Load Image from Run Folder")
        self.resize(720, 420)
        self._selected: Optional[str] = None

        layout = QVBoxLayout(self)

        # Folder row
        folder_row = QHBoxLayout()
        self._folder_edit = QLineEdit()
        self._folder_edit.setPlaceholderText("Run folder path…")
        # Restore last run folder
        saved_run = QSettings("rsimpson-bc", "PA-GUI").value("roi/run_folder", "")
        if saved_run:
            self._folder_edit.setText(str(saved_run))
        btn_browse = QPushButton("Browse…")
        btn_browse.clicked.connect(self._on_browse_folder)
        folder_row.addWidget(QLabel("Run Folder:"))
        folder_row.addWidget(self._folder_edit)
        folder_row.addWidget(btn_browse)
        layout.addLayout(folder_row)

        # Two-panel list
        lists = QHBoxLayout()
        self._sub_list = QListWidget()
        self._sub_list.itemSelectionChanged.connect(self._on_subfolder_selected)
        lists.addWidget(self._sub_list, 1)

        self._file_list = QListWidget()
        self._file_list.itemDoubleClicked.connect(self._on_file_double_clicked)
        lists.addWidget(self._file_list, 2)
        layout.addLayout(lists, stretch=1)

        btns = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok
            | QDialogButtonBox.StandardButton.Cancel)
        btns.accepted.connect(self._on_accept)
        btns.rejected.connect(self.reject)
        layout.addWidget(btns)

    def selected_path(self) -> Optional[str]:
        return self._selected

    def _on_browse_folder(self) -> None:
        folder = QFileDialog.getExistingDirectory(
            self, "Select Run Folder", self._folder_edit.text())
        if folder:
            self._folder_edit.setText(folder)
            QSettings("rsimpson-bc", "PA-GUI").setValue("roi/run_folder", folder)
            self._populate_subfolders(folder)

    def _populate_subfolders(self, folder: str) -> None:
        self._sub_list.clear()
        self._file_list.clear()
        for sub in sorted(Path(folder).iterdir()):
            if sub.is_dir():
                self._sub_list.addItem(sub.name)

    def _on_subfolder_selected(self) -> None:
        sel = self._sub_list.selectedItems()
        if not sel:
            return
        folder = Path(self._folder_edit.text()) / sel[0].text()
        self._file_list.clear()
        exts = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
        for f in sorted(folder.iterdir()):
            if f.is_file() and f.suffix.lower() in exts:
                item = self._file_list.addItem(f.name)
                self._file_list.item(
                    self._file_list.count() - 1
                ).setData(Qt.ItemDataRole.UserRole, str(f))

    def _on_file_double_clicked(self, item) -> None:
        self._selected = item.data(Qt.ItemDataRole.UserRole)
        self.accept()

    def _on_accept(self) -> None:
        sel = self._file_list.selectedItems()
        if sel:
            self._selected = sel[0].data(Qt.ItemDataRole.UserRole)
        self.accept()
