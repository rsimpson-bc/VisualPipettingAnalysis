"""
Mode Compare Dialog.

Opens from PipelineConfigEditor → "Mode Compare…".  Allows the user to pick
any subset of saved presets for the current mode and view their signal graphs
side-by-side, all aligned to a global z-axis (so roi_expansion_px differences
are naturally absorbed).  A shared crosshair tracks across every panel.

Layout
------
 ┌──────────────────────────────────────────────────────────────────────────┐
 │ Source: [__________] [Browse…]  Frame: [0 ▼]  Pip: [▲▼] 1 2 3…         │
 ├─────────────────────────────────────────────────────────────────────────┤
 │ LEFT (~260 px)            │  RIGHT (QScrollArea, horizontal)            │
 │ Presets                   │  [Image strip] [Chart A] [Chart B] …        │
 │ ☑ preset 1                │                                             │
 │ ☐ preset 2                │   All panels same height, aligned to        │
 │ ☑ preset 3                │   global z_min … z_max.                    │
 │ …                         │                                             │
 │ [Sel All] [Clear All]     │   Header per chart:                         │
 │ [▶ Run]   status…         │     ◀ ▶ ▼ description                      │
 └─────────────────────────────────────────────────────────────────────────┘
"""

from __future__ import annotations

import copy
import os
from typing import Any, Dict, List, Optional, TYPE_CHECKING

import cv2
import numpy as np

from PySide6.QtCore import Qt, QRectF, QSettings, QTimer, Signal
from PySide6.QtGui import (
    QColor, QImage, QPainter, QPen, QPixmap, QFont,
)
from PySide6.QtWidgets import (
    QApplication, QCheckBox, QComboBox, QDialog, QFileDialog,
    QFrame, QGraphicsScene, QGraphicsView, QHBoxLayout, QLabel,
    QListWidget, QListWidgetItem, QMenu, QPushButton,
    QScrollArea, QSizePolicy, QSpinBox, QVBoxLayout, QWidget,
    QSplitter,
)

from pa_gui.analysis import ab_presets
from pa_gui.analysis.ab_compare_dialog import (
    _ZoomableView,
    _ndarray_to_pixmap,
    _signal_pixmap_arr,
)
from pa_gui.analysis.mode_compare_worker import ModeCompareWorker

if TYPE_CHECKING:
    from pa.pipeline.types import DebugStage

_SETTINGS_ORG = "rsimpson-bc"
_SETTINGS_APP = "PA-GUI"
_SETTINGS_KEY = "ab_compare_dialog"   # shared with ABCompareDialog

_CHART_W    = 187        # px wide per chart column  (560 / 3)
_DISPLAY_H  = 700        # px tall for the comparison area
_IMG_W      = 200        # px wide for the image strip
_HEADER_H   = 28         # px tall header row above each panel
_COLLAPSED_W = 28        # px wide when a chart column is collapsed
_DEBOUNCE_MS = 600


# ---------------------------------------------------------------------------
# Collapsible/reorderable chart panel
# ---------------------------------------------------------------------------

class _ChartPanel(QWidget):
    """
    One chart column in the Mode Compare right area.

    Header (top):  [◀ reorder] [▶ reorder] [▼/▶ collapse]  description label
    Body:          _ZoomableView showing the signal chart
    """

    # emitted when the user clicks a reorder arrow: positive = move right, negative = left
    move_requested = Signal(object, int)   # (self, delta)
    # emitted when the mouse enters/leaves the panel (for list highlight)
    hovered = Signal(bool)

    def __init__(self, preset_idx: int, description: str, parent=None):
        super().__init__(parent)
        self.preset_idx  = preset_idx
        self.description = description
        self._collapsed  = False
        self._expanded_w = _CHART_W   # updated by set_expanded_width()

        self.setFixedWidth(_CHART_W)
        # Expand vertically so HBoxLayout gives the full scroll-area height
        self.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Expanding)

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        # Header bar
        hdr = QWidget()
        hdr.setFixedHeight(_HEADER_H)
        hdr.setStyleSheet("background:#2a2a2a;")
        hdr_lay = QHBoxLayout(hdr)
        hdr_lay.setContentsMargins(3, 0, 3, 0)
        hdr_lay.setSpacing(2)

        self._btn_left = QPushButton("◀")
        self._btn_left.setFixedSize(20, 20)
        self._btn_left.setToolTip("Move this column left")
        self._btn_left.clicked.connect(lambda: self.move_requested.emit(self, -1))
        hdr_lay.addWidget(self._btn_left)

        self._btn_right = QPushButton("▶")
        self._btn_right.setFixedSize(20, 20)
        self._btn_right.setToolTip("Move this column right")
        self._btn_right.clicked.connect(lambda: self.move_requested.emit(self, +1))
        hdr_lay.addWidget(self._btn_right)

        self._btn_collapse = QPushButton("▼")
        self._btn_collapse.setFixedSize(20, 20)
        self._btn_collapse.setToolTip("Collapse / expand this column")
        self._btn_collapse.clicked.connect(self._toggle_collapse)
        hdr_lay.addWidget(self._btn_collapse)

        lbl = QLabel(description[:60] + ("…" if len(description) > 60 else ""))
        lbl.setToolTip(description)
        lbl.setStyleSheet("color:#ccc; font-size:10px;")
        lbl.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        hdr_lay.addWidget(lbl, 1)

        root.addWidget(hdr)

        # Chart view
        self.view = _ZoomableView(f"Preset {preset_idx + 1}", stretch_fit=True)
        self.view.setMinimumHeight(100)
        root.addWidget(self.view, 1)

    # ------------------------------------------------------------------
    def enterEvent(self, event):
        self.hovered.emit(True)
        super().enterEvent(event)

    def leaveEvent(self, event):
        self.hovered.emit(False)
        super().leaveEvent(event)

    def set_expanded_width(self, w: int):
        """Update the expanded width and apply immediately if not collapsed."""
        self._expanded_w = w
        if not self._collapsed:
            self.setFixedWidth(w)

    def _toggle_collapse(self):
        self._collapsed = not self._collapsed
        self.view.setVisible(not self._collapsed)
        if self._collapsed:
            self.setFixedWidth(_COLLAPSED_W)
            self._btn_collapse.setText("▶")
        else:
            self.setFixedWidth(self._expanded_w)
            self._btn_collapse.setText("▼")


# ---------------------------------------------------------------------------
# Image strip panel
# ---------------------------------------------------------------------------

class _StripPanel(QWidget):
    """Left-most panel: tip image strip cropped to global z-range."""

    def __init__(self, parent=None):
        super().__init__(parent)
        # Width is managed dynamically via set_aspect_ratio + resizeEvent
        self.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Expanding)
        self._aspect_ratio: Optional[float] = None  # w/h of the cropped ROI image

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        hdr = QWidget()
        hdr.setFixedHeight(_HEADER_H)
        hdr.setStyleSheet("background:#1e2e1e;")
        hdr_lay = QHBoxLayout(hdr)
        hdr_lay.setContentsMargins(4, 0, 4, 0)

        lbl = QLabel("Image")
        lbl.setStyleSheet("color:#aaa; font-size:10px; font-weight:bold;")
        hdr_lay.addWidget(lbl)
        hdr_lay.addStretch()

        self._toggle_btn = QPushButton("Contrast off")
        self._toggle_btn.setCheckable(True)
        self._toggle_btn.setFixedHeight(20)
        self._toggle_btn.setStyleSheet(
            "QPushButton { font-size:9px; padding:0 4px; background:#333; color:#aaa; }"
            "QPushButton:checked { background:#2a5f9e; color:white; }"
        )
        self._toggle_btn.toggled.connect(self._on_toggle)
        hdr_lay.addWidget(self._toggle_btn)

        root.addWidget(hdr)

        self.view = _ZoomableView("Image", stretch_fit=True)
        root.addWidget(self.view, 1)

        self._source_pm: Optional[QPixmap] = None
        self._contrast_pm: Optional[QPixmap] = None

    # ------------------------------------------------------------------
    def set_aspect_ratio(self, ratio: float):
        """Store w/h ratio so resizeEvent can keep the strip proportional."""
        self._aspect_ratio = ratio

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self._aspect_ratio is not None:
            body_h = max(1, event.size().height() - _HEADER_H)
            new_w = max(40, int(body_h * self._aspect_ratio))
            if new_w != self.width():
                self.setFixedWidth(new_w)

    # ------------------------------------------------------------------
    def set_images(self, source_pm: Optional[QPixmap], contrast_pm: Optional[QPixmap]):
        self._source_pm  = source_pm
        self._contrast_pm = contrast_pm
        self._toggle_btn.setEnabled(contrast_pm is not None)
        self._refresh()

    def _on_toggle(self, checked: bool):
        self._toggle_btn.setText("Contrast on" if checked else "Contrast off")
        self._refresh()

    def _refresh(self):
        use_contrast = self._toggle_btn.isChecked() and self._contrast_pm is not None
        pm = self._contrast_pm if use_contrast else self._source_pm
        if pm is not None:
            self.view.set_pixmap(pm)


# ---------------------------------------------------------------------------
# Main dialog
# ---------------------------------------------------------------------------

class ModeCompareDialog(QDialog):
    """
    Compare signal profiles of multiple saved presets for one mode.
    """

    def __init__(
        self,
        mode_name: str,
        instrument_config_path: str = "",
        parent=None,
    ):
        super().__init__(parent)
        self.setWindowTitle("Mode Compare")
        self.resize(1400, 880)
        self.setWindowFlag(Qt.WindowType.Window)
        self.setWindowFlags(
            self.windowFlags()
            | Qt.WindowType.WindowMaximizeButtonHint
        )

        self._mode_name = mode_name
        self._instrument_config_path = instrument_config_path

        # Chart display width (px) — user-adjustable via spinbox
        self._chart_w: int = _CHART_W

        # Cached data from last successful _render_all for cheap re-render
        self._cached_src_bgr:       Optional[object] = None
        self._cached_ctr_bgr:       Optional[object] = None
        self._cached_z_range:       Optional[tuple]  = None
        self._cached_bbox:          Optional[list]   = None
        self._cached_roi1_points:   Optional[list]   = None
        self._cached_signal_data:   Dict[int, dict]  = {}   # pre-offset per-preset data
        self._cached_strip_display_w: int = _IMG_W
        self._cached_strip:         Optional[object] = None
        # Map chart panel preset_idx → QListWidget row for hover highlight
        self._panel_list_row:       Dict[int, int]   = {}

        # Runtime state
        self._image_paths:     List[str] = []
        self._reference_paths: List[str] = []
        self._worker:          Optional[ModeCompareWorker] = None
        self._running_entries: List[Dict[str, Any]] = []  # entries at run-time
        self._debounce = QTimer(self)
        self._debounce.setSingleShot(True)
        self._debounce.setInterval(_DEBOUNCE_MS)
        self._debounce.timeout.connect(self._run_comparison)

        # Per-preset results accumulated while worker runs
        # Each entry: {"entry": preset_dict, "stages": [...], "source": ndarray,
        #              "contrast": ndarray|None, "roi_bbox": tuple|None}
        self._results: Dict[int, dict] = {}

        # Registered _ZoomableViews for crosshair sync
        self._all_views: List[_ZoomableView] = []

        # Ordered list of _ChartPanel widgets currently in the right area
        self._chart_panels: List[_ChartPanel] = []

        self._setup_ui()
        self._populate_preset_list()
        self._restore_saved_source()
        self._restore_saved_reference()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _setup_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(6, 6, 6, 6)
        root.setSpacing(4)

        # ── Top source/frame/pipette row ──────────────────────────────────
        top = QHBoxLayout()

        top.addWidget(QLabel("Source:"))
        self._src_edit = QLabel("<i>no image selected</i>")
        self._src_edit.setFixedWidth(260)
        self._src_edit.setStyleSheet(
            "background:#1a1a1a; padding:2px 4px; border:1px solid #444; color:#888;"
        )
        top.addWidget(self._src_edit)

        browse_btn = QPushButton("File…")
        browse_btn.setFixedWidth(52)
        browse_btn.clicked.connect(self._on_browse)
        top.addWidget(browse_btn)
        browse_folder_btn = QPushButton("Folder…")
        browse_folder_btn.setFixedWidth(58)
        browse_folder_btn.clicked.connect(self._on_browse_folder)
        top.addWidget(browse_folder_btn)

        top.addSpacing(12)
        top.addWidget(QLabel("Frame:"))
        self._frame_combo = QComboBox()
        self._frame_combo.setFixedWidth(160)
        self._frame_combo.currentIndexChanged.connect(self._on_frame_changed)
        top.addWidget(self._frame_combo)

        top.addSpacing(12)
        top.addWidget(QLabel("Pipette:"))
        self._pipette_spin = QSpinBox()
        self._pipette_spin.setRange(1, 32)
        self._pipette_spin.setValue(1)
        self._pipette_spin.setFixedWidth(44)
        self._pipette_spin.valueChanged.connect(self._on_pipette_changed)
        top.addWidget(self._pipette_spin)

        self._pipette_btns: List[QPushButton] = []
        for i in range(1, 9):
            btn = QPushButton(str(i))
            btn.setFixedSize(22, 22)
            btn.clicked.connect(lambda _c, n=i: self._pipette_spin.setValue(n))
            top.addWidget(btn)
            self._pipette_btns.append(btn)
        self._refresh_pipette_buttons()

        top.addStretch()

        # ── Display options ───────────────────────────────────────────────
        top.addWidget(QLabel("Overlay:"))
        self._roi1_chk = QCheckBox("ROI1")
        self._roi1_chk.setChecked(True)
        self._roi1_chk.setToolTip("Draw the ROI1 polygon on the image strip")
        self._roi1_chk.stateChanged.connect(self._on_display_option_changed)
        top.addWidget(self._roi1_chk)

        self._poi_chk = QCheckBox("POI")
        self._poi_chk.setChecked(False)
        self._poi_chk.setToolTip("Show detected points-of-interest on signal graphs")
        self._poi_chk.stateChanged.connect(self._on_display_option_changed)
        top.addWidget(self._poi_chk)

        top.addSpacing(8)
        top.addWidget(QLabel("Graph w:"))
        self._chart_w_spin = QSpinBox()
        self._chart_w_spin.setRange(60, 600)
        self._chart_w_spin.setValue(_CHART_W)
        self._chart_w_spin.setSingleStep(10)
        self._chart_w_spin.setFixedWidth(56)
        self._chart_w_spin.setToolTip("Chart column width in pixels")
        self._chart_w_spin.valueChanged.connect(self._on_display_option_changed)
        top.addWidget(self._chart_w_spin)

        top.addSpacing(8)
        self._status_lbl = QLabel("")
        self._status_lbl.setStyleSheet("color:#888; font-style:italic;")
        top.addWidget(self._status_lbl)

        self._run_btn = QPushButton("▶  Run")
        self._run_btn.setEnabled(False)
        self._run_btn.setFixedWidth(80)
        self._run_btn.setStyleSheet(
            "QPushButton { background:#2a5f9e; color:white; font-weight:bold; }"
            "QPushButton:disabled { background:#444; color:#888; }"
        )
        self._run_btn.clicked.connect(self._run_comparison)
        top.addWidget(self._run_btn)

        root.addLayout(top)

        # ── Horizontal separator ──────────────────────────────────────────
        sep = QFrame()
        sep.setFrameShape(QFrame.Shape.HLine)
        sep.setStyleSheet("color:#444;")
        root.addWidget(sep)

        # ── Main body: left checklist + right comparison area ─────────────
        body = QSplitter(Qt.Orientation.Horizontal)

        # Left: preset checklist
        left = QWidget()
        left.setMinimumWidth(220)
        left.setMaximumWidth(340)
        left_lay = QVBoxLayout(left)
        left_lay.setContentsMargins(0, 0, 4, 0)
        left_lay.setSpacing(4)

        left_lay.addWidget(QLabel("<b>Presets</b>"))

        self._preset_list = QListWidget()
        self._preset_list.setToolTip(
            "Check presets to include in the comparison.\n"
            "Presets are listed newest-first."
        )
        self._preset_list.setContextMenuPolicy(
            Qt.ContextMenuPolicy.CustomContextMenu
        )
        self._preset_list.customContextMenuRequested.connect(
            self._on_preset_list_context_menu
        )
        left_lay.addWidget(self._preset_list, 1)

        sel_row = QHBoxLayout()
        sel_all_btn  = QPushButton("Select All")
        clr_all_btn  = QPushButton("Clear All")
        sel_all_btn.clicked.connect(self._select_all)
        clr_all_btn.clicked.connect(self._clear_all)
        sel_row.addWidget(sel_all_btn)
        sel_row.addWidget(clr_all_btn)
        left_lay.addLayout(sel_row)

        ref_row = QHBoxLayout()
        ref_row.addWidget(QLabel("Ref:"))
        self._ref_edit = QLabel("<i>none</i>")
        self._ref_edit.setFixedWidth(120)
        self._ref_edit.setStyleSheet(
            "background:#1a1a1a; padding:1px 3px; border:1px solid #444; "
            "color:#888; font-size:9px;"
        )
        ref_row.addWidget(self._ref_edit, 1)
        ref_browse = QPushButton("…")
        ref_browse.setFixedWidth(24)
        ref_browse.setToolTip("Browse reference file")
        ref_browse.clicked.connect(self._on_browse_ref)
        ref_row.addWidget(ref_browse)
        ref_browse_folder = QPushButton("📁")
        ref_browse_folder.setFixedWidth(24)
        ref_browse_folder.setToolTip("Browse reference folder")
        ref_browse_folder.clicked.connect(self._on_browse_ref_folder)
        ref_row.addWidget(ref_browse_folder)
        ref_clear = QPushButton("✕")
        ref_clear.setFixedWidth(22)
        ref_clear.setToolTip("Clear reference image")
        ref_clear.clicked.connect(self._on_clear_ref)
        ref_row.addWidget(ref_clear)
        left_lay.addLayout(ref_row)

        body.addWidget(left)

        # Right: horizontally scrollable panels area
        right_outer = QWidget()
        right_outer_lay = QVBoxLayout(right_outer)
        right_outer_lay.setContentsMargins(0, 0, 0, 0)
        right_outer_lay.setSpacing(0)

        self._scroll = QScrollArea()
        self._scroll.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAsNeeded
        )
        self._scroll.setVerticalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )
        self._scroll.setWidgetResizable(True)

        self._panels_container = QWidget()
        self._panels_layout = QHBoxLayout(self._panels_container)
        self._panels_layout.setContentsMargins(4, 0, 4, 0)
        self._panels_layout.setSpacing(4)
        self._panels_layout.addStretch()

        self._scroll.setWidget(self._panels_container)
        right_outer_lay.addWidget(self._scroll, 1)

        body.addWidget(right_outer)
        body.setStretchFactor(0, 0)
        body.setStretchFactor(1, 1)

        root.addWidget(body, 1)

    # ------------------------------------------------------------------
    # Preset list
    # ------------------------------------------------------------------

    def _populate_preset_list(self):
        from collections import OrderedDict
        from PySide6.QtGui import QColor
        self._preset_list.clear()
        # Load ALL presets from ALL modes, grouped by mode name (newest-first)
        data = ab_presets.load_presets()
        all_entries = list(reversed(data.get("presets", [])))
        by_mode: dict = OrderedDict()
        for entry in all_entries:
            m = entry.get("mode", "Unknown")
            by_mode.setdefault(m, []).append(entry)
        # Put the initial mode first
        ordered: list = []
        if self._mode_name in by_mode:
            ordered.append((self._mode_name, by_mode.pop(self._mode_name)))
        for m, entries in by_mode.items():
            ordered.append((m, entries))
        for mode_name, entries in ordered:
            # Non-checkable mode header
            hdr = QListWidgetItem(f"── {mode_name} ──")
            hdr.setFlags(Qt.ItemFlag.NoItemFlags)
            hdr.setForeground(QColor("#4a8fd8"))
            hdr.setData(Qt.ItemDataRole.UserRole, None)  # sentinel
            self._preset_list.addItem(hdr)
            for entry in entries:
                label = ab_presets.format_combo_label(entry)
                item = QListWidgetItem(f"  {label}")
                item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
                # Pre-check presets belonging to the initial mode
                init = Qt.CheckState.Checked if entry.get("mode") == self._mode_name else Qt.CheckState.Unchecked
                item.setCheckState(init)
                item.setToolTip(
                    f"Mode: {entry.get('mode', '')}\n"
                    f"{entry.get('description', '')}\n\n"
                    f"Created: {entry.get('created', '')}"
                )
                item.setData(Qt.ItemDataRole.UserRole, entry)
                self._preset_list.addItem(item)

    def _selected_entries(self) -> List[Dict[str, Any]]:
        result = []
        for i in range(self._preset_list.count()):
            item = self._preset_list.item(i)
            if item.data(Qt.ItemDataRole.UserRole) is None:
                continue  # mode header
            if item.checkState() == Qt.CheckState.Checked:
                result.append(item.data(Qt.ItemDataRole.UserRole))
        return result

    def _select_all(self):
        for i in range(self._preset_list.count()):
            item = self._preset_list.item(i)
            if item.data(Qt.ItemDataRole.UserRole) is not None:
                item.setCheckState(Qt.CheckState.Checked)

    def _clear_all(self):
        for i in range(self._preset_list.count()):
            item = self._preset_list.item(i)
            if item.data(Qt.ItemDataRole.UserRole) is not None:
                item.setCheckState(Qt.CheckState.Unchecked)

    def _on_preset_list_context_menu(self, pos):
        """Right-click context menu on the preset list."""
        clicked_item = self._preset_list.itemAt(pos)

        # Determine which mode the clicked item belongs to
        # (walk upward in the list to find the nearest header above it).
        clicked_mode: Optional[str] = None
        if clicked_item is not None:
            row = self._preset_list.row(clicked_item)
            for r in range(row, -1, -1):
                candidate = self._preset_list.item(r)
                if candidate.data(Qt.ItemDataRole.UserRole) is None:
                    # This is a mode-header item; extract the mode name
                    text = candidate.text().strip().strip("\u2500").strip()
                    clicked_mode = text
                    break

        menu = QMenu(self)

        if clicked_mode:
            act_mode = menu.addAction(f'Select all in "{clicked_mode}"')
        else:
            act_mode = None

        act_all = menu.addAction("Select all presets")
        menu.addSeparator()
        act_clear = menu.addAction("Clear all")

        chosen = menu.exec(self._preset_list.viewport().mapToGlobal(pos))

        if chosen is None:
            return
        if act_mode is not None and chosen is act_mode:
            # Check only presets belonging to this mode; uncheck all others
            current_mode_header = False
            for i in range(self._preset_list.count()):
                item = self._preset_list.item(i)
                if item.data(Qt.ItemDataRole.UserRole) is None:
                    # Header — switch tracking flag
                    header_text = item.text().strip().strip("\u2500").strip()
                    current_mode_header = (header_text == clicked_mode)
                else:
                    state = Qt.CheckState.Checked if current_mode_header else Qt.CheckState.Unchecked
                    item.setCheckState(state)
        elif chosen is act_all:
            self._select_all()
        elif chosen is act_clear:
            self._clear_all()

    # ------------------------------------------------------------------
    # Source / reference browse
    # ------------------------------------------------------------------

    def _on_browse(self):
        s = QSettings(_SETTINGS_ORG, _SETTINGS_APP)
        start = s.value(f"{_SETTINGS_KEY}/last_browse_dir", "")
        path, _ = QFileDialog.getOpenFileName(
            self, "Select source image", start,
            "Images (*.jpg *.jpeg *.png *.bmp);;All files (*)",
        )
        if path:
            s.setValue(f"{_SETTINGS_KEY}/last_browse_dir", os.path.dirname(path))
            s.setValue(f"{_SETTINGS_KEY}/last_file", path)
            s.remove(f"{_SETTINGS_KEY}/last_folder")
            self._load_source_file(path)

    def _on_browse_folder(self):
        s = QSettings(_SETTINGS_ORG, _SETTINGS_APP)
        start = s.value(f"{_SETTINGS_KEY}/last_browse_dir", "")
        folder = QFileDialog.getExistingDirectory(self, "Select image folder", start)
        if folder:
            s.setValue(f"{_SETTINGS_KEY}/last_browse_dir", folder)
            s.setValue(f"{_SETTINGS_KEY}/last_folder", folder)
            s.remove(f"{_SETTINGS_KEY}/last_file")
            self._load_source_folder(folder)

    def _load_source_file(self, path: str):
        self._image_paths = [path]
        self._src_edit.setText(os.path.basename(path))
        self._src_edit.setToolTip(path)
        self._src_edit.setStyleSheet(
            "background:#1a1a1a; padding:2px 4px; border:1px solid #444; color:#ccc;"
        )
        self._populate_frame_combo()
        self._run_btn.setEnabled(True)

    def _load_source_folder(self, folder: str):
        exts = (".jpg", ".jpeg", ".png", ".bmp")
        paths = sorted(
            os.path.join(folder, f)
            for f in os.listdir(folder)
            if f.lower().endswith(exts)
        )
        if not paths:
            self._status_lbl.setText("No images in folder.")
            return
        self._image_paths = paths
        self._src_edit.setText(f"{os.path.basename(folder)}/  ({len(paths)} images)")
        self._src_edit.setToolTip(folder)
        self._src_edit.setStyleSheet(
            "background:#1a1a1a; padding:2px 4px; border:1px solid #444; color:#ccc;"
        )
        self._populate_frame_combo()
        self._run_btn.setEnabled(True)

    def _on_browse_ref(self):
        s = QSettings(_SETTINGS_ORG, _SETTINGS_APP)
        start = s.value(f"{_SETTINGS_KEY}/last_ref_dir", "")
        path, _ = QFileDialog.getOpenFileName(
            self, "Select reference image", start,
            "Images (*.jpg *.jpeg *.png *.bmp);;All files (*)",
        )
        if path:
            s.setValue(f"{_SETTINGS_KEY}/last_ref_dir", os.path.dirname(path))
            s.setValue(f"{_SETTINGS_KEY}/last_ref_file", path)
            s.remove(f"{_SETTINGS_KEY}/last_ref_folder")
            self._load_ref_file(path)

    def _on_browse_ref_folder(self):
        s = QSettings(_SETTINGS_ORG, _SETTINGS_APP)
        start = s.value(f"{_SETTINGS_KEY}/last_ref_dir", "")
        folder = QFileDialog.getExistingDirectory(self, "Select reference folder", start)
        if folder:
            s.setValue(f"{_SETTINGS_KEY}/last_ref_dir", folder)
            s.setValue(f"{_SETTINGS_KEY}/last_ref_folder", folder)
            s.remove(f"{_SETTINGS_KEY}/last_ref_file")
            self._load_ref_folder(folder)

    def _load_ref_file(self, path: str):
        self._reference_paths = [path]
        self._ref_edit.setText(os.path.basename(path))
        self._ref_edit.setStyleSheet(
            "background:#1a1a1a; padding:1px 3px; border:1px solid #2a6; "
            "color:#ccc; font-size:9px;"
        )
        self._ref_edit.setToolTip(path)

    def _load_ref_folder(self, folder: str):
        exts = (".jpg", ".jpeg", ".png", ".bmp")
        paths = sorted(
            os.path.join(folder, f)
            for f in os.listdir(folder)
            if f.lower().endswith(exts)
        )
        if not paths:
            self._status_lbl.setText("No images in reference folder.")
            return
        self._reference_paths = paths
        self._ref_edit.setText(f"{os.path.basename(folder)}/  ({len(paths)} images)")
        self._ref_edit.setStyleSheet(
            "background:#1a1a1a; padding:1px 3px; border:1px solid #2a6; "
            "color:#ccc; font-size:9px;"
        )
        self._ref_edit.setToolTip(folder)

    def _on_clear_ref(self):
        self._reference_paths = []
        self._ref_edit.setText("<i>none</i>")
        self._ref_edit.setStyleSheet(
            "background:#1a1a1a; padding:1px 3px; border:1px solid #444; "
            "color:#888; font-size:9px;"
        )
        self._ref_edit.setToolTip("")

    def _restore_saved_source(self):
        """Auto-load the last source used by A/B Compare (or Mode Compare)."""
        s = QSettings(_SETTINGS_ORG, _SETTINGS_APP)
        saved_file   = s.value(f"{_SETTINGS_KEY}/last_file", "")
        saved_folder = s.value(f"{_SETTINGS_KEY}/last_folder", "")
        if saved_file and os.path.isfile(saved_file):
            self._load_source_file(saved_file)
        elif saved_folder and os.path.isdir(saved_folder):
            self._load_source_folder(saved_folder)

    def _restore_saved_reference(self):
        """Auto-load the last reference used by A/B Compare (or Mode Compare)."""
        s = QSettings(_SETTINGS_ORG, _SETTINGS_APP)
        saved_file   = s.value(f"{_SETTINGS_KEY}/last_ref_file", "")
        saved_folder = s.value(f"{_SETTINGS_KEY}/last_ref_folder", "")
        if saved_file and os.path.isfile(saved_file):
            self._load_ref_file(saved_file)
        elif saved_folder and os.path.isdir(saved_folder):
            self._load_ref_folder(saved_folder)

    def _populate_frame_combo(self):
        self._frame_combo.blockSignals(True)
        self._frame_combo.clear()
        for i, p in enumerate(self._image_paths):
            self._frame_combo.addItem(f"Frame {i}  ({os.path.basename(p)})")
        self._frame_combo.blockSignals(False)

    def _on_frame_changed(self):
        if self._image_paths:
            self._debounce.start()

    def _on_pipette_changed(self):
        self._refresh_pipette_buttons()
        if self._image_paths:
            self._debounce.start()

    def _refresh_pipette_buttons(self):
        cur = self._pipette_spin.value()
        for i, btn in enumerate(self._pipette_btns, start=1):
            if i == cur:
                btn.setStyleSheet(
                    "QPushButton { background:#2a5f9e; color:white; font-weight:bold;"
                    " border:1px solid #4a8fd8; border-radius:3px; }"
                )
            else:
                btn.setStyleSheet(
                    "QPushButton { background:#2d2d2d; color:#ccc;"
                    " border:1px solid #555; border-radius:3px; }"
                    "QPushButton:hover { background:#3d3d3d; }"
                )

    # ------------------------------------------------------------------
    # Run
    # ------------------------------------------------------------------

    def _run_comparison(self):
        if not self._image_paths:
            self._status_lbl.setText("Select an image first.")
            return

        entries = self._selected_entries()
        if not entries:
            self._status_lbl.setText("Check at least one preset.")
            return

        if self._worker and self._worker.isRunning():
            self._worker.requestInterruption()
            self._worker.wait(2000)

        self._results = {}
        self._running_entries = entries  # snapshot for callbacks
        self._clear_panels()

        frame_idx    = max(0, self._frame_combo.currentIndex())
        pipette_idx  = self._pipette_spin.value() - 1

        # Mode name for the worker is just a fallback; _run_one will use entry["mode"]
        self._worker = ModeCompareWorker(
            mode_name=entries[0].get("mode", self._mode_name) if entries else self._mode_name,
            preset_entries=entries,
            image_paths=self._image_paths,
            reference_paths=self._reference_paths,
            frame_index=frame_idx,
            pipette_index=pipette_idx,
            instrument_config_path=self._instrument_config_path,
            parent=self,
        )
        self._worker.progress.connect(self._on_progress)
        self._worker.preset_done.connect(self._on_preset_done)
        self._worker.all_done.connect(self._on_all_done)
        self._worker.error.connect(self._on_error)

        self._run_btn.setEnabled(False)
        self._status_lbl.setText("Running…")
        self._worker.start()

    def _on_progress(self, msg: str):
        self._status_lbl.setText(msg)

    def _on_display_option_changed(self):
        """ROI1/POI toggle or chart-width change — re-render pixmaps in place."""
        if self._cached_signal_data:
            self._rerender_pixmaps()

    def _rerender_pixmaps(self):
        """Re-render all pixmaps using cached data without re-running the pipeline."""
        show_roi1 = self._roi1_chk.isChecked()
        show_poi  = self._poi_chk.isChecked()
        chart_w   = self._chart_w_spin.value()

        # Strip
        if self._cached_strip is not None and self._cached_src_bgr is not None:
            z_min, z_max = self._cached_z_range
            strip_pm = self._make_strip_pixmap(
                self._cached_src_bgr, z_min, z_max,
                _DISPLAY_H, self._cached_strip_display_w, self._cached_bbox,
                roi1_points=self._cached_roi1_points if show_roi1 else None,
            )
            ctr_pm = None
            if self._cached_ctr_bgr is not None:
                ctr_pm = self._make_strip_pixmap(
                    self._cached_ctr_bgr, z_min, z_max,
                    _DISPLAY_H, self._cached_strip_display_w, self._cached_bbox,
                    roi1_points=self._cached_roi1_points if show_roi1 else None,
                )
            self._cached_strip.set_images(strip_pm, ctr_pm)

        # Charts
        w_changed = (chart_w != self._chart_w)
        self._chart_w = chart_w
        for panel in self._chart_panels:
            sig_data = self._cached_signal_data.get(panel.preset_idx)
            if sig_data is None:
                continue
            chart_arr = _signal_pixmap_arr(
                sig_data["signal"],
                sig_data["z_axis"],   # already offset to full-image coords
                sig_data["poi_z"],
                w=chart_w,
                h=_DISPLAY_H,
                line_thickness=2,
                log_scale=False,
                show_poi=show_poi,
                extra_signals=sig_data["extra_signals"],
                z_range_override=self._cached_z_range,
            )
            panel.view.set_pixmap(_ndarray_to_pixmap(chart_arr))
            if w_changed:
                panel.set_expanded_width(chart_w)

        if w_changed:
            margins = self._panels_layout.contentsMargins()
            sp = self._panels_layout.spacing()
            n = len(self._chart_panels)
            content_w = (
                margins.left() + margins.right()
                + self._cached_strip_display_w
                + sp + n * chart_w
                + max(0, n - 1) * sp
            )
            self._panels_container.setMinimumWidth(content_w)

    def _on_preset_done(
        self,
        preset_idx: int,
        stages: list,
        source_image: object,
        contrast_image: object,
        roi_bbox: object,
    ):
        entry = self._running_entries[preset_idx] if preset_idx < len(self._running_entries) else {}
        self._results[preset_idx] = {
            "entry":    entry,
            "stages":   stages,
            "source":   source_image,
            "contrast": contrast_image,
            "roi_bbox": roi_bbox,
        }

    def _on_all_done(self):
        self._run_btn.setEnabled(True)
        self._status_lbl.setText(
            f"Done — {len(self._results)} preset(s) compared."
        )
        try:
            self._render_all()
        except Exception as exc:
            import traceback
            self._status_lbl.setText(f"Render error: {exc}")
            traceback.print_exc()

    def _on_error(self, msg: str):
        self._status_lbl.setText(f"Error: {msg[:80]}")
        self._run_btn.setEnabled(True)

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

    def _clear_panels(self):
        """Remove all panels from the layout and unregister crosshair views."""
        self._all_views.clear()
        self._chart_panels.clear()
        self._panel_list_row.clear()
        self._panels_container.setMinimumWidth(0)
        # Remove all widgets except the trailing stretch
        while self._panels_layout.count() > 1:
            item = self._panels_layout.takeAt(0)
            w = item.widget()
            if w:
                w.deleteLater()

    def _render_all(self):
        self._clear_panels()
        if not self._results:
            return

        # ── 1. Find global z-range across all preset signal stages ────────
        global_z_min: Optional[float] = None
        global_z_max: Optional[float] = None

        signal_stages: Dict[int, "DebugStage"] = {}  # preset_idx → stage

        roi_y_offsets: Dict[int, float] = {}   # full-image y-start for each preset's expanded ROI
        for idx, res in self._results.items():
            sig_stage = next(
                (s for s in res["stages"] if s.signal is not None), None
            )
            if sig_stage is None or sig_stage.z_axis_px is None or len(sig_stage.z_axis_px) < 2:
                continue
            signal_stages[idx] = sig_stage
            # Use display_bbox[1] (the expanded/contracted ROI's y-start in
            # full-image coords) so that roi_expansion_px shifts are reflected.
            # Fall back to roi_bbox[1] (base ROI) if no display_bbox is found.
            display_bb = next(
                (s.metadata.get("display_bbox") for s in res["stages"]
                 if s.metadata.get("display_bbox") is not None),
                None,
            )
            base_bb = res.get("roi_bbox")
            effective_bb = display_bb if display_bb is not None else base_bb
            roi_y = float(effective_bb[1]) if (effective_bb is not None and len(effective_bb) >= 2) else 0.0
            roi_y_offsets[idx] = roi_y
            lo = float(sig_stage.z_axis_px[0]) + roi_y
            hi = float(sig_stage.z_axis_px[-1]) + roi_y
            if global_z_min is None or lo < global_z_min:
                global_z_min = lo
            if global_z_max is None or hi > global_z_max:
                global_z_max = hi

        if global_z_min is None:
            self._status_lbl.setText("No signal stages found in results.")
            return

        z_range = (global_z_min, global_z_max)

        # ── 2. Resolve ROI bbox and source images ───────────────────────────
        first_res = self._results[min(self._results.keys())]
        src_bgr: Optional[np.ndarray] = first_res.get("source")
        ctr_bgr: Optional[np.ndarray] = None

        # LineContinuity signal stages have no display_bbox; image stages do.
        # Search every stage across all results for the first available bbox.
        _bbox_src = None
        for res in self._results.values():
            for s in res["stages"]:
                bb = s.metadata.get("display_bbox")
                if bb is not None:
                    _bbox_src = bb
                    break
            if _bbox_src is not None:
                break
        if _bbox_src is None:
            _bbox_src = first_res.get("roi_bbox")

        # Find a contrast image — use the first result that has one
        for res in self._results.values():
            if res.get("contrast") is not None:
                ctr_bgr = res["contrast"]
                break

        # Strip pixel width = ROI width (capped). Height and chart heights
        # are not pre-fixed — panels fill the scroll area height and
        # fitInView() in _ZoomableView.resizeEvent handles scaling.
        strip_display_w = _IMG_W
        if _bbox_src is not None:
            _bx, _by, _bw, _bh = _bbox_src
            strip_display_w = max(60, min(_bw, 400))

        # Render pixmaps at _DISPLAY_H; fitInView scales them at display time.
        display_h = _DISPLAY_H

        strip_pm:   Optional[QPixmap] = None
        strip_ctr_pm: Optional[QPixmap] = None

        if src_bgr is not None:
            strip_pm    = self._make_strip_pixmap(src_bgr,  global_z_min, global_z_max,
                                                  display_h, strip_display_w, _bbox_src)
        if ctr_bgr is not None:
            strip_ctr_pm = self._make_strip_pixmap(ctr_bgr, global_z_min, global_z_max,
                                                   display_h, strip_display_w, _bbox_src)

        # ── 3. Create strip panel ──────────────────────────────────────────
        strip = _StripPanel()
        strip.setFixedWidth(strip_display_w)   # initial width; resizeEvent adjusts it
        strip.set_aspect_ratio(strip_display_w / max(1, display_h))
        # No setFixedHeight — height fills the scroll area viewport
        self._panels_layout.insertWidget(
            self._panels_layout.count() - 1,  # before trailing stretch
            strip,
        )
        strip.set_images(strip_pm, strip_ctr_pm)
        self._all_views.append(strip.view)
        strip.view.row_hovered.connect(self._on_row_hovered)

        # ── 4. Create a chart panel per preset (in selection order) ───────
        show_poi  = self._poi_chk.isChecked()
        show_roi1 = self._roi1_chk.isChecked()
        self._chart_w = self._chart_w_spin.value()
        self._cached_signal_data = {}
        for idx in sorted(self._results.keys()):
            res = self._results[idx]
            sig_stage = signal_stages.get(idx)
            if sig_stage is None:
                continue

            entry = res["entry"]
            desc  = entry.get("description", f"Preset {idx + 1}")

            # Offset z values from ROI-local to full-image coordinates so all
            # presets (possibly with different roi_expansion_px) align correctly.
            roi_y = roi_y_offsets.get(idx, 0.0)
            z_axis_full = sig_stage.z_axis_px + roi_y
            poi_z_full = (
                [p + roi_y for p in sig_stage.poi_z_px]
                if sig_stage.poi_z_px else None
            )

            chart_arr = _signal_pixmap_arr(
                sig_stage.signal,
                z_axis_full,
                poi_z_full,
                w=self._chart_w,
                h=display_h,
                line_thickness=2,
                log_scale=False,
                show_poi=show_poi,
                extra_signals=sig_stage.extra_signals,
                z_range_override=z_range,
            )
            pm = _ndarray_to_pixmap(chart_arr)

            panel = _ChartPanel(idx, desc)
            panel.set_expanded_width(self._chart_w)
            # No setFixedHeight — height fills the scroll area viewport
            panel.move_requested.connect(self._on_move_panel)
            panel.hovered.connect(lambda v, _idx=idx: self._on_chart_hovered(_idx, v))
            self._panels_layout.insertWidget(
                self._panels_layout.count() - 1,
                panel,
            )
            panel.view.set_pixmap(pm)
            self._chart_panels.append(panel)
            self._all_views.append(panel.view)
            panel.view.row_hovered.connect(self._on_row_hovered)

            # Cache pre-offset signal data so re-renders (toggle/resize) stay aligned
            self._cached_signal_data[idx] = {
                "signal": sig_stage.signal,
                "z_axis": z_axis_full,
                "poi_z":  poi_z_full,
                "extra_signals": sig_stage.extra_signals,
            }
            # Map panel → list row for hover highlight
            self._panel_list_row[idx] = -1
            for _row in range(self._preset_list.count()):
                _item = self._preset_list.item(_row)
                if _item.data(Qt.ItemDataRole.UserRole) == entry:
                    self._panel_list_row[idx] = _row
                    break

        # Set minimum width on the container so the horizontal scrollbar
        # appears correctly.  Height is unconstrained — setWidgetResizable(True)
        # on the scroll area fills the viewport height automatically.
        margins = self._panels_layout.contentsMargins()
        sp = self._panels_layout.spacing()
        n = len(self._chart_panels)
        content_w = (
            margins.left() + margins.right()
            + strip_display_w
            + sp
            + n * self._chart_w
            + max(0, n - 1) * sp
        )
        self._panels_container.setMinimumWidth(content_w)
        # ── 5. Cache data for cheap re-renders (toggle ROI1/POI, resize width) ──
        self._cached_src_bgr        = src_bgr
        self._cached_ctr_bgr        = ctr_bgr
        self._cached_z_range        = z_range   # full-image row coordinates
        self._cached_bbox           = _bbox_src
        self._cached_strip_display_w = strip_display_w
        self._cached_strip          = strip
        # _cached_signal_data is populated per-preset in step 4 above
        # Collect roi1_points from any stage that has them
        self._cached_roi1_points = None
        for res in self._results.values():
            for s in res["stages"]:
                pts = s.metadata.get("roi1_points")
                if pts:
                    self._cached_roi1_points = pts
                    break
            if self._cached_roi1_points:
                break

        # Re-render immediately to apply current ROI1/POI toggle state.
        self._rerender_pixmaps()
    # ------------------------------------------------------------------

    def _make_strip_pixmap(
        self,
        bgr,
        z_min: float,
        z_max: float,
        display_h: int,
        display_w: int,
        bbox=None,
        roi1_points=None,
    ) -> QPixmap:
        """
        Crop bgr to the ROI x-range and global z rows, then scale to display size.
        Optionally draw the ROI1 polygon overlay.

        z_min / z_max are in full-image row coordinates (roi_y already added by caller).
        bbox is (x, y, w, h) in full-image coordinates; only x/w are used for cropping.
        roi1_points is a list of [x, y] full-image coordinates.
        """
        h, w = bgr.shape[:2]
        if bbox is not None:
            bx, by, bw, bh = bbox
            x0 = max(0, int(bx))
            x1 = min(w, int(bx + bw))
        else:
            x0, x1 = 0, w
        # z_min/z_max are in full-image row coordinates (roi_y already added)
        y0 = max(0, int(z_min))
        y1 = min(h, int(z_max) + 1)
        if y1 <= y0:
            y0, y1 = 0, h
        if x1 <= x0:
            x0, x1 = 0, w
        cropped = bgr[y0:y1, x0:x1].copy()
        if cropped.size == 0:
            cropped = bgr.copy()
        scaled = cv2.resize(cropped, (display_w, display_h), interpolation=cv2.INTER_LINEAR)

        # Draw ROI1 polygon (if provided) in crop→scale coords
        if roi1_points:
            crop_h = max(1, y1 - y0)
            crop_w = max(1, x1 - x0)
            sx = display_w / crop_w
            sy = display_h / crop_h
            pts = np.array(
                [[int((p[0] - x0) * sx), int((p[1] - y0) * sy)]
                 for p in roi1_points],
                dtype=np.int32,
            )
            cv2.polylines(scaled, [pts], isClosed=True,
                          color=(220, 220, 0), thickness=1, lineType=cv2.LINE_AA)

        rgb = cv2.cvtColor(scaled, cv2.COLOR_BGR2RGB)
        qimg = QImage(
            rgb.data, rgb.shape[1], rgb.shape[0],
            rgb.strides[0], QImage.Format.Format_RGB888,
        )
        return QPixmap.fromImage(qimg)

    # ------------------------------------------------------------------
    # Crosshair sync
    # ------------------------------------------------------------------

    def _on_row_hovered(self, frac: float):
        for view in self._all_views:
            view.set_crosshair_frac(frac)

    def _on_chart_hovered(self, preset_idx: int, entered: bool):
        """Highlight the corresponding preset list item when a chart panel is hovered."""
        row = self._panel_list_row.get(preset_idx, -1)
        if row < 0:
            return
        item = self._preset_list.item(row)
        if item is None:
            return
        if entered:
            item.setBackground(QColor("#2a5f9e"))
            self._preset_list.scrollToItem(item)
        else:
            item.setBackground(QColor(0, 0, 0, 0))   # fully transparent — restores default

    # ------------------------------------------------------------------
    # Column reorder
    # ------------------------------------------------------------------

    def _on_move_panel(self, panel: _ChartPanel, delta: int):
        """Move *panel* left (delta=-1) or right (delta=+1) among chart panels."""
        if panel not in self._chart_panels:
            return
        old_pos = self._chart_panels.index(panel)
        new_pos = old_pos + delta
        if new_pos < 0 or new_pos >= len(self._chart_panels):
            return

        # Swap in our tracking list
        self._chart_panels[old_pos], self._chart_panels[new_pos] = (
            self._chart_panels[new_pos], self._chart_panels[old_pos],
        )

        # The layout has: strip (idx 0), chart panels (idx 1…N), stretch (idx N+1)
        # chart panel layout indices start at 1
        lo = self._panels_layout
        old_layout_idx = old_pos + 1   # +1 for strip
        new_layout_idx = new_pos + 1

        # Remove and re-insert
        item = lo.takeAt(old_layout_idx)
        lo.insertWidget(new_layout_idx, item.widget())
