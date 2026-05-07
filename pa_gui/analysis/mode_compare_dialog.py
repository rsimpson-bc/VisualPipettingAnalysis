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

from PySide6.QtCore import Qt, QRectF, QSettings, QThread, QTimer, Signal
from PySide6.QtGui import (
    QColor, QImage, QPainter, QPen, QPixmap, QFont,
)
from PySide6.QtWidgets import (
    QApplication, QCheckBox, QComboBox, QDialog, QFileDialog,
    QFormLayout, QFrame, QGraphicsScene, QGraphicsView, QGroupBox,
    QHBoxLayout, QLabel,
    QListWidget, QListWidgetItem, QMenu, QPushButton,
    QScrollArea, QSizePolicy, QSpinBox, QToolButton, QVBoxLayout, QWidget,
    QWidgetAction, QSplitter,
)

from pa_gui.analysis import ab_annotations, ab_presets
from pa_gui.analysis.rule_editor_dialog import BehaviorRulesEditorDialog
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
_HEADER_H   = 56         # px tall header row above each panel (two rows)
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
    # emitted when the user right-clicks on the chart (annotation mode)
    annotation_requested = Signal(object, float)   # (self, z_frac)

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

        # Header bar — description label on top row, nav buttons on bottom row
        hdr = QWidget()
        hdr.setFixedHeight(_HEADER_H)
        hdr.setStyleSheet("background:#2a2a2a;")
        hdr_root = QVBoxLayout(hdr)
        hdr_root.setContentsMargins(3, 3, 3, 2)
        hdr_root.setSpacing(2)

        # Top row: description text fills the full width
        lbl = QLabel(description[:60] + ("…" if len(description) > 60 else ""))
        lbl.setToolTip(description)
        lbl.setStyleSheet("color:#ccc; font-size:10px;")
        lbl.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        hdr_root.addWidget(lbl)

        # Bottom row: nav buttons left-aligned
        btn_row = QHBoxLayout()
        btn_row.setContentsMargins(0, 0, 0, 0)
        btn_row.setSpacing(2)

        self._btn_left = QPushButton("◀")
        self._btn_left.setFixedSize(20, 20)
        self._btn_left.setToolTip("Move this column left")
        self._btn_left.clicked.connect(lambda: self.move_requested.emit(self, -1))
        btn_row.addWidget(self._btn_left)

        self._btn_right = QPushButton("▶")
        self._btn_right.setFixedSize(20, 20)
        self._btn_right.setToolTip("Move this column right")
        self._btn_right.clicked.connect(lambda: self.move_requested.emit(self, +1))
        btn_row.addWidget(self._btn_right)

        self._btn_collapse = QPushButton("▼")
        self._btn_collapse.setFixedSize(20, 20)
        self._btn_collapse.setToolTip("Collapse / expand this column")
        self._btn_collapse.clicked.connect(self._toggle_collapse)
        btn_row.addWidget(self._btn_collapse)
        btn_row.addStretch()
        hdr_root.addLayout(btn_row)

        root.addWidget(hdr)

        # Chart view
        self.view = _ZoomableView(f"Preset {preset_idx + 1}", stretch_fit=True)
        self.view.setMinimumHeight(100)
        self.view.annotation_mode = True
        self.view.row_right_clicked.connect(
            lambda frac: self.annotation_requested.emit(self, frac)
        )
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
        self._cached_interp_z_range: Optional[tuple]  = None   # narrowed to valid rows
        self._cached_bbox:          Optional[list]   = None
        self._cached_roi1_points:   Optional[list]   = None
        self._cached_signal_data:   Dict[int, dict]  = {}   # pre-offset per-preset data
        self._cached_strip_display_w: int = _IMG_W
        self._cached_strip:         Optional[object] = None
        # Per-tip strip data (All Tips mode): key = tip idx
        self._cached_strips:        Dict[int, dict]  = {}  # {idx: {strip, src_bgr, ctr_bgr, bbox, strip_w, roi1_pts}}
        self._strip_panels:         List[object]     = []  # all _StripPanel instances
        # Per-tip interpretation z ranges (All Tips mode); None in normal mode
        self._cached_per_tip_interp_z_ranges: Optional[Dict[int, tuple]] = None
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

        # Ground-truth annotation for the current (image_filename, pipette_index)
        self._annotation: Optional[Dict[str, Any]] = None

        # Background optimizer thread (one at a time)
        self._optimizer_thread: Optional[_RuleOptimizerThread] = None

        # Signal-interpretation results per preset (populated after each run)
        self._cached_interp_results: Dict[int, Any] = {}
        # Combined (cross-preset mean density) result
        self._cached_combined_interp: Optional[Dict[str, Any]] = None
        # Reference to the combined chart panel (preset_idx == -1)
        self._combined_panel: Optional[_ChartPanel] = None
        # All Tips mode: show all 8 tips for one preset instead of many presets
        self._all_tips_mode: bool = False

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
        self._pipette_spin.setFixedWidth(72)
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

        top.addSpacing(4)
        self._all_tips_btn = QPushButton("All Tips")
        self._all_tips_btn.setCheckable(True)
        self._all_tips_btn.setChecked(False)
        self._all_tips_btn.setFixedHeight(22)
        self._all_tips_btn.setToolTip(
            "Show all 8 tips for the selected preset side-by-side.\n"
            "Exactly one preset must be checked.\n"
            "The pipette selector is disabled in this mode."
        )
        self._all_tips_btn.clicked.connect(self._on_all_tips_toggled)
        top.addWidget(self._all_tips_btn)

        top.addSpacing(4)
        self._images_btn = QPushButton("Images")
        self._images_btn.setCheckable(True)
        self._images_btn.setChecked(True)
        self._images_btn.setFixedHeight(22)
        self._images_btn.setToolTip("Show / hide the image strip columns")
        self._images_btn.clicked.connect(self._on_images_toggled)
        top.addWidget(self._images_btn)

        top.addStretch()

        # ── Display options — grouped ────────────────────────────────────
        def _grp(title: str) -> tuple:
            """Return (QGroupBox, inner QHBoxLayout) for a compact toggle group."""
            gb = QGroupBox(title)
            gb.setFlat(True)
            gb.setStyleSheet(
                "QGroupBox { border:1px solid #555; border-radius:3px; "
                "margin-top:6px; padding-top:2px; font-size:9px; color:#aaa; }"
                "QGroupBox::title { subcontrol-origin:margin; left:4px; }"
            )
            lay = QHBoxLayout(gb)
            lay.setContentsMargins(4, 10, 4, 2)
            lay.setSpacing(4)
            return gb, lay

        # ── Overlay group (image-level) ───────────────────────────────────
        _grp_ov, _lay_ov = _grp("Overlay")
        self._roi1_chk = QCheckBox("ROI1")
        self._roi1_chk.setChecked(True)
        self._roi1_chk.setToolTip("Draw the ROI1 polygon on the image strip")
        self._roi1_chk.stateChanged.connect(self._on_display_option_changed)
        _lay_ov.addWidget(self._roi1_chk)
        self._poi_chk = QCheckBox("POI")
        self._poi_chk.setChecked(False)
        self._poi_chk.setToolTip("Show detected points-of-interest on signal graphs")
        self._poi_chk.stateChanged.connect(self._on_display_option_changed)
        _lay_ov.addWidget(self._poi_chk)

        # ── Known (annotation) group ──────────────────────────────────────
        _grp_kn, _lay_kn = _grp("Known")
        self._ann_tb_chk = QCheckBox("Tip↓")
        self._ann_tb_chk.setChecked(True)
        self._ann_tb_chk.setToolTip(
            "Show the manually annotated tip-bottom position\n"
            "(dashed green line; set by right-clicking a chart)"
        )
        self._ann_tb_chk.stateChanged.connect(self._on_display_option_changed)
        _lay_kn.addWidget(self._ann_tb_chk)
        self._ann_mn_chk = QCheckBox("Mn")
        self._ann_mn_chk.setChecked(True)
        self._ann_mn_chk.setToolTip(
            "Show the manually annotated meniscus position\n"
            "(dashed orange line; set by right-clicking a chart)"
        )
        self._ann_mn_chk.stateChanged.connect(self._on_display_option_changed)
        _lay_kn.addWidget(self._ann_mn_chk)

        # ── Calculated (interpretation) group ─────────────────────────────
        _grp_ca, _lay_ca = _grp("Calculated")
        self._calc_band_chk = QCheckBox("Band")
        self._calc_band_chk.setChecked(True)
        self._calc_band_chk.setToolTip(
            "Show the narrow state-colour band on the left edge of each chart\n"
            "(gray = gas, blue = liquid, brown = below tip)"
        )
        self._calc_band_chk.stateChanged.connect(self._on_display_option_changed)
        _lay_ca.addWidget(self._calc_band_chk)
        self._calc_soft_chk = QCheckBox("Soft")
        self._calc_soft_chk.setChecked(False)
        self._calc_soft_chk.setToolTip(
            "When Band is on: blend state colours by score probability instead of\n"
            "using the hard Viterbi decision. Uncertain rows appear as mixed\n"
            "colours; fully confident rows appear as the pure state colour."
        )
        self._calc_soft_chk.stateChanged.connect(self._on_display_option_changed)
        _lay_ca.addWidget(self._calc_soft_chk)
        self._calc_tb_chk = QCheckBox("Tip↓")
        self._calc_tb_chk.setChecked(True)
        self._calc_tb_chk.setToolTip(
            "Show the calculated tip-bottom position\n"
            "(solid green line; requires behavior_rules)"
        )
        self._calc_tb_chk.stateChanged.connect(self._on_display_option_changed)
        _lay_ca.addWidget(self._calc_tb_chk)
        self._calc_mn_chk = QCheckBox("Mn")
        self._calc_mn_chk.setChecked(True)
        self._calc_mn_chk.setToolTip(
            "Show the calculated meniscus position\n"
            "(solid orange line; requires behavior_rules)"
        )
        self._calc_mn_chk.stateChanged.connect(self._on_display_option_changed)
        _lay_ca.addWidget(self._calc_mn_chk)
        self._calc_density_chk = QCheckBox("Density")
        self._calc_density_chk.setChecked(False)
        self._calc_density_chk.setToolTip(
            "Overlay the probability density curves for tip-bottom (green)\n"
            "and meniscus (orange) as semi-transparent filled bars.\n"
            "Bar width at each row is proportional to the KDE density value,\n"
            "normalised so the peak value fills the maximum bar width.\n"
            "Tip↓ / Mn toggles above control which curves are shown."
        )
        self._calc_density_chk.stateChanged.connect(self._on_display_option_changed)
        _lay_ca.addWidget(self._calc_density_chk)
        self._calc_combined_chk = QCheckBox("Comb")
        self._calc_combined_chk.setChecked(True)
        self._calc_combined_chk.setToolTip(
            "Show the cross-preset combined estimate on the image strip.\n"
            "Tip-bottom (cyan) and meniscus (yellow) lines mark the peak of\n"
            "the mean density across all presets that have behavior_rules."
        )
        self._calc_combined_chk.stateChanged.connect(self._on_display_option_changed)
        _lay_ca.addWidget(self._calc_combined_chk)

        # ── Pack the three groups into a single drop-down button ───────────
        # Using QWidgetAction so the menu stays open while checkboxes are
        # toggled — the user can adjust several options in one click.
        _oc_widget = QWidget()
        _oc_lay = QHBoxLayout(_oc_widget)
        _oc_lay.setContentsMargins(6, 4, 6, 4)
        _oc_lay.setSpacing(10)
        _oc_lay.addWidget(_grp_ov)
        _oc_lay.addWidget(_grp_kn)
        _oc_lay.addWidget(_grp_ca)
        _overlay_menu = QMenu(self)
        _overlay_action = QWidgetAction(_overlay_menu)
        _overlay_action.setDefaultWidget(_oc_widget)
        _overlay_menu.addAction(_overlay_action)
        _overlay_btn = QToolButton()
        _overlay_btn.setText("Overlays")
        _overlay_btn.setMenu(_overlay_menu)
        _overlay_btn.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
        _overlay_btn.setToolTip(
            "Show/hide annotation and analysis overlays.\n"
            "The panel stays open so multiple options can be toggled at once."
        )
        top.addWidget(_overlay_btn)

        top.addSpacing(8)
        self._chart_w_spin = QSpinBox()
        self._chart_w_spin.setRange(60, 600)
        self._chart_w_spin.setValue(_CHART_W)
        self._chart_w_spin.setSingleStep(10)
        self._chart_w_spin.setFixedWidth(84)
        self._chart_w_spin.setToolTip("Chart column width in pixels")
        self._chart_w_spin.valueChanged.connect(self._on_display_option_changed)

        self._band_w_spin = QSpinBox()
        self._band_w_spin.setRange(4, 80)
        self._band_w_spin.setValue(8)
        self._band_w_spin.setSingleStep(4)
        self._band_w_spin.setFixedWidth(80)
        self._band_w_spin.setToolTip(
            "Width of the state-colour band on the left edge of each chart.\n"
            "\u25a0 Cyan/teal  = gas in tip (above meniscus)\n"
            "\u25a0 Green      = liquid in tip\n"
            "\u25a0 Red/orange = below tip (shank region)"
        )
        self._band_w_spin.valueChanged.connect(self._on_display_option_changed)

        _ly_widget = QWidget()
        _ly_form = QFormLayout(_ly_widget)
        _ly_form.setContentsMargins(8, 6, 8, 6)
        _ly_form.setSpacing(6)
        _ly_form.setLabelAlignment(Qt.AlignmentFlag.AlignRight)
        _ly_form.addRow("Graph w:", self._chart_w_spin)
        _ly_form.addRow("Band w:",  self._band_w_spin)
        _ly_menu = QMenu(self)
        _ly_action = QWidgetAction(_ly_menu)
        _ly_action.setDefaultWidget(_ly_widget)
        _ly_menu.addAction(_ly_action)
        _ly_btn = QToolButton()
        _ly_btn.setText("Layout")
        _ly_btn.setMenu(_ly_menu)
        _ly_btn.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
        _ly_btn.setToolTip(
            "Graph w: chart column width in pixels\n"
            "Band w:  state-colour band width in pixels"
        )
        top.addWidget(_ly_btn)

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

        self._optimize_btn = QPushButton("⚙ Optimize…")
        self._optimize_btn.setEnabled(False)
        self._optimize_btn.setFixedWidth(90)
        self._optimize_btn.setToolTip(
            "Optimise behavior_rules for a selected preset.\n"
            "Right-click on a chart first to mark tip_bottom / meniscus positions."
        )
        self._optimize_btn.clicked.connect(self._on_optimize_clicked)
        top.addWidget(self._optimize_btn)

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
        self._preset_list.itemChanged.connect(self._on_preset_item_changed)
        left_lay.addWidget(self._preset_list, 1)

        sel_row = QHBoxLayout()
        sel_all_btn  = QPushButton("Select All")
        clr_all_btn  = QPushButton("Clear All")
        sel_all_btn.clicked.connect(self._select_all)
        clr_all_btn.clicked.connect(self._clear_all)
        sel_row.addWidget(sel_all_btn)
        sel_row.addWidget(clr_all_btn)
        left_lay.addLayout(sel_row)

        reload_btn = QPushButton("↺  Reload Presets")
        reload_btn.setToolTip(
            "Re-read ab_presets.json from disk.\n"
            "Use this after editing the file externally."
        )
        reload_btn.clicked.connect(self._reload_presets)
        left_lay.addWidget(reload_btn)

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

    def _populate_preset_list(self, checked_ids: Optional[set] = None):
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
                rules = entry.get("behavior_rules") or []
                rules_badge = f"  [{len(rules)} rule{'s' if len(rules) != 1 else ''}]" if rules else ""
                item = QListWidgetItem(f"  {label}{rules_badge}")
                item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
                # Determine initial check state:
                # • on reload: use checked_ids snapshot
                # • on first load: pre-check presets of the initial mode
                if checked_ids is not None:
                    init = (Qt.CheckState.Checked
                            if entry.get("id") in checked_ids
                            else Qt.CheckState.Unchecked)
                else:
                    init = (Qt.CheckState.Checked
                            if entry.get("mode") == self._mode_name
                            else Qt.CheckState.Unchecked)
                item.setCheckState(init)
                rule_summary = (
                    "\n\nbehavior_rules: " + str(len(rules)) + " defined"
                    if rules else "\n\nbehavior_rules: (none)"
                )
                item.setToolTip(
                    f"Mode: {entry.get('mode', '')}\n"
                    f"{entry.get('description', '')}\n"
                    f"Created: {entry.get('created', '')}"
                    f"{rule_summary}"
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

        # ── Edit behavior rules / Rename (only when a real preset item is clicked) ──
        clicked_entry = None
        if clicked_item is not None:
            clicked_entry = clicked_item.data(Qt.ItemDataRole.UserRole)
        act_edit_rules = None
        act_rename = None
        if clicked_entry is not None:   # not a header
            desc = clicked_entry.get("description", "preset")
            act_rename = menu.addAction(f'Rename…  "{desc[:40]}"')
            act_edit_rules = menu.addAction(
                f'Edit behavior rules…  "{desc[:40]}"'
            )
            menu.addSeparator()

        if clicked_mode:
            act_mode = menu.addAction(f'Select all in "{clicked_mode}"')
        else:
            act_mode = None

        act_all   = menu.addAction("Select all presets")
        menu.addSeparator()
        act_clear = menu.addAction("Clear all")

        chosen = menu.exec(self._preset_list.viewport().mapToGlobal(pos))

        if chosen is None:
            return
        if act_rename is not None and chosen is act_rename:
            self._on_rename_preset_item(clicked_item, clicked_entry)
        elif act_edit_rules is not None and chosen is act_edit_rules:
            self._open_behavior_rules_gui(clicked_entry)
        elif act_mode is not None and chosen is act_mode:
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

    def _open_behavior_rules_gui(self, entry: dict) -> None:
        """Open the GUI behavior-rules editor dialog for *entry*.

        Changes made in the dialog trigger a live re-render (debounced 250 ms).
        "Save & Close" persists to disk; "Discard & Close" reverts in-memory.
        """
        def _on_rules_changed(rules):
            """Called by the dialog when any field changes (debounced)."""
            # Patch the in-memory entry for the result that owns this preset
            preset_id = entry.get("id")
            for res in self._results.values():
                if res.get("entry", {}).get("id") == preset_id:
                    res["entry"]["behavior_rules"] = rules
            # Also patch cached signal data entries that carry behavior_rules
            self._recompute_interpretations()
            self._rerender_pixmaps()

        dlg = BehaviorRulesEditorDialog(entry, _on_rules_changed, parent=self)
        dlg.exec()
        # After close (save or discard) reload the presets list so that any
        # name/rules change is reflected in the sidebar.
        self._reload_presets()

    def _open_behavior_rules_in_editor(self, entry: dict) -> None:
        """
        Open ab_presets.json in Notepad++ (or the system default text editor
        if Notepad++ is not found) so the user can edit behavior_rules.
        The file path is shown in the status label so the user knows which
        file to save.
        """
        import subprocess
        import shutil
        presets_path = ab_presets.get_presets_path()
        preset_id    = entry.get("id", "")
        desc         = entry.get("description", "")

        # Try Notepad++ first (common install locations on Windows)
        npp_candidates = [
            r"C:\Program Files\Notepad++\notepad++.exe",
            r"C:\Program Files (x86)\Notepad++\notepad++.exe",
            shutil.which("notepad++") or "",
        ]
        npp_exe = next((p for p in npp_candidates if p and os.path.isfile(p)), None)

        try:
            if npp_exe:
                subprocess.Popen([npp_exe, presets_path])
            else:
                # Fall back to the system default (works on Windows, macOS, Linux)
                import platform
                if platform.system() == "Windows":
                    os.startfile(presets_path)   # type: ignore[attr-defined]
                elif platform.system() == "Darwin":
                    subprocess.Popen(["open", "-t", presets_path])
                else:
                    subprocess.Popen(["xdg-open", presets_path])
            editor_name = os.path.basename(npp_exe) if npp_exe else "default editor"
            self._status_lbl.setText(
                f'Opened in {editor_name} — id: {preset_id[:16]}…  '
                f'Click "↺ Reload Presets" after saving.'
            )
        except Exception as exc:
            self._status_lbl.setText(f"Could not open editor: {exc}")

    def _on_rename_preset_item(self, item, entry: dict) -> None:
        """Prompt the user for a new name and update the preset on disk."""
        from PySide6.QtWidgets import QInputDialog
        current = entry.get("description", "")
        new_desc, ok = QInputDialog.getText(
            self,
            "Rename preset",
            "New description:",
            text=current,
        )
        if not ok:
            return
        new_desc = new_desc.strip()
        if new_desc == current.strip():
            return
        try:
            ab_presets.rename_preset(entry["id"], new_desc)
        except Exception as exc:
            from PySide6.QtWidgets import QMessageBox
            QMessageBox.warning(self, "Rename failed", str(exc))
            return
        # Update in-memory entry so cached results stay consistent
        entry["description"] = new_desc
        self._reload_presets()

    def _reload_presets(self) -> None:
        """Re-read ab_presets.json and rebuild the preset list, preserving
        the current check state of presets whose id is still present."""
        # Snapshot which preset ids are currently checked
        checked_ids: set = set()
        for i in range(self._preset_list.count()):
            item = self._preset_list.item(i)
            entry = item.data(Qt.ItemDataRole.UserRole)
            if entry is not None and item.checkState() == Qt.CheckState.Checked:
                checked_ids.add(entry.get("id"))

        self._populate_preset_list(checked_ids=checked_ids)
        # Sync behavior_rules in cached _results with the freshly reloaded file
        self._refresh_result_entries()
        # Recompute interpretations and refresh display if a run exists
        if self._cached_signal_data:
            self._recompute_interpretations()
            self._rerender_pixmaps()
        self._status_lbl.setText("Presets reloaded.")

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
        self._load_annotation_for_current()

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
    # All Tips mode
    # ------------------------------------------------------------------

    def _on_all_tips_toggled(self, checked: bool) -> None:
        """Toggle All Tips mode.  Guard: exactly one preset must be checked."""
        if checked:
            entries = self._selected_entries()
            if len(entries) != 1:
                from PySide6.QtWidgets import QMessageBox
                QMessageBox.warning(
                    self,
                    "All Tips — single preset required",
                    "All Tips mode requires exactly one preset selected.\n\n"
                    f"Currently {len(entries)} preset(s) are checked.\n"
                    "Please select exactly one preset, then click All Tips.",
                )
                self._all_tips_btn.setChecked(False)
                return
        self._all_tips_mode = checked
        # Disable the per-tip pipette selector while All Tips is active
        self._pipette_spin.setEnabled(not checked)
        for btn in self._pipette_btns:
            btn.setEnabled(not checked)
        # Clear stale results so the next Run starts fresh
        if checked:
            self._results = {}
            self._clear_panels()

    def _on_preset_item_changed(self, item) -> None:
        """When All Tips is active, block a second preset from being checked."""
        if not self._all_tips_mode:
            return
        if item.data(Qt.ItemDataRole.UserRole) is None:
            return  # header row
        if item.checkState() != Qt.CheckState.Checked:
            return  # unchecking is always allowed
        # Count checked presets; if >1 reject and warn
        checked_rows = [
            i for i in range(self._preset_list.count())
            if (self._preset_list.item(i).data(Qt.ItemDataRole.UserRole) is not None
                and self._preset_list.item(i).checkState() == Qt.CheckState.Checked)
        ]
        if len(checked_rows) > 1:
            self._preset_list.blockSignals(True)
            item.setCheckState(Qt.CheckState.Unchecked)
            self._preset_list.blockSignals(False)
            from PySide6.QtWidgets import QMessageBox
            QMessageBox.warning(
                self,
                "All Tips mode",
                "All Tips mode requires exactly one preset selected.\n\n"
                "Uncheck the current preset before selecting a different one.",
            )

    def _on_images_toggled(self, checked: bool) -> None:
        """Show / hide all image strip panels and update the scroll min-width."""
        for strip in self._strip_panels:
            strip.setVisible(checked)
        # Recalculate min-width so the scrollbar adjusts
        chart_w = self._chart_w_spin.value()
        margins = self._panels_layout.contentsMargins()
        sp      = self._panels_layout.spacing()
        n_charts = len([p for p in self._chart_panels if p.isVisible()])
        if self._all_tips_mode:
            strip_total_w = (
                sum(sd["strip_w"] for sd in self._cached_strips.values())
                if checked else 0
            )
            n_strips = len(self._cached_strips) if checked else 0
        else:
            strip_total_w = self._cached_strip_display_w if checked else 0
            n_strips = 1 if checked and self._cached_strip is not None else 0
        content_w = (
            margins.left() + margins.right()
            + strip_total_w + n_strips * sp
            + n_charts * chart_w + max(0, n_charts - 1) * sp
        )
        self._panels_container.setMinimumWidth(content_w)

    # ------------------------------------------------------------------
    # Run
    # ------------------------------------------------------------------

    def _run_comparison(self):
        if not self._image_paths:
            self._status_lbl.setText("Select an image first.")
            return

        if self._worker and self._worker.isRunning():
            self._worker.requestInterruption()
            self._worker.wait(2000)

        self._results = {}
        self._clear_panels()

        frame_idx = max(0, self._frame_combo.currentIndex())

        if self._all_tips_mode:
            # All Tips: run the same preset for each of the 8 tips.
            entries = self._selected_entries()
            if len(entries) != 1:
                from PySide6.QtWidgets import QMessageBox
                QMessageBox.warning(
                    self,
                    "All Tips — single preset required",
                    "All Tips mode requires exactly one preset selected.\n\n"
                    f"Currently {len(entries)} preset(s) are checked.\n"
                    "Please select exactly one preset, then click Run.",
                )
                return
            run_entries = [entries[0]] * 8
            per_pip     = list(range(8))
            self._running_entries = run_entries
            self._worker = ModeCompareWorker(
                mode_name=entries[0].get("mode", self._mode_name),
                preset_entries=run_entries,
                image_paths=self._image_paths,
                reference_paths=self._reference_paths,
                frame_index=frame_idx,
                pipette_index=0,
                instrument_config_path=self._instrument_config_path,
                per_entry_pipette=per_pip,
                parent=self,
            )
        else:
            entries = self._selected_entries()
            if not entries:
                self._status_lbl.setText("Check at least one preset.")
                return
            pipette_idx = self._pipette_spin.value() - 1
            self._running_entries = entries  # snapshot for callbacks
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

        # Strip(s)
        if self._all_tips_mode:
            # All Tips: re-render each tip's strip independently
            for idx, sd in self._cached_strips.items():
                if sd["src_bgr"] is None:
                    continue
                z_min, z_max = self._cached_z_range
                strip_pm = self._make_strip_pixmap(
                    sd["src_bgr"], z_min, z_max,
                    _DISPLAY_H, sd["strip_w"], sd["bbox"],
                    roi1_points=sd["roi1_pts"] if show_roi1 else None,
                )
                ctr_pm = None
                if sd["ctr_bgr"] is not None:
                    ctr_pm = self._make_strip_pixmap(
                        sd["ctr_bgr"], z_min, z_max,
                        _DISPLAY_H, sd["strip_w"], sd["bbox"],
                        roi1_points=sd["roi1_pts"] if show_roi1 else None,
                    )
                strip_pm = self._draw_interp_on_strip(strip_pm, tip_idx=idx)
                if ctr_pm is not None:
                    ctr_pm = self._draw_interp_on_strip(ctr_pm, tip_idx=idx)
                sd["strip"].set_images(strip_pm, ctr_pm)
        elif self._cached_strip is not None and self._cached_src_bgr is not None:
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
            strip_pm = self._draw_interp_on_strip(strip_pm)
            if ctr_pm is not None:
                ctr_pm = self._draw_interp_on_strip(ctr_pm)
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
            pm = _ndarray_to_pixmap(chart_arr)
            pm = self._draw_interp_on_chart(pm, panel.preset_idx)
            pm = self._draw_annotation_lines(pm)
            panel.view.set_pixmap(pm)
            if w_changed:
                panel.set_expanded_width(chart_w)

        # Combined panel (preset_idx == -1)
        if self._combined_panel is not None and self._calc_combined_chk.isChecked():
            self._combined_panel.setVisible(True)
            comb = self._cached_combined_interp
            if comb is not None:
                pm = self._draw_combined_chart(chart_w, _DISPLAY_H, comb)
            else:
                # No combined data yet — blank chart
                pm = _ndarray_to_pixmap(
                    np.zeros((_DISPLAY_H, chart_w, 3), dtype=np.uint8)
                )
            self._combined_panel.view.set_pixmap(pm)
            if w_changed:
                self._combined_panel.set_expanded_width(chart_w)
        elif self._combined_panel is not None:
            self._combined_panel.setVisible(False)

        if w_changed:
            margins = self._panels_layout.contentsMargins()
            sp = self._panels_layout.spacing()
            # Count only visible panels for min-width
            visible_panels = [p for p in self._chart_panels if p.isVisible()]
            n = len(visible_panels)
            show_images = self._images_btn.isChecked()
            if self._all_tips_mode:
                strip_total_w = (
                    sum(sd["strip_w"] for sd in self._cached_strips.values())
                    if show_images else 0
                )
                n_strips = len(self._cached_strips) if show_images else 0
            else:
                strip_total_w = self._cached_strip_display_w if show_images else 0
                n_strips = 1 if show_images and self._cached_strip is not None else 0
            content_w = (
                margins.left() + margins.right()
                + strip_total_w
                + n_strips * sp
                + n * chart_w
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
        if self._all_tips_mode:
            self._status_lbl.setText("Done — all 8 tips.")
        else:
            self._status_lbl.setText(
                f"Done — {len(self._results)} preset(s) compared."
            )
        try:
            self._render_all()
        except Exception as exc:
            import traceback
            self._status_lbl.setText(f"Render error: {exc}")
            traceback.print_exc()
        # Load any existing annotation for the current image + pipette
        self._load_annotation_for_current()
        self._optimize_btn.setEnabled(bool(self._cached_signal_data))

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
        self._strip_panels.clear()
        self._combined_panel = None
        self._cached_strip = None
        self._cached_strips.clear()
        self._cached_per_tip_interp_z_ranges = None
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

        # ── 1b. Narrow z_range to the rows that are valid in every preset ──
        # Each preset may have ignore_top_rows / ignore_bottom_rows which zero
        # out edge rows via mask_signal_edges.  For interpretation we only want
        # the intersection of valid windows so those artificially-zeroed edge
        # rows (which look like "low_signal") don't pollute the analysis.
        # roi_expansion_px is implicitly handled: a preset with larger expansion
        # has a z_axis that starts earlier, so indexing by ignore_top_rows in
        # absolute z coordinates automatically gives the right calibrated offset.
        interp_z_min: float = global_z_min
        interp_z_max: float = global_z_max
        # In All Tips mode each tip has its own valid window — compute per-tip.
        per_tip_interp_z_ranges: Dict[int, tuple] = {}
        for idx, sig_stage in signal_stages.items():
            entry  = self._results[idx].get("entry", {})
            params = entry.get("params_b", {})
            ignore_top    = int(params.get("ignore_top_rows",    0) or 0)
            ignore_bottom = int(params.get("ignore_bottom_rows", 0) or 0)
            roi_y  = roi_y_offsets.get(idx, 0.0)
            z_full = sig_stage.z_axis_px + roi_y  # full-image coords, same length as signal
            n_rows = len(z_full)
            tip_iz_min = float(z_full[0])
            tip_iz_max = float(z_full[-1])
            if ignore_top > 0 and n_rows > ignore_top:
                tip_iz_min = float(z_full[ignore_top])
            if ignore_bottom > 0 and n_rows > ignore_bottom:
                tip_iz_max = float(z_full[n_rows - 1 - ignore_bottom])
            if tip_iz_min < tip_iz_max:
                per_tip_interp_z_ranges[idx] = (tip_iz_min, tip_iz_max)
            # For the shared (normal-mode) intersection range, aggregate as before
            if not self._all_tips_mode:
                interp_z_min = max(interp_z_min, tip_iz_min)
                interp_z_max = min(interp_z_max, tip_iz_max)
        if interp_z_min >= interp_z_max:
            # Degenerate — fall back to the full range
            interp_z_range: tuple = z_range
        else:
            interp_z_range = (interp_z_min, interp_z_max)

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

        # ── 3. Create strip panel(s) + chart panels ────────────────────────
        show_poi  = self._poi_chk.isChecked()
        show_roi1 = self._roi1_chk.isChecked()
        self._chart_w = self._chart_w_spin.value()
        self._cached_signal_data = {}
        show_images = self._images_btn.isChecked()

        if self._all_tips_mode:
            # ── ALL TIPS: one strip + one chart per tip, interleaved ───────
            for idx in sorted(self._results.keys()):
                res = self._results[idx]
                sig_stage = signal_stages.get(idx)
                if sig_stage is None:
                    continue

                desc = f"Tip {idx + 1}"

                # Resolve per-tip bbox from this result's stages
                tip_bbox = None
                for s in res["stages"]:
                    bb = s.metadata.get("display_bbox")
                    if bb is not None:
                        tip_bbox = bb
                        break
                if tip_bbox is None:
                    tip_bbox = res.get("roi_bbox")

                tip_strip_w = _IMG_W
                if tip_bbox is not None:
                    _bx, _by, _bw, _bh = tip_bbox
                    tip_strip_w = max(60, min(_bw, 400))

                tip_src_bgr = res.get("source")
                tip_ctr_bgr = res.get("contrast")

                tip_roi1_pts = None
                for s in res["stages"]:
                    pts = s.metadata.get("roi1_points")
                    if pts:
                        tip_roi1_pts = pts
                        break

                tip_strip_pm = None
                tip_strip_ctr = None
                if tip_src_bgr is not None:
                    tip_strip_pm = self._make_strip_pixmap(
                        tip_src_bgr, global_z_min, global_z_max,
                        display_h, tip_strip_w, tip_bbox,
                        roi1_points=tip_roi1_pts if show_roi1 else None,
                    )
                if tip_ctr_bgr is not None:
                    tip_strip_ctr = self._make_strip_pixmap(
                        tip_ctr_bgr, global_z_min, global_z_max,
                        display_h, tip_strip_w, tip_bbox,
                        roi1_points=tip_roi1_pts if show_roi1 else None,
                    )

                strip = _StripPanel()
                strip.setFixedWidth(tip_strip_w)
                strip.set_aspect_ratio(tip_strip_w / max(1, display_h))
                strip.setVisible(show_images)
                self._panels_layout.insertWidget(
                    self._panels_layout.count() - 1, strip
                )
                strip.set_images(tip_strip_pm, tip_strip_ctr)
                self._all_views.append(strip.view)
                strip.view.row_hovered.connect(self._on_row_hovered)
                self._strip_panels.append(strip)
                _zmin, _zmax = z_range
                strip.view.coord_formatter = (
                    lambda rf, cf, zm=_zmin, zs=_zmax - _zmin:
                        f"z: {zm + rf * zs:.1f} px"
                )

                self._cached_strips[idx] = {
                    "strip":    strip,
                    "src_bgr":  tip_src_bgr,
                    "ctr_bgr":  tip_ctr_bgr,
                    "bbox":     tip_bbox,
                    "strip_w":  tip_strip_w,
                    "roi1_pts": tip_roi1_pts,
                }

                # Chart
                roi_y = roi_y_offsets.get(idx, 0.0)
                z_axis_full = sig_stage.z_axis_px + roi_y
                poi_z_full = (
                    [p + roi_y for p in sig_stage.poi_z_px]
                    if sig_stage.poi_z_px else None
                )
                chart_arr = _signal_pixmap_arr(
                    sig_stage.signal, z_axis_full, poi_z_full,
                    w=self._chart_w, h=display_h,
                    line_thickness=2, log_scale=False,
                    show_poi=show_poi,
                    extra_signals=sig_stage.extra_signals,
                    z_range_override=z_range,
                )
                panel = _ChartPanel(idx, desc)
                panel.set_expanded_width(self._chart_w)
                panel.move_requested.connect(self._on_move_panel)
                panel.hovered.connect(lambda v, _idx=idx: self._on_chart_hovered(_idx, v))
                self._panels_layout.insertWidget(
                    self._panels_layout.count() - 1, panel
                )
                panel.view.set_pixmap(_ndarray_to_pixmap(chart_arr))
                self._chart_panels.append(panel)
                self._all_views.append(panel.view)
                panel.view.row_hovered.connect(self._on_row_hovered)
                panel.annotation_requested.connect(self._on_chart_annotation_requested)
                _zmin, _zmax = z_range
                panel.view.coord_formatter = (
                    lambda rf, cf, zm=_zmin, zs=_zmax - _zmin:
                        f"z: {zm + rf * zs:.1f} px"
                )

                self._cached_signal_data[idx] = {
                    "signal": sig_stage.signal,
                    "z_axis": z_axis_full,
                    "poi_z":  poi_z_full,
                    "extra_signals": sig_stage.extra_signals,
                }
                self._panel_list_row[idx] = -1

            # Shared caches
            self._cached_src_bgr = src_bgr
            self._cached_ctr_bgr = ctr_bgr
            self._cached_strip   = None   # unused in all-tips mode
            self._cached_strip_display_w = (
                next(iter(self._cached_strips.values()))["strip_w"]
                if self._cached_strips else _IMG_W
            )

        else:
            # ── NORMAL MODE: single shared strip then all chart panels ──────

            strip_pm:   Optional[QPixmap] = None
            strip_ctr_pm: Optional[QPixmap] = None
            if src_bgr is not None:
                strip_pm = self._make_strip_pixmap(
                    src_bgr, global_z_min, global_z_max,
                    display_h, strip_display_w, _bbox_src)
            if ctr_bgr is not None:
                strip_ctr_pm = self._make_strip_pixmap(
                    ctr_bgr, global_z_min, global_z_max,
                    display_h, strip_display_w, _bbox_src)

            strip = _StripPanel()
            strip.setFixedWidth(strip_display_w)
            strip.set_aspect_ratio(strip_display_w / max(1, display_h))
            strip.setVisible(show_images)
            self._panels_layout.insertWidget(
                self._panels_layout.count() - 1, strip
            )
            strip.set_images(strip_pm, strip_ctr_pm)
            self._all_views.append(strip.view)
            strip.view.row_hovered.connect(self._on_row_hovered)
            self._strip_panels.append(strip)
            _zmin, _zmax = z_range
            strip.view.coord_formatter = (
                lambda rf, cf, zm=_zmin, zs=_zmax - _zmin:
                    f"z: {zm + rf * zs:.1f} px"
            )
            self._cached_strip = strip
            self._cached_src_bgr = src_bgr
            self._cached_ctr_bgr = ctr_bgr
            self._cached_strip_display_w = strip_display_w

            for idx in sorted(self._results.keys()):
                res = self._results[idx]
                sig_stage = signal_stages.get(idx)
                if sig_stage is None:
                    continue

                entry = res["entry"]
                desc  = entry.get("description", f"Preset {idx + 1}")

                roi_y = roi_y_offsets.get(idx, 0.0)
                z_axis_full = sig_stage.z_axis_px + roi_y
                poi_z_full = (
                    [p + roi_y for p in sig_stage.poi_z_px]
                    if sig_stage.poi_z_px else None
                )

                chart_arr = _signal_pixmap_arr(
                    sig_stage.signal, z_axis_full, poi_z_full,
                    w=self._chart_w, h=display_h,
                    line_thickness=2, log_scale=False,
                    show_poi=show_poi,
                    extra_signals=sig_stage.extra_signals,
                    z_range_override=z_range,
                )
                panel = _ChartPanel(idx, desc)
                panel.set_expanded_width(self._chart_w)
                panel.move_requested.connect(self._on_move_panel)
                panel.hovered.connect(lambda v, _idx=idx: self._on_chart_hovered(_idx, v))
                self._panels_layout.insertWidget(
                    self._panels_layout.count() - 1, panel
                )
                panel.view.set_pixmap(_ndarray_to_pixmap(chart_arr))
                self._chart_panels.append(panel)
                self._all_views.append(panel.view)
                panel.view.row_hovered.connect(self._on_row_hovered)
                panel.annotation_requested.connect(self._on_chart_annotation_requested)
                _zmin, _zmax = z_range
                panel.view.coord_formatter = (
                    lambda rf, cf, zm=_zmin, zs=_zmax - _zmin:
                        f"z: {zm + rf * zs:.1f} px"
                )

                self._cached_signal_data[idx] = {
                    "signal": sig_stage.signal,
                    "z_axis": z_axis_full,
                    "poi_z":  poi_z_full,
                    "extra_signals": sig_stage.extra_signals,
                }
                self._panel_list_row[idx] = -1
                for _row in range(self._preset_list.count()):
                    _item = self._preset_list.item(_row)
                    if _item.data(Qt.ItemDataRole.UserRole) == entry:
                        self._panel_list_row[idx] = _row
                        break

        # ── 4b. Placeholder combined panel (only in normal preset-compare mode) ──
        if not self._all_tips_mode:
            combined_panel = _ChartPanel(-1, "\u2211 Combined")
            combined_panel.set_expanded_width(self._chart_w)
            combined_panel._btn_left.setVisible(False)
            combined_panel._btn_right.setVisible(False)
            combined_panel.view.annotation_mode = False
            self._panels_layout.insertWidget(
                self._panels_layout.count() - 1,
                combined_panel,
            )
            self._combined_panel = combined_panel
            self._chart_panels.append(combined_panel)
            self._all_views.append(combined_panel.view)
            combined_panel.view.row_hovered.connect(self._on_row_hovered)

        margins = self._panels_layout.contentsMargins()
        sp = self._panels_layout.spacing()
        n_charts = len(self._chart_panels)
        n_strips  = len(self._strip_panels)
        show_images = self._images_btn.isChecked()
        if show_images:
            strip_total_w = (
                sum(sd["strip_w"] for sd in self._cached_strips.values())
                if self._all_tips_mode
                else self._cached_strip_display_w
            )
        else:
            strip_total_w = 0
        content_w = (
            margins.left() + margins.right()
            + strip_total_w
            + (sp * n_strips if show_images else 0)
            + n_charts * self._chart_w
            + max(0, n_charts - 1) * sp
        )
        self._panels_container.setMinimumWidth(content_w)
        self._cached_src_bgr        = src_bgr
        self._cached_ctr_bgr        = ctr_bgr
        self._cached_z_range        = z_range          # full display range
        self._cached_interp_z_range = interp_z_range   # narrowed to valid rows (normal mode)
        self._cached_per_tip_interp_z_ranges = per_tip_interp_z_ranges if self._all_tips_mode else None
        self._cached_bbox           = _bbox_src
        # _cached_strip / _cached_src_bgr etc set in the branch above
        # _cached_signal_data is populated per-preset in step 3 above
        # Collect roi1_points from any stage that has them (normal mode fallback)
        self._cached_roi1_points = None
        for res in self._results.values():
            for s in res["stages"]:
                pts = s.metadata.get("roi1_points")
                if pts:
                    self._cached_roi1_points = pts
                    break
            if self._cached_roi1_points:
                break

        # Compute signal interpretations for presets that have behavior_rules.
        self._recompute_interpretations()

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

    # ------------------------------------------------------------------
    # Annotation helpers
    # ------------------------------------------------------------------

    def _current_image_path(self) -> str:
        """Full path of the currently selected source image (used as sidecar key)."""
        idx = max(0, self._frame_combo.currentIndex())
        if idx < len(self._image_paths):
            return self._image_paths[idx]
        return ""

    def _load_annotation_for_current(self) -> None:
        """Refresh ``self._annotation`` from the sidecar for the current image+pipette."""
        fpath = self._current_image_path()
        pip   = self._pipette_spin.value() - 1
        if fpath:
            self._annotation = ab_annotations.get_annotation(fpath, pip)
        else:
            self._annotation = None

    # ------------------------------------------------------------------
    # Phase C: signal-interpretation overlays
    # ------------------------------------------------------------------

    def _refresh_result_entries(self) -> None:
        """Update behavior_rules in cached _results from the freshly loaded
        presets file so that _recompute_interpretations uses current rules."""
        data = ab_presets.load_presets()
        by_id = {p["id"]: p for p in data.get("presets", [])}
        for res in self._results.values():
            pid = res.get("entry", {}).get("id")
            if pid and pid in by_id:
                res["entry"]["behavior_rules"] = by_id[pid].get("behavior_rules", [])

    def _recompute_interpretations(self) -> None:
        """Run ``interpret_signals`` for every preset in ``_cached_signal_data``
        that has behavior_rules.  Stores results in ``_cached_interp_results``."""
        from pa.pipeline.interpreters.signal_interpreter import interpret_signals

        # In All Tips mode each tip has its own valid z window; in normal mode
        # a single shared interp_z_range (intersection of all presets) is used.
        per_tip_ranges = self._cached_per_tip_interp_z_ranges  # None in normal mode
        shared_z_range = self._cached_interp_z_range or self._cached_z_range

        if per_tip_ranges is None and shared_z_range is None:
            self._cached_interp_results = {}
            return

        # Shared global_z_axis for normal mode only
        if per_tip_ranges is None:
            z_min, z_max = shared_z_range
            shared_global_z_axis = np.arange(int(z_min), int(z_max) + 1, dtype=np.float32)
        else:
            shared_global_z_axis = None

        results: Dict[int, Any] = {}
        for idx, sig_data in self._cached_signal_data.items():
            res   = self._results.get(idx)
            entry = res.get("entry", {}) if res else {}
            rules = list(entry.get("behavior_rules") or [])
            if not rules:
                continue

            # Determine the z axis for this entry
            if per_tip_ranges is not None:
                tip_range = per_tip_ranges.get(idx)
                if tip_range is None:
                    continue
                t_min, t_max = tip_range
                global_z_axis = np.arange(int(t_min), int(t_max) + 1, dtype=np.float32)
            else:
                global_z_axis = shared_global_z_axis

            z_axis  = sig_data["z_axis"]
            signals = [(sig_data["signal"], z_axis)]
            for es in (sig_data.get("extra_signals") or []):
                s = es.get("signal")
                z = es.get("z_axis", z_axis)
                if s is not None and len(s) > 0:
                    signals.append((
                        np.asarray(s, dtype=np.float32),
                        np.asarray(z, dtype=np.float32)
                        if z is not None else z_axis,
                    ))

            try:
                interp = interpret_signals(
                    signals=signals,
                    rules_per_signal=[rules] * len(signals),
                    global_z_axis=global_z_axis,
                )
                results[idx] = interp
            except Exception as _exc:
                import traceback
                traceback.print_exc()   # visible in terminal; overlay just skipped

        self._cached_interp_results = results

        # ── Combined cross-preset density (normal mode only) ───────────────
        # In All Tips mode each tip has its own z axis, so stacking densities
        # across tips is not meaningful; the combined panel is hidden anyway.
        if results and not self._all_tips_mode:
            from pa.pipeline.interpreters.signal_interpreter import _kde_peak
            first = next(iter(results.values()))
            gza   = first.z_axis
            tb_stack = np.stack([r.tip_bottom_density for r in results.values()], axis=0)
            mn_stack = np.stack([r.meniscus_density   for r in results.values()], axis=0)
            combined_tb = tb_stack.mean(axis=0)
            combined_mn = mn_stack.mean(axis=0)
            self._cached_combined_interp = {
                "z_axis":            gza,
                "tip_bottom_density": combined_tb,
                "meniscus_density":   combined_mn,
                "tip_bottom_z":       _kde_peak(combined_tb, gza),
                "meniscus_z":         _kde_peak(combined_mn, gza),
            }
        else:
            self._cached_combined_interp = None

    def _draw_interp_on_strip(
        self, pm: QPixmap, tip_idx: Optional[int] = None
    ) -> QPixmap:
        """Draw solid boundary lines from interpretation results onto *pm*.

        Green  = tip_bottom estimates
        Orange = meniscus estimates
        When *tip_idx* is given (All Tips mode) only that tip's own result is
        drawn; combined lines are also suppressed in that case.
        """
        show_tb   = self._calc_tb_chk.isChecked()
        show_mn   = self._calc_mn_chk.isChecked()
        show_comb = self._calc_combined_chk.isChecked() and tip_idx is None
        if not (show_tb or show_mn or show_comb):
            return pm
        if not self._cached_interp_results and not self._cached_combined_interp:
            return pm
        z_range = self._cached_z_range
        if z_range is None:
            return pm

        z_min, z_max = z_range
        z_span = max(1.0, z_max - z_min)
        h = pm.height()
        w = pm.width()

        # Filter to one tip when tip_idx is specified
        if tip_idx is not None:
            single = self._cached_interp_results.get(tip_idx)
            interp_iter = [single] if single is not None else []
        else:
            interp_iter = list(self._cached_interp_results.values())

        tb_zs = ([r.tip_bottom_z for r in interp_iter
                  if r.tip_bottom_z is not None] if show_tb else [])
        mn_zs = ([r.meniscus_z   for r in interp_iter
                  if r.meniscus_z is not None]  if show_mn else [])

        comb_tb_z = (self._cached_combined_interp or {}).get("tip_bottom_z") if show_comb else None
        comb_mn_z = (self._cached_combined_interp or {}).get("meniscus_z")   if show_comb else None

        if not tb_zs and not mn_zs and comb_tb_z is None and comb_mn_z is None:
            return pm

        result = QPixmap(pm)
        painter = QPainter(result)
        font = painter.font()
        font.setPointSize(7)
        painter.setFont(font)

        # Per-preset thin lines (1 px)
        for z_val in tb_zs:
            y = int((float(z_val) - z_min) / z_span * h)
            y = max(1, min(h - 1, y))
            painter.setPen(QPen(QColor(0, 220, 80), 1))
            painter.drawLine(0, y, w, y)

        for z_val in mn_zs:
            y = int((float(z_val) - z_min) / z_span * h)
            y = max(1, min(h - 1, y))
            painter.setPen(QPen(QColor(255, 160, 0), 1))
            painter.drawLine(0, y, w, y)

        # Combined thick lines (2 px, distinct colours: cyan / yellow)
        if comb_tb_z is not None:
            y = int((float(comb_tb_z) - z_min) / z_span * h)
            y = max(1, min(h - 1, y))
            painter.setPen(QPen(QColor(0, 230, 230), 2))
            painter.drawLine(0, y, w, y)
            painter.setPen(QPen(QColor(0, 230, 230)))
            painter.drawText(2, max(10, y - 1), "tb")

        if comb_mn_z is not None:
            y = int((float(comb_mn_z) - z_min) / z_span * h)
            y = max(1, min(h - 1, y))
            painter.setPen(QPen(QColor(230, 230, 0), 2))
            painter.drawLine(0, y, w, y)
            painter.setPen(QPen(QColor(230, 230, 0)))
            painter.drawText(2, max(10, y - 1), "mn")

        painter.end()
        return result

    def _draw_interp_on_chart(self, pm: QPixmap, preset_idx: int) -> QPixmap:
        """Draw interpretation overlays on a chart pixmap.

        Layers (bottom to top):
          1. State band (8 px, left edge) — hard Viterbi or soft-blended.
          2. Probability density bars — tip_bottom (green) and meniscus
             (orange), extending rightward from the band edge.
          3. Solid boundary lines + text labels.
        """
        show_band    = self._calc_band_chk.isChecked()
        show_soft    = self._calc_soft_chk.isChecked()
        show_tb      = self._calc_tb_chk.isChecked()
        show_mn      = self._calc_mn_chk.isChecked()
        show_density = self._calc_density_chk.isChecked()
        if not (show_band or show_tb or show_mn or show_density):
            return pm
        interp = self._cached_interp_results.get(preset_idx)
        if interp is None:
            return pm
        z_range = self._cached_z_range
        if z_range is None:
            return pm

        z_min, z_max = z_range
        z_span = max(1.0, z_max - z_min)
        h = pm.height()
        w = pm.width()
        BAND_W = self._band_w_spin.value()

        gza = interp.z_axis
        seq = interp.state_sequence
        if (show_band or show_density) and (len(gza) == 0 or len(seq) == 0):
            show_band    = False
            show_density = False
        if not (show_band or show_tb or show_mn or show_density):
            return pm

        # Row mapping: display row index → position in gza
        y_z     = np.linspace(float(z_min), float(z_max), h, dtype=np.float32)
        row_idx = np.searchsorted(gza.astype(np.float32), y_z).clip(0, len(gza) - 1)

        result  = QPixmap(pm)
        painter = QPainter(result)

        # ── Layer 1: state band ───────────────────────────────────────────
        # BGR palette: vivid, clearly distinct colours at full opacity.
        # gas_in_tip    = cyan-teal   (cool — gas/air is "cold")
        # liquid_in_tip = bright green
        # below_tip     = red-orange  (warm — tip bottom / shank region)
        _STATE_BGR_F = np.array([
            [ 40, 200, 200],   # STATE_GAS   — cyan-teal
            [ 20, 200,  20],   # STATE_LIQ   — bright green
            [ 30,  60, 220],   # STATE_BELOW — red-orange (BGR: B=30,G=60,R=220)
        ], dtype=np.float32)

        if show_band:
            if show_soft and len(interp.state_scores) == len(gza):
                # Soft: blend colours by softmax of state_scores.
                # state_scores are log-likelihood accumulators whose absolute
                # magnitudes vary widely; a plain softmax often makes one state
                # dominate everywhere (usually GAS, which looks all-grey).
                # Fix: per-row range-normalise to a fixed spread of 5.0 before
                # softmax so the winner always wins by a meaningful margin while
                # gradual transitions still produce blended colours.
                sc = interp.state_scores[row_idx].astype(np.float64)   # (h, 3)
                row_min   = sc.min(axis=1, keepdims=True)
                row_max   = sc.max(axis=1, keepdims=True)
                row_range = np.maximum(row_max - row_min, 1e-9)
                sc = (sc - row_min) / row_range * 5.0                   # winner=5, rest in [0,5]
                sc -= sc.max(axis=1, keepdims=True)                      # numerical stability
                exp_s  = np.exp(np.clip(sc, -20.0, 0.0))
                soft_p = (exp_s / (exp_s.sum(axis=1, keepdims=True) + 1e-9)).astype(np.float32)
                row_bgr = (soft_p @ _STATE_BGR_F).clip(0, 255).astype(np.uint8)  # (h, 3)
            else:
                # Hard: use Viterbi state sequence
                row_states = seq[row_idx].clip(0, 2)
                row_bgr    = _STATE_BGR_F[row_states].astype(np.uint8)            # (h, 3)

            band_bgr = np.broadcast_to(row_bgr[:, np.newaxis, :], (h, BAND_W, 3)).copy()
            band_rgb = np.ascontiguousarray(cv2.cvtColor(band_bgr, cv2.COLOR_BGR2RGB))
            painter.setOpacity(0.55)
            qimg = QImage(
                band_rgb.data, BAND_W, h,
                int(band_rgb.strides[0]), QImage.Format.Format_RGB888,
            )
            painter.drawImage(0, 0, qimg)

        # ── Layer 2: probability density bars ────────────────────────────
        if show_density:
            # Max bar width: up to 25 % of chart area (min 6 px)
            max_bar_w = max(6, min(60, (w - BAND_W) // 4))
            gza_f = gza.astype(np.float32)
            xs    = np.arange(max_bar_w, dtype=np.int32)

            def _density_bars_rgba(density_1d: np.ndarray,
                                   r: int, g: int, b: int) -> None:
                d    = np.interp(y_z, gza_f, density_1d.astype(np.float32),
                                 left=0.0, right=0.0)
                dmax = float(d.max())
                if dmax < 1e-9:
                    return
                bar_ws = (d / dmax * max_bar_w).astype(np.int32).clip(0, max_bar_w)
                mask   = xs[np.newaxis, :] < bar_ws[:, np.newaxis]   # (h, max_bar_w)
                arr    = np.zeros((h, max_bar_w, 4), dtype=np.uint8)
                arr[mask, 0] = r
                arr[mask, 1] = g
                arr[mask, 2] = b
                arr[mask, 3] = 140   # ~55 % alpha
                arr = np.ascontiguousarray(arr)
                qi  = QImage(arr.data, max_bar_w, h,
                             int(arr.strides[0]), QImage.Format.Format_RGBA8888)
                painter.setOpacity(1.0)
                painter.drawImage(BAND_W, 0, qi)

            if show_tb:
                _density_bars_rgba(interp.tip_bottom_density,  60, 220,  60)  # green
            if show_mn:
                _density_bars_rgba(interp.meniscus_density,   255, 160,   0)  # orange

        # ── Layer 3: boundary lines + labels ─────────────────────────────
        painter.setOpacity(1.0)
        font = painter.font()
        font.setPointSize(7)
        painter.setFont(font)

        if show_tb and interp.tip_bottom_z is not None:
            y = int((float(interp.tip_bottom_z) - z_min) / z_span * h)
            y = max(1, min(h - 1, y))
            painter.setPen(QPen(QColor(0, 220, 80), 1))
            painter.drawLine(BAND_W, y, w, y)
            painter.setPen(QPen(QColor(0, 220, 80)))
            painter.drawText(BAND_W + 2, max(10, y - 1), "tb")

        if show_mn and interp.meniscus_z is not None:
            y = int((float(interp.meniscus_z) - z_min) / z_span * h)
            y = max(1, min(h - 1, y))
            painter.setPen(QPen(QColor(255, 160, 0), 1))
            painter.drawLine(BAND_W, y, w, y)
            painter.setPen(QPen(QColor(255, 160, 0)))
            painter.drawText(BAND_W + 2, max(10, y - 1), "mn")

        painter.end()
        return result

    def _draw_combined_chart(
        self,
        w: int,
        h: int,
        comb: Dict[str, Any],
    ) -> QPixmap:
        """Render a chart for the combined cross-preset result.

        Layout (left → right):
          • 8 px left edge: dark background (reserved, no state band — combined
            state sequence is not computed).
          • Remaining width: tip_bottom density (green) and meniscus density
            (orange) bars drawn as filled horizontal bars; bar extends rightward
            from x=8, width proportional to normalised density.
          • Solid cyan line for combined tip_bottom_z; solid yellow for meniscus_z.
        """
        show_tb = self._calc_tb_chk.isChecked()
        show_mn = self._calc_mn_chk.isChecked()

        z_range = self._cached_z_range
        if z_range is None:
            return _ndarray_to_pixmap(np.zeros((h, w, 3), dtype=np.uint8))

        z_min, z_max = z_range
        z_span = max(1.0, z_max - z_min)
        BAND_W   = self._band_w_spin.value()
        BAR_AREA = max(6, w - BAND_W)

        gza   = comb["z_axis"].astype(np.float32)
        y_z   = np.linspace(float(z_min), float(z_max), h, dtype=np.float32)
        xs    = np.arange(BAR_AREA, dtype=np.int32)

        # Dark background
        canvas = np.zeros((h, w, 3), dtype=np.uint8)
        canvas[:, :, :] = 22   # near-black

        def _fill_bars(density_1d: np.ndarray, r: int, g: int, b: int) -> None:
            d    = np.interp(y_z, gza, density_1d.astype(np.float32),
                             left=0.0, right=0.0)
            dmax = float(d.max())
            if dmax < 1e-9:
                return
            bar_ws = (d / dmax * BAR_AREA).astype(np.int32).clip(0, BAR_AREA)
            for row in range(h):
                bw = bar_ws[row]
                if bw > 0:
                    canvas[row, BAND_W:BAND_W + bw, 0] = \
                        np.maximum(canvas[row, BAND_W:BAND_W + bw, 0], r)
                    canvas[row, BAND_W:BAND_W + bw, 1] = \
                        np.maximum(canvas[row, BAND_W:BAND_W + bw, 1], g)
                    canvas[row, BAND_W:BAND_W + bw, 2] = \
                        np.maximum(canvas[row, BAND_W:BAND_W + bw, 2], b)

        if show_tb:
            _fill_bars(comb["tip_bottom_density"], 30, 160, 30)   # dark green
        if show_mn:
            _fill_bars(comb["meniscus_density"],   160, 100, 0)   # dark orange

        # Convert to RGB pixmap and add boundary lines via QPainter
        rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
        qimg = QImage(rgb.data, w, h, int(rgb.strides[0]),
                      QImage.Format.Format_RGB888)
        pm = QPixmap.fromImage(qimg.copy())

        painter = QPainter(pm)
        font = painter.font()
        font.setPointSize(7)
        font.setBold(True)
        painter.setFont(font)

        tb_z = comb.get("tip_bottom_z")
        mn_z = comb.get("meniscus_z")

        if show_tb and tb_z is not None:
            y = int((float(tb_z) - z_min) / z_span * h)
            y = max(1, min(h - 1, y))
            painter.setPen(QPen(QColor(0, 230, 230), 2))   # cyan
            painter.drawLine(BAND_W, y, w, y)
            painter.setPen(QPen(QColor(0, 230, 230)))
            painter.drawText(BAND_W + 2, max(12, y - 1), "tb")

        if show_mn and mn_z is not None:
            y = int((float(mn_z) - z_min) / z_span * h)
            y = max(1, min(h - 1, y))
            painter.setPen(QPen(QColor(230, 230, 0), 2))   # yellow
            painter.drawLine(BAND_W, y, w, y)
            painter.setPen(QPen(QColor(230, 230, 0)))
            painter.drawText(BAND_W + 2, max(12, y - 1), "mn")

        painter.end()
        return pm

    # ------------------------------------------------------------------
    # Annotation helpers
    # ------------------------------------------------------------------

    def _draw_annotation_lines(self, pm: QPixmap) -> QPixmap:
        """Overlay horizontal dashed lines for any annotated z positions.

        Green  = tip_bottom_z
        Orange = meniscus_z

        The pixmap is returned as a new copy with the lines drawn on top.
        """
        ann = self._annotation
        if ann is None:
            return pm
        tip_z = ann.get("tip_bottom_z") if self._ann_tb_chk.isChecked() else None
        men_z = ann.get("meniscus_z")   if self._ann_mn_chk.isChecked() else None
        if tip_z is None and men_z is None:
            return pm

        z_range = self._cached_z_range
        if z_range is None:
            return pm
        z_min, z_max = z_range
        z_span = max(1.0, z_max - z_min)

        result = QPixmap(pm)
        painter = QPainter(result)
        h = result.height()
        w = result.width()

        def _draw_line(z_val: float, color: QColor, label: str) -> None:
            y = int((z_val - z_min) / z_span * h)
            y = max(1, min(h - 1, y))
            pen = QPen(color, 2)
            pen.setStyle(Qt.PenStyle.DashLine)
            painter.setPen(pen)
            painter.drawLine(0, y, w, y)
            # Small label
            font = painter.font()
            font.setPointSize(7)
            painter.setFont(font)
            painter.setPen(QPen(color))
            painter.drawText(2, max(10, y - 2), label)

        if tip_z is not None:
            _draw_line(float(tip_z), QColor(60, 220, 60),   "tip_bottom")
        if men_z is not None:
            _draw_line(float(men_z), QColor(255, 160, 50),  "meniscus")

        painter.end()
        return result

    def _on_chart_annotation_requested(self, panel: "_ChartPanel", z_frac: float) -> None:
        """Called when the user right-clicks on a chart at z-fraction *z_frac*."""
        if self._cached_z_range is None:
            return

        z_min, z_max = self._cached_z_range
        z_val = z_min + z_frac * (z_max - z_min)
        z_int = int(round(z_val))

        fpath  = self._current_image_path()
        pip    = self._pipette_spin.value() - 1
        preset_idx = panel.preset_idx

        # Resolve preset entry from running results
        entry: Dict[str, Any] = {}
        res = self._results.get(preset_idx)
        if res:
            entry = res.get("entry", {})

        ann = self._annotation or {}
        current_tb = ann.get("tip_bottom_z")
        current_mn = ann.get("meniscus_z")

        menu = QMenu(self)
        menu.setTitle(f"z = {z_int} px")

        act_tb  = menu.addAction(f"Mark tip_bottom here  (z={z_int})")
        act_mn  = menu.addAction(f"Mark meniscus here    (z={z_int})")
        menu.addSeparator()

        act_clr_tb = menu.addAction(
            f"Clear tip_bottom  (was {int(current_tb)} px)" if current_tb is not None
            else "Clear tip_bottom  (not set)"
        )
        act_clr_mn = menu.addAction(
            f"Clear meniscus    (was {int(current_mn)} px)" if current_mn is not None
            else "Clear meniscus    (not set)"
        )
        act_clr_tb.setEnabled(current_tb is not None)
        act_clr_mn.setEnabled(current_mn is not None)

        if entry.get("behavior_rules"):
            has_ann = (current_tb is not None or current_mn is not None
                       or ann.get("tip_bottom_z") is not None
                       or ann.get("meniscus_z") is not None)
            # include the new mark in the check
            menu.addSeparator()
            desc = entry.get("description", f"preset {preset_idx + 1}")[:40]
            act_opt = menu.addAction(f'Optimize rules for "{desc}"…')
            act_opt.setEnabled(True)   # user can start optimizer; annotation created first
        else:
            act_opt = None

        from PySide6.QtGui import QCursor
        chosen = menu.exec(QCursor.pos())

        if chosen is None:
            return

        if chosen is act_tb:
            if fpath:
                self._annotation = ab_annotations.upsert_annotation(
                    fpath, pip, tip_bottom_z=float(z_val)
                )
            self._status_lbl.setText(f"Annotated tip_bottom = {z_int} px  (image: {os.path.basename(fpath)})")
            self._rerender_pixmaps()

        elif chosen is act_mn:
            if fpath:
                self._annotation = ab_annotations.upsert_annotation(
                    fpath, pip, meniscus_z=float(z_val)
                )
            self._status_lbl.setText(f"Annotated meniscus = {z_int} px  (image: {os.path.basename(fpath)})")
            self._rerender_pixmaps()

        elif chosen is act_clr_tb:
            if fpath:
                ab_annotations.clear_annotation_field(fpath, pip, "tip_bottom_z")
                self._load_annotation_for_current()
            self._status_lbl.setText("Cleared tip_bottom annotation.")
            self._rerender_pixmaps()

        elif chosen is act_clr_mn:
            if fpath:
                ab_annotations.clear_annotation_field(fpath, pip, "meniscus_z")
                self._load_annotation_for_current()
            self._status_lbl.setText("Cleared meniscus annotation.")
            self._rerender_pixmaps()

        elif act_opt is not None and chosen is act_opt:
            self._run_optimize_for(preset_idx, entry)

    def _on_optimize_clicked(self) -> None:
        """Optimize button: pick the first checked preset that has rules."""
        # Collect checked presets that have behavior_rules
        candidates: List[tuple] = []   # (preset_idx, entry)
        for idx, res in self._results.items():
            entry = res.get("entry", {})
            if entry.get("behavior_rules"):
                candidates.append((idx, entry))

        if not candidates:
            self._status_lbl.setText(
                "No checked presets have behavior_rules defined. "
                "Add rules first via right-click → Edit behavior rules…"
            )
            return

        ann = self._annotation
        has_ann = ann and (ann.get("tip_bottom_z") is not None
                           or ann.get("meniscus_z") is not None)
        if not has_ann:
            self._status_lbl.setText(
                "No ground-truth annotation. "
                "Right-click on a chart to mark tip_bottom / meniscus first."
            )
            return

        # Use the first candidate (user can right-click for a specific one)
        preset_idx, entry = candidates[0]
        self._run_optimize_for(preset_idx, entry)

    def _run_optimize_for(
        self, preset_idx: int, entry: Dict[str, Any]
    ) -> None:
        """Start the optimizer for the given preset in a background thread."""
        if self._optimizer_thread and self._optimizer_thread.isRunning():
            self._status_lbl.setText("Optimizer already running…")
            return

        rules = list(entry.get("behavior_rules") or [])
        if not rules:
            self._status_lbl.setText("Preset has no behavior_rules to optimize.")
            return

        sig_data = self._cached_signal_data.get(preset_idx)
        if sig_data is None:
            self._status_lbl.setText("Run the comparison first to populate signal data.")
            return

        ann = self._annotation
        if ann is None or (ann.get("tip_bottom_z") is None and ann.get("meniscus_z") is None):
            self._status_lbl.setText(
                "No annotation for the current image. "
                "Right-click on the chart to mark tip_bottom / meniscus."
            )
            return

        # Build signal list: primary + extra signals
        z_axis = sig_data["z_axis"]
        signals = [(sig_data["signal"], z_axis)]
        for es in (sig_data.get("extra_signals") or []):
            s = es.get("signal")
            z = es.get("z_axis") if "z_axis" in es else z_axis
            if s is not None and len(s) > 0:
                signals.append((s, z))

        annotations = [ann]   # single annotation for the current image
        preset_id   = entry.get("id", "")
        desc        = entry.get("description", f"preset {preset_idx + 1}")[:40]

        self._optimize_btn.setEnabled(False)
        self._status_lbl.setText(f'Optimizing "{desc}"…')

        self._optimizer_thread = _RuleOptimizerThread(
            preset_idx=preset_idx,
            rules=rules,
            signals=signals,
            global_z_axis=z_axis,
            annotations=annotations,
            parent=self,
        )
        self._optimizer_thread.progress.connect(self._status_lbl.setText)
        self._optimizer_thread.finished_result.connect(
            lambda idx, res: self._on_optimize_finished(idx, res, entry)
        )
        self._optimizer_thread.error.connect(
            lambda msg: self._on_optimize_error(msg)
        )
        self._optimizer_thread.start()

    def _on_optimize_finished(
        self,
        preset_idx: int,
        result: object,   # OptimizationResult
        entry: Dict[str, Any],
    ) -> None:
        self._optimize_btn.setEnabled(bool(self._cached_signal_data))
        preset_id = entry.get("id", "")
        desc      = entry.get("description", f"preset {preset_idx + 1}")[:40]
        self._status_lbl.setText(
            f'"{desc}" — loss {result.initial_loss:.1f}→{result.final_loss:.1f} px '
            f'in {result.iterations} iterations.  Rules saved.'
        )
        if preset_id:
            ab_presets.update_behavior_rules(preset_id, result.optimized_rules)
        # Update cached entry so a subsequent optimize uses the new rules
        res = self._results.get(preset_idx)
        if res:
            res["entry"]["behavior_rules"] = result.optimized_rules

    def _on_optimize_error(self, msg: str) -> None:
        self._optimize_btn.setEnabled(bool(self._cached_signal_data))
        self._status_lbl.setText(f"Optimize error: {msg}")


# ---------------------------------------------------------------------------
# Background optimizer thread
# ---------------------------------------------------------------------------

class _RuleOptimizerThread(QThread):
    """Runs ``rule_optimizer.optimize_rules`` in a background thread."""

    progress       = Signal(str)            # status updates during optimization
    finished_result = Signal(int, object)   # (preset_idx, OptimizationResult)
    error          = Signal(str)

    def __init__(
        self,
        preset_idx: int,
        rules: list,
        signals: list,
        global_z_axis: object,
        annotations: list,
        parent=None,
    ):
        super().__init__(parent)
        self._preset_idx  = preset_idx
        self._rules       = rules
        self._signals     = signals
        self._global_z    = global_z_axis
        self._annotations = annotations
        self._iter_count  = 0

    def run(self) -> None:
        try:
            from pa.optimization.rule_optimizer import optimize_rules

            def _cb(x):
                self._iter_count += 1
                if self._iter_count % 20 == 0:
                    self.progress.emit(f"Optimizing… iteration {self._iter_count}")

            result = optimize_rules(
                rules=self._rules,
                signals=self._signals,
                global_z_axis=self._global_z,
                annotations=self._annotations,
                method="Nelder-Mead",
                max_iter=400,
                callback=_cb,
            )
            self.finished_result.emit(self._preset_idx, result)
        except Exception as exc:
            self.error.emit(str(exc))
