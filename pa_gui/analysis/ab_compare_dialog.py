"""
A/B Parameter Comparison Dialog.

Opened from PipelineConfigEditor when a mode is selected.  Shows two
side-by-side pipeline stage images — one computed with the baseline (A)
params and one with modified (B) params.

Layout:
    ┌──────────────────────────────────────────────────────────────────┐
    │ Source: [__________________________________________] [Browse…]   │
    │ Frame: [0 ▼]  Pipette: [0 ▲▼]           [status]    [▶ Run]    │
    ├────────────────────┬─────────────────────────────────────────────┤
    │ B Parameters       │ Stage: [────────────────────────]           │
    │                    │ Overlay α: [──────•────] 30%               │
    │ [ParamFormWidget   │ ┌──────────────┐  ┌──────────────┐        │
    │  with A annots]    │ │      A       │  │      B       │        │
    │                    │ │ [ZoomView]   │  │ [ZoomView]   │        │
    │ [Reset B → A]      │ └──────────────┘  └──────────────┘        │
    └────────────────────┴─────────────────────────────────────────────┘

Params in the B form that differ from A are highlighted gold; a small
"A:<val>" annotation appears to the right of each changed widget.
B images auto-update 800 ms after the last param change (debounced).
"""

from __future__ import annotations

import copy
import os
from typing import Any, Dict, List, Optional, TYPE_CHECKING

import cv2
import numpy as np

from PySide6.QtCore import Qt, QRectF, QSettings, QTimer, QPointF, Signal
from PySide6.QtGui import (
    QColor, QImage, QPainter, QPen, QPolygonF, QPixmap, QFont,
)
from PySide6.QtWidgets import (
    QCheckBox, QComboBox, QDialog, QFileDialog, QFrame,
    QGraphicsPixmapItem, QGraphicsScene, QGraphicsView,
    QHBoxLayout, QLabel, QPushButton,
    QSizePolicy, QSlider, QSpinBox, QSplitter,
    QVBoxLayout, QWidget,
)

from pa_gui.params.param_form import ParamFormWidget

if TYPE_CHECKING:
    from pa.pipeline.types import DebugStage

_SETTINGS_ORG  = "rsimpson-bc"
_SETTINGS_APP  = "PA-GUI"
_SETTINGS_KEY  = "ab_compare_dialog"
_CHART_W = 560
_CHART_H = 320
_DEBOUNCE_MS = 800


# ---------------------------------------------------------------------------
# Zoomable graphics view
# ---------------------------------------------------------------------------

class _ZoomableView(QGraphicsView):

    # Emitted while the mouse is over this view.
    # Value is a 0.0–1.0 fraction of scene height; -1.0 on leave.
    row_hovered: Signal = Signal(float)

    def __init__(self, title: str, parent=None):
        super().__init__(parent)
        self._title = title
        self._scene = QGraphicsScene(self)
        self.setScene(self._scene)
        self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
        self.setRenderHint(QPainter.RenderHint.Antialiasing)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.ViewportAnchor.AnchorViewCenter)
        self.setBackgroundBrush(QColor(18, 18, 18))
        self.setMouseTracking(True)
        self._crosshair_item = None
        self._user_zoomed = False
        self._show_placeholder()

    def _show_placeholder(self):
        pm = QPixmap(320, 220)
        pm.fill(QColor(30, 30, 30))
        p = QPainter(pm)
        p.setPen(QPen(QColor(90, 90, 90)))
        font = QFont()
        font.setPointSize(11)
        p.setFont(font)
        p.drawText(pm.rect(), Qt.AlignmentFlag.AlignCenter,
                   f"{self._title}\n\n(select an image and run)")
        p.end()
        self.set_pixmap(pm)

    def set_pixmap(self, pm: QPixmap):
        self._scene.clear()
        self._crosshair_item = None   # cleared with scene
        self._user_zoomed = False     # reset zoom tracking on new image
        self.resetTransform()
        self._scene.addPixmap(pm)
        self._scene.setSceneRect(QRectF(pm.rect()))
        self.fitInView(self._scene.sceneRect(),
                       Qt.AspectRatioMode.KeepAspectRatio)

    def set_crosshair_frac(self, frac: float) -> None:
        """Draw a dashed horizontal crosshair at *frac* (0–1) of scene height.
        Pass frac < 0 to clear."""
        if self._crosshair_item is not None:
            try:
                self._scene.removeItem(self._crosshair_item)
            except RuntimeError:
                pass
            self._crosshair_item = None
        if frac < 0:
            return
        sr = self._scene.sceneRect()
        if not sr.isValid():
            return
        from PySide6.QtWidgets import QGraphicsLineItem
        y = sr.top() + frac * sr.height()
        pen = QPen(QColor(255, 160, 0), 0)   # width 0 = cosmetic (1 px regardless of zoom)
        pen.setStyle(Qt.PenStyle.DashLine)
        item = QGraphicsLineItem(sr.left(), y, sr.right(), y)
        item.setPen(pen)
        self._scene.addItem(item)
        self._crosshair_item = item

    def resizeEvent(self, event):
        super().resizeEvent(event)
        # Only auto-fit while the user hasn't manually zoomed
        if not self._user_zoomed and self._scene.sceneRect().isValid():
            self.fitInView(self._scene.sceneRect(),
                           Qt.AspectRatioMode.KeepAspectRatio)

    def wheelEvent(self, event):
        self._user_zoomed = True
        factor = 1.15 if event.angleDelta().y() > 0 else 1.0 / 1.15
        self.scale(factor, factor)

    def mouseMoveEvent(self, event):
        sr = self._scene.sceneRect()
        if sr.isValid() and sr.height() > 0:
            scene_y = self.mapToScene(event.pos()).y()
            frac = max(0.0, min(1.0, (scene_y - sr.top()) / sr.height()))
            self.set_crosshair_frac(frac)
            self.row_hovered.emit(frac)
        super().mouseMoveEvent(event)

    def leaveEvent(self, event):
        self.set_crosshair_frac(-1.0)
        self.row_hovered.emit(-1.0)
        super().leaveEvent(event)


# ---------------------------------------------------------------------------
# Image-rendering helpers
# ---------------------------------------------------------------------------

def _ndarray_to_pixmap(arr: np.ndarray) -> QPixmap:
    if arr.dtype != np.uint8:
        mn, mx = arr.min(), arr.max()
        arr = ((arr - mn) / (mx - mn) * 255).astype(np.uint8) if mx > mn \
              else np.zeros_like(arr, dtype=np.uint8)
    if arr.ndim == 2:
        arr = cv2.cvtColor(arr, cv2.COLOR_GRAY2RGB)
    elif arr.ndim == 3 and arr.shape[2] == 3:
        arr = cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)
    qimg = QImage(arr.data, arr.shape[1], arr.shape[0],
                  arr.strides[0], QImage.Format.Format_RGB888)
    return QPixmap.fromImage(qimg)


def _signal_pixmap_arr(
    signal: np.ndarray,
    z_axis: Optional[np.ndarray] = None,
    poi_z: Optional[List[float]] = None,
    w: int = _CHART_W,
    h: int = _CHART_H,
    line_thickness: int = 2,
) -> np.ndarray:
    """Return a BGR numpy array (h, w, 3) of the horizontal signal chart."""
    canvas = np.zeros((h, w, 3), dtype=np.uint8)
    if signal is None or len(signal) < 2:
        return canvas
    mn, mx = float(signal.min()), float(signal.max())
    if mx == mn:
        return canvas
    norm = (signal - mn) / (mx - mn)
    n = len(norm)

    if z_axis is not None and len(z_axis) >= 2:
        z_min, z_max = float(z_axis[0]), float(z_axis[-1])
    else:
        z_min, z_max = 0.0, float(n - 1)

    # Signal line: x = intensity (left→right), y = row (top→bottom)
    pts = []
    for i, v in enumerate(norm):
        z = float(z_axis[i]) if z_axis is not None and i < len(z_axis) else float(i)
        x = int(v * (w - 14)) + 7
        y = int((z - z_min) / max(z_max - z_min, 1) * (h - 1))
        pts.append((x, y))
    for i in range(len(pts) - 1):
        cv2.line(canvas, pts[i], pts[i + 1], (100, 220, 100), line_thickness)

    # POI markers — horizontal lines
    if poi_z is not None and z_axis is not None and len(z_axis) >= 2:
        for z in poi_z:
            if z_max > z_min:
                y = int((z - z_min) / (z_max - z_min) * (h - 1))
                cv2.line(canvas, (0, y), (w - 1, y), (0, 90, 255), 2)

    return canvas


def _signal_pixmap(
    signal: np.ndarray,
    z_axis: Optional[np.ndarray] = None,
    poi_z: Optional[List[float]] = None,
    w: int = _CHART_W,
    h: int = _CHART_H,
) -> QPixmap:
    """Render a 1-D signal as a horizontal chart (Y = row/Z, X = intensity)."""
    return _ndarray_to_pixmap(_signal_pixmap_arr(signal, z_axis, poi_z, w, h))


def _stage_to_pixmap(
    stage: "DebugStage",
    source_image: Optional[np.ndarray] = None,
    overlay_alpha: float = 0.0,
    line_thickness: int = 2,
    show_roi1: bool = False,
    show_roi3: bool = False,
) -> QPixmap:
    """Render a DebugStage as QPixmap, optionally blended with the source."""
    if stage.image is not None:
        img = stage.image
        if img.dtype != np.uint8:
            mn, mx = img.min(), img.max()
            img = ((img - mn) / (mx - mn) * 255).astype(np.uint8) if mx > mn \
                  else np.zeros_like(img, dtype=np.uint8)

        if overlay_alpha > 0.0 and source_image is not None:
            src = source_image
            if src.shape[:2] != img.shape[:2]:
                src = cv2.resize(src, (img.shape[1], img.shape[0]))
            if img.ndim == 2:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            else:
                img = img.copy()
            img = cv2.addWeighted(img, 1.0 - overlay_alpha, src, overlay_alpha, 0)

        return _ndarray_to_pixmap(img)

    if stage.signal is not None:
        if source_image is not None:
            src = source_image
            if src.ndim == 2:
                src = cv2.cvtColor(src, cv2.COLOR_GRAY2BGR)
            else:
                src = src.copy()   # don't mutate caller's array when drawing ROI
            src_h, src_w = src.shape[:2]

            # Scale source to a sensible display height (cap at 700 px tall).
            _MAX_H = 700
            scale = 1.0
            if src_h > _MAX_H:
                scale = _MAX_H / src_h
                src = cv2.resize(src, (max(1, int(src_w * scale)), _MAX_H))
                src_h, src_w = src.shape[:2]

            # Draw ROI polygon(s) onto the source portion before compositing.
            bbox = stage.metadata.get("display_bbox")
            if bbox is not None and (show_roi1 or show_roi3):
                px0, py0 = float(bbox[0]), float(bbox[1])

                def _xform(pts):
                    return np.array(
                        [[int(round((p[0] - px0) * scale)),
                          int(round((p[1] - py0) * scale))] for p in pts],
                        dtype=np.int32,
                    )

                if show_roi3:
                    roi3_pts = stage.metadata.get("roi3_points")
                    if roi3_pts:
                        cv2.polylines(src, [_xform(roi3_pts)], True,
                                      (0, 140, 255), 2, cv2.LINE_AA)
                if show_roi1:
                    roi1_pts = stage.metadata.get("roi1_points")
                    if roi1_pts:
                        cv2.polylines(src, [_xform(roi1_pts)], True,
                                      (220, 220, 0), 1, cv2.LINE_AA)

            # Chart width: ~25% of total composite, min 120 px, max 300 px.
            # This keeps the chart narrower than the image so it doesn't dominate.
            chart_w = max(120, min(300, src_w // 4))
            chart_arr = _signal_pixmap_arr(stage.signal, stage.z_axis_px, stage.poi_z_px,
                                           w=chart_w, h=src_h,
                                           line_thickness=line_thickness)
            sep = np.full((src_h, 2, 3), 55, dtype=np.uint8)
            composite = np.concatenate([src, sep, chart_arr], axis=1)
            return _ndarray_to_pixmap(composite)

        # No source image — chart only, at default size
        chart_arr = _signal_pixmap_arr(stage.signal, stage.z_axis_px, stage.poi_z_px,
                                       w=_CHART_W, h=_CHART_H,
                                       line_thickness=line_thickness)
        return _ndarray_to_pixmap(chart_arr)

    pm = QPixmap(320, 220)
    pm.fill(QColor(30, 30, 30))
    return pm


# ---------------------------------------------------------------------------
# Stage metadata → info text
# ---------------------------------------------------------------------------

_LABEL_MAP = {
    # ROIEdgeFit / ROIContrastFit — heatmap stage
    "fit_score":        "fit score (raw)",
    "contrast_score":   "contrast score (raw)",
    "normalised":       "confidence (0–1)  [= matched boundary / total boundary px]",
    "best_dx_px":       "best offset Δx (px)",
    "best_dy_px":       "best offset Δy (px)",
    # fit result stage
    "confidence":       "confidence (0–1)",
    "center_top":       "center-top output",
    "offset_dx":        "fitted Δx (px)",
    "offset_dy":        "fitted Δy (px)",
    "n_candidates":     "candidates",
}

def _format_stage_info(stages: list, idx: int) -> str:
    """Return a compact human-readable summary of a stage's metadata dict."""
    if not stages or idx < 0 or idx >= len(stages):
        return ""
    stage = stages[idx]
    meta = {k: v for k, v in stage.metadata.items()
            if k not in ("display_bbox", "roi1_points", "roi3_points") and v is not None}
    if not meta:
        return ""
    lines = []
    for k, v in meta.items():
        label = _LABEL_MAP.get(k, k.replace("_", " "))
        if isinstance(v, float):
            lines.append(f"{label}: {v:.4f}")
        else:
            lines.append(f"{label}: {v}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Dialog
# ---------------------------------------------------------------------------

class ABCompareDialog(QDialog):
    """
    Side-by-side A/B pipeline parameter comparison window.

    Parameters
    ----------
    mode_name : str
        Pipeline mode identifier (e.g. "InwardEdgeScan").
    schema : dict
        Resolved JSON Schema for the mode (from ``load_mode_schema``).
    params_a : dict
        Baseline (A) parameter values.  Passed read-only; B starts as a copy.
    instrument_config_path : str, optional
        Path to instrument_config.json.  When provided the worker resolves the
        ROI polygon/bounding box for the selected pipette so the analysis is
        cropped correctly.
    tip_type : str, optional
        Tip type string used for ROI lookup (e.g. "standard_1000ul").
    parent : QWidget, optional
    """

    # Emitted when the user clicks "Apply B → Mode".
    # Carries a copy of the current B params dict.
    params_applied = Signal(dict)

    def __init__(
        self,
        mode_name: str,
        schema: Dict[str, Any],
        params_a: Dict[str, Any],
        instrument_config_path: str = "",
        tip_type: str = "",
        parent=None,
    ) -> None:
        super().__init__(parent)
        self._mode_name = mode_name
        self._schema    = schema
        self._params_a  = copy.deepcopy(params_a)
        self._params_b  = copy.deepcopy(params_a)   # B starts equal to A
        self._instrument_config_path = instrument_config_path
        self._tip_type  = tip_type

        self._image_paths: List[str] = []
        self._reference_paths: List[str] = []
        self._stages_a: List["DebugStage"] = []
        self._stages_b: List["DebugStage"] = []
        self._source_image: Optional[np.ndarray] = None
        self._contrast_image_a: Optional[np.ndarray] = None
        self._contrast_image_b: Optional[np.ndarray] = None
        self._roi_bbox: Optional[tuple] = None
        self._worker = None
        self._line_thickness: int = 2

        self._debounce = QTimer(self)
        self._debounce.setSingleShot(True)
        self._debounce.setInterval(_DEBOUNCE_MS)
        self._debounce.timeout.connect(self._run_comparison)

        self.setWindowTitle(f"A/B Comparison — {mode_name}")
        self.setWindowFlag(Qt.WindowType.Window)

        self._build_ui()
        self._restore_state()

    # ── UI construction ──────────────────────────────────────────────────────

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(6, 6, 6, 6)
        root.setSpacing(4)

        # ── Top bar: image source ─────────────────────────────────────────
        src_row = QHBoxLayout()
        src_row.addWidget(QLabel("Source:"))
        self._src_edit = QLabel("<i>no image selected</i>")
        self._src_edit.setStyleSheet(
            "background:#1a1a1a; padding:2px 4px; border:1px solid #444;"
        )
        self._src_edit.setSizePolicy(QSizePolicy.Policy.Expanding,
                                     QSizePolicy.Policy.Fixed)
        self._src_edit.setFixedHeight(24)
        src_row.addWidget(self._src_edit, 1)
        btn_browse_file = QPushButton("File…")
        btn_browse_file.setFixedWidth(52)
        btn_browse_file.setToolTip("Select a single image file.")
        btn_browse_file.clicked.connect(self._on_browse_file)
        src_row.addWidget(btn_browse_file)
        btn_browse_folder = QPushButton("Folder…")
        btn_browse_folder.setFixedWidth(62)
        btn_browse_folder.setToolTip(
            "Select a folder of images.\n"
            "For A:B contrast accumulation, place both frames in the same folder."
        )
        btn_browse_folder.clicked.connect(self._on_browse_folder)
        src_row.addWidget(btn_browse_folder)
        root.addLayout(src_row)

        # ── Reference source row (for use_contrast mode) ──────────────────
        ref_row = QHBoxLayout()
        ref_row.addWidget(QLabel("Reference:"))
        self._ref_edit = QLabel("<i>no reference (optional — needed when use_contrast is enabled)</i>")
        self._ref_edit.setStyleSheet(
            "background:#1a1a1a; padding:2px 4px; border:1px solid #444; color:#888;"
        )
        self._ref_edit.setSizePolicy(QSizePolicy.Policy.Expanding,
                                     QSizePolicy.Policy.Fixed)
        self._ref_edit.setFixedHeight(24)
        ref_row.addWidget(self._ref_edit, 1)
        btn_ref_file = QPushButton("File…")
        btn_ref_file.setFixedWidth(52)
        btn_ref_file.setToolTip(
            "Select a single reference image (e.g. tip without liquid).\n"
            "Used when use_contrast is enabled in the mode params."
        )
        btn_ref_file.clicked.connect(self._on_browse_ref_file)
        ref_row.addWidget(btn_ref_file)
        btn_ref_folder = QPushButton("Folder…")
        btn_ref_folder.setFixedWidth(62)
        btn_ref_folder.setToolTip(
            "Select a folder of reference images.\n"
            "Matched to sample frames by sort order (frame 0 ↔ frame 0, etc.)."
        )
        btn_ref_folder.clicked.connect(self._on_browse_ref_folder)
        ref_row.addWidget(btn_ref_folder)
        btn_ref_clear = QPushButton("Clear")
        btn_ref_clear.setFixedWidth(48)
        btn_ref_clear.setToolTip("Clear the reference source.")
        btn_ref_clear.clicked.connect(self._on_clear_reference)
        ref_row.addWidget(btn_ref_clear)
        root.addLayout(ref_row)

        # ── Frame / pipette / tip row ──────────────────────────────────────
        src_row = QHBoxLayout()

        src_row.addSpacing(12)
        src_row.addWidget(QLabel("Frame:"))
        self._frame_combo = QComboBox()
        self._frame_combo.setFixedWidth(180)
        self._frame_combo.setToolTip(
            "Reference frame shown in the overlay.\n"
            "When A:B contrast is enabled all frames are used for accumulation —\n"
            "this only controls the overlay source image."
        )
        self._frame_combo.currentIndexChanged.connect(self._on_frame_changed)
        src_row.addWidget(self._frame_combo)

        src_row.addSpacing(8)
        src_row.addWidget(QLabel("Pipette idx:"))
        self._pipette_spin = QSpinBox()
        self._pipette_spin.setRange(1, 32)
        self._pipette_spin.setValue(1)
        self._pipette_spin.setFixedWidth(44)
        self._pipette_spin.setToolTip(
            "1-based pipette number (1 = first pipette).  Used for ROI lookup "
            "from the instrument config and for pipeline cache keying."
        )
        self._pipette_spin.valueChanged.connect(self._on_pipette_changed)
        src_row.addWidget(self._pipette_spin)

        src_row.addSpacing(8)
        src_row.addWidget(QLabel("Tip type:"))
        from PySide6.QtWidgets import QLineEdit
        self._tip_edit = QLineEdit(self._tip_type or "")
        self._tip_edit.setFixedWidth(120)
        self._tip_edit.setPlaceholderText("(optional)")
        self._tip_edit.setToolTip(
            "Tip type for ROI lookup (e.g. \"standard_1000ul\").\n"
            "Leave blank to use the first/default ROI from the instrument config."
        )
        self._tip_edit.textChanged.connect(self._on_pipette_changed)
        src_row.addWidget(self._tip_edit)

        src_row.addStretch()
        self._status_lbl = QLabel("")
        self._status_lbl.setStyleSheet("color: #888; font-style: italic;")
        src_row.addWidget(self._status_lbl)

        self._run_btn = QPushButton("▶  Run")
        self._run_btn.setEnabled(False)
        self._run_btn.setToolTip("Select an image first.")
        self._run_btn.setFixedWidth(80)
        self._run_btn.setStyleSheet(
            "QPushButton { background:#2a5f9e; color:white; font-weight:bold; }"
            "QPushButton:disabled { background:#444; color:#888; }"
        )
        self._run_btn.clicked.connect(self._run_comparison)
        src_row.addWidget(self._run_btn)

        root.addLayout(src_row)

        # ── Horizontal divider ────────────────────────────────────────────
        sep = QFrame()
        sep.setFrameShape(QFrame.Shape.HLine)
        sep.setStyleSheet("color:#444;")
        root.addWidget(sep)

        # ── Main splitter: B params (left) | image area (right) ──────────
        splitter = QSplitter(Qt.Orientation.Horizontal)

        # Left: B params
        left = QWidget()
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(0, 0, 4, 0)
        left_layout.setSpacing(4)

        left_hdr = QLabel(f"<b>B Parameters</b>  <span style='color:#888'>(A = baseline)</span>")
        left_layout.addWidget(left_hdr)

        self._b_form = ParamFormWidget(
            self._schema,
            self._params_b,
            show_weight=False,
            reference_params=self._params_a,
        )
        self._b_form.params_changed.connect(self._on_b_params_changed)
        left_layout.addWidget(self._b_form, 1)

        reset_btn = QPushButton("Reset B → A")
        reset_btn.setToolTip("Revert all B parameters to the baseline A values.")
        reset_btn.clicked.connect(self._on_reset_b)

        apply_btn = QPushButton("Apply B \u2192 Mode")
        apply_btn.setToolTip(
            "Copy all current B parameters back to the mode in the pipeline editor."
        )
        apply_btn.setStyleSheet(
            "QPushButton { background:#2a5f9e; color:white; font-weight:bold; }"
        )
        apply_btn.clicked.connect(self._on_apply_to_mode)

        btn_row = QHBoxLayout()
        btn_row.addWidget(reset_btn)
        btn_row.addWidget(apply_btn)
        left_layout.addLayout(btn_row)

        splitter.addWidget(left)

        # Right: controls + two views
        right = QWidget()
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(4, 0, 0, 0)
        right_layout.setSpacing(4)

        # Controls row: stage selector + overlay
        ctrl_row = QHBoxLayout()
        ctrl_row.addWidget(QLabel("Stage:"))
        self._stage_combo = QComboBox()
        self._stage_combo.setMinimumWidth(240)
        self._stage_combo.setEnabled(False)
        self._stage_combo.currentIndexChanged.connect(self._on_stage_changed)
        ctrl_row.addWidget(self._stage_combo, 1)

        ctrl_row.addSpacing(16)
        ctrl_row.addWidget(QLabel("Overlay original:"))
        self._overlay_slider = QSlider(Qt.Orientation.Horizontal)
        self._overlay_slider.setRange(0, 100)
        self._overlay_slider.setValue(0)
        self._overlay_slider.setFixedWidth(120)
        self._overlay_slider.setToolTip(
            "Blend the original camera image over the pipeline stage output.\n"
            "0% = pure stage, 100% = pure original."
        )
        self._overlay_slider.valueChanged.connect(self._on_overlay_changed)
        ctrl_row.addWidget(self._overlay_slider)
        self._overlay_lbl = QLabel("0%")
        self._overlay_lbl.setFixedWidth(32)
        ctrl_row.addWidget(self._overlay_lbl)

        ctrl_row.addSpacing(16)
        self._chk_roi1 = QCheckBox("ROI1")
        self._chk_roi1.setChecked(True)
        self._chk_roi1.setToolTip(
            "ROI1 — the original calibrated ROI polygon as drawn in the ROI Calibration tab.\n"
            "This is the template used for matching and the shape of the output result.\n"
            "Shown in cyan."
        )
        self._chk_roi1.stateChanged.connect(self._render_current_stage)
        ctrl_row.addWidget(self._chk_roi1)
        self._chk_roi2 = QCheckBox("ROI3")
        self._chk_roi2.setChecked(True)
        self._chk_roi2.setToolTip(
            "ROI3 — ROI1 expanded outward by roi_expansion_px perpendicular to each edge.\n"
            "This is the search boundary: the template (ROI1) is allowed to slide anywhere\n"
            "within the rectangular region that bounds ROI3 (ROI2).\n"
            "Shown in orange."
        )
        self._chk_roi2.stateChanged.connect(self._render_current_stage)
        ctrl_row.addWidget(self._chk_roi2)

        ctrl_row.addSpacing(16)
        ctrl_row.addWidget(QLabel("Line:"))
        self._line_spin = QSpinBox()
        self._line_spin.setRange(1, 8)
        self._line_spin.setValue(2)
        self._line_spin.setFixedWidth(44)
        self._line_spin.setToolTip("Signal chart line thickness (px)")
        self._line_spin.valueChanged.connect(self._on_line_thickness_changed)
        ctrl_row.addWidget(self._line_spin)

        right_layout.addLayout(ctrl_row)

        # Two image views side by side
        views_splitter = QSplitter(Qt.Orientation.Horizontal)

        a_wrap = QWidget()
        a_lay = QVBoxLayout(a_wrap)
        a_lay.setContentsMargins(0, 0, 0, 0)
        a_lay.setSpacing(2)
        a_hdr = QLabel("<b>A</b>  (baseline)")
        a_hdr.setAlignment(Qt.AlignmentFlag.AlignCenter)
        a_hdr.setStyleSheet("background:#1a2a1a; padding:2px;")
        a_lay.addWidget(a_hdr)
        self._view_a = _ZoomableView("A")
        a_lay.addWidget(self._view_a, 1)
        self._score_lbl_a = QLabel("")
        self._score_lbl_a.setWordWrap(True)
        self._score_lbl_a.setStyleSheet(
            "font-size:10px; color:#aaa; background:#111; padding:3px 5px;"
        )
        a_lay.addWidget(self._score_lbl_a)
        views_splitter.addWidget(a_wrap)

        b_wrap = QWidget()
        b_lay = QVBoxLayout(b_wrap)
        b_lay.setContentsMargins(0, 0, 0, 0)
        b_lay.setSpacing(2)
        b_hdr = QLabel("<b>B</b>  (modified)")
        b_hdr.setAlignment(Qt.AlignmentFlag.AlignCenter)
        b_hdr.setStyleSheet("background:#2a1a1a; padding:2px;")
        b_lay.addWidget(b_hdr)
        self._view_b = _ZoomableView("B")
        b_lay.addWidget(self._view_b, 1)
        self._score_lbl_b = QLabel("")
        self._score_lbl_b.setWordWrap(True)
        self._score_lbl_b.setStyleSheet(
            "font-size:10px; color:#aaa; background:#111; padding:3px 5px;"
        )
        b_lay.addWidget(self._score_lbl_b)
        views_splitter.addWidget(b_wrap)

        right_layout.addWidget(views_splitter, 1)
        splitter.addWidget(right)

        # Link crosshairs: hovering over one view moves the cursor in the other
        self._view_a.row_hovered.connect(self._view_b.set_crosshair_frac)
        self._view_b.row_hovered.connect(self._view_a.set_crosshair_frac)

        splitter.setStretchFactor(0, 2)
        splitter.setStretchFactor(1, 5)
        root.addWidget(splitter, 1)

        # Keep references for state save/restore
        self._main_splitter  = splitter
        self._views_splitter = views_splitter

        # ── ROI / status row below views ───────────────────────────────────
        self._roi_lbl = QLabel("ROI: not resolved")
        self._roi_lbl.setStyleSheet("color: #888; font-style: italic; font-size: 10px;")
        root.addWidget(self._roi_lbl)

    # ── Slots ──────────────────────────────────────────────────────────────

    def _on_browse_file(self) -> None:
        s = QSettings(_SETTINGS_ORG, _SETTINGS_APP)
        start_dir = s.value(f"{_SETTINGS_KEY}/last_browse_dir", "")
        choice, _ = QFileDialog.getOpenFileName(
            self, "Select source image",
            start_dir, "Images (*.jpg *.jpeg *.png *.bmp);;All files (*)",
        )
        if choice:
            s.setValue(f"{_SETTINGS_KEY}/last_browse_dir", os.path.dirname(choice))
            s.setValue(f"{_SETTINGS_KEY}/last_file", choice)
            s.remove(f"{_SETTINGS_KEY}/last_folder")
            self._load_file(choice)

    def _on_browse_folder(self) -> None:
        s = QSettings(_SETTINGS_ORG, _SETTINGS_APP)
        start_dir = s.value(f"{_SETTINGS_KEY}/last_browse_dir", "")
        folder = QFileDialog.getExistingDirectory(
            self, "Select image folder", start_dir
        )
        if folder:
            s.setValue(f"{_SETTINGS_KEY}/last_browse_dir", folder)
            s.setValue(f"{_SETTINGS_KEY}/last_folder", folder)
            s.remove(f"{_SETTINGS_KEY}/last_file")
            self._load_folder(folder)

    def _load_file(self, path: str) -> None:
        self._image_paths = [path]
        self._src_edit.setText(os.path.basename(path))
        self._src_edit.setToolTip(path)
        self._populate_frame_combo()
        self._run_btn.setEnabled(True)
        self._run_btn.setToolTip("")
        self._schedule_run()

    def _load_folder(self, folder: str) -> None:
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
        self._populate_frame_combo()
        self._run_btn.setEnabled(True)
        self._run_btn.setToolTip("")
        self._schedule_run()

    # ── Reference source browse / load ────────────────────────────────────

    def _on_browse_ref_file(self) -> None:
        s = QSettings(_SETTINGS_ORG, _SETTINGS_APP)
        start_dir = s.value(f"{_SETTINGS_KEY}/last_ref_dir", "")
        choice, _ = QFileDialog.getOpenFileName(
            self, "Select reference image",
            start_dir, "Images (*.jpg *.jpeg *.png *.bmp);;All files (*)",
        )
        if choice:
            s.setValue(f"{_SETTINGS_KEY}/last_ref_dir", os.path.dirname(choice))
            s.setValue(f"{_SETTINGS_KEY}/last_ref_file", choice)
            s.remove(f"{_SETTINGS_KEY}/last_ref_folder")
            self._load_ref_file(choice)

    def _on_browse_ref_folder(self) -> None:
        s = QSettings(_SETTINGS_ORG, _SETTINGS_APP)
        start_dir = s.value(f"{_SETTINGS_KEY}/last_ref_dir", "")
        folder = QFileDialog.getExistingDirectory(
            self, "Select reference image folder", start_dir
        )
        if folder:
            s.setValue(f"{_SETTINGS_KEY}/last_ref_dir", folder)
            s.setValue(f"{_SETTINGS_KEY}/last_ref_folder", folder)
            s.remove(f"{_SETTINGS_KEY}/last_ref_file")
            self._load_ref_folder(folder)

    def _on_clear_reference(self) -> None:
        self._reference_paths = []
        self._ref_edit.setText(
            "<i>no reference (optional — needed when use_contrast is enabled)</i>"
        )
        self._ref_edit.setStyleSheet(
            "background:#1a1a1a; padding:2px 4px; border:1px solid #444; color:#888;"
        )
        self._ref_edit.setToolTip("")
        s = QSettings(_SETTINGS_ORG, _SETTINGS_APP)
        s.remove(f"{_SETTINGS_KEY}/last_ref_file")
        s.remove(f"{_SETTINGS_KEY}/last_ref_folder")
        self._schedule_run()

    def _load_ref_file(self, path: str) -> None:
        self._reference_paths = [path]
        self._ref_edit.setText(os.path.basename(path))
        self._ref_edit.setStyleSheet(
            "background:#1a1a1a; padding:2px 4px; border:1px solid #2a6;"
        )
        self._ref_edit.setToolTip(path)
        self._schedule_run()

    def _load_ref_folder(self, folder: str) -> None:
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
            "background:#1a1a1a; padding:2px 4px; border:1px solid #2a6;"
        )
        self._ref_edit.setToolTip(folder)
        self._schedule_run()

    def _populate_frame_combo(self) -> None:
        self._frame_combo.blockSignals(True)
        self._frame_combo.clear()
        for i, p in enumerate(self._image_paths):
            self._frame_combo.addItem(f"Frame {i}  ({os.path.basename(p)})")
        self._frame_combo.blockSignals(False)

    def _on_frame_changed(self, idx: int) -> None:
        if self._image_paths:
            self._schedule_run()

    def _on_pipette_changed(self) -> None:
        """Pipette index or tip type changed — schedule a re-run."""
        if self._image_paths:
            self._schedule_run()

    def _on_b_params_changed(self) -> None:
        self._params_b = self._b_form.get_params()
        self._schedule_run()

    def _on_reset_b(self) -> None:
        self._params_b = copy.deepcopy(self._params_a)
        self._b_form.set_params(self._params_b)
        # set_params triggers params_changed → _on_b_params_changed → schedule run

    def _on_apply_to_mode(self) -> None:
        """Emit the current B params so the parent editor can apply them."""
        self.params_applied.emit(copy.deepcopy(self._params_b))

    def _on_stage_changed(self, idx: int) -> None:
        self._render_current_stage()

    def _on_overlay_changed(self, value: int) -> None:
        self._overlay_lbl.setText(f"{value}%")
        self._render_current_stage()

    def _on_line_thickness_changed(self, value: int) -> None:
        self._line_thickness = value
        self._render_current_stage()

    def _schedule_run(self) -> None:
        """Restart the debounce timer; runs after _DEBOUNCE_MS of inactivity."""
        if self._image_paths:
            self._debounce.start()

    # ── Pipeline execution ─────────────────────────────────────────────────

    def _run_comparison(self) -> None:
        if not self._image_paths:
            return
        if self._worker and self._worker.isRunning():
            self._worker.requestInterruption()

        frame_idx   = max(0, self._frame_combo.currentIndex())
        pipette_idx = self._pipette_spin.value() - 1  # spin is 1-based; worker is 0-based

        from pa_gui.analysis.ab_compare_worker import ABCompareWorker
        self._worker = ABCompareWorker(
            mode_name=self._mode_name,
            params_a=self._params_a,
            params_b=self._params_b,
            image_paths=self._image_paths,
            reference_paths=self._reference_paths,
            frame_index=frame_idx,
            pipette_index=pipette_idx,
            instrument_config_path=self._instrument_config_path,
            tip_type=self._tip_edit.text().strip(),
            parent=self,
        )
        self._worker.progress.connect(self._on_progress)
        self._worker.result_ready.connect(self._on_result)
        self._worker.error.connect(self._on_worker_error)
        self._run_btn.setEnabled(False)
        self._status_lbl.setText("Running…")
        self._worker.start()

    def _on_progress(self, msg: str) -> None:
        self._status_lbl.setText(msg)
        # Surface ROI resolution messages prominently
        if msg.startswith("ROI"):
            self._roi_lbl.setText(msg)
            self._roi_lbl.setStyleSheet(
                "color: #f5a623; font-style: italic; font-size: 10px;"
                if "warning" in msg or "not" in msg
                else "color: #74c97a; font-size: 10px;"
            )

    def _on_result(
        self,
        stages_a: List["DebugStage"],
        stages_b: List["DebugStage"],
        source_image: np.ndarray,
        contrast_image_a: Optional[np.ndarray],
        contrast_image_b: Optional[np.ndarray],
        roi_bbox: Optional[tuple],
    ) -> None:
        self._stages_a        = stages_a
        self._stages_b        = stages_b
        self._source_image    = source_image
        self._contrast_image_a = contrast_image_a
        self._contrast_image_b = contrast_image_b
        self._roi_bbox        = roi_bbox
        self._status_lbl.setText("Done")
        self._run_btn.setEnabled(True)
        self._populate_stage_combo()

    def _on_worker_error(self, msg: str) -> None:
        self._status_lbl.setText("Error — see console")
        self._run_btn.setEnabled(True)
        import traceback
        print(f"[ABCompare] Worker error:\n{msg}")

    # ── Stage display ──────────────────────────────────────────────────────

    def _populate_stage_combo(self) -> None:
        prev = self._stage_combo.currentIndex()
        self._stage_combo.blockSignals(True)
        self._stage_combo.clear()
        # Use stage names from A; fall back to B if A is empty
        names = (
            [s.name for s in self._stages_a]
            if self._stages_a else
            [s.name for s in self._stages_b]
        )
        for name in names:
            self._stage_combo.addItem(name)
        self._stage_combo.setEnabled(bool(names))
        # Try to restore previous selection
        new_idx = min(prev, len(names) - 1) if names else -1
        if new_idx >= 0:
            self._stage_combo.setCurrentIndex(new_idx)
        self._stage_combo.blockSignals(False)
        self._render_current_stage()

    def _render_current_stage(self) -> None:
        idx = self._stage_combo.currentIndex()
        alpha = self._overlay_slider.value() / 100.0

        def _crop_src_for_stage(stages, src_override=None):
            """Crop the given source image using that stage's display_bbox."""
            src = src_override if src_override is not None else self._source_image
            if src is None:
                return None
            bbox = None
            if stages and 0 <= idx < len(stages):
                bbox = stages[idx].metadata.get("display_bbox")
            if bbox is None:
                bbox = self._roi_bbox
            if bbox is None:
                return src
            x, y, w, h = bbox
            x1 = max(0, int(x))
            y1 = max(0, int(y))
            x2 = min(src.shape[1], int(x + w))
            y2 = min(src.shape[0], int(y + h))
            return src[y1:y2, x1:x2] if x2 > x1 and y2 > y1 else src

        def _get_pixmap(stages, label):
            if not stages or idx < 0 or idx >= len(stages):
                pm = QPixmap(320, 220)
                pm.fill(QColor(30, 30, 30))
                return pm
            stage = stages[idx]
            if stage.signal is not None:
                # For signal stages: show the contrast-adjusted image in the
                # left strip if use_contrast produced one; otherwise raw source.
                contrast_img = (self._contrast_image_a if label == "A"
                                else self._contrast_image_b)
                src = _crop_src_for_stage(stages, contrast_img)
                # Signal stages handle ROI overlays inside _stage_to_pixmap so
                # the polygon is restricted to the source-image portion of the
                # composite.  _with_roi_overlays is bypassed for these.
                return _stage_to_pixmap(
                    stage, src, alpha,
                    line_thickness=self._line_thickness,
                    show_roi1=self._chk_roi1.isChecked(),
                    show_roi3=self._chk_roi2.isChecked(),
                )
            else:
                # For image stages: overlay always uses the raw original.
                src = _crop_src_for_stage(stages)
            return _stage_to_pixmap(stage, src, alpha,
                                    line_thickness=self._line_thickness)

        def _with_roi_overlays(stages, pm: QPixmap) -> QPixmap:
            stage = stages[idx] if stages and 0 <= idx < len(stages) else None
            if stage is None:
                return pm
            # Signal stages already have their ROI polygon drawn inside
            # _stage_to_pixmap (restricted to the source-image portion of
            # the composite); skip the QPainter pass here.
            if stage.signal is not None:
                return pm
            roi1_pts = stage.metadata.get("roi1_points")
            roi3_pts = stage.metadata.get("roi3_points")
            bbox = stage.metadata.get("display_bbox")  # (px0, py0, pw, ph)
            show_roi1 = self._chk_roi1.isChecked() and roi1_pts and bbox
            show_roi3 = self._chk_roi2.isChecked() and roi3_pts and bbox
            if not show_roi1 and not show_roi3:
                return pm
            pm = QPixmap(pm)  # copy so we don't mutate the original
            painter = QPainter(pm)
            painter.setRenderHint(QPainter.RenderHint.Antialiasing)
            px0_r, py0_r = bbox[0], bbox[1]
            if show_roi3:
                pen = QPen(QColor(255, 140, 0))
                pen.setWidth(2)
                painter.setPen(pen)
                painter.setBrush(Qt.BrushStyle.NoBrush)
                poly3 = QPolygonF(
                    [QPointF(p[0] - px0_r, p[1] - py0_r) for p in roi3_pts]
                )
                painter.drawPolygon(poly3)
            if show_roi1:
                pen = QPen(QColor(0, 220, 220))
                pen.setWidth(1)
                painter.setPen(pen)
                poly1 = QPolygonF(
                    [QPointF(p[0] - px0_r, p[1] - py0_r) for p in roi1_pts]
                )
                painter.drawPolygon(poly1)
            painter.end()
            return pm

        self._view_a.set_pixmap(_with_roi_overlays(self._stages_a, _get_pixmap(self._stages_a, "A")))
        self._view_b.set_pixmap(_with_roi_overlays(self._stages_b, _get_pixmap(self._stages_b, "B")))
        self._score_lbl_a.setText(_format_stage_info(self._stages_a, idx))
        self._score_lbl_b.setText(_format_stage_info(self._stages_b, idx))

    # ── State save / restore ────────────────────────────────────────────────

    def _restore_state(self) -> None:
        s = QSettings(_SETTINGS_ORG, _SETTINGS_APP)
        geo = s.value(f"{_SETTINGS_KEY}/geometry")
        if geo:
            self.restoreGeometry(geo)
        else:
            self.resize(1200, 700)
        main_sp = s.value(f"{_SETTINGS_KEY}/main_splitter")
        if main_sp:
            self._main_splitter.restoreState(main_sp)
        views_sp = s.value(f"{_SETTINGS_KEY}/views_splitter")
        if views_sp:
            self._views_splitter.restoreState(views_sp)
        last_file = s.value(f"{_SETTINGS_KEY}/last_file", "")
        last_folder = s.value(f"{_SETTINGS_KEY}/last_folder", "")
        if last_file and os.path.isfile(last_file):
            self._load_file(last_file)
        elif last_folder and os.path.isdir(last_folder):
            self._load_folder(last_folder)
        last_ref_file = s.value(f"{_SETTINGS_KEY}/last_ref_file", "")
        last_ref_folder = s.value(f"{_SETTINGS_KEY}/last_ref_folder", "")
        if last_ref_file and os.path.isfile(last_ref_file):
            self._load_ref_file(last_ref_file)
        elif last_ref_folder and os.path.isdir(last_ref_folder):
            self._load_ref_folder(last_ref_folder)

    def _save_state(self) -> None:
        s = QSettings(_SETTINGS_ORG, _SETTINGS_APP)
        s.setValue(f"{_SETTINGS_KEY}/geometry",       self.saveGeometry())
        s.setValue(f"{_SETTINGS_KEY}/main_splitter",  self._main_splitter.saveState())
        s.setValue(f"{_SETTINGS_KEY}/views_splitter", self._views_splitter.saveState())

    # ── Close behaviour ──────────────────────────────────────────────────

    def closeEvent(self, event):
        self._save_state()
        if self._worker and self._worker.isRunning():
            self._worker.requestInterruption()
            self._worker.wait(500)
        super().closeEvent(event)
