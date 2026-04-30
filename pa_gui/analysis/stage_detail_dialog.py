"""
StageDetailDialog — full-screen pipeline stage viewer.

Layout:
  ┌───────────────────────────────────────────────────────────┐
  │  Pipette N  │  Stage N/M: "Stage Name"   │  [◄ Prev] [Next ►]│
  ├───────────────────────────────────────────────────────────┤
  │                                                           │
  │    [  ←  ]     ┌─────────────────────────┐    [  →  ]   │
  │                │   QGraphicsView / chart  │              │
  │                └─────────────────────────┘              │
  │                                                           │
  ├───────────────────────────────────────────────────────────┤
  │  Description text                                         │
  │  Metadata:  key=value  key=value  …                       │
  └───────────────────────────────────────────────────────────┘

Navigation:
  - Previous/Next buttons + left/right keyboard arrows cycle through stages.
  - The filmstrip strip at the bottom shows all stages as small thumbnails;
    clicking one jumps directly to that stage.

For image stages:
  - Displayed in a zoomable QGraphicsView (same as the calibration image viewer).
  - Mouse wheel zooms, click+drag pans.

For signal stages (1-D):
  - Rendered as a line chart in the same area (using a small custom widget
    that draws via QPainter — no matplotlib required).
  - Detected POI positions shown as vertical tick marks.
"""

from __future__ import annotations

from typing import List, Optional, TYPE_CHECKING

import numpy as np

from PySide6.QtCore import Qt, QRectF, QPointF, Signal
from PySide6.QtGui import (
    QImage, QPixmap, QFont, QPainter, QPen, QColor, QKeySequence,
)
from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QScrollArea, QFrame, QWidget, QSizePolicy, QDialogButtonBox,
    QGraphicsView, QGraphicsScene, QGraphicsPixmapItem, QSplitter,
    QSpinBox,
)

if TYPE_CHECKING:
    from pa.pipeline.types import DebugData, DebugStage

_FILMSTRIP_THUMB_W = 72
_FILMSTRIP_THUMB_H = 60


# ---------------------------------------------------------------------------
# Helper: numpy → QPixmap (full-res)
# ---------------------------------------------------------------------------

def _to_qpixmap(arr: np.ndarray) -> QPixmap:
    """Convert a HxW or HxWx3 uint8 numpy array to QPixmap."""
    if arr.dtype != np.uint8:
        mn, mx = arr.min(), arr.max()
        if mx > mn:
            arr = ((arr - mn) / (mx - mn) * 255).astype(np.uint8)
        else:
            arr = np.zeros_like(arr, dtype=np.uint8)

    if arr.ndim == 2:
        import cv2
        arr = cv2.cvtColor(arr, cv2.COLOR_GRAY2RGB)
    elif arr.ndim == 3 and arr.shape[2] == 3:
        import cv2
        arr = cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)

    qimg = QImage(
        arr.data, arr.shape[1], arr.shape[0],
        arr.strides[0], QImage.Format.Format_RGB888,
    )
    return QPixmap.fromImage(qimg)


def _thumb_pixmap(stage: "DebugStage", w=_FILMSTRIP_THUMB_W, h=_FILMSTRIP_THUMB_H) -> QPixmap:
    from pa_gui.analysis.debug_inspector import _ndarray_to_pixmap, _signal_to_pixmap
    if stage.image is not None:
        return _ndarray_to_pixmap(stage.image, w, h)
    elif stage.signal is not None:
        return _signal_to_pixmap(stage.signal, stage.z_axis_px, stage.poi_z_px, w, h)
    pm = QPixmap(w, h)
    pm.fill(Qt.GlobalColor.darkGray)
    return pm


# ---------------------------------------------------------------------------
# Zoomable image view
# ---------------------------------------------------------------------------

class _ZoomableView(QGraphicsView):

    row_hovered: Signal = Signal(float)   # scene-space y, or -1 on leave

    def __init__(self, parent=None):
        super().__init__(parent)
        self._scene = QGraphicsScene(self)
        self.setScene(self._scene)
        self._item: Optional[QGraphicsPixmapItem] = None
        self._crosshair_item = None      # QGraphicsLineItem for the crosshair
        self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
        self.setRenderHint(QPainter.RenderHint.Antialiasing)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.ViewportAnchor.AnchorViewCenter)
        self.setBackgroundBrush(QColor(20, 20, 20))
        self.setMouseTracking(True)

    def set_pixmap(self, pm: QPixmap):
        self._scene.clear()
        self._item = self._scene.addPixmap(pm)
        self._crosshair_item = None
        self._scene.setSceneRect(QRectF(pm.rect()))
        self.fitInView(self._scene.sceneRect(),
                       Qt.AspectRatioMode.KeepAspectRatio)

    def set_crosshair_row(self, image_y: Optional[float]) -> None:
        """Draw (or clear) a horizontal crosshair at the given image-space y."""
        if self._crosshair_item is not None:
            try:
                self._scene.removeItem(self._crosshair_item)
            except RuntimeError:
                pass
            self._crosshair_item = None
        if image_y is None or image_y < 0:
            return
        sr = self._scene.sceneRect()
        if not sr.isValid():
            return
        from PySide6.QtWidgets import QGraphicsLineItem
        pen = QPen(QColor(255, 160, 0), 0)   # cosmetic (1px regardless of zoom)
        pen.setStyle(Qt.PenStyle.DashLine)
        item = QGraphicsLineItem(sr.left(), image_y, sr.right(), image_y)
        item.setPen(pen)
        self._scene.addItem(item)
        self._crosshair_item = item

    def wheelEvent(self, event):
        factor = 1.15 if event.angleDelta().y() > 0 else 1.0 / 1.15
        self.scale(factor, factor)

    def mouseMoveEvent(self, event):
        if self._item is not None:
            scene_pt = self.mapToScene(event.pos())
            self.row_hovered.emit(scene_pt.y())
        super().mouseMoveEvent(event)

    def leaveEvent(self, event):
        self.row_hovered.emit(-1.0)
        super().leaveEvent(event)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self._item:
            self.fitInView(self._scene.sceneRect(),
                           Qt.AspectRatioMode.KeepAspectRatio)


# ---------------------------------------------------------------------------
# 1-D signal chart (QPainter-based, no matplotlib)
# ---------------------------------------------------------------------------

class _SignalChart(QWidget):
    """
    Renders a 1-D per-row intensity signal as a *horizontal* line chart.

    Axes
    ----
    Y-axis (top → bottom) = image row / Z-position in pixels  (matches image)
    X-axis (left → right) = signal intensity value

    Crosshair
    ---------
    Moving the mouse over the chart emits ``row_hovered(z_px)`` and draws a
    dashed orange horizontal line at the hovered row.  The companion image
    view can call ``set_crosshair_row(z_px)`` to mirror the line from the
    other direction.
    """

    row_hovered: Signal = Signal(float)   # z_px value under cursor, or -1 on leave

    def __init__(self, parent=None):
        super().__init__(parent)
        self._signal: Optional[np.ndarray] = None
        self._z_axis: Optional[np.ndarray] = None
        self._poi_z: Optional[List[float]] = None
        self._mode_name: str = ""
        self._crosshair_row: Optional[float] = None   # z_px value to highlight
        self._line_thickness: int = 2
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setMinimumSize(120, 200)
        self.setMouseTracking(True)

    def set_data(
        self,
        signal: np.ndarray,
        z_axis: Optional[np.ndarray] = None,
        poi_z: Optional[List[float]] = None,
        mode_name: str = "",
    ):
        self._signal = signal
        self._z_axis = z_axis
        self._poi_z = poi_z
        self._mode_name = mode_name
        self._crosshair_row = None
        self.update()

    def set_crosshair_row(self, z_px: Optional[float]) -> None:
        """Set the crosshair position (z-px in signal space). None clears it."""
        self._crosshair_row = z_px
        self.update()

    def set_line_thickness(self, thickness: int) -> None:
        self._line_thickness = max(1, thickness)
        self.update()

    # ── Events ────────────────────────────────────────────────────────

    def mouseMoveEvent(self, event):
        z_px = self._y_to_zpx(event.pos().y())
        if z_px is not None:
            self._crosshair_row = z_px
            self.row_hovered.emit(z_px)
            self.update()
        super().mouseMoveEvent(event)

    def leaveEvent(self, event):
        self._crosshair_row = None
        self.row_hovered.emit(-1.0)
        self.update()
        super().leaveEvent(event)

    # ── Paint ─────────────────────────────────────────────────────────

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        w, h = self.width(), self.height()
        # Padding: left = row-number labels, right = "Intensity" label area,
        # top/bottom = small margin
        pad_l, pad_r, pad_t, pad_b = 44, 12, 8, 28
        chart_w = max(1, w - pad_l - pad_r)
        chart_h = max(1, h - pad_t - pad_b)

        # Background
        painter.fillRect(0, 0, w, h, QColor(22, 22, 22))
        painter.fillRect(pad_l, pad_t, chart_w, chart_h, QColor(30, 30, 30))

        # Chart border
        painter.setPen(QPen(QColor(80, 80, 80), 1))
        painter.drawRect(pad_l, pad_t, chart_w, chart_h)

        if self._signal is None or len(self._signal) < 2:
            painter.setPen(QPen(QColor(160, 160, 160), 1))
            painter.drawText(pad_l + 4, pad_t + chart_h // 2, "No signal data")
            return

        sig = self._signal
        mn, mx = float(sig.min()), float(sig.max())
        if mx == mn:
            mx = mn + 1.0
        n = len(sig)

        # Z-axis range (row numbers)
        if self._z_axis is not None and len(self._z_axis) >= 2:
            z_min = float(self._z_axis[0])
            z_max = float(self._z_axis[-1])
        else:
            z_min, z_max = 0.0, float(n - 1)
        if z_max == z_min:
            z_max = z_min + 1.0

        font = QFont("", 7)
        painter.setFont(font)

        # Y-axis ticks (row positions)
        painter.setPen(QPen(QColor(160, 160, 160), 1))
        n_yticks = 5
        for ti in range(n_yticks + 1):
            frac = ti / n_yticks
            z_val = z_min + frac * (z_max - z_min)
            y = int(pad_t + frac * chart_h)
            painter.setPen(QPen(QColor(160, 160, 160), 1))
            painter.drawText(0, y - 6, pad_l - 4, 12,
                             Qt.AlignmentFlag.AlignRight,
                             f"{int(z_val)}")
            painter.setPen(QPen(QColor(45, 45, 45), 1, Qt.PenStyle.DotLine))
            painter.drawLine(pad_l, y, pad_l + chart_w, y)

        # X-axis ticks (intensity values)
        painter.setPen(QPen(QColor(160, 160, 160), 1))
        n_xticks = 4
        for ti in range(n_xticks + 1):
            frac = ti / n_xticks
            v = mn + frac * (mx - mn)
            x = int(pad_l + frac * chart_w)
            painter.setPen(QPen(QColor(160, 160, 160), 1))
            painter.drawText(x - 16, h - pad_b + 2, 32, pad_b - 2,
                             Qt.AlignmentFlag.AlignHCenter,
                             f"{v:.2f}")
            painter.setPen(QPen(QColor(45, 45, 45), 1, Qt.PenStyle.DotLine))
            painter.drawLine(x, pad_t, x, pad_t + chart_h)

        # X-axis label
        painter.setPen(QPen(QColor(120, 120, 120), 1))
        painter.drawText(0, h - pad_b + 2, w, pad_b - 2,
                         Qt.AlignmentFlag.AlignHCenter, "Intensity")

        # Signal line  (horizontal: x = intensity, y = row)
        pen = QPen(QColor(100, 220, 100), float(self._line_thickness))
        painter.setPen(pen)
        pts = []
        for i, v in enumerate(sig):
            if self._z_axis is not None and i < len(self._z_axis):
                z = float(self._z_axis[i])
            else:
                z = float(i)
            x = int(pad_l + (v - mn) / (mx - mn) * chart_w)
            y = int(pad_t + (z - z_min) / (z_max - z_min) * chart_h)
            pts.append(QPointF(x, y))
        for i in range(len(pts) - 1):
            painter.drawLine(pts[i], pts[i + 1])

        # POI horizontal lines (candidate transition rows)
        if self._poi_z is not None:
            poi_pen = QPen(QColor(0, 120, 255), 1, Qt.PenStyle.DashLine)
            painter.setPen(poi_pen)
            for z in self._poi_z:
                if z_max > z_min:
                    frac = (z - z_min) / (z_max - z_min)
                    y = int(pad_t + frac * chart_h)
                    painter.drawLine(pad_l, y, pad_l + chart_w, y)

        # Crosshair
        if self._crosshair_row is not None:
            frac = (self._crosshair_row - z_min) / (z_max - z_min)
            y = int(pad_t + frac * chart_h)
            if pad_t <= y <= pad_t + chart_h:
                ch_pen = QPen(QColor(255, 160, 0), 1, Qt.PenStyle.DashLine)
                painter.setPen(ch_pen)
                painter.drawLine(pad_l, y, pad_l + chart_w, y)

    # ── Helpers ───────────────────────────────────────────────────────

    def _y_to_zpx(self, widget_y: int) -> Optional[float]:
        """Map a widget y-coordinate to z-px signal space. None if outside chart."""
        if self._signal is None or len(self._signal) < 2:
            return None
        w, h = self.width(), self.height()
        pad_l, pad_r, pad_t, pad_b = 44, 12, 8, 28
        chart_h = max(1, h - pad_t - pad_b)
        if not (pad_t <= widget_y <= pad_t + chart_h):
            return None
        frac = (widget_y - pad_t) / chart_h
        n = len(self._signal)
        if self._z_axis is not None and len(self._z_axis) >= 2:
            z_min, z_max = float(self._z_axis[0]), float(self._z_axis[-1])
        else:
            z_min, z_max = 0.0, float(n - 1)
        return z_min + frac * (z_max - z_min)


# ---------------------------------------------------------------------------
# Filmstrip thumbnail button
# ---------------------------------------------------------------------------

class _FilmThumb(QLabel):
    def __init__(self, stage_index: int, parent=None):
        super().__init__(parent)
        self._stage_index = stage_index
        self.setFixedSize(_FILMSTRIP_THUMB_W + 4, _FILMSTRIP_THUMB_H + 2)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self._selected = False
        self._apply_style()

    def set_selected(self, selected: bool):
        self._selected = selected
        self._apply_style()

    def _apply_style(self):
        if self._selected:
            self.setStyleSheet("border: 2px solid #89b4fa; background: #2a2a3a;")
        else:
            self.setStyleSheet("border: 1px solid #444; background: #1a1a1a;")

    def mousePressEvent(self, event):
        # Handled by parent via findChild pattern; emit via parent
        # The dialog connects to each thumb individually.
        pass

    def stage_index(self) -> int:
        return self._stage_index


# ---------------------------------------------------------------------------
# Main dialog
# ---------------------------------------------------------------------------

class StageDetailDialog(QDialog):
    """
    Full pipeline stage inspector for one pipette's DebugData.

    Parameters
    ----------
    debug_data : DebugData
    initial_stage : int
        Which stage index to show on open (use -1 for the result overlay).
    parent : QWidget or None
    """

    def __init__(
        self,
        debug_data: "DebugData",
        initial_stage: int = 0,
        parent=None,
    ):
        super().__init__(parent)
        self._debug_data = debug_data
        # Build ordered stage list: result overlay first (index -1), then stages
        self._stages: List["DebugStage"] = list(debug_data.stages)
        self._has_overlay = debug_data.result_overlay is not None
        # Offset: if overlay present, stage 0 in our list = result overlay;
        # debug stages start at index 1.
        self._offset = 1 if self._has_overlay else 0
        self._total = len(self._stages) + self._offset
        self._current = max(0, initial_stage + self._offset
                            if initial_stage >= 0 else 0)
        self._roi_y_offset: float = 0.0   # for chart↔image crosshair mapping
        self._crosshair_conns: list = []  # active signal connections to disconnect

        self.setWindowTitle(
            f"Stage Inspector — Pipette {debug_data.pipette_index + 1} — "
            f"{debug_data.step_id}"
        )
        self.resize(1000, 640)
        self._build_ui()
        self._navigate_to(self._current)

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(6)

        # ── Top navigation bar ─────────────────────────────────────────
        nav = QHBoxLayout()
        nav.setSpacing(8)

        self._prev_btn = QPushButton("◄  Prev")
        self._prev_btn.setFixedWidth(80)
        self._prev_btn.clicked.connect(self._go_prev)
        nav.addWidget(self._prev_btn)

        self._stage_label = QLabel()
        self._stage_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._stage_label.setFont(QFont("", 10, QFont.Weight.Bold))
        nav.addWidget(self._stage_label, stretch=1)

        self._next_btn = QPushButton("Next  ►")
        self._next_btn.setFixedWidth(80)
        self._next_btn.clicked.connect(self._go_next)
        nav.addWidget(self._next_btn)

        nav.addSpacing(16)
        nav.addWidget(QLabel("Line:"))
        self._line_spin = QSpinBox()
        self._line_spin.setRange(1, 8)
        self._line_spin.setValue(2)
        self._line_spin.setFixedWidth(44)
        self._line_spin.setToolTip("Signal line thickness (px)")
        self._line_spin.valueChanged.connect(lambda v: self._chart.set_line_thickness(v))
        nav.addWidget(self._line_spin)

        root.addLayout(nav)

        # ── Main content area ──────────────────────────────────────────
        # Image stages: show _content_stack (view fills it).
        # Signal stages: show _signal_split (image | chart side-by-side).

        # --- Image-only stack -----------------------------------------
        self._view = _ZoomableView()

        self._content_stack = QFrame()
        stack_layout = QVBoxLayout(self._content_stack)
        stack_layout.setContentsMargins(0, 0, 0, 0)
        stack_layout.addWidget(self._view)

        # --- Signal side-by-side splitter -----------------------------
        self._signal_split = QSplitter(Qt.Orientation.Horizontal)

        self._sig_image_view = _ZoomableView()
        self._sig_image_view.setMinimumWidth(120)

        self._chart = _SignalChart()
        self._chart.setMinimumWidth(120)

        self._signal_split.addWidget(self._sig_image_view)
        self._signal_split.addWidget(self._chart)
        self._signal_split.setStretchFactor(0, 1)
        self._signal_split.setStretchFactor(1, 1)
        self._signal_split.setVisible(False)

        # Container that holds both (only one visible at a time)
        content_container = QFrame()
        cc_layout = QVBoxLayout(content_container)
        cc_layout.setContentsMargins(0, 0, 0, 0)
        cc_layout.addWidget(self._content_stack)
        cc_layout.addWidget(self._signal_split)

        root.addWidget(content_container, stretch=1)

        # ── Info panel (description + metadata) ───────────────────────
        info_frame = QFrame()
        info_frame.setFrameStyle(QFrame.Shape.StyledPanel)
        info_layout = QVBoxLayout(info_frame)
        info_layout.setContentsMargins(8, 4, 8, 4)
        info_layout.setSpacing(2)
        self._desc_label = QLabel()
        self._desc_label.setWordWrap(True)
        self._meta_label = QLabel()
        self._meta_label.setWordWrap(True)
        self._meta_label.setFont(QFont("", 8))
        self._meta_label.setStyleSheet("color: #aaa;")
        info_layout.addWidget(self._desc_label)
        info_layout.addWidget(self._meta_label)
        root.addWidget(info_frame)

        # ── Filmstrip ─────────────────────────────────────────────────
        film_scroll = QScrollArea()
        film_scroll.setFixedHeight(_FILMSTRIP_THUMB_H + 20)
        film_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        film_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        film_scroll.setWidgetResizable(True)

        film_widget = QWidget()
        film_row = QHBoxLayout(film_widget)
        film_row.setContentsMargins(2, 2, 2, 2)
        film_row.setSpacing(4)

        self._film_thumbs: List[_FilmThumb] = []

        # Overlay thumb
        if self._has_overlay:
            t = _FilmThumb(0)
            pm = _to_qpixmap(self._debug_data.result_overlay)
            t.setPixmap(pm.scaled(
                _FILMSTRIP_THUMB_W, _FILMSTRIP_THUMB_H,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            ))
            t.setToolTip("Result Overlay")
            idx_capture = 0
            t.mousePressEvent = lambda e, i=idx_capture: self._navigate_to(i)
            film_row.addWidget(t)
            self._film_thumbs.append(t)

        for i, stage in enumerate(self._stages):
            t = _FilmThumb(i + self._offset)
            pm = _thumb_pixmap(stage)
            t.setPixmap(pm)
            t.setToolTip(stage.name)
            idx_capture = i + self._offset
            t.mousePressEvent = lambda e, i=idx_capture: self._navigate_to(i)
            film_row.addWidget(t)
            self._film_thumbs.append(t)

        film_row.addStretch()
        film_scroll.setWidget(film_widget)
        root.addWidget(film_scroll)

        # ── Close button ──────────────────────────────────────────────
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        buttons.rejected.connect(self.reject)
        root.addWidget(buttons)

    # ------------------------------------------------------------------
    # Navigation
    # ------------------------------------------------------------------

    def _navigate_to(self, index: int):
        if not (0 <= index < self._total):
            return
        self._current = index

        # Disconnect any previous crosshair connections
        for sig, slot in self._crosshair_conns:
            try:
                sig.disconnect(slot)
            except RuntimeError:
                pass
        self._crosshair_conns = []

        # Update filmstrip highlight
        for t in self._film_thumbs:
            t.set_selected(t.stage_index() == index)

        self._prev_btn.setEnabled(index > 0)
        self._next_btn.setEnabled(index < self._total - 1)

        if index == 0 and self._has_overlay:
            # Result overlay (not a DebugStage)
            stage_name = "Result Overlay"
            pm = _to_qpixmap(self._debug_data.result_overlay)
            self._view.set_pixmap(pm)
            self._content_stack.setVisible(True)
            self._signal_split.setVisible(False)
            self._stage_label.setText(
                f"Stage {index + 1}/{self._total}: {stage_name}"
            )
            self._desc_label.setText(
                "Final pipeline result overlaid on the source image."
            )
            self._meta_label.setText("")
        else:
            # DebugStage
            stage = self._stages[index - self._offset]
            self._stage_label.setText(
                f"Stage {index + 1}/{self._total}: {stage.name}"
                + (f"  [{stage.mode_name}]" if stage.mode_name else "")
            )
            self._desc_label.setText(stage.description or "")
            meta_parts = [f"{k}: {v}" for k, v in stage.metadata.items()
                          if k != "roi"]   # omit raw roi tuple from display
            self._meta_label.setText("  |  ".join(meta_parts))

            if stage.image is not None:
                pm = _to_qpixmap(stage.image)
                self._view.set_pixmap(pm)
                self._content_stack.setVisible(True)
                self._signal_split.setVisible(False)

            elif stage.signal is not None:
                # ── Side-by-side: source image left, chart right ───────
                self._content_stack.setVisible(False)
                self._signal_split.setVisible(True)

                # Load signal into chart
                self._chart.set_data(
                    stage.signal,
                    stage.z_axis_px,
                    stage.poi_z_px,
                    stage.mode_name,
                )

                # Load source image (ROI-cropped if roi metadata available)
                roi = stage.metadata.get("roi")   # [x, y, w, h] or None
                src = self._debug_data.source_image
                self._roi_y_offset = 0.0
                if src is not None:
                    if roi is not None:
                        rx, ry, rw, rh = int(roi[0]), int(roi[1]), int(roi[2]), int(roi[3])
                        self._roi_y_offset = float(ry)
                        crop = src[ry:ry + rh, rx:rx + rw]
                        img_pm = _to_qpixmap(crop)
                    else:
                        img_pm = _to_qpixmap(src)
                    self._sig_image_view.set_pixmap(img_pm)
                else:
                    blank = QPixmap(200, 400)
                    blank.fill(QColor(40, 40, 40))
                    self._sig_image_view.set_pixmap(blank)

                # Wire crosshair sync
                roi_y = self._roi_y_offset

                def _chart_hovered(z_px: float, _roi_y=roi_y):
                    # z_px is in ROI-space (relative to ROI top)
                    image_y = _roi_y + z_px if z_px >= 0 else None
                    self._sig_image_view.set_crosshair_row(image_y)

                def _image_hovered(image_y: float, _roi_y=roi_y):
                    z_px = image_y - _roi_y if image_y >= 0 else None
                    self._chart.set_crosshair_row(z_px)

                c1 = self._chart.row_hovered.connect(_chart_hovered)
                c2 = self._sig_image_view.row_hovered.connect(_image_hovered)
                self._crosshair_conns = [
                    (self._chart.row_hovered, _chart_hovered),
                    (self._sig_image_view.row_hovered, _image_hovered),
                ]

            else:
                # Stage with no visual — show placeholder
                blank = QPixmap(400, 300)
                blank.fill(Qt.GlobalColor.darkGray)
                self._view.set_pixmap(blank)
                self._content_stack.setVisible(True)
                self._signal_split.setVisible(False)

    def _go_prev(self):
        self._navigate_to(self._current - 1)

    def _go_next(self):
        self._navigate_to(self._current + 1)

    def keyPressEvent(self, event):
        if event.key() == Qt.Key.Key_Left:
            self._go_prev()
        elif event.key() == Qt.Key.Key_Right:
            self._go_next()
        else:
            super().keyPressEvent(event)
