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

from PySide6.QtCore import Qt, QRectF, QPointF
from PySide6.QtGui import (
    QImage, QPixmap, QFont, QPainter, QPen, QColor, QKeySequence,
)
from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QScrollArea, QFrame, QWidget, QSizePolicy, QDialogButtonBox,
    QGraphicsView, QGraphicsScene, QGraphicsPixmapItem,
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
    def __init__(self, parent=None):
        super().__init__(parent)
        self._scene = QGraphicsScene(self)
        self.setScene(self._scene)
        self._item: Optional[QGraphicsPixmapItem] = None
        self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
        self.setRenderHint(QPainter.RenderHint.Antialiasing)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.ViewportAnchor.AnchorViewCenter)
        self.setBackgroundBrush(QColor(20, 20, 20))

    def set_pixmap(self, pm: QPixmap):
        self._scene.clear()
        self._item = self._scene.addPixmap(pm)
        self._scene.setSceneRect(QRectF(pm.rect()))
        self.fitInView(self._scene.sceneRect(),
                       Qt.AspectRatioMode.KeepAspectRatio)

    def wheelEvent(self, event):
        factor = 1.15 if event.angleDelta().y() > 0 else 1.0 / 1.15
        self.scale(factor, factor)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self._item:
            self.fitInView(self._scene.sceneRect(),
                           Qt.AspectRatioMode.KeepAspectRatio)


# ---------------------------------------------------------------------------
# 1-D signal chart (QPainter-based, no matplotlib)
# ---------------------------------------------------------------------------

class _SignalChart(QWidget):
    """Renders a 1-D signal as a line chart using QPainter."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._signal: Optional[np.ndarray] = None
        self._z_axis: Optional[np.ndarray] = None
        self._poi_z: Optional[List[float]] = None
        self._mode_name: str = ""
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setMinimumSize(200, 150)

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
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        w, h = self.width(), self.height()
        pad_l, pad_r, pad_t, pad_b = 50, 20, 20, 30
        chart_w = w - pad_l - pad_r
        chart_h = h - pad_t - pad_b

        # Background
        painter.fillRect(0, 0, w, h, QColor(22, 22, 22))
        painter.fillRect(pad_l, pad_t, chart_w, chart_h, QColor(30, 30, 30))

        # Axes
        pen = QPen(QColor(80, 80, 80), 1)
        painter.setPen(pen)
        painter.drawRect(pad_l, pad_t, chart_w, chart_h)

        if self._signal is None or len(self._signal) < 2:
            painter.setPen(QPen(QColor(160, 160, 160), 1))
            painter.drawText(pad_l + 10, pad_t + chart_h // 2, "No signal data")
            return

        sig = self._signal
        mn, mx = sig.min(), sig.max()
        if mx == mn:
            mx = mn + 1.0
        n = len(sig)

        # Y-axis labels
        font = QFont("", 7)
        painter.setFont(font)
        painter.setPen(QPen(QColor(160, 160, 160), 1))
        for tick_v in [mn, (mn + mx) / 2, mx]:
            y = int(pad_t + (1.0 - (tick_v - mn) / (mx - mn)) * chart_h)
            painter.drawText(2, y + 4, pad_l - 6, 12,
                             Qt.AlignmentFlag.AlignRight,
                             f"{tick_v:.2f}")
            painter.setPen(QPen(QColor(55, 55, 55), 1, Qt.PenStyle.DotLine))
            painter.drawLine(pad_l, y, pad_l + chart_w, y)
            painter.setPen(QPen(QColor(160, 160, 160), 1))

        # Signal line
        pen = QPen(QColor(100, 220, 100), 1.5)
        painter.setPen(pen)
        pts = []
        for i, v in enumerate(sig):
            x = int(pad_l + i / (n - 1) * chart_w)
            y = int(pad_t + (1.0 - (v - mn) / (mx - mn)) * chart_h)
            pts.append(QPointF(x, y))
        for i in range(len(pts) - 1):
            painter.drawLine(pts[i], pts[i + 1])

        # POI vertical lines
        if self._poi_z is not None and self._z_axis is not None and len(self._z_axis) >= 2:
            z_min, z_max = self._z_axis[0], self._z_axis[-1]
            poi_pen = QPen(QColor(0, 120, 255), 1, Qt.PenStyle.DashLine)
            painter.setPen(poi_pen)
            for z in self._poi_z:
                if z_max > z_min:
                    frac = (z - z_min) / (z_max - z_min)
                    x = int(pad_l + frac * chart_w)
                    painter.drawLine(x, pad_t, x, pad_t + chart_h)

        # X-axis label
        painter.setPen(QPen(QColor(160, 160, 160), 1))
        painter.drawText(0, h - pad_b + 2, w, pad_b - 2,
                         Qt.AlignmentFlag.AlignHCenter, "Z position (px)")


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

        self.setWindowTitle(
            f"Stage Inspector — Pipette {debug_data.pipette_index + 1} — "
            f"{debug_data.step_id}"
        )
        self.resize(900, 620)
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

        root.addLayout(nav)

        # ── Main content area ──────────────────────────────────────────
        content = QHBoxLayout()
        content.setSpacing(0)

        self._view = _ZoomableView()
        self._chart = _SignalChart()
        self._chart.setVisible(False)

        # Stack them in the same cell (only one visible at a time)
        self._content_stack = QFrame()
        stack_layout = QVBoxLayout(self._content_stack)
        stack_layout.setContentsMargins(0, 0, 0, 0)
        stack_layout.addWidget(self._view)
        stack_layout.addWidget(self._chart)

        content.addWidget(self._content_stack)
        root.addLayout(content, stretch=1)

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
            self._view.setVisible(True)
            self._chart.setVisible(False)
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
            meta_parts = [f"{k}: {v}" for k, v in stage.metadata.items()]
            self._meta_label.setText("  |  ".join(meta_parts))

            if stage.image is not None:
                pm = _to_qpixmap(stage.image)
                self._view.set_pixmap(pm)
                self._view.setVisible(True)
                self._chart.setVisible(False)
            elif stage.signal is not None:
                self._view.setVisible(False)
                self._chart.setVisible(True)
                self._chart.set_data(
                    stage.signal,
                    stage.z_axis_px,
                    stage.poi_z_px,
                    stage.mode_name,
                )
            else:
                # Stage with no visual — show placeholder text
                pm = QPixmap(400, 300)
                pm.fill(Qt.GlobalColor.darkGray)
                self._view.set_pixmap(pm)
                self._view.setVisible(True)
                self._chart.setVisible(False)

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
