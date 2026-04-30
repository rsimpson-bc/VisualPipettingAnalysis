"""
DebugInspectorWidget — inline summary panel shown in the "Inspect" tab of
the Run Analysis right-panel when a step is selected in the results tree.

Layout (per pipette):
  ┌──────────────────────────────────────────────────────────────────────┐
  │  Pipette N                                             [Inspect…]    │
  │  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐            │
  │  │ Overlay  │  │ Stage 0  │  │ Stage 1  │  │ Stage 2  │  ◄  ►       │
  │  └──────────┘  └──────────┘  └──────────┘  └──────────┘            │
  │  "Blur • Ridge Mask • Termination Signal"                            │
  └──────────────────────────────────────────────────────────────────────┘

Clicking a thumbnail (or pressing Inspect…) opens StageDetailDialog.
"""

from __future__ import annotations

from typing import List, Optional, TYPE_CHECKING

import numpy as np

from PySide6.QtCore import Qt, Signal, QSize
from PySide6.QtGui import QImage, QPixmap, QFont
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QScrollArea, QFrame, QSizePolicy, QSplitter, QPlainTextEdit,
    QApplication,
)

if TYPE_CHECKING:
    from pa.pipeline.types import DebugData, DebugStage
    from pa_gui.analysis.models import StepDef

_THUMB_W = 96
_THUMB_H = 80
_MAX_VISIBLE_THUMBS = 6


def _ndarray_to_pixmap(arr: np.ndarray, w: int = _THUMB_W, h: int = _THUMB_H) -> QPixmap:
    """Convert a numpy image (grayscale or BGR) to a scaled QPixmap."""
    if arr is None or arr.size == 0:
        pm = QPixmap(w, h)
        pm.fill(Qt.GlobalColor.darkGray)
        return pm

    # Ensure uint8
    if arr.dtype != np.uint8:
        mn, mx = arr.min(), arr.max()
        if mx > mn:
            arr = ((arr - mn) / (mx - mn) * 255).astype(np.uint8)
        else:
            arr = np.zeros_like(arr, dtype=np.uint8)

    if arr.ndim == 2:
        # Grayscale → RGB
        import cv2
        arr = cv2.cvtColor(arr, cv2.COLOR_GRAY2RGB)
    elif arr.ndim == 3 and arr.shape[2] == 3:
        import cv2
        arr = cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)
    else:
        # Fallback: try as-is
        pass

    qimg = QImage(
        arr.data,
        arr.shape[1],
        arr.shape[0],
        arr.strides[0],
        QImage.Format.Format_RGB888,
    )
    pm = QPixmap.fromImage(qimg)
    return pm.scaled(w, h, Qt.AspectRatioMode.KeepAspectRatio,
                     Qt.TransformationMode.SmoothTransformation)


def _signal_to_pixmap(
    signal: np.ndarray,
    z_axis: Optional[np.ndarray] = None,
    poi_z: Optional[List[float]] = None,
    w: int = _THUMB_W,
    h: int = _THUMB_H,
) -> QPixmap:
    """Render a 1-D signal as a small chart QPixmap (horizontal orientation)."""
    import cv2
    canvas = np.zeros((h, w, 3), dtype=np.uint8)

    if signal is None or len(signal) < 2:
        return _ndarray_to_pixmap(canvas, w, h)

    # Normalise
    mn, mx = signal.min(), signal.max()
    if mx == mn:
        return _ndarray_to_pixmap(canvas, w, h)
    norm = (signal - mn) / (mx - mn)

    n = len(norm)
    if z_axis is not None and len(z_axis) >= 2:
        z_min, z_max = float(z_axis[0]), float(z_axis[-1])
    else:
        z_min, z_max = 0.0, float(n - 1)

    # Horizontal chart: x = intensity, y = row (top→bottom)
    pts = []
    for i, v in enumerate(norm):
        if z_axis is not None and i < len(z_axis):
            z = float(z_axis[i])
        else:
            z = float(i)
        x = int(v * (w - 3)) + 1
        y = int((z - z_min) / max(z_max - z_min, 1) * (h - 1))
        pts.append((x, y))

    for i in range(len(pts) - 1):
        cv2.line(canvas, pts[i], pts[i + 1], (100, 220, 100), 1)

    # POI markers — horizontal lines
    if poi_z is not None and z_axis is not None and len(z_axis) >= 2:
        for z in poi_z:
            if z_max > z_min:
                frac = (z - z_min) / (z_max - z_min)
                y = int(frac * (h - 1))
                cv2.line(canvas, (0, y), (w - 1, y), (0, 80, 255), 1)

    return _ndarray_to_pixmap(canvas, w, h)


class _StageThumbnail(QLabel):
    """A single clickable thumbnail in the filmstrip."""
    clicked = Signal(int)  # stage index

    def __init__(self, stage_index: int, label: str, parent=None):
        super().__init__(parent)
        self._stage_index = stage_index
        self.setFixedSize(_THUMB_W + 4, _THUMB_H + 22)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setStyleSheet("border: 1px solid #555; background: #1a1a1a;")
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self._name_label = QLabel(label, self)
        self._name_label.setAlignment(Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignBottom)
        self._name_label.setFont(QFont("", 7))
        self._name_label.setFixedWidth(_THUMB_W + 4)
        self._name_label.move(0, _THUMB_H + 2)
        self._name_label.setWordWrap(False)

    def set_pixmap(self, pm: QPixmap):
        self.setPixmap(pm)

    def mousePressEvent(self, event):
        self.clicked.emit(self._stage_index)
        super().mousePressEvent(event)

    def enterEvent(self, event):
        self.setStyleSheet("border: 1px solid #89b4fa; background: #222;")
        super().enterEvent(event)

    def leaveEvent(self, event):
        self.setStyleSheet("border: 1px solid #555; background: #1a1a1a;")
        super().leaveEvent(event)


class _PipettePanel(QFrame):
    """Summary panel for one pipette's DebugData."""
    inspect_requested = Signal(object, int)  # DebugData, initial_stage_index

    def __init__(self, debug_data: "DebugData", parent=None):
        super().__init__(parent)
        self._debug_data = debug_data
        self.setFrameStyle(QFrame.Shape.StyledPanel)
        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 4, 6, 4)
        layout.setSpacing(4)

        # Header row
        hdr = QHBoxLayout()
        hdr.setContentsMargins(0, 0, 0, 0)
        title = QLabel(f"Pipette {self._debug_data.pipette_index + 1}")
        title.setFont(QFont("", 9, QFont.Weight.Bold))
        hdr.addWidget(title)
        hdr.addStretch()
        btn = QPushButton("Inspect…")
        btn.setFixedHeight(22)
        btn.setToolTip("Open full filmstrip viewer")
        btn.clicked.connect(lambda: self.inspect_requested.emit(self._debug_data, 0))
        hdr.addWidget(btn)
        layout.addLayout(hdr)

        # Filmstrip scroll area
        scroll = QScrollArea()
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setFixedHeight(_THUMB_H + 34)
        scroll.setWidgetResizable(True)

        filmstrip_widget = QWidget()
        row = QHBoxLayout(filmstrip_widget)
        row.setContentsMargins(2, 2, 2, 2)
        row.setSpacing(4)

        self._thumbs: List[_StageThumbnail] = []

        # Always show the result overlay first (as "stage -1")
        if self._debug_data.result_overlay is not None:
            t = _StageThumbnail(-1, "Result")
            t.set_pixmap(_ndarray_to_pixmap(self._debug_data.result_overlay))
            t.clicked.connect(lambda idx: self.inspect_requested.emit(self._debug_data, 0))
            row.addWidget(t)
            self._thumbs.append(t)

        # One thumbnail per stage
        for i, stage in enumerate(self._debug_data.stages):
            t = _StageThumbnail(i, stage.name)
            if stage.image is not None:
                t.set_pixmap(_ndarray_to_pixmap(stage.image))
            elif stage.signal is not None:
                t.set_pixmap(_signal_to_pixmap(
                    stage.signal, stage.z_axis_px, stage.poi_z_px))
            t.clicked.connect(lambda idx=i: self.inspect_requested.emit(self._debug_data, idx))
            row.addWidget(t)
            self._thumbs.append(t)

        row.addStretch()
        scroll.setWidget(filmstrip_widget)
        layout.addWidget(scroll)

        # Stage name summary
        names = [s.name for s in self._debug_data.stages]
        summary = QLabel("  •  ".join(names) if names else "No stages captured.")
        summary.setFont(QFont("", 7))
        summary.setWordWrap(True)
        layout.addWidget(summary)


class DebugInspectorWidget(QWidget):
    """
    Inline 'Inspect' tab embedded in the right panel of analysis_tab.
    Populated by calling ``load(step)``.
    Emits ``inspect_requested(DebugData, initial_stage_index)`` to open
    the StageDetailDialog.
    """
    inspect_requested = Signal(object, int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._layout = QVBoxLayout(self)
        self._layout.setContentsMargins(4, 4, 4, 4)
        self._layout.setSpacing(6)

        self._placeholder = QLabel("Select a completed step to inspect its pipeline stages.")
        self._placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._placeholder.setWordWrap(True)
        self._layout.addWidget(self._placeholder)
        self._layout.addStretch()

        self._panels: List[_PipettePanel] = []

    def load(self, step: "StepDef") -> None:
        """Populate the inspector with a step's DebugData and/or error."""
        self._clear()
        debug_data_list = getattr(step, "debug_data", None)
        error_text = getattr(step, "error", None)

        # ── Error panel (always shown first when an error occurred) ──────────
        if error_text:
            err_frame = QFrame()
            err_frame.setFrameStyle(QFrame.Shape.StyledPanel)
            err_frame.setStyleSheet(
                "QFrame { border: 1px solid #c0392b; border-radius: 4px; background: #1a0a0a; }"
            )
            err_layout = QVBoxLayout(err_frame)
            err_layout.setContentsMargins(6, 4, 6, 4)
            err_layout.setSpacing(4)

            hdr = QHBoxLayout()
            hdr.setContentsMargins(0, 0, 0, 0)
            title = QLabel("⚠  Step Error")
            title.setFont(QFont("", 9, QFont.Weight.Bold))
            title.setStyleSheet("color: #e74c3c; border: none;")
            hdr.addWidget(title)
            hdr.addStretch()
            copy_btn = QPushButton("Copy")
            copy_btn.setFixedHeight(20)
            copy_btn.setToolTip("Copy traceback to clipboard")
            copy_btn.clicked.connect(
                lambda: QApplication.clipboard().setText(error_text))
            hdr.addWidget(copy_btn)
            err_layout.addLayout(hdr)

            tb_box = QPlainTextEdit()
            tb_box.setReadOnly(True)
            tb_box.setPlainText(error_text)
            tb_box.setFont(QFont("Courier New", 8))
            tb_box.setStyleSheet(
                "QPlainTextEdit { background: #0d0d0d; color: #e06c75; border: none; }"
            )
            tb_box.setMinimumHeight(120)
            tb_box.setMaximumHeight(220)
            err_layout.addWidget(tb_box)

            self._layout.insertWidget(0, err_frame)

        # ── No debug data ─────────────────────────────────────────────────────
        if debug_data_list is None:
            if not error_text:
                self._placeholder.setText(
                    f"No debug data for step '{step.step_id}'.\n"
                    "Run the step first to capture pipeline stages."
                )
            else:
                self._placeholder.setText("No pipeline stage data was captured before the error.")
            self._placeholder.setVisible(True)
            return

        if len(debug_data_list) == 0:
            if not error_text:
                self._placeholder.setText(
                    f"Step '{step.step_id}' ran with no pipettes.\n"
                    "Add pipettes in the Step Editor and re-run to capture debug data."
                )
            else:
                self._placeholder.setText("No pipettes were processed before the error.")
            self._placeholder.setVisible(True)
            return

        # ── Stage panels (one per pipette) ────────────────────────────────────
        self._placeholder.setVisible(False)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        inner = QWidget()
        inner_layout = QVBoxLayout(inner)
        inner_layout.setContentsMargins(0, 0, 0, 0)
        inner_layout.setSpacing(8)

        for dbg in debug_data_list:
            panel = _PipettePanel(dbg)
            panel.inspect_requested.connect(self.inspect_requested)
            inner_layout.addWidget(panel)
            self._panels.append(panel)

        inner_layout.addStretch()
        scroll.setWidget(inner)
        # Insert after any error frame already added
        insert_pos = 0 if not error_text else 1
        self._layout.insertWidget(insert_pos, scroll)

    def _clear(self):
        """Remove all dynamically added panels and error frames."""
        for panel in self._panels:
            panel.setParent(None)
        self._panels.clear()
        # Remove all dynamically added widgets (leave placeholder + stretch at end)
        while self._layout.count() > 2:
            item = self._layout.takeAt(0)
            w = item.widget()
            if w:
                w.deleteLater()
        self._placeholder.setVisible(True)
