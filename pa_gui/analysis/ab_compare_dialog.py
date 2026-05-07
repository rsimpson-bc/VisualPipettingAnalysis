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
import json
import os
from typing import Any, Dict, List, Optional, TYPE_CHECKING

import cv2
import numpy as np

from PySide6.QtCore import Qt, QRectF, QSettings, QTimer, QPointF, Signal
from PySide6.QtGui import QClipboard
from PySide6.QtGui import (
    QColor, QImage, QPainter, QPen, QPolygonF, QPixmap, QFont,
)
from PySide6.QtWidgets import (
    QApplication, QCheckBox, QComboBox, QDialog, QFileDialog, QFrame,
    QGraphicsPixmapItem, QGraphicsScene, QGraphicsView, QGroupBox,
    QHBoxLayout, QLabel, QMenu, QPlainTextEdit, QPushButton,
    QScrollArea, QSizePolicy, QSlider, QSpinBox, QSplitter,
    QVBoxLayout, QWidget,
)

from pa_gui.params.param_form import ParamFormWidget
from pa_gui.analysis import ab_presets

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

    # Emitted on right-click when annotation_mode is True.
    # Value is the 0.0–1.0 row fraction (scene height) at the click point.
    row_right_clicked: Signal = Signal(float)

    def __init__(self, title: str, parent=None, stretch_fit: bool = False):
        super().__init__(parent)
        self._title = title
        self._stretch_fit = stretch_fit
        self._scene = QGraphicsScene(self)
        self.setScene(self._scene)
        if not stretch_fit:
            self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
        self.setRenderHint(QPainter.RenderHint.Antialiasing)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.ViewportAnchor.AnchorViewCenter)
        self.setBackgroundBrush(QColor(18, 18, 18))
        self.setMouseTracking(True)
        self._crosshair_item = None    # horizontal line — synced across views
        self._crosshair_v_item = None  # vertical line   — local to this view
        self._user_zoomed = False
        # When True, right-click emits row_right_clicked instead of the copy menu
        self.annotation_mode: bool = False
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
        self._crosshair_item = None    # cleared with scene
        self._crosshair_v_item = None  # cleared with scene
        self._user_zoomed = False     # reset zoom tracking on new image
        self.resetTransform()
        self._scene.addPixmap(pm)
        self._scene.setSceneRect(QRectF(pm.rect()))
        mode = (Qt.AspectRatioMode.IgnoreAspectRatio
                if self._stretch_fit
                else Qt.AspectRatioMode.KeepAspectRatio)
        self.fitInView(self._scene.sceneRect(), mode)

    def set_crosshair_frac(self, frac: float) -> None:
        """Draw a dashed horizontal line at *frac* (0–1) of scene height.
        Pass frac < 0 to clear. Called externally for cross-view sync."""
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

    def set_col_crosshair_frac(self, frac: float) -> None:
        """Draw a dashed vertical line at *frac* (0–1) of scene width.
        Pass frac < 0 to clear. Local to this view only."""
        if self._crosshair_v_item is not None:
            try:
                self._scene.removeItem(self._crosshair_v_item)
            except RuntimeError:
                pass
            self._crosshair_v_item = None
        if frac < 0:
            return
        sr = self._scene.sceneRect()
        if not sr.isValid():
            return
        from PySide6.QtWidgets import QGraphicsLineItem
        x = sr.left() + frac * sr.width()
        pen = QPen(QColor(255, 160, 0), 0)
        pen.setStyle(Qt.PenStyle.DashLine)
        item = QGraphicsLineItem(x, sr.top(), x, sr.bottom())
        item.setPen(pen)
        self._scene.addItem(item)
        self._crosshair_v_item = item

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if not self._user_zoomed and self._scene.sceneRect().isValid():
            mode = (Qt.AspectRatioMode.IgnoreAspectRatio
                    if self._stretch_fit
                    else Qt.AspectRatioMode.KeepAspectRatio)
            self.fitInView(self._scene.sceneRect(), mode)

    def wheelEvent(self, event):
        if self._stretch_fit:
            event.ignore()
            return
        self._user_zoomed = True
        factor = 1.15 if event.angleDelta().y() > 0 else 1.0 / 1.15
        self.scale(factor, factor)

    def mouseMoveEvent(self, event):
        sr = self._scene.sceneRect()
        if sr.isValid() and sr.height() > 0:
            scene_pos = self.mapToScene(event.pos())
            frac = max(0.0, min(1.0, (scene_pos.y() - sr.top()) / sr.height()))
            col_frac = (max(0.0, min(1.0, (scene_pos.x() - sr.left()) / sr.width()))
                        if sr.width() > 0 else 0.0)
            self.set_crosshair_frac(frac)
            self.set_col_crosshair_frac(col_frac)
            self.row_hovered.emit(frac)
        super().mouseMoveEvent(event)

    def leaveEvent(self, event):
        self.set_crosshair_frac(-1.0)
        self.set_col_crosshair_frac(-1.0)
        self.row_hovered.emit(-1.0)
        super().leaveEvent(event)

    def contextMenuEvent(self, event):
        """Right-click menu.

        When ``annotation_mode`` is True the view emits ``row_right_clicked``
        with the z-fraction and suppresses the copy menu so the caller can
        show its own context menu.
        """
        if self.annotation_mode:
            sr = self._scene.sceneRect()
            if sr.isValid() and sr.height() > 0:
                scene_pos = self.mapToScene(event.pos())
                frac = max(0.0, min(1.0, (scene_pos.y() - sr.top()) / sr.height()))
                self.row_right_clicked.emit(frac)
            return

        # Default: copy / save menu
        # Grab the pixmap from the first item in the scene (the composite
        # image+chart that was last set via set_pixmap).
        items = self._scene.items()
        pixmap: Optional[QPixmap] = None
        for item in items:
            from PySide6.QtWidgets import QGraphicsPixmapItem as _GPI
            if isinstance(item, _GPI):
                pixmap = item.pixmap()
                break
        if pixmap is None or pixmap.isNull():
            return
        menu = QMenu(self)
        act_copy = menu.addAction("Copy image")
        act_save = menu.addAction("Save image as…")
        chosen = menu.exec(event.globalPos())
        if chosen == act_copy:
            QApplication.clipboard().setPixmap(pixmap)
        elif chosen == act_save:
            from PySide6.QtWidgets import QFileDialog as _QFD
            path, _ = _QFD.getSaveFileName(
                self, "Save image", "",
                "PNG image (*.png);;JPEG image (*.jpg *.jpeg);;All files (*)",
            )
            if path:
                pixmap.save(path)


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


def _band_distribution_arr(
    signal: np.ndarray,
    z_axis: Optional[np.ndarray],
    poi_z: Optional[List[float]],
    extra_signals: List[dict],
    w: int,
    h: int,
    line_thickness: int,
    show_poi: bool,
    z_range_override: Optional[tuple] = None,
) -> np.ndarray:
    """Render peak_count_bands_h as a per-row horizontal color distribution strip.

    Each row is a horizontal bar whose width-segments are proportional to the
    fraction of peaks in each prominence band:
        [band0_color × frac0][band1_color × frac1] ...
    Rows with zero peaks are dark.  The result is built at native row
    resolution then nearest-neighbor-resized to (h, w).
    """
    n = len(signal)

    # Parse band colors to BGR (OpenCV order)
    band_colors: List[tuple] = []
    for band in extra_signals:
        col_hex = band.get("color", "#888888").lstrip("#")
        try:
            r = int(col_hex[0:2], 16)
            g = int(col_hex[2:4], 16)
            b = int(col_hex[4:6], 16)
            band_colors.append((b, g, r))
        except Exception:
            band_colors.append((128, 128, 128))

    if z_range_override is not None:
        z_min, z_max = float(z_range_override[0]), float(z_range_override[1])
    elif z_axis is not None and len(z_axis) >= 2:
        z_min, z_max = float(z_axis[0]), float(z_axis[-1])
    else:
        z_min, z_max = 0.0, float(n - 1)

    # Reserve right-side legend panel so it doesn't overlap the strip.
    # Use a fixed per-character estimate (cv2 can't measure text precisely).
    lh_cv = 12
    swatch_cv = 8
    pad_cv = 3
    legend_entries = [(b.get("color", "#888"), b.get("label", "")) for b in extra_signals]
    max_lbl_chars = max((len(lbl) for _, lbl in legend_entries), default=4)
    legend_panel_w = max_lbl_chars * 6 + swatch_cv + pad_cv * 3 + 8  # px estimate
    legend_panel_w = max(legend_panel_w, 55)
    strip_w = max(1, w - legend_panel_w - 4)

    # Build at native row resolution then resize — clean pixel boundaries
    strip = np.zeros((n, strip_w, 3), dtype=np.uint8)
    for i in range(n):
        x_pos = 0
        for bi, band in enumerate(extra_signals):
            sig_arr = band.get("signal")
            frac = float(sig_arr[i]) if (sig_arr is not None and i < len(sig_arr)) else 0.0
            bw = int(frac * strip_w)
            if bw > 0 and x_pos < strip_w:
                x_end = min(x_pos + bw, strip_w)
                strip[i, x_pos:x_end] = band_colors[bi]
            x_pos = min(x_pos + bw, strip_w)

    canvas = np.zeros((h, w, 3), dtype=np.uint8)
    canvas[:, :strip_w] = cv2.resize(strip, (strip_w, h), interpolation=cv2.INTER_NEAREST)
    # Legend background
    canvas[:, strip_w:] = (20, 20, 20)

    # POI markers (strip only)
    if show_poi and poi_z is not None and z_axis is not None and len(z_axis) >= 2:
        for z in poi_z:
            if z_max > z_min:
                y = int((z - z_min) / (z_max - z_min) * (h - 1))
                y = max(0, min(y, h - 1))
                cv2.line(canvas, (0, y), (strip_w - 1, y), (50, 210, 50), line_thickness)

    # Legend in right panel
    if legend_entries:
        legend_h_total = lh_cv * len(legend_entries) + pad_cv * 2
        lx = strip_w + 4
        ly = max(4, (h - legend_h_total) // 2)  # vertically centred
        for ei, (col_hex, lbl) in enumerate(legend_entries):
            col_hex = col_hex.lstrip("#")
            try:
                r = int(col_hex[0:2], 16); g = int(col_hex[2:4], 16); b = int(col_hex[4:6], 16)
                bgr_l = (b, g, r)
            except Exception:
                bgr_l = (128, 128, 128)
            ey = ly + pad_cv + ei * lh_cv
            cv2.rectangle(canvas, (lx, ey + 2), (lx + swatch_cv, ey + lh_cv - 2), bgr_l, -1)
            cv2.putText(canvas, lbl, (lx + swatch_cv + pad_cv, ey + lh_cv - 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.30, (200, 200, 200), 1, cv2.LINE_AA)

    return canvas


def _signal_pixmap_arr(
    signal: np.ndarray,
    z_axis: Optional[np.ndarray] = None,
    poi_z: Optional[List[float]] = None,
    w: int = _CHART_W,
    h: int = _CHART_H,
    line_thickness: int = 2,
    log_scale: bool = False,
    show_poi: bool = True,
    extra_signals: Optional[List[dict]] = None,
    z_range_override: Optional[tuple] = None,
) -> np.ndarray:
    """Return a BGR numpy array (h, w, 3) of the horizontal signal chart."""
    canvas = np.zeros((h, w, 3), dtype=np.uint8)
    if signal is None or len(signal) < 2:
        return canvas

    # Band distribution mode: delegate to dedicated renderer
    if (extra_signals is not None
            and len(extra_signals) > 0
            and extra_signals[0].get("viz") == "band_distribution"):
        return _band_distribution_arr(
            signal, z_axis, poi_z, extra_signals, w, h, line_thickness, show_poi,
            z_range_override=z_range_override,
        )

    mn, mx = float(signal.min()), float(signal.max())
    # Extend range to include band signals so all lines share one axis
    if extra_signals:
        for band in extra_signals:
            bs = band.get("signal")
            if bs is not None and len(bs) > 0:
                mn = min(mn, float(bs.min()))
                mx = max(mx, float(bs.max()))
    if mx == mn:
        return canvas

    if log_scale:
        import numpy as _np
        log_mn = float(_np.log1p(max(0.0, mn)))
        log_mx = float(_np.log1p(max(0.0, mx)))
        if log_mx == log_mn:
            log_mx = log_mn + 1.0
        norm = (_np.log1p(_np.maximum(0.0, signal.astype(float))) - log_mn) / (log_mx - log_mn)
    else:
        norm = (signal - mn) / (mx - mn)
    n = len(norm)

    if z_range_override is not None:
        z_min, z_max = float(z_range_override[0]), float(z_range_override[1])
    elif z_axis is not None and len(z_axis) >= 2:
        z_min, z_max = float(z_axis[0]), float(z_axis[-1])
    else:
        z_min, z_max = 0.0, float(n - 1)

    def _norm_val(v_raw: float) -> float:
        if log_scale:
            lmn = float(np.log1p(max(0.0, mn)))
            lmx = float(np.log1p(max(0.0, mx)))
            return (float(np.log1p(max(0.0, v_raw))) - lmn) / max(lmx - lmn, 1e-9)
        return (v_raw - mn) / max(mx - mn, 1e-9)

    # Signal line: x = intensity (left→right), y = row (top→bottom)
    # Color: blue BGR(255, 130, 70)
    pts = []
    for i, v in enumerate(norm):
        z = float(z_axis[i]) if z_axis is not None and i < len(z_axis) else float(i)
        x = int(v * (w - 14)) + 7
        y = int((z - z_min) / max(z_max - z_min, 1) * (h - 1))
        pts.append((x, y))
    for i in range(len(pts) - 1):
        cv2.line(canvas, pts[i], pts[i + 1], (255, 130, 70), line_thickness)

    # Extra band lines
    if extra_signals:
        band_thick = max(1, line_thickness - 1)
        for band in extra_signals:
            bs = band.get("signal")
            if bs is None or len(bs) < 2:
                continue
            # Parse hex color to BGR
            col_hex = band.get("color", "#888888").lstrip("#")
            try:
                r = int(col_hex[0:2], 16)
                g = int(col_hex[2:4], 16)
                b = int(col_hex[4:6], 16)
                bgr = (b, g, r)
            except Exception:
                bgr = (128, 128, 128)
            bpts = []
            for i, v_raw in enumerate(bs):
                z = float(z_axis[i]) if z_axis is not None and i < len(z_axis) else float(i)
                x = int(_norm_val(float(v_raw)) * (w - 14)) + 7
                y = int((z - z_min) / max(z_max - z_min, 1) * (h - 1))
                bpts.append((x, y))
            for i in range(len(bpts) - 1):
                cv2.line(canvas, bpts[i], bpts[i + 1], bgr, band_thick)

        # Compact legend (top-right)
        legend_entries = [("#4682ff", "total")] + [
            (b.get("color", "#888"), b.get("label", "")) for b in extra_signals
        ]
        lh = 12
        swatch_w = 8
        pad = 3
        max_lbl_w = max(len(lbl) * 5 + swatch_w + pad * 3 for _, lbl in legend_entries)
        legend_h = lh * len(legend_entries) + pad * 2
        lx = w - max_lbl_w - 4
        ly = 4
        cv2.rectangle(canvas, (lx - 2, ly - 2), (lx + max_lbl_w + 2, ly + legend_h + 2),
                      (0, 0, 0), -1)
        for ei, (col_hex, lbl) in enumerate(legend_entries):
            col_hex = col_hex.lstrip("#")
            try:
                r = int(col_hex[0:2], 16); g = int(col_hex[2:4], 16); b = int(col_hex[4:6], 16)
                bgr_l = (b, g, r)
            except Exception:
                bgr_l = (128, 128, 128)
            ey = ly + pad + ei * lh
            cv2.rectangle(canvas, (lx, ey + 2), (lx + swatch_w, ey + lh - 2), bgr_l, -1)
            cv2.putText(canvas, lbl, (lx + swatch_w + pad, ey + lh - 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.30, (200, 200, 200), 1, cv2.LINE_AA)

    # POI markers — horizontal lines
    # Color: green BGR(50, 210, 50)
    if show_poi and poi_z is not None and z_axis is not None and len(z_axis) >= 2:
        for z in poi_z:
            if z_max > z_min:
                y = int((z - z_min) / (z_max - z_min) * (h - 1))
                cv2.line(canvas, (0, y), (w - 1, y), (50, 210, 50), line_thickness)

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
    log_scale: bool = False,
    show_poi: bool = True,
    companion_signal_stage: Optional["DebugStage"] = None,
) -> QPixmap:
    """Render a DebugStage as QPixmap, optionally blended with the source.

    If *companion_signal_stage* is provided and this is an image stage, the
    signal chart from the companion stage is appended on the right (same
    layout as signal stages: [image | chart]).
    """
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
        elif img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        else:
            img = img.copy()

        # Append companion signal chart if available
        if companion_signal_stage is not None and companion_signal_stage.signal is not None:
            img_h, img_w = img.shape[:2]
            chart_w = max(120, min(300, img_w // 4))
            chart_arr = _signal_pixmap_arr(
                companion_signal_stage.signal,
                companion_signal_stage.z_axis_px,
                companion_signal_stage.poi_z_px,
                w=chart_w, h=img_h,
                line_thickness=line_thickness,
                log_scale=log_scale,
                show_poi=show_poi,
                extra_signals=companion_signal_stage.extra_signals,
            )
            sep = np.full((img_h, 2, 3), 55, dtype=np.uint8)
            img = np.concatenate([img, sep, chart_arr], axis=1)

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
                                           line_thickness=line_thickness,
                                           log_scale=log_scale,
                                           show_poi=show_poi,
                                           extra_signals=stage.extra_signals)
            sep = np.full((src_h, 2, 3), 55, dtype=np.uint8)
            composite = np.concatenate([src, sep, chart_arr], axis=1)
            return _ndarray_to_pixmap(composite)

        # No source image — chart only, at default size
        chart_arr = _signal_pixmap_arr(stage.signal, stage.z_axis_px, stage.poi_z_px,
                                       w=_CHART_W, h=_CHART_H,
                                       line_thickness=line_thickness,
                                       log_scale=log_scale,
                                       show_poi=show_poi,
                                       extra_signals=stage.extra_signals)
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
        self._log_scale: bool = False
        self._show_poi: bool = True
        self._multi_results: List[dict] = []
        self._tip_row_widgets: List[tuple] = []  # (lbl_a, lbl_b) per pipette
        self._hover_stage_a: Optional["DebugStage"] = None
        self._hover_stage_b: Optional["DebugStage"] = None

        self._debounce = QTimer(self)
        self._debounce.setSingleShot(True)
        self._debounce.setInterval(_DEBOUNCE_MS)
        self._debounce.timeout.connect(self._run_comparison)

        self.setWindowTitle(f"A/B Comparison — {mode_name}")
        self.setWindowFlag(Qt.WindowType.Window)

        self._build_ui()
        self._restore_state()
        self._refresh_preset_combo()

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

        self._btn_prev = QPushButton("◀")
        self._btn_prev.setFixedWidth(28)
        self._btn_prev.setToolTip("Previous image file")
        self._btn_prev.setEnabled(False)
        self._btn_prev.clicked.connect(self._on_nav_prev)
        src_row.addWidget(self._btn_prev)

        self._btn_next = QPushButton("▶")
        self._btn_next.setFixedWidth(28)
        self._btn_next.setToolTip("Next image file")
        self._btn_next.setEnabled(False)
        self._btn_next.clicked.connect(self._on_nav_next)
        src_row.addWidget(self._btn_next)

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

        # Quick-select buttons 1–8
        self._pipette_btns: list[QPushButton] = []
        for _i in range(1, 9):
            _btn = QPushButton(str(_i))
            _btn.setFixedSize(22, 22)
            _btn.setCheckable(False)
            _btn.setToolTip(f"Select pipette {_i}")
            _btn.clicked.connect(lambda _checked, n=_i: self._on_pipette_btn_clicked(n))
            src_row.addWidget(_btn)
            self._pipette_btns.append(_btn)
        self._refresh_pipette_buttons()

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

        # ── Presets group ─────────────────────────────────────────────────
        presets_box = QGroupBox("Presets")
        presets_box.setStyleSheet(
            "QGroupBox { font-weight:bold; margin-top:6px; }"
            "QGroupBox::title { subcontrol-origin:margin; left:6px; padding:0 2px; }"
        )
        pb_layout = QVBoxLayout(presets_box)
        pb_layout.setSpacing(4)
        pb_layout.setContentsMargins(6, 10, 6, 6)

        # File row
        file_row = QHBoxLayout()
        file_row.addWidget(QLabel("File:"))
        self._preset_file_lbl = QLabel()
        self._preset_file_lbl.setStyleSheet(
            "background:#1a1a1a; padding:1px 4px; border:1px solid #444;"
            "font-size:10px; color:#aaa;"
        )
        self._preset_file_lbl.setSizePolicy(QSizePolicy.Policy.Expanding,
                                             QSizePolicy.Policy.Fixed)
        self._preset_file_lbl.setFixedHeight(20)
        self._preset_file_lbl.setToolTip(ab_presets.get_presets_path())
        self._preset_file_lbl.setText(
            os.path.basename(ab_presets.get_presets_path())
        )
        file_row.addWidget(self._preset_file_lbl, 1)
        change_file_btn = QPushButton("…")
        change_file_btn.setFixedWidth(24)
        change_file_btn.setToolTip("Choose a different presets file")
        change_file_btn.clicked.connect(self._on_change_preset_file)
        file_row.addWidget(change_file_btn)
        pb_layout.addLayout(file_row)

        # Separator
        sep_p = QFrame()
        sep_p.setFrameShape(QFrame.Shape.HLine)
        sep_p.setStyleSheet("color:#333;")
        pb_layout.addWidget(sep_p)

        # Description label + text area
        pb_layout.addWidget(QLabel("Description (explain why these settings work):"))
        self._preset_desc = QPlainTextEdit()
        self._preset_desc.setPlaceholderText(
            "e.g. Wider ROI avoids glare on meniscus at 50 µL…"
        )
        self._preset_desc.setFixedHeight(58)  # ~3 lines
        pb_layout.addWidget(self._preset_desc)

        # Save + Copy row
        save_copy_row = QHBoxLayout()
        save_btn = QPushButton("Save Preset")
        save_btn.setToolTip("Save current B parameters + description to the presets file.")
        save_btn.clicked.connect(self._on_save_preset)
        save_copy_row.addWidget(save_btn)
        copy_btn = QPushButton("Copy JSON")
        copy_btn.setToolTip(
            "Copy the current B parameters and description as JSON to the clipboard."
        )
        copy_btn.clicked.connect(self._on_copy_preset_json)
        save_copy_row.addWidget(copy_btn)
        pb_layout.addLayout(save_copy_row)

        # Separator
        sep_p2 = QFrame()
        sep_p2.setFrameShape(QFrame.Shape.HLine)
        sep_p2.setStyleSheet("color:#333;")
        pb_layout.addWidget(sep_p2)

        # Load section
        load_hdr_row = QHBoxLayout()
        load_hdr_row.addWidget(QLabel("Load preset:"))
        refresh_btn = QPushButton("↺")
        refresh_btn.setFixedWidth(24)
        refresh_btn.setToolTip("Reload the presets file")
        refresh_btn.clicked.connect(self._refresh_preset_combo)
        load_hdr_row.addWidget(refresh_btn)
        pb_layout.addLayout(load_hdr_row)

        self._preset_combo = QComboBox()
        self._preset_combo.setSizePolicy(QSizePolicy.Policy.Expanding,
                                          QSizePolicy.Policy.Fixed)
        self._preset_combo.setToolTip("Saved presets for this mode (newest first)")
        self._preset_combo.currentIndexChanged.connect(self._on_preset_selected)
        pb_layout.addWidget(self._preset_combo)

        self._preset_preview = QPlainTextEdit()
        self._preset_preview.setReadOnly(True)
        self._preset_preview.setFixedHeight(72)  # ~4 lines
        self._preset_preview.setStyleSheet(
            "background:#161616; color:#aaa; font-family:monospace; font-size:10px;"
        )
        self._preset_preview.setPlaceholderText("Select a preset above to preview it.")
        pb_layout.addWidget(self._preset_preview)

        load_delete_row = QHBoxLayout()
        self._load_preset_btn = QPushButton("Load into B")
        self._load_preset_btn.setEnabled(False)
        self._load_preset_btn.setToolTip(
            "Apply the previewed preset's parameters to the B parameter form."
        )
        self._load_preset_btn.setStyleSheet(
            "QPushButton { background:#4a3a00; color:#f5c542; font-weight:bold; }"
            "QPushButton:disabled { background:#333; color:#666; }"
        )
        self._load_preset_btn.clicked.connect(self._on_load_preset)
        load_delete_row.addWidget(self._load_preset_btn, 1)

        self._delete_preset_btn = QPushButton("Delete")
        self._delete_preset_btn.setEnabled(False)
        self._delete_preset_btn.setToolTip("Permanently delete the selected preset.")
        self._delete_preset_btn.setStyleSheet(
            "QPushButton { background:#3a0000; color:#ff6666; }"
            "QPushButton:disabled { background:#333; color:#666; }"
        )
        self._delete_preset_btn.clicked.connect(self._on_delete_preset)
        load_delete_row.addWidget(self._delete_preset_btn)
        pb_layout.addLayout(load_delete_row)

        left_layout.addWidget(presets_box)

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

        self._log_chk = QCheckBox("Log")
        self._log_chk.setToolTip("Display signal (x) axis on a logarithmic scale")
        self._log_chk.stateChanged.connect(self._on_log_scale_changed)
        ctrl_row.addWidget(self._log_chk)

        self._all_tips_chk = QCheckBox("All tips")
        self._all_tips_chk.setToolTip(
            "Run and display all pipettes simultaneously.\n"
            "Shows a compact signal chart grid (one row per tip).\n"
            "The Pipette idx spinner is disabled in this mode."
        )
        self._all_tips_chk.stateChanged.connect(self._on_all_tips_toggled)
        ctrl_row.addWidget(self._all_tips_chk)

        self._poi_chk = QCheckBox("POI")
        self._poi_chk.setChecked(True)
        self._poi_chk.setToolTip("Toggle peak/POI marker lines on signal charts")
        self._poi_chk.stateChanged.connect(self._on_show_poi_changed)
        ctrl_row.addWidget(self._poi_chk)

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
        self._hover_lbl_a = QLabel("")
        self._hover_lbl_a.setFixedHeight(16)
        self._hover_lbl_a.setStyleSheet(
            "font-size:10px; color:#ffc850; background:#1a1a1a; padding:0 5px;"
        )
        a_lay.addWidget(self._hover_lbl_a)
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
        self._hover_lbl_b = QLabel("")
        self._hover_lbl_b.setFixedHeight(16)
        self._hover_lbl_b.setStyleSheet(
            "font-size:10px; color:#ffc850; background:#1a1a1a; padding:0 5px;"
        )
        b_lay.addWidget(self._hover_lbl_b)
        self._score_lbl_b = QLabel("")
        self._score_lbl_b.setWordWrap(True)
        self._score_lbl_b.setStyleSheet(
            "font-size:10px; color:#aaa; background:#111; padding:3px 5px;"
        )
        b_lay.addWidget(self._score_lbl_b)
        views_splitter.addWidget(b_wrap)

        right_layout.addWidget(views_splitter, 1)

        # ── All-tips scroll area (hidden by default) ───────────────────
        self._all_tips_inner = QWidget()
        self._all_tips_layout = QVBoxLayout(self._all_tips_inner)
        self._all_tips_layout.setContentsMargins(4, 4, 4, 4)
        self._all_tips_layout.setSpacing(2)
        self._all_tips_layout.addStretch()

        self._all_tips_scroll = QScrollArea()
        self._all_tips_scroll.setWidget(self._all_tips_inner)
        self._all_tips_scroll.setWidgetResizable(True)
        self._all_tips_scroll.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )
        self._all_tips_scroll.setVisible(False)
        right_layout.addWidget(self._all_tips_scroll, 1)

        splitter.addWidget(right)

        # Link crosshairs: hovering over one view moves the cursor in the other
        self._view_a.row_hovered.connect(self._view_b.set_crosshair_frac)
        self._view_b.row_hovered.connect(self._view_a.set_crosshair_frac)
        self._view_a.row_hovered.connect(self._on_row_hovered_a)
        self._view_b.row_hovered.connect(self._on_row_hovered_b)

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
        self._update_nav_buttons()

    def _update_nav_buttons(self) -> None:
        n = len(self._image_paths)
        idx = self._frame_combo.currentIndex()
        self._btn_prev.setEnabled(n > 1 and idx > 0)
        self._btn_next.setEnabled(n > 1 and idx < n - 1)

    def _on_nav_prev(self) -> None:
        idx = self._frame_combo.currentIndex()
        if idx > 0:
            self._frame_combo.setCurrentIndex(idx - 1)  # triggers _on_frame_changed

    def _on_nav_next(self) -> None:
        idx = self._frame_combo.currentIndex()
        if idx < len(self._image_paths) - 1:
            self._frame_combo.setCurrentIndex(idx + 1)  # triggers _on_frame_changed

    def _on_frame_changed(self, idx: int) -> None:
        self._update_nav_buttons()
        if self._image_paths:
            self._schedule_run()

    def _on_pipette_changed(self) -> None:
        """Pipette index or tip type changed — schedule a re-run."""
        self._refresh_pipette_buttons()
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

    # ── Preset slots ───────────────────────────────────────────────────────

    def _refresh_preset_combo(self) -> None:
        """Reload the preset combo from the current presets file."""
        self._preset_combo.blockSignals(True)
        self._preset_combo.clear()
        self._preset_combo.addItem("— select a preset —", userData=None)
        entries = ab_presets.presets_for_mode(self._mode_name)
        for entry in entries:
            self._preset_combo.addItem(
                ab_presets.format_combo_label(entry), userData=entry
            )
        self._preset_combo.blockSignals(False)
        self._preset_preview.clear()
        self._load_preset_btn.setEnabled(False)

    def _on_preset_selected(self, idx: int) -> None:
        """Show a preview when a preset is chosen in the combo."""
        entry = self._preset_combo.itemData(idx)
        if entry is None:
            self._preset_preview.clear()
            self._load_preset_btn.setEnabled(False)
            self._delete_preset_btn.setEnabled(False)
        else:
            self._preset_preview.setPlainText(ab_presets.format_preview(entry))
            self._load_preset_btn.setEnabled(True)
            self._delete_preset_btn.setEnabled(True)

    def _on_load_preset(self) -> None:
        """Load the currently previewed preset's params_b into the B form."""
        idx = self._preset_combo.currentIndex()
        entry = self._preset_combo.itemData(idx)
        if entry is None:
            return
        params = entry.get("params_b", {})
        desc = entry.get("description", "")
        self._params_b = copy.deepcopy(params)
        self._b_form.set_params(self._params_b)
        # Pre-fill the description field with the loaded preset's description
        self._preset_desc.setPlainText(desc)

    def _on_save_preset(self) -> None:
        """Append current B params + description to the presets file."""
        desc = self._preset_desc.toPlainText().strip()
        if not desc:
            from PySide6.QtWidgets import QMessageBox
            reply = QMessageBox.question(
                self, "No description",
                "Save preset without a description?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            )
            if reply != QMessageBox.StandardButton.Yes:
                return
        try:
            ab_presets.add_preset(
                mode=self._mode_name,
                description=desc,
                params_b=copy.deepcopy(self._b_form.get_params()),
            )
        except Exception as exc:
            from PySide6.QtWidgets import QMessageBox
            QMessageBox.warning(self, "Save failed", str(exc))
            return
        self._refresh_preset_combo()
        # Select the newly saved entry (index 1 = first real entry after placeholder)
        if self._preset_combo.count() > 1:
            self._preset_combo.setCurrentIndex(1)

    def _on_delete_preset(self) -> None:
        """Delete the currently selected preset after confirmation."""
        idx = self._preset_combo.currentIndex()
        entry = self._preset_combo.itemData(idx)
        if entry is None:
            return
        from PySide6.QtWidgets import QMessageBox
        desc = entry.get("description", "").strip() or "(no description)"
        reply = QMessageBox.question(
            self,
            "Delete preset",
            f"Delete preset:\n\n\"{desc}\"\n\nThis cannot be undone.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if reply != QMessageBox.StandardButton.Yes:
            return
        try:
            ab_presets.delete_preset(entry["id"])
        except Exception as exc:
            QMessageBox.warning(self, "Delete failed", str(exc))
            return
        self._refresh_preset_combo()

    def _on_copy_preset_json(self) -> None:
        """Copy current B params + description as a JSON snippet to clipboard."""
        payload = {
            "mode": self._mode_name,
            "description": self._preset_desc.toPlainText().strip(),
            "params_b": copy.deepcopy(self._b_form.get_params()),
        }
        text = json.dumps(payload, indent=2, ensure_ascii=False)
        QApplication.clipboard().setText(text)
        # Brief visual feedback via the status label
        self._status_lbl.setText("Copied to clipboard.")

    def _on_change_preset_file(self) -> None:
        """Let the user pick a different (or new) presets JSON file."""
        current = ab_presets.get_presets_path()
        choice, _ = QFileDialog.getSaveFileName(
            self,
            "Select or create presets file",
            current,
            "JSON files (*.json);;All files (*)",
            options=QFileDialog.Option.DontConfirmOverwrite,
        )
        if not choice:
            return
        ab_presets.set_presets_path(choice)
        self._preset_file_lbl.setText(os.path.basename(choice))
        self._preset_file_lbl.setToolTip(choice)
        self._refresh_preset_combo()

    def _on_stage_changed(self, idx: int) -> None:
        self._render_current_stage()

    def _on_overlay_changed(self, value: int) -> None:
        self._overlay_lbl.setText(f"{value}%")
        self._render_current_stage()

    def _on_line_thickness_changed(self, value: int) -> None:
        self._line_thickness = value
        self._render_current_stage()

    def _on_log_scale_changed(self, state: int) -> None:
        self._log_scale = bool(state)
        self._render_current_stage()

    def _on_show_poi_changed(self, state: int) -> None:
        self._show_poi = bool(state)
        self._render_current_stage()

    def _on_pipette_btn_clicked(self, n: int) -> None:
        """Jump to pipette *n* when a quick-select button is clicked."""
        self._pipette_spin.setValue(n)  # triggers valueChanged → _on_pipette_changed

    def _refresh_pipette_buttons(self) -> None:
        """Highlight the button matching the current spinbox value; clear the rest."""
        current = self._pipette_spin.value()
        for i, btn in enumerate(self._pipette_btns, start=1):
            if i == current:
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

    def _on_all_tips_toggled(self, state: int) -> None:
        """Switch between single-tip and all-tips display modes."""
        enabled = bool(state)
        self._pipette_spin.setEnabled(not enabled)
        for btn in self._pipette_btns:
            btn.setEnabled(not enabled)
        self._views_splitter.setVisible(not enabled)
        self._all_tips_scroll.setVisible(enabled)
        self._schedule_run()

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

        frame_idx = max(0, self._frame_combo.currentIndex())

        if self._all_tips_chk.isChecked():
            # Derive pipette count from IC; fall back to 8.
            import os, json
            pipette_count = 8
            if self._instrument_config_path and os.path.isfile(self._instrument_config_path):
                try:
                    with open(self._instrument_config_path, encoding="utf-8") as f:
                        ic = json.load(f)
                    entries = ic.get("calibrated_pixel_yz", [])
                    if entries:
                        pipette_count = len(entries)
                except Exception:
                    pass

            from pa_gui.analysis.ab_compare_worker import ABMultiTipWorker
            self._worker = ABMultiTipWorker(
                mode_name=self._mode_name,
                params_a=self._params_a,
                params_b=self._params_b,
                image_paths=self._image_paths,
                reference_paths=self._reference_paths,
                frame_index=frame_idx,
                pipette_count=pipette_count,
                instrument_config_path=self._instrument_config_path,
                tip_type=self._tip_edit.text().strip(),
                parent=self,
            )
            self._worker.progress.connect(self._on_progress)
            self._worker.multi_result_ready.connect(self._on_multi_result)
            self._worker.error.connect(self._on_worker_error)
        else:
            pipette_idx = self._pipette_spin.value() - 1
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

    def _rebuild_tip_rows(self, n: int) -> None:
        """Recreate the all-tips scroll area to hold n tip rows."""
        # Clear old content (remove all items except the trailing stretch)
        while self._all_tips_layout.count():
            item = self._all_tips_layout.takeAt(0)
            w = item.widget()
            if w is not None:
                w.deleteLater()
        self._tip_row_widgets = []

        _CHART_W_TIP = 260
        _CHART_H_TIP = 130

        for i in range(n):
            row_widget = QWidget()
            row = QHBoxLayout(row_widget)
            row.setContentsMargins(2, 2, 2, 2)
            row.setSpacing(6)

            lbl_tip = QLabel(f"Tip {i + 1}")
            lbl_tip.setFixedWidth(52)
            lbl_tip.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            lbl_tip.setStyleSheet("color:#aaa; font-size:10px;")
            row.addWidget(lbl_tip)

            lbl_a = QLabel()
            lbl_a.setFixedSize(_CHART_W_TIP, _CHART_H_TIP)
            lbl_a.setStyleSheet("background:#1a1a1a; border:1px solid #2a4a2a;")
            row.addWidget(lbl_a)

            sep = QLabel("|")
            sep.setFixedWidth(8)
            sep.setAlignment(Qt.AlignmentFlag.AlignCenter)
            sep.setStyleSheet("color:#555;")
            row.addWidget(sep)

            lbl_b = QLabel()
            lbl_b.setFixedSize(_CHART_W_TIP, _CHART_H_TIP)
            lbl_b.setStyleSheet("background:#1a1a1a; border:1px solid #4a2a2a;")
            row.addWidget(lbl_b)

            row.addStretch()
            self._tip_row_widgets.append((lbl_a, lbl_b))
            self._all_tips_layout.addWidget(row_widget)

            if i < n - 1:
                sep_line = QFrame()
                sep_line.setFrameShape(QFrame.Shape.HLine)
                sep_line.setStyleSheet("color:#333;")
                self._all_tips_layout.addWidget(sep_line)

        self._all_tips_layout.addStretch()

    def _render_all_tips(self) -> None:
        """Render all tip rows from _multi_results for the current stage."""
        if not self._multi_results or not self._tip_row_widgets:
            return
        combo_name = self._stage_combo.currentText()
        combo_idx  = self._stage_combo.currentIndex()
        _CHART_W_TIP = 260
        _CHART_H_TIP = 130

        blank = QPixmap(_CHART_W_TIP, _CHART_H_TIP)
        blank.fill(QColor(26, 26, 26))

        def _find_stage(stages):
            """Find the stage matching the combo by name (positional fallback)."""
            if not stages:
                return None
            for s in stages:
                if s.name == combo_name:
                    return s
            # Fallback: use the last signal stage
            for s in reversed(stages):
                if s.signal is not None:
                    return s
            return None

        for result in self._multi_results:
            i = result["pipette_index"]
            if i >= len(self._tip_row_widgets):
                continue
            lbl_a, lbl_b = self._tip_row_widgets[i]

            for lbl, stages in ((lbl_a, result["stages_a"]), (lbl_b, result["stages_b"])):
                stage = _find_stage(stages)
                # For image stages use the companion signal stage instead
                if stage is not None and stage.signal is None:
                    stage = next((s for s in stages if s.signal is not None), None)
                if stage is not None and stage.signal is not None:
                    arr = _signal_pixmap_arr(
                        stage.signal, stage.z_axis_px, stage.poi_z_px,
                        w=_CHART_W_TIP, h=_CHART_H_TIP,
                        line_thickness=self._line_thickness,
                        log_scale=self._log_scale,
                        show_poi=self._show_poi,
                        extra_signals=stage.extra_signals,
                    )
                    lbl.setPixmap(_ndarray_to_pixmap(arr))
                else:
                    lbl.setPixmap(blank)

    def _on_multi_result(self, results: list) -> None:
        """Receive all-tips result from ABMultiTipWorker."""
        self._multi_results = results
        if results:
            self._stages_a = results[0]["stages_a"]
            self._stages_b = results[0]["stages_b"]
        self._rebuild_tip_rows(len(results))
        self._populate_stage_combo()   # uses _stages_a/_stages_b for names
        self._render_all_tips()
        self._status_lbl.setText(f"Done ({len(results)} tips)")
        self._run_btn.setEnabled(True)

    def _populate_stage_combo(self) -> None:
        prev_name = self._stage_combo.currentText()
        self._stage_combo.blockSignals(True)
        self._stage_combo.clear()
        # Merge stage names from A and B (preserve order; union so stages that
        # appear in only one panel — e.g. Gradient only in B — are still listed).
        seen: set = set()
        names: list = []
        for s in list(self._stages_a) + list(self._stages_b):
            if s.name not in seen:
                seen.add(s.name)
                names.append(s.name)
        for name in names:
            self._stage_combo.addItem(name)
        self._stage_combo.setEnabled(bool(names))
        # Restore previous selection by name, then fall back to last item
        new_idx = 0
        if prev_name and prev_name in names:
            new_idx = names.index(prev_name)
        elif names:
            new_idx = len(names) - 1
        if names:
            self._stage_combo.setCurrentIndex(new_idx)
        self._stage_combo.blockSignals(False)
        self._render_current_stage()

    def _render_current_stage(self) -> None:
        if self._all_tips_chk.isChecked():
            self._render_all_tips()
            return

        idx = self._stage_combo.currentIndex()
        alpha = self._overlay_slider.value() / 100.0

        # Resolve per-panel stage index by matching the combo name first.
        # This handles the case where A and B have different stage counts
        # (e.g., B has an extra Gradient stage not present in A).
        combo_name = self._stage_combo.currentText() if idx >= 0 else ""

        def _resolve_idx(stages) -> int:
            """Return the index of the stage whose name == combo_name, or idx."""
            if not stages:
                return idx
            for i, s in enumerate(stages):
                if s.name == combo_name:
                    return i
            return idx  # positional fallback if no name match

        idx_a = _resolve_idx(self._stages_a)
        idx_b = _resolve_idx(self._stages_b)

        def _crop_src_for_stage(stages, stage_idx, src_override=None):
            """Crop the given source image using that stage's display_bbox."""
            src = src_override if src_override is not None else self._source_image
            if src is None:
                return None
            bbox = None
            if stages and 0 <= stage_idx < len(stages):
                bbox = stages[stage_idx].metadata.get("display_bbox")
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

        def _get_pixmap(stages, stage_idx, label):
            if not stages or stage_idx < 0 or stage_idx >= len(stages):
                pm = QPixmap(320, 220)
                pm.fill(QColor(30, 30, 30))
                return pm
            stage = stages[stage_idx]
            # Companion signal stage: the first signal stage in this panel's
            # list; used by image stages to show [image | chart] composite.
            companion = next((s for s in stages if s.signal is not None), None)
            if stage.signal is not None:
                # For signal stages: show the contrast-adjusted image in the
                # left strip if use_contrast produced one; otherwise raw source.
                contrast_img = (self._contrast_image_a if label == "A"
                                else self._contrast_image_b)
                src = _crop_src_for_stage(stages, stage_idx, contrast_img)
                # Signal stages handle ROI overlays inside _stage_to_pixmap so
                # the polygon is restricted to the source-image portion of the
                # composite.  _with_roi_overlays is bypassed for these.
                return _stage_to_pixmap(
                    stage, src, alpha,
                    line_thickness=self._line_thickness,
                    show_roi1=self._chk_roi1.isChecked(),
                    show_roi3=self._chk_roi2.isChecked(),
                    log_scale=self._log_scale,
                    show_poi=self._show_poi,
                )
            else:
                # For image stages: overlay always uses the raw original.
                # Also pass the companion signal stage so the chart is shown
                # alongside the image (same [image | chart] layout).
                src = _crop_src_for_stage(stages, stage_idx)
                return _stage_to_pixmap(
                    stage, src, alpha,
                    line_thickness=self._line_thickness,
                    show_roi1=self._chk_roi1.isChecked(),
                    show_roi3=self._chk_roi2.isChecked(),
                    log_scale=self._log_scale,
                    show_poi=self._show_poi,
                    companion_signal_stage=companion,
                )

        def _with_roi_overlays(stages, stage_idx, pm: QPixmap) -> QPixmap:
            stage = stages[stage_idx] if stages and 0 <= stage_idx < len(stages) else None
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

        self._view_a.set_pixmap(_with_roi_overlays(self._stages_a, idx_a, _get_pixmap(self._stages_a, idx_a, "A")))
        self._view_b.set_pixmap(_with_roi_overlays(self._stages_b, idx_b, _get_pixmap(self._stages_b, idx_b, "B")))
        self._score_lbl_a.setText(_format_stage_info(self._stages_a, idx_a))
        self._score_lbl_b.setText(_format_stage_info(self._stages_b, idx_b))
        # Stash current signal stage for hover readout
        self._hover_stage_a = (self._stages_a[idx_a]
                               if self._stages_a and 0 <= idx_a < len(self._stages_a)
                               else None)
        self._hover_stage_b = (self._stages_b[idx_b]
                               if self._stages_b and 0 <= idx_b < len(self._stages_b)
                               else None)
        self._hover_lbl_a.setText("")
        self._hover_lbl_b.setText("")

    # ── Hover readout ────────────────────────────────────────────────────────

    @staticmethod
    def _interp_signal(stage: Optional["DebugStage"], frac: float) -> Optional[float]:
        """Interpolate the signal value at the given scene fraction (0–1)."""
        if stage is None or stage.signal is None or len(stage.signal) < 2:
            return None
        sig = stage.signal
        n = len(sig)
        z_axis = stage.z_axis_px
        if z_axis is not None and len(z_axis) >= 2:
            z_min, z_max = float(z_axis[0]), float(z_axis[-1])
        else:
            z_min, z_max = 0.0, float(n - 1)
        if z_max == z_min:
            return float(sig[0])
        z_px = z_min + frac * (z_max - z_min)
        idx = (z_px - z_min) / (z_max - z_min) * (n - 1)
        i0 = max(0, min(int(idx), n - 2))
        t = idx - i0
        return float(sig[i0] * (1.0 - t) + sig[i0 + 1] * t)

    def _on_row_hovered_a(self, frac: float) -> None:
        if frac < 0:
            self._hover_lbl_a.setText("")
            return
        val = self._interp_signal(self._hover_stage_a, frac)
        if val is None:
            self._hover_lbl_a.setText("")
            return
        stage = self._hover_stage_a
        z_axis = stage.z_axis_px if stage else None
        if z_axis is not None and len(z_axis) >= 2:
            z_px = float(z_axis[0]) + frac * (float(z_axis[-1]) - float(z_axis[0]))
        else:
            n = len(stage.signal) if stage and stage.signal is not None else 1
            z_px = frac * (n - 1)
        self._hover_lbl_a.setText(f"Row {int(round(z_px))}  ·  {val:.4g}")

    def _on_row_hovered_b(self, frac: float) -> None:
        if frac < 0:
            self._hover_lbl_b.setText("")
            return
        val = self._interp_signal(self._hover_stage_b, frac)
        if val is None:
            self._hover_lbl_b.setText("")
            return
        stage = self._hover_stage_b
        z_axis = stage.z_axis_px if stage else None
        if z_axis is not None and len(z_axis) >= 2:
            z_px = float(z_axis[0]) + frac * (float(z_axis[-1]) - float(z_axis[0]))
        else:
            n = len(stage.signal) if stage and stage.signal is not None else 1
            z_px = frac * (n - 1)
        self._hover_lbl_b.setText(f"Row {int(round(z_px))}  ·  {val:.4g}")

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
