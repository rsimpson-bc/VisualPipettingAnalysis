"""
RotationPanel — widget for determining and saving the camera rotation correction.

Offers two methods:
  Auto   — fit a line through mandrel pixel positions in instrument_config.
  Manual — load any image, click two points to define a horizontal reference.

The final rotation angle (CCW positive, degrees) can always be edited directly
and is saved via the "Apply & Save" button.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
from PySide6.QtCore import Qt, QPointF, QRectF, Signal
from PySide6.QtGui import QColor, QImage, QMouseEvent, QPainter, QPen, QPixmap
from PySide6.QtWidgets import (
    QDoubleSpinBox,
    QFileDialog,
    QGraphicsEllipseItem,
    QGraphicsLineItem,
    QGraphicsPixmapItem,
    QGraphicsScene,
    QGraphicsView,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)


class _LineDrawView(QGraphicsView):
    """
    Zoomable QGraphicsView that lets the user click two points to define a line.
    Emits angle_computed(degrees) after the second click.

    Coordinate system: click positions are reported in image (scene) coordinates,
    so they are independent of zoom level.
    """

    angle_computed: Signal = Signal(float)

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._scene = QGraphicsScene(self)
        self.setScene(self._scene)
        self._pixmap_item: Optional[QGraphicsPixmapItem] = None
        self._points: List[QPointF] = []          # image-space coordinates
        self._overlay_items: list = []            # QGraphicsItems for the line overlay
        self._user_zoomed = False                 # suppress fitInView on resize once user zooms

        self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
        self.setRenderHint(QPainter.RenderHint.Antialiasing)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.ViewportAnchor.AnchorViewCenter)
        self.setBackgroundBrush(QColor(17, 17, 17))

        placeholder = QLabel("(load an image to draw a reference line)")
        placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
        placeholder.setStyleSheet("color:#555;")
        self._placeholder = self._scene.addWidget(placeholder)

    # ── Public API ────────────────────────────────────────────────────

    def load_pixmap(self, pixmap: QPixmap) -> None:
        self._scene.clear()
        self._pixmap_item = self._scene.addPixmap(pixmap)
        self._scene.setSceneRect(QRectF(pixmap.rect()))
        self._points = []
        self._overlay_items = []
        self._user_zoomed = False
        self.fitInView(self._scene.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)

    def clear_line(self) -> None:
        for item in self._overlay_items:
            self._scene.removeItem(item)
        self._overlay_items = []
        self._points = []

    # ── Events ────────────────────────────────────────────────────────

    def mousePressEvent(self, event: QMouseEvent) -> None:
        if self._pixmap_item is None:
            super().mousePressEvent(event)
            return
        if event.button() != Qt.MouseButton.LeftButton:
            super().mousePressEvent(event)
            return

        scene_pt = self.mapToScene(event.pos())
        # Clamp to pixmap bounds
        r = self._scene.sceneRect()
        if not r.contains(scene_pt):
            super().mousePressEvent(event)
            return

        if len(self._points) >= 2:
            self.clear_line()

        self._points.append(scene_pt)
        self._draw_overlay()

        if len(self._points) == 2:
            dx = self._points[1].x() - self._points[0].x()
            dy = self._points[1].y() - self._points[0].y()
            angle_deg = math.degrees(math.atan2(-dy, dx))
            self.angle_computed.emit(angle_deg)

    def wheelEvent(self, event) -> None:
        factor = 1.15 if event.angleDelta().y() > 0 else 1.0 / 1.15
        self.scale(factor, factor)
        self._user_zoomed = True

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        if not self._user_zoomed and self._scene.sceneRect().isValid():
            self.fitInView(self._scene.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)

    # ── Overlay drawing ───────────────────────────────────────────────

    def _draw_overlay(self) -> None:
        r = 6
        pen_dot  = QPen(QColor("#e74c3c"), 2)
        pen_line = QPen(QColor("#e74c3c"), 2)

        for pt in self._points:
            item = QGraphicsEllipseItem(pt.x() - r, pt.y() - r, r * 2, r * 2)
            item.setPen(pen_dot)
            item.setBrush(Qt.BrushStyle.NoBrush)
            self._scene.addItem(item)
            self._overlay_items.append(item)

        if len(self._points) == 2:
            line = QGraphicsLineItem(
                self._points[0].x(), self._points[0].y(),
                self._points[1].x(), self._points[1].y(),
            )
            line.setPen(pen_line)
            self._scene.addItem(line)
            self._overlay_items.append(line)


        self._pixmap_original = pixmap
        self._points = []
        self._refresh()

    def clear_line(self) -> None:
        self._points = []
        self._refresh()

    def mousePressEvent(self, event: QMouseEvent) -> None:  # noqa: N802
        if self._pixmap_original is None:
            return
        if event.button() != Qt.MouseButton.LeftButton:
            return

        # Map click position to image coordinates (accounting for centred scaling)
        lw, lh = self.width(), self.height()
        pm = self._pixmap_original.scaled(
            lw, lh,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        offset_x = (lw - pm.width()) / 2
        offset_y = (lh - pm.height()) / 2

        img_x = event.position().x() - offset_x
        img_y = event.position().y() - offset_y

        if img_x < 0 or img_y < 0 or img_x > pm.width() or img_y > pm.height():
            return

class RotationPanel(QWidget):
    """
    Right side of the Camera Calibration tab.

    Signals
    -------
    rotation_saved(float)
        Emitted (with angle in degrees) when the user clicks "Apply & Save".
    """

    rotation_saved: Signal = Signal(float)

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._mandrels: List[Dict[str, Any]] = []
        self._cal_matrix: Optional[np.ndarray] = None
        self._cal_dist:   Optional[np.ndarray] = None
        self._build_ui()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def set_mandrels(self, mandrels: List[Dict[str, Any]]) -> None:
        """Update mandrel list from instrument_config['mandrels']."""
        self._mandrels = mandrels or []
        self._update_auto_status()

    def set_rotation_deg(self, angle: float) -> None:
        """Pre-fill the angle spinbox (e.g. from loaded instrument_config)."""
        self._angle_spin.setValue(angle)

    def set_calibration(
        self,
        camera_matrix: Optional[List[List[float]]],
        dist_coeffs: Optional[List[float]],
    ) -> None:
        """Pass checkerboard calibration data so loaded images are undistorted."""
        if camera_matrix and dist_coeffs:
            self._cal_matrix = np.array(camera_matrix, dtype=np.float64)
            self._cal_dist   = np.array(dist_coeffs,   dtype=np.float64)
        else:
            self._cal_matrix = None
            self._cal_dist   = None
        self._update_undistort_notice()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(8, 8, 8, 8)

        root.addWidget(QLabel(
            "<b>Rotation correction</b> — correct camera tilt so horizontal "
            "features appear level in analysis images."
        ))

        # ── Auto section ──────────────────────────────────────────────
        auto_group = QGroupBox("Auto — compute from mandrel positions")
        auto_layout = QVBoxLayout(auto_group)

        self._auto_status = QLabel("Loading…")
        self._auto_status.setWordWrap(True)
        auto_layout.addWidget(self._auto_status)

        self._compute_btn = QPushButton("Compute from mandrels")
        self._compute_btn.clicked.connect(self._on_compute_auto)
        auto_layout.addWidget(self._compute_btn)

        root.addWidget(auto_group)

        # ── Manual section ────────────────────────────────────────────
        manual_group = QGroupBox("Manual — draw a horizontal reference line")
        manual_layout = QVBoxLayout(manual_group)

        load_row = QHBoxLayout()
        self._load_img_btn = QPushButton("Load Image…")
        self._load_img_btn.clicked.connect(self._on_load_image)
        load_row.addWidget(self._load_img_btn)
        self._clear_line_btn = QPushButton("Clear line")
        load_row.addWidget(self._clear_line_btn)
        load_row.addStretch()
        manual_layout.addLayout(load_row)

        self._undistort_notice = QLabel("")
        self._undistort_notice.setWordWrap(True)
        manual_layout.addWidget(self._undistort_notice)

        instr = QLabel(
            "Click two points on a feature that should be horizontal "
            "(e.g. a platform edge or the mandrel row).  "
            "Scroll to zoom, drag to pan."
        )
        instr.setWordWrap(True)
        instr.setStyleSheet("color: #888;")
        manual_layout.addWidget(instr)

        self._line_view = _LineDrawView()
        self._line_view.angle_computed.connect(self._on_line_angle)
        self._line_view.setMinimumHeight(180)
        self._line_view.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )
        manual_layout.addWidget(self._line_view, 1)

        self._clear_line_btn.clicked.connect(self._line_view.clear_line)

        root.addWidget(manual_group, 1)

        # ── Angle result + save ───────────────────────────────────────
        angle_group = QGroupBox("Rotation angle")
        angle_layout = QHBoxLayout(angle_group)

        angle_layout.addWidget(QLabel("Correction (°):"))
        self._angle_spin = QDoubleSpinBox()
        self._angle_spin.setRange(-45.0, 45.0)
        self._angle_spin.setDecimals(4)
        self._angle_spin.setValue(0.0)
        self._angle_spin.setSingleStep(0.01)
        self._angle_spin.setToolTip(
            "Positive = counter-clockwise.  Applied after lens undistortion."
        )
        angle_layout.addWidget(self._angle_spin)

        angle_layout.addStretch()

        self._save_btn = QPushButton("Apply && Save")
        self._save_btn.setFixedHeight(32)
        self._save_btn.setStyleSheet(
            "QPushButton { background: #2a7a2a; color: white; font-weight: bold; }"
        )
        self._save_btn.clicked.connect(self._on_save)
        angle_layout.addWidget(self._save_btn)

        root.addWidget(angle_group)

        # Initialise auto status and undistort notice
        self._update_auto_status()
        self._update_undistort_notice()

    # ------------------------------------------------------------------
    # Slots
    # ------------------------------------------------------------------

    def _on_compute_auto(self) -> None:
        valid = [
            m for m in self._mandrels
            if m.get("calibrated_pixel_yz") and len(m["calibrated_pixel_yz"]) >= 2
        ]
        if len(valid) < 2:
            self._auto_status.setText(
                f"<span style='color:#e74c3c'>Need at least 2 mandrels with calibrated_pixel_yz "
                f"(found {len(valid)}).</span>"
            )
            return

        # calibrated_pixel_yz = [y_px, z_px]
        # y_px = horizontal position in image; z_px = vertical.
        # We fit z_px as a function of y_px (treat y as x-axis, z as y-axis).
        ys = np.array([m["calibrated_pixel_yz"][0] for m in valid], dtype=float)
        zs = np.array([m["calibrated_pixel_yz"][1] for m in valid], dtype=float)

        if len(valid) == 2:
            slope = (zs[1] - zs[0]) / (ys[1] - ys[0]) if (ys[1] - ys[0]) != 0 else 0.0
        else:
            coeffs = np.polyfit(ys, zs, 1)
            slope = float(coeffs[0])

        # In image coordinates y increases downward.
        # A slope > 0 means the mandrel row goes downward left→right, i.e. the
        # camera is tilted CW → we need a CCW correction (positive angle).
        angle_deg = math.degrees(math.atan(slope))

        self._angle_spin.setValue(angle_deg)
        self._auto_status.setText(
            f"<span style='color:#2ecc71'>Computed from {len(valid)} mandrels: "
            f"<b>{angle_deg:.4f}°</b></span>"
        )

    def _on_load_image(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Load Reference Image",
            "",
            "Images (*.jpg *.jpeg *.png *.bmp *.tif *.tiff)",
        )
        if not path:
            return
        img_bgr = cv2.imread(path)
        if img_bgr is None:
            return

        # Apply lens undistortion if checkerboard calibration is available.
        if self._cal_matrix is not None and self._cal_dist is not None:
            img_bgr = cv2.undistort(img_bgr, self._cal_matrix, self._cal_dist)

        rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        qimg = QImage(rgb.data, w, h, ch * w, QImage.Format.Format_RGB888)
        self._line_view.load_pixmap(QPixmap.fromImage(qimg))

    def _on_line_angle(self, angle_deg: float) -> None:
        self._angle_spin.setValue(angle_deg)

    def _on_save(self) -> None:
        self.rotation_saved.emit(self._angle_spin.value())

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _update_undistort_notice(self) -> None:
        if not hasattr(self, "_undistort_notice"):
            return
        if self._cal_matrix is not None and self._cal_dist is not None:
            self._undistort_notice.setText(
                "<span style='color:#2ecc71'>&#10003; Checkerboard calibration loaded — "
                "images will be undistorted before display.</span>"
            )
        else:
            self._undistort_notice.setText(
                "<span style='color:#f5a623'>&#9888; No checkerboard calibration — "
                "image will be shown as-is (raw from camera). "
                "Complete Camera Calibration first for best accuracy.</span>"
            )

    def _update_auto_status(self) -> None:
        valid = [
            m for m in self._mandrels
            if m.get("calibrated_pixel_yz") and len(m["calibrated_pixel_yz"]) >= 2
        ]
        n = len(valid)
        if n >= 2:
            self._auto_status.setText(
                f"<span style='color:#2ecc71'>{n} mandrel(s) with pixel positions available.</span>"
            )
            self._compute_btn.setEnabled(True)
        else:
            self._auto_status.setText(
                f"<span style='color:#888'>{n} mandrel(s) with calibrated_pixel_yz — "
                "need at least 2 to compute automatically.  Use Manual method below.</span>"
            )
            self._compute_btn.setEnabled(n >= 2)
