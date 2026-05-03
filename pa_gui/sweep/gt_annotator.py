"""Ground-truth annotator widget for the Parameter Sweep tab.

Layout::

    ┌─ GTAnnotatorWidget ─────────────────────────────────────────────────────┐
    │  Toolbar: [Add] [type ▼] [tip_bottom] [liquid] [bubble] [Del]           │
    │           Tolerance: [10 ▲▼] px                                         │
    │  ──────────────────────────────────────────────────────────────────────  │
    │  ┌─ image list (left 25%) ──┐  ┌─ LineAnnotatorView (right 75%) ──────┐ │
    │  │ img_001.png  3 ann       │  │                                       │ │
    │  │ img_002.png  0 ann   ←   │  │   [full image with coloured H-lines]  │ │
    │  │ img_003.png  1 ann       │  │                                       │ │
    │  └──────────────────────────┘  └───────────────────────────────────────┘ │
    │  ┌─ Annotations for current image ─────────────────────────────────────┐ │
    │  │ Type       Y px   [Delete]                                          │ │
    │  │ tip_bottom  312   [ × ]                                             │ │
    │  │ liquid      480   [ × ]                                             │ │
    │  └─────────────────────────────────────────────────────────────────────┘ │
    └─────────────────────────────────────────────────────────────────────────┘
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional

import cv2
import numpy as np

from PySide6.QtCore import Qt, QRectF, Signal
from PySide6.QtGui import (
    QColor, QFont, QPainter, QPen, QPixmap, QImage, QPolygonF,
)
from PySide6.QtCore import QPointF
from PySide6.QtWidgets import (
    QComboBox, QGraphicsLineItem, QGraphicsPolygonItem, QGraphicsScene,
    QGraphicsTextItem, QGraphicsView, QHBoxLayout, QLabel,
    QListWidget, QListWidgetItem, QPushButton, QSizePolicy,
    QSpinBox, QSplitter, QToolButton, QVBoxLayout, QWidget,
    QTableWidget, QTableWidgetItem, QHeaderView, QAbstractItemView,
    QFrame,
)

from pa_gui.sweep.gt_data import (
    ANN_COLORS, ANN_TYPES, GTAnnotation, ImageGT,
    load_gt, save_gt,
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _qcolor(ann_type: str) -> QColor:
    r, g, b = ANN_COLORS.get(ann_type, (200, 200, 200))
    return QColor(r, g, b)


def _load_pixmap(path: str) -> Optional[QPixmap]:
    img = cv2.imread(path)
    if img is None:
        return None
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w, ch = rgb.shape
    qi = QImage(rgb.data, w, h, ch * w, QImage.Format.Format_RGB888)
    return QPixmap.fromImage(qi)


# ── Line annotator view ───────────────────────────────────────────────────────

class LineAnnotatorView(QGraphicsView):
    """QGraphicsView that displays an image and allows placing / dragging
    horizontal annotation lines.

    Interaction
    -----------
    - Scroll wheel: zoom (anchored to cursor)
    - Left-click on empty image space: place a new annotation at that Y
      (only when self._add_mode is True)
    - Left-click on an existing line: select it (highlights it)
    - Drag a selected line vertically: reposition it
    - Delete key: remove selected line
    """

    annotation_added   = Signal(str, int)   # (type, y_px)
    annotation_moved   = Signal(int, int)   # (old_y_px, new_y_px)  [full-image coords]
    annotation_deleted = Signal(int)         # (y_px)
    annotation_selected = Signal(int)        # (y_px) or -1 for deselect

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._scene = QGraphicsScene(self)
        self.setScene(self._scene)
        self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
        self.setRenderHint(QPainter.RenderHint.Antialiasing)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.ViewportAnchor.AnchorViewCenter)
        self.setBackgroundBrush(QColor(18, 18, 18))
        self.setMouseTracking(True)

        self._user_zoomed        = False
        self._add_mode           = False
        self._add_type           = "liquid"
        self._image_height       = 0
        self._line_items: Dict[int, QGraphicsLineItem] = {}  # y_px → line item
        self._label_items: Dict[int, QGraphicsTextItem] = {} # y_px → label
        self._selected_y: Optional[int] = None
        self._drag_y: Optional[int] = None   # y_px being dragged
        self._drag_offset: float = 0.0       # scene-y - line-y at drag start
        self._annotations: List[GTAnnotation] = []
        # ROI overlay (full-image coordinates)
        self._roi_bbox: Optional[tuple] = None           # (x, y, w, h)
        self._roi_points: Optional[List[tuple]] = None   # polygon [(x,y),...]
        self._roi_item: Optional[QGraphicsPolygonItem] = None

    # ── Public API ────────────────────────────────────────────────────────

    def set_roi(self, roi_bbox: Optional[tuple], roi_points: Optional[List[tuple]]) -> None:
        """Set ROI context. Call before or after set_image — overlay is redrawn either way."""
        self._roi_bbox   = roi_bbox
        self._roi_points = roi_points
        self._draw_roi_overlay()
        if not self._user_zoomed:
            self._fit_to_roi()

    def set_image(self, pixmap: QPixmap, annotations: List[GTAnnotation], keep_zoom: bool = False) -> None:
        # Capture current transform before clearing if we want to preserve it
        saved_transform = self.transform() if keep_zoom and self._user_zoomed else None
        saved_hbar = self.horizontalScrollBar().value() if saved_transform is not None else None
        saved_vbar = self.verticalScrollBar().value()   if saved_transform is not None else None

        self._scene.clear()
        self._line_items.clear()
        self._label_items.clear()
        self._roi_item = None
        self._selected_y = None
        self._drag_y = None
        self._annotations = list(annotations)
        if not keep_zoom:
            self._user_zoomed = False
            self.resetTransform()

        if pixmap and not pixmap.isNull():
            self._scene.addPixmap(pixmap)
            self._scene.setSceneRect(QRectF(pixmap.rect()))
            self._image_height = pixmap.height()
        else:
            self._image_height = 0
            return

        # Draw ROI overlay, then fit (or restore saved transform)
        self._draw_roi_overlay()
        if saved_transform is not None:
            self.setTransform(saved_transform)
            self.horizontalScrollBar().setValue(saved_hbar)
            self.verticalScrollBar().setValue(saved_vbar)
        else:
            self._fit_to_roi()   # zoom to ROI (or full image if no ROI)

        for ann in annotations:
            self._draw_line(ann.y_px, ann.type, selected=False)

    def set_add_mode(self, enabled: bool, ann_type: str = "liquid") -> None:
        self._add_mode = enabled
        self._add_type = ann_type
        if enabled:
            self.setDragMode(QGraphicsView.DragMode.NoDrag)
            self.viewport().setCursor(Qt.CursorShape.CrossCursor)
        else:
            self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
            self.viewport().unsetCursor()

    def reload_annotations(self, annotations: List[GTAnnotation]) -> None:
        """Redraw all lines from an updated annotation list."""
        # Remove old line/label items
        for item in list(self._line_items.values()) + list(self._label_items.values()):
            self._scene.removeItem(item)
        self._line_items.clear()
        self._label_items.clear()
        self._annotations = list(annotations)
        self._selected_y = None
        for ann in annotations:
            self._draw_line(ann.y_px, ann.type, selected=False)

    # ── ROI helpers ───────────────────────────────────────────────────────

    def _draw_roi_overlay(self) -> None:
        """Draw (or redraw) the ROI polygon as a scene overlay."""
        if self._roi_item is not None:
            try:
                self._scene.removeItem(self._roi_item)
            except RuntimeError:
                pass
            self._roi_item = None

        if not self._scene.sceneRect().isValid():
            return

        pts = self._roi_points
        if pts and len(pts) >= 2:
            poly = QPolygonF([QPointF(p[0], p[1]) for p in pts])
            pen  = QPen(QColor(255, 100, 0), 0)   # orange, cosmetic
            pen.setStyle(Qt.PenStyle.DashLine)
            item = QGraphicsPolygonItem(poly)
            item.setPen(pen)
            item.setBrush(Qt.BrushStyle.NoBrush)
            item.setZValue(5)   # below annotation lines (z=10)
            self._scene.addItem(item)
            self._roi_item = item
        elif self._roi_bbox:
            x, y, w, h = self._roi_bbox
            poly = QPolygonF([
                QPointF(x, y), QPointF(x + w, y),
                QPointF(x + w, y + h), QPointF(x, y + h),
            ])
            pen = QPen(QColor(255, 100, 0), 0)
            pen.setStyle(Qt.PenStyle.DashLine)
            item = QGraphicsPolygonItem(poly)
            item.setPen(pen)
            item.setBrush(Qt.BrushStyle.NoBrush)
            item.setZValue(5)
            self._scene.addItem(item)
            self._roi_item = item

    def _fit_to_roi(self) -> None:
        """Fit view to ROI bounding box with 10% padding, or full image if no ROI."""
        if self._roi_bbox:
            x, y, w, h = self._roi_bbox
            pad_x = max(10, int(w * 0.10))
            pad_y = max(10, int(h * 0.10))
            roi_rect = QRectF(x - pad_x, y - pad_y, w + 2 * pad_x, h + 2 * pad_y)
            # Clamp to scene rect
            sr = self._scene.sceneRect()
            roi_rect = roi_rect.intersected(sr) if sr.isValid() else roi_rect
            if roi_rect.isValid():
                self.fitInView(roi_rect, Qt.AspectRatioMode.KeepAspectRatio)
                return
        # Fallback: full image
        if self._scene.sceneRect().isValid():
            self.fitInView(self._scene.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)

    # ── Drawing helpers ───────────────────────────────────────────────────

    def _draw_line(self, y_px: int, ann_type: str, selected: bool) -> None:
        sr = self._scene.sceneRect()
        color = _qcolor(ann_type)
        pen = QPen(color, 0)
        pen.setStyle(Qt.PenStyle.SolidLine if not selected else Qt.PenStyle.DashLine)

        line = QGraphicsLineItem(sr.left(), y_px, sr.right(), y_px)
        line.setPen(pen)
        line.setZValue(10)
        self._scene.addItem(line)
        self._line_items[y_px] = line

        # Label on right edge
        label = QGraphicsTextItem(f"{ann_type}  y={y_px}")
        label.setDefaultTextColor(color)
        font = QFont("Arial", 8)
        label.setFont(font)
        label.setPos(sr.right() - 150, y_px - 14)
        label.setZValue(11)
        self._scene.addItem(label)
        self._label_items[y_px] = label

    def _remove_line_items(self, y_px: int) -> None:
        for store in (self._line_items, self._label_items):
            item = store.pop(y_px, None)
            if item is not None:
                try:
                    self._scene.removeItem(item)
                except RuntimeError:
                    pass

    def _highlight_selected(self, y_px: Optional[int]) -> None:
        """Refresh pen style on all lines to reflect new selection."""
        for ann in self._annotations:
            item = self._line_items.get(ann.y_px)
            if item is None:
                continue
            color = _qcolor(ann.type)
            pen = QPen(color, 0 if ann.y_px != y_px else 1)
            pen.setStyle(
                Qt.PenStyle.DashLine if ann.y_px == y_px else Qt.PenStyle.SolidLine
            )
            item.setPen(pen)

    # ── Qt event overrides ────────────────────────────────────────────────

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if not self._user_zoomed and self._scene.sceneRect().isValid():
            self._fit_to_roi()

    def wheelEvent(self, event):
        self._user_zoomed = True
        factor = 1.15 if event.angleDelta().y() > 0 else 1.0 / 1.15
        self.scale(factor, factor)

    def mousePressEvent(self, event):
        if event.button() != Qt.MouseButton.LeftButton:
            super().mousePressEvent(event)
            return

        scene_pos = self.mapToScene(event.pos())
        y_scene   = scene_pos.y()

        if self._add_mode:
            y_px = int(round(y_scene))
            if 0 <= y_px < max(1, self._image_height):
                self._draw_line(y_px, self._add_type, selected=False)
                self._annotations.append(GTAnnotation(type=self._add_type, y_px=y_px))
                self.annotation_added.emit(self._add_type, y_px)
            return

        # Check if clicking near an existing line (within 5 scene-pixels)
        hit = self._find_nearest_line(y_scene, threshold=5.0)
        if hit is not None:
            self._drag_y      = hit
            self._drag_offset = y_scene - hit
            self._selected_y  = hit
            self._highlight_selected(hit)
            self.annotation_selected.emit(hit)
            self.viewport().setCursor(Qt.CursorShape.SizeVerCursor)
        else:
            if self._selected_y is not None:
                self._selected_y = None
                self._highlight_selected(None)
                self.annotation_selected.emit(-1)
            super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self._drag_y is not None:
            scene_y   = self.mapToScene(event.pos()).y()
            new_y_px  = int(round(scene_y - self._drag_offset))
            new_y_px  = max(0, min(self._image_height - 1, new_y_px))
            old_y_px  = self._drag_y

            # Find the annotation
            ann = next((a for a in self._annotations if a.y_px == old_y_px), None)
            if ann is not None:
                # Remove old graphics items
                self._remove_line_items(old_y_px)
                # Update data
                ann.y_px = new_y_px
                self._drag_y = new_y_px
                self._selected_y = new_y_px
                # Redraw
                self._draw_line(new_y_px, ann.type, selected=True)
                self._highlight_selected(new_y_px)
        else:
            super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if self._drag_y is not None and event.button() == Qt.MouseButton.LeftButton:
            # Emit the final moved position
            old_anns = [a for a in self._annotations if a.y_px == self._drag_y]
            if old_anns:
                self.annotation_moved.emit(self._drag_y, self._drag_y)
            self._drag_y = None
            self.viewport().unsetCursor()
        else:
            super().mouseReleaseEvent(event)

    def keyPressEvent(self, event):
        if event.key() == Qt.Key.Key_Delete and self._selected_y is not None:
            y = self._selected_y
            self._remove_line_items(y)
            self._annotations = [a for a in self._annotations if a.y_px != y]
            self._selected_y = None
            self.annotation_deleted.emit(y)
        else:
            super().keyPressEvent(event)

    # ── Private ───────────────────────────────────────────────────────────

    def _find_nearest_line(self, y_scene: float, threshold: float) -> Optional[int]:
        """Return y_px of the nearest annotation line within threshold scene units."""
        best_y  = None
        best_d  = threshold
        # Convert threshold from scene to view pixels for accurate hit test
        for y_px in self._line_items:
            dist = abs(y_px - y_scene)
            if dist < best_d:
                best_d = dist
                best_y = y_px
        return best_y


# ── Full annotator widget ─────────────────────────────────────────────────────

class GTAnnotatorWidget(QWidget):
    """Full GT annotator with image list, line view, and annotation table.

    Signals
    -------
    gt_changed()
        Emitted after every annotation edit (add, move, delete).
    """

    gt_changed = Signal()

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._primary_folder: Optional[str] = None
        self._image_paths: List[str] = []
        self._image_gts: Dict[str, ImageGT] = {}
        self._tolerance_px: int = 10
        self._current_path: Optional[str] = None

        self._build_ui()

    # ── Public API ────────────────────────────────────────────────────────

    def set_folder(self, primary_folder: str) -> None:
        self._primary_folder = primary_folder
        exts = (".jpg", ".jpeg", ".png", ".bmp")
        self._image_paths = sorted(
            os.path.join(primary_folder, f)
            for f in os.listdir(primary_folder)
            if f.lower().endswith(exts)
        )
        self._image_gts, self._tolerance_px = load_gt(primary_folder)
        self._tol_spin.setValue(self._tolerance_px)
        self._populate_image_list()

    def set_roi_context(
        self,
        ic_path: str,
        pipette_index: int,
        tip_type: str,
    ) -> None:
        """Resolve ROI from instrument config and push it to the view."""
        roi_bbox   = None
        roi_points = None
        if ic_path and os.path.isfile(ic_path):
            try:
                import json
                from pa.analysis.shared_geometry import lookup_roi
                with open(ic_path, "r", encoding="utf-8") as fh:
                    ic = json.load(fh)
                roi_bbox, roi_points = lookup_roi(ic, pipette_index, tip_type or None)
            except Exception:
                pass
        self._view.set_roi(roi_bbox, roi_points)

    def get_image_gts(self) -> Dict[str, ImageGT]:
        return dict(self._image_gts)

    def get_tolerance_px(self) -> int:
        return self._tol_spin.value()

    def get_image_paths(self) -> List[str]:
        return list(self._image_paths)

    # ── UI construction ───────────────────────────────────────────────────

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(4)

        # ── Toolbar ───────────────────────────────────────────────────────
        tb = QHBoxLayout()

        self._add_btn = QPushButton("+ Add line")
        self._add_btn.setCheckable(True)
        self._add_btn.setToolTip(
            "Click to enter Add mode, then click on the image to place a line."
        )
        self._add_btn.toggled.connect(self._on_add_mode_toggled)
        tb.addWidget(self._add_btn)

        self._type_combo = QComboBox()
        for t in ANN_TYPES:
            self._type_combo.addItem(t)
        self._type_combo.setCurrentIndex(1)  # default = liquid
        self._type_combo.currentTextChanged.connect(self._on_type_changed)
        tb.addWidget(self._type_combo)

        tb.addStretch()

        tb.addWidget(QLabel("Tolerance:"))
        self._tol_spin = QSpinBox()
        self._tol_spin.setRange(1, 100)
        self._tol_spin.setValue(10)
        self._tol_spin.setSuffix(" px")
        self._tol_spin.setToolTip(
            "Search radius around each GT line when looking for a signal extremum."
        )
        self._tol_spin.valueChanged.connect(self._on_tolerance_changed)
        tb.addWidget(self._tol_spin)

        root.addLayout(tb)

        # ── Separator ─────────────────────────────────────────────────────
        sep = QFrame()
        sep.setFrameShape(QFrame.Shape.HLine)
        sep.setStyleSheet("color:#444;")
        root.addWidget(sep)

        # ── Main splitter: image list (left) | view + table (right) ──────
        splitter = QSplitter(Qt.Orientation.Horizontal)

        # Left: image list
        left = QWidget()
        left_lay = QVBoxLayout(left)
        left_lay.setContentsMargins(0, 0, 0, 0)
        left_lay.setSpacing(2)
        left_lay.addWidget(QLabel("<b>Images</b>"))
        self._img_list = QListWidget()
        self._img_list.setMinimumWidth(140)
        self._img_list.currentRowChanged.connect(self._on_image_selected)
        left_lay.addWidget(self._img_list, 1)
        splitter.addWidget(left)

        # Right: view stacked above annotation table
        right = QWidget()
        right_lay = QVBoxLayout(right)
        right_lay.setContentsMargins(0, 0, 0, 0)
        right_lay.setSpacing(4)

        self._view = LineAnnotatorView()
        self._view.annotation_added.connect(self._on_annotation_added)
        self._view.annotation_moved.connect(self._on_annotation_moved)
        self._view.annotation_deleted.connect(self._on_annotation_deleted)
        right_lay.addWidget(self._view, 3)

        # Annotation table
        right_lay.addWidget(QLabel("Annotations for this image:"))
        self._ann_table = QTableWidget(0, 3)
        self._ann_table.setHorizontalHeaderLabels(["Type", "Y (px)", ""])
        self._ann_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        self._ann_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        self._ann_table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
        self._ann_table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self._ann_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self._ann_table.verticalHeader().setVisible(False)
        self._ann_table.setFixedHeight(120)
        right_lay.addWidget(self._ann_table, 0)

        splitter.addWidget(right)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 4)

        root.addWidget(splitter, 1)

    # ── Slots ─────────────────────────────────────────────────────────────

    def _on_add_mode_toggled(self, checked: bool) -> None:
        self._view.set_add_mode(checked, self._type_combo.currentText())
        self._add_btn.setText("✓ Adding…" if checked else "+ Add line")

    def _on_type_changed(self, text: str) -> None:
        self._view._add_type = text

    def _on_tolerance_changed(self, value: int) -> None:
        self._tolerance_px = value
        self._autosave()

    def _on_image_selected(self, row: int) -> None:
        if row < 0 or row >= len(self._image_paths):
            return
        path = self._image_paths[row]
        prev_path = self._current_path
        self._current_path = path
        gt = self._image_gts.get(path, ImageGT(image_path=path))
        pm = _load_pixmap(path)
        if pm:
            # Preserve zoom/pan when navigating between images in the same folder
            keep = (prev_path is not None and prev_path != path)
            self._view.set_image(pm, gt.annotations, keep_zoom=keep)
        self._refresh_table(gt.annotations)

    def _on_annotation_added(self, ann_type: str, y_px: int) -> None:
        if self._current_path is None:
            return
        gt = self._image_gts.setdefault(
            self._current_path, ImageGT(image_path=self._current_path)
        )
        gt.annotations.append(GTAnnotation(type=ann_type, y_px=y_px))
        self._refresh_table(gt.annotations)
        self._refresh_image_list_item()
        self._autosave()
        self.gt_changed.emit()

    def _on_annotation_moved(self, _old_y: int, _new_y: int) -> None:
        # The view already updated gt.annotations in place via the shared list
        if self._current_path is None:
            return
        gt = self._image_gts.get(self._current_path)
        if gt:
            self._refresh_table(gt.annotations)
        self._autosave()
        self.gt_changed.emit()

    def _on_annotation_deleted(self, y_px: int) -> None:
        if self._current_path is None:
            return
        gt = self._image_gts.get(self._current_path)
        if gt:
            gt.annotations = [a for a in gt.annotations if a.y_px != y_px]
            self._refresh_table(gt.annotations)
        self._refresh_image_list_item()
        self._autosave()
        self.gt_changed.emit()

    # ── Helpers ───────────────────────────────────────────────────────────

    def _populate_image_list(self) -> None:
        self._img_list.blockSignals(True)
        self._img_list.clear()
        for path in self._image_paths:
            gt = self._image_gts.get(path)
            n = len(gt.annotations) if gt else 0
            item = QListWidgetItem(f"{os.path.basename(path)}  ({n})")
            self._img_list.addItem(item)
        self._img_list.blockSignals(False)
        if self._image_paths:
            self._img_list.setCurrentRow(0)

    def _refresh_image_list_item(self) -> None:
        if self._current_path is None:
            return
        try:
            row = self._image_paths.index(self._current_path)
        except ValueError:
            return
        gt = self._image_gts.get(self._current_path)
        n  = len(gt.annotations) if gt else 0
        item = self._img_list.item(row)
        if item:
            item.setText(f"{os.path.basename(self._current_path)}  ({n})")

    def _refresh_table(self, annotations: List[GTAnnotation]) -> None:
        anns = sorted(annotations, key=lambda a: a.y_px)
        self._ann_table.setRowCount(len(anns))
        for row, ann in enumerate(anns):
            self._ann_table.setItem(row, 0, QTableWidgetItem(ann.type))
            self._ann_table.setItem(row, 1, QTableWidgetItem(str(ann.y_px)))
            del_btn = QToolButton()
            del_btn.setText("×")
            del_btn.setToolTip("Delete this annotation")
            del_btn.clicked.connect(lambda _, y=ann.y_px: self._delete_annotation(y))
            self._ann_table.setCellWidget(row, 2, del_btn)
            # Colour the type cell
            color = _qcolor(ann.type)
            self._ann_table.item(row, 0).setForeground(color)

    def _delete_annotation(self, y_px: int) -> None:
        if self._current_path is None:
            return
        gt = self._image_gts.get(self._current_path)
        if gt:
            gt.annotations = [a for a in gt.annotations if a.y_px != y_px]
            self._view.reload_annotations(gt.annotations)
            self._refresh_table(gt.annotations)
        self._refresh_image_list_item()
        self._autosave()
        self.gt_changed.emit()

    def _autosave(self) -> None:
        if self._primary_folder:
            try:
                save_gt(self._primary_folder, self._image_gts, self._tol_spin.value())
            except Exception:
                pass  # Non-fatal; user will see stale data on next load
