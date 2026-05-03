"""
Interactive image canvas for ROI polygon drawing and editing.

Interaction model:
    Draw mode ON  (toggle button):
        Left-click 1 → places the FIRST point of a pair (yellow pending dot).
        Left-click 2 → places the SECOND point, completing the pair.
                        The two points are sorted by image_x to determine
                        left vs right automatically (zig-zag friendly).
        Escape        → cancels a pending first click.
        Left-click on an existing handle → drags the handle instead.

    Draw mode OFF:
        Left-click + drag → pans the view.

    Both modes:
        Middle-mouse drag → pan.
        Scroll-wheel      → zoom (anchor = cursor).
        Right-click on handle → context menu (Delete pair).

Signals:
    pair_added(int, AbsolutePair)  — new pair index + data
    pair_moved(int, AbsolutePair)  — existing pair was dragged
    pair_deleted(int)              — pair was removed
"""

from __future__ import annotations

from typing import List, Optional

from PySide6.QtCore import QPointF, QRectF, Qt, Signal
from PySide6.QtGui import (
    QBrush, QColor, QCursor, QImage, QPainter, QPen, QPixmap, QPolygonF,
)
from PySide6.QtWidgets import (
    QGraphicsEllipseItem, QGraphicsItem, QGraphicsLineItem, QGraphicsObject,
    QGraphicsPixmapItem, QGraphicsPolygonItem, QGraphicsScene, QGraphicsView,
    QMenu,
)

from pa_gui.roi_calibration.models import AbsolutePair

# ── Visual constants ───────────────────────────────────────────────────────────
_C_LEFT     = QColor( 30, 170, 255)        # handle: left point  (blue)
_C_RIGHT    = QColor(255, 120,  30)        # handle: right point (orange)
_C_PENDING  = QColor(255, 255,   0)        # first-click dot     (yellow)
_C_POLY_E   = QColor(  0, 220,  60, 220)  # active polygon edge
_C_POLY_F   = QColor(  0, 220,  60,  40)  # active polygon fill
_C_OTHER_E  = QColor(180, 180, 180, 160)  # other-tip polygon edge
_C_OTHER_F  = QColor(180, 180, 180,  25)  # other-tip polygon fill
_C_ANCHOR   = QColor(255,  50, 255)        # anchor cross-hair
_HANDLE_R   = 6.0                          # handle radius (scene px)


# ── Draggable handle ───────────────────────────────────────────────────────────

class PointHandle(QGraphicsObject):
    """
    A draggable circular handle representing one vertex of the polygon.
    Uses QGraphicsObject so it can emit Qt signals.
    """
    position_changed = Signal()   # fired after each drag step

    def __init__(
        self,
        pair_idx: int,
        side: str,          # "left" | "right"
        pos: QPointF,
        radius: float = _HANDLE_R,
    ) -> None:
        super().__init__()
        self.pair_idx = pair_idx
        self.side = side
        self._r = radius
        self.setPos(pos)
        self.setFlags(
            QGraphicsItem.GraphicsItemFlag.ItemIsMovable
            | QGraphicsItem.GraphicsItemFlag.ItemIsSelectable
            | QGraphicsItem.GraphicsItemFlag.ItemSendsGeometryChanges
        )
        self.setAcceptHoverEvents(True)
        self.setZValue(10.0)
        self.setToolTip(f"Pair {pair_idx} — {side}")

    # QGraphicsItem interface
    def boundingRect(self) -> QRectF:
        r = self._r
        return QRectF(-r, -r, 2 * r, 2 * r)

    def paint(self, painter: QPainter, option, widget=None) -> None:
        color = _C_LEFT if self.side == "left" else _C_RIGHT
        painter.setBrush(QBrush(color))
        painter.setPen(QPen(Qt.GlobalColor.black, 1.0))
        painter.drawEllipse(self.boundingRect())

    def itemChange(self, change, value):
        if change == QGraphicsItem.GraphicsItemChange.ItemPositionHasChanged:
            self.position_changed.emit()
        return super().itemChange(change, value)

    def hoverEnterEvent(self, event) -> None:
        self.setCursor(QCursor(Qt.CursorShape.SizeAllCursor))
        super().hoverEnterEvent(event)

    def hoverLeaveEvent(self, event) -> None:
        self.unsetCursor()
        super().hoverLeaveEvent(event)


class PairCenterHandle(QGraphicsObject):
    """
    A draggable diamond handle at the midpoint of a pair.
    Dragging it moves both left and right vertices of that pair together.
    Visible only in Move mode.
    """
    position_delta = Signal(float, float)  # (dx, dy) incremental step per move event

    def __init__(self, pair_idx: int, pos: QPointF, radius: float = _HANDLE_R) -> None:
        super().__init__()
        self.pair_idx = pair_idx
        self._r = radius
        self.setPos(pos)
        self.setFlags(
            QGraphicsItem.GraphicsItemFlag.ItemIsMovable
            | QGraphicsItem.GraphicsItemFlag.ItemIsSelectable
            | QGraphicsItem.GraphicsItemFlag.ItemSendsGeometryChanges
        )
        self.setAcceptHoverEvents(True)
        self.setZValue(12.0)   # above vertex handles (z=10)
        self.setToolTip(f"Pair {pair_idx} \u2014 drag to move entire row")

    def boundingRect(self) -> QRectF:
        r = self._r * 1.5
        return QRectF(-r, -r, 2 * r, 2 * r)

    def paint(self, painter: QPainter, option, widget=None) -> None:
        r = self._r * 1.2
        diamond = QPolygonF([
            QPointF(0,  -r),
            QPointF( r,  0),
            QPointF(0,   r),
            QPointF(-r,  0),
        ])
        painter.setBrush(QBrush(QColor(255, 255, 160, 220)))
        painter.setPen(QPen(QColor(100, 80, 0), 1.2))
        painter.drawPolygon(diamond)

    def itemChange(self, change, value):
        if change == QGraphicsItem.GraphicsItemChange.ItemPositionChange:
            # value = proposed new position; self.pos() = current (old) position
            dx = value.x() - self.pos().x()
            dy = value.y() - self.pos().y()
            if abs(dx) > 1e-9 or abs(dy) > 1e-9:
                self.position_delta.emit(dx, dy)
        return super().itemChange(change, value)

    def hoverEnterEvent(self, event) -> None:
        self.setCursor(QCursor(Qt.CursorShape.SizeAllCursor))
        super().hoverEnterEvent(event)

    def hoverLeaveEvent(self, event) -> None:
        self.unsetCursor()
        super().hoverLeaveEvent(event)


# ── Main canvas ────────────────────────────────────────────────────────────────

class ImageCanvas(QGraphicsView):
    """Zoomable image canvas with interactive ROI polygon overlay."""

    pair_added        = Signal(int, object)   # (pair_index, AbsolutePair)
    pair_moved        = Signal(int, object)   # (pair_index, AbsolutePair)
    pair_deleted      = Signal(int)           # pair_index
    mouse_image_pos   = Signal(float, float)  # (image_x, image_y) while hovering
    mouse_left        = Signal()              # mouse left the canvas area
    other_tip_offset_changed = Signal(int, float, float)  # (array_idx, dx, dy)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._scene = QGraphicsScene(self)
        self.setScene(self._scene)

        # ── Internal state ────────────────────────────────────────────────────
        self._image_item: Optional[QGraphicsPixmapItem] = None
        self._pairs:      List[AbsolutePair] = []
        self._other_tip_pairs: List[List[AbsolutePair]] = []
        self._show_other_tips: bool = True
        self._show_anchor:     bool = True

        # Rendered items (rebuilt on each change)
        self._handles:         List[PointHandle] = []
        self._poly_items:      List[QGraphicsItem] = []
        self._other_tip_items: List[QGraphicsItem] = []
        self._anchor_item:     Optional[QGraphicsEllipseItem] = None

        # Drawing state machine
        self._draw_mode:    bool = False
        self._pending_pt:   Optional[QPointF] = None
        self._pending_dot:  Optional[QGraphicsEllipseItem] = None
        self._rubber_line:  Optional[QGraphicsLineItem] = None

        # Middle-mouse pan
        self._pan_origin: Optional[QPointF] = None  # in viewport coords

        # Move-mode whole-polygon drag
        self._move_mode: bool = False
        self._poly_drag_origin: Optional[QPointF] = None
        self._poly_drag_pairs_snapshot: List[AbsolutePair] = []

        # Per-tip overlay drag (move mode)
        self._other_tip_item_map: dict = {}          # id(item) → array_index
        self._other_tip_drag_idx: Optional[int] = None
        self._other_tip_drag_origin: Optional[QPointF] = None

        # ── View settings ─────────────────────────────────────────────────────
        self.setRenderHint(QPainter.RenderHint.Antialiasing)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.ViewportAnchor.AnchorViewCenter)
        self.setDragMode(QGraphicsView.DragMode.NoDrag)
        self.setMouseTracking(True)

    # ── Public API ─────────────────────────────────────────────────────────────

    def load_image(self, path: str) -> bool:
        """Load an image file.  Tries OpenCV first for broad format support."""
        pixmap = _load_pixmap(path)
        if pixmap is None or pixmap.isNull():
            return False

        self._scene.clear()
        self._handles.clear()
        self._poly_items.clear()
        self._other_tip_items.clear()
        self._anchor_item  = None
        self._pending_pt   = None
        self._pending_dot  = None
        self._rubber_line  = None

        self._image_item = self._scene.addPixmap(pixmap)
        self._scene.setSceneRect(QRectF(pixmap.rect()))
        self.fitInView(self._scene.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)
        self._rebuild_overlay()
        return True

    def set_pairs(self, pairs: List[AbsolutePair]) -> None:
        self._pairs = list(pairs)
        self._rebuild_overlay()

    def get_pairs(self) -> List[AbsolutePair]:
        return list(self._pairs)

    def set_other_tip_overlays(self, tip_pair_lists: List[List[AbsolutePair]]) -> None:
        self._other_tip_pairs = tip_pair_lists
        self._rebuild_other_overlays()

    def set_show_other_tips(self, visible: bool) -> None:
        self._show_other_tips = visible
        self._rebuild_other_overlays()

    def set_show_anchor(self, visible: bool) -> None:
        self._show_anchor = visible
        self._rebuild_anchor()

    def set_draw_mode(self, enabled: bool) -> None:
        if enabled:
            # Disengage move mode when entering draw mode
            self._move_mode = False
            self._poly_drag_origin = None
        self._draw_mode = enabled
        self.setDragMode(QGraphicsView.DragMode.NoDrag)   # always NoDrag; pan = middle
        cursor = Qt.CursorShape.CrossCursor if enabled else Qt.CursorShape.ArrowCursor
        self.viewport().setCursor(QCursor(cursor))
        if not enabled:
            self._cancel_pending()
        self._rebuild_overlay()  # refresh center-handle visibility

    def set_move_mode(self, enabled: bool) -> None:
        """Toggle move mode: drag polygon body to reposition all pairs."""
        if enabled:
            # Disengage draw mode when entering move mode
            self._draw_mode = False
            self._cancel_pending()
        self._move_mode = enabled
        self.setDragMode(QGraphicsView.DragMode.NoDrag)
        cursor = Qt.CursorShape.SizeAllCursor if enabled else Qt.CursorShape.ArrowCursor
        self.viewport().setCursor(QCursor(cursor))
        if not enabled:
            self._poly_drag_origin = None
            self._poly_drag_pairs_snapshot = []
        self._rebuild_overlay()  # show/hide center handles

    def leaveEvent(self, event) -> None:
        self.mouse_left.emit()
        super().leaveEvent(event)

    def delete_pair(self, pair_idx: int) -> None:
        if 0 <= pair_idx < len(self._pairs):
            self._pairs.pop(pair_idx)
            self._rebuild_overlay()
            self.pair_deleted.emit(pair_idx)

    def clear_pairs(self) -> None:
        self._pairs.clear()
        self._rebuild_overlay()

    # ── Mouse / keyboard events ────────────────────────────────────────────────

    def mousePressEvent(self, event) -> None:
        # Middle-mouse → start pan
        if event.button() == Qt.MouseButton.MiddleButton:
            self._pan_origin = event.position()
            self.viewport().setCursor(QCursor(Qt.CursorShape.ClosedHandCursor))
            event.accept()
            return

        if event.button() == Qt.MouseButton.LeftButton and self._move_mode:
            item = self.itemAt(event.pos())
            if isinstance(item, (PointHandle, PairCenterHandle)):
                # Individual handle drag (vertex or whole-pair center)
                super().mousePressEvent(event)
                return
            # Check if the click landed on the active polygon body
            if item is not None and item in self._poly_items:
                self._poly_drag_origin = self.mapToScene(event.pos())
                self._poly_drag_pairs_snapshot = [
                    AbsolutePair(p.left_x, p.left_y, p.right_x, p.right_y)
                    for p in self._pairs
                ]
                self.viewport().setCursor(QCursor(Qt.CursorShape.ClosedHandCursor))
                event.accept()
                return

            # Check other-tip overlay drag
            if item is not None and id(item) in self._other_tip_item_map:
                self._other_tip_drag_idx = self._other_tip_item_map[id(item)]
                self._other_tip_drag_origin = self.mapToScene(event.pos())
                self.viewport().setCursor(QCursor(Qt.CursorShape.ClosedHandCursor))
                event.accept()
                return

        if self._draw_mode and event.button() == Qt.MouseButton.LeftButton:
            # If clicking on an existing handle, let it drag
            item = self.itemAt(event.pos())
            if isinstance(item, PointHandle):
                super().mousePressEvent(event)
                return
            scene_pos = self.mapToScene(event.pos())
            self._process_draw_click(scene_pos)
            return

        if event.button() == Qt.MouseButton.RightButton:
            # Context menu on handles
            item = self.itemAt(event.pos())
            if isinstance(item, (PointHandle, PairCenterHandle)):
                self._show_handle_menu(item, event.globalPosition().toPoint())
                event.accept()
                return

        super().mousePressEvent(event)

    def mouseMoveEvent(self, event) -> None:
        # Emit image-pixel coordinates for position display (always)
        if self._image_item is not None:
            sp = self.mapToScene(event.pos())
            if self._scene.sceneRect().contains(sp):
                self.mouse_image_pos.emit(sp.x(), sp.y())

        # Middle-mouse pan
        if self._pan_origin is not None:
            delta = event.position() - self._pan_origin
            self._pan_origin = event.position()
            self.horizontalScrollBar().setValue(
                self.horizontalScrollBar().value() - int(delta.x()))
            self.verticalScrollBar().setValue(
                self.verticalScrollBar().value() - int(delta.y()))
            event.accept()
            return

        # Whole-polygon drag
        if self._poly_drag_origin is not None and self._poly_drag_pairs_snapshot:
            sp = self.mapToScene(event.pos())
            dx = sp.x() - self._poly_drag_origin.x()
            dy = sp.y() - self._poly_drag_origin.y()
            for pair, orig in zip(self._pairs, self._poly_drag_pairs_snapshot):
                pair.left_x  = orig.left_x  + dx
                pair.left_y  = orig.left_y  + dy
                pair.right_x = orig.right_x + dx
                pair.right_y = orig.right_y + dy
            self._rebuild_overlay()
            for i, pair in enumerate(self._pairs):
                self.pair_moved.emit(i, pair)
            event.accept()
            return

        # Other-tip overlay drag (move mode)
        if self._other_tip_drag_idx is not None and self._other_tip_drag_origin is not None:
            sp = self.mapToScene(event.pos())
            dx = sp.x() - self._other_tip_drag_origin.x()
            dy = sp.y() - self._other_tip_drag_origin.y()
            if self._other_tip_drag_idx < len(self._other_tip_items):
                self._other_tip_items[self._other_tip_drag_idx].setPos(QPointF(dx, dy))
            event.accept()
            return

        # Update rubber-band
        if self._draw_mode and self._rubber_line and self._pending_pt:
            sp = self.mapToScene(event.pos())
            self._rubber_line.setLine(
                self._pending_pt.x(), self._pending_pt.y(),
                sp.x(), sp.y(),
            )

        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event) -> None:
        if event.button() == Qt.MouseButton.MiddleButton:
            self._pan_origin = None
            if self._draw_mode:
                cursor = Qt.CursorShape.CrossCursor
            elif self._move_mode:
                cursor = Qt.CursorShape.SizeAllCursor
            else:
                cursor = Qt.CursorShape.ArrowCursor
            self.viewport().setCursor(QCursor(cursor))
            event.accept()
            return
        if event.button() == Qt.MouseButton.LeftButton and self._poly_drag_origin is not None:
            self._poly_drag_origin = None
            self._poly_drag_pairs_snapshot = []
            self.viewport().setCursor(QCursor(Qt.CursorShape.SizeAllCursor))
            event.accept()
            return
        if event.button() == Qt.MouseButton.LeftButton and self._other_tip_drag_idx is not None:
            sp = self.mapToScene(event.pos())
            dx = sp.x() - self._other_tip_drag_origin.x()
            dy = sp.y() - self._other_tip_drag_origin.y()
            # Reset visual translation — overlay refresh will reposition correctly
            if self._other_tip_drag_idx < len(self._other_tip_items):
                self._other_tip_items[self._other_tip_drag_idx].setPos(QPointF(0, 0))
            tip_idx = self._other_tip_drag_idx
            self._other_tip_drag_idx = None
            self._other_tip_drag_origin = None
            self.viewport().setCursor(QCursor(Qt.CursorShape.SizeAllCursor))
            self.other_tip_offset_changed.emit(tip_idx, dx, dy)
            event.accept()
            return
        super().mouseReleaseEvent(event)

    def wheelEvent(self, event) -> None:
        factor = 1.15 if event.angleDelta().y() > 0 else 1.0 / 1.15
        self.scale(factor, factor)

    def keyPressEvent(self, event) -> None:
        if event.key() == Qt.Key.Key_Escape:
            self._cancel_pending()
        super().keyPressEvent(event)

    # ── Drawing state machine ──────────────────────────────────────────────────

    def _process_draw_click(self, scene_pos: QPointF) -> None:
        if self._pending_pt is None:
            # First click: show yellow dot + rubber-band
            self._pending_pt = scene_pos
            r = _HANDLE_R
            self._pending_dot = self._scene.addEllipse(
                scene_pos.x() - r, scene_pos.y() - r, 2 * r, 2 * r,
                QPen(Qt.GlobalColor.black, 1),
                QBrush(_C_PENDING),
            )
            self._pending_dot.setZValue(20.0)
            self._rubber_line = self._scene.addLine(
                scene_pos.x(), scene_pos.y(),
                scene_pos.x(), scene_pos.y(),
                QPen(_C_PENDING, 1.0, Qt.PenStyle.DashLine),
            )
            self._rubber_line.setZValue(15.0)
        else:
            # Second click: sort the two points left/right and complete pair.
            # Average the vertical (Z) coordinate so both points sit on the
            # same row — each pair should share an identical image-y value.
            p1, p2 = self._pending_pt, scene_pos
            self._cancel_pending()

            left_pt, right_pt = (p1, p2) if p1.x() <= p2.x() else (p2, p1)
            avg_y = (left_pt.y() + right_pt.y()) / 2.0
            pair = AbsolutePair(
                left_x=left_pt.x(),  left_y=avg_y,
                right_x=right_pt.x(), right_y=avg_y,
            )
            self._pairs.append(pair)
            new_idx = len(self._pairs) - 1
            self._rebuild_overlay()
            self.pair_added.emit(new_idx, pair)

    def _cancel_pending(self) -> None:
        if self._pending_dot is not None:
            self._scene.removeItem(self._pending_dot)
            self._pending_dot = None
        if self._rubber_line is not None:
            self._scene.removeItem(self._rubber_line)
            self._rubber_line = None
        self._pending_pt = None

    # ── Context menu ───────────────────────────────────────────────────────────

    def _show_handle_menu(self, handle, global_pos) -> None:
        menu = QMenu(self)
        action = menu.addAction(f"Delete pair {handle.pair_idx}")
        chosen = menu.exec(global_pos)
        if chosen == action:
            self.delete_pair(handle.pair_idx)

    # ── Overlay rebuild ────────────────────────────────────────────────────────

    def _rebuild_overlay(self) -> None:
        """Remove and redraw all editable polygon items and handles."""
        for h in self._handles:
            self._scene.removeItem(h)
        self._handles.clear()
        for it in self._poly_items:
            self._scene.removeItem(it)
        self._poly_items.clear()

        if self._pairs:
            item = _make_polygon_item(self._scene, self._pairs, _C_POLY_E, _C_POLY_F, z=1.0)
            if item:
                self._poly_items.append(item)

            for i, pair in enumerate(self._pairs):
                for side, x, y in (
                    ("left",  pair.left_x,  pair.left_y),
                    ("right", pair.right_x, pair.right_y),
                ):
                    h = PointHandle(i, side, QPointF(x, y))
                    h.position_changed.connect(lambda h=h: self._on_handle_moved(h))
                    self._scene.addItem(h)
                    self._handles.append(h)

                if self._move_mode:
                    cx = (pair.left_x + pair.right_x) / 2.0
                    cy = (pair.left_y + pair.right_y) / 2.0
                    ch = PairCenterHandle(i, QPointF(cx, cy))
                    ch.position_delta.connect(
                        lambda dx, dy, i=i: self._on_center_handle_moved(i, dx, dy))
                    self._scene.addItem(ch)
                    self._handles.append(ch)

        self._rebuild_anchor()

    def _rebuild_other_overlays(self) -> None:
        # Cancel any in-progress other-tip drag
        self._other_tip_drag_idx = None
        self._other_tip_drag_origin = None
        for it in self._other_tip_items:
            self._scene.removeItem(it)
        self._other_tip_items.clear()
        self._other_tip_item_map.clear()
        if not self._show_other_tips:
            return
        for arr_idx, tip_pairs in enumerate(self._other_tip_pairs):
            item = _make_polygon_item(
                self._scene, tip_pairs, _C_OTHER_E, _C_OTHER_F, z=0.5)
            if item:
                self._other_tip_items.append(item)
                self._other_tip_item_map[id(item)] = arr_idx

    def _rebuild_anchor(self) -> None:
        if self._anchor_item is not None:
            self._scene.removeItem(self._anchor_item)
            self._anchor_item = None
        if not self._show_anchor or not self._pairs:
            return
        topmost = min(self._pairs, key=lambda p: p.midpoint_y())
        ax = (topmost.left_x + topmost.right_x) / 2.0
        ay = (topmost.left_y + topmost.right_y) / 2.0
        r = 4.0
        self._anchor_item = self._scene.addEllipse(
            ax - r, ay - r, 2 * r, 2 * r,
            QPen(_C_ANCHOR, 1.5),
            QBrush(QColor(255, 50, 255, 80)),
        )
        self._anchor_item.setZValue(5.0)

    def _redraw_polygon_only(self) -> None:
        """Lightweight redraw: only the polygon fill/outline, not the handles."""
        for it in self._poly_items:
            self._scene.removeItem(it)
        self._poly_items.clear()
        if self._pairs:
            item = _make_polygon_item(
                self._scene, self._pairs, _C_POLY_E, _C_POLY_F, z=1.0)
            if item:
                self._poly_items.append(item)
        self._rebuild_anchor()

    # ── Handle drag callback ───────────────────────────────────────────────────

    def _on_handle_moved(self, handle: PointHandle) -> None:
        idx = handle.pair_idx
        if idx >= len(self._pairs):
            return
        pair = self._pairs[idx]
        pos = handle.pos()
        if handle.side == "left":
            pair.left_x,  pair.left_y  = pos.x(), pos.y()
        else:
            pair.right_x, pair.right_y = pos.x(), pos.y()
        # Keep the center handle for this pair in sync
        for h in self._handles:
            if isinstance(h, PairCenterHandle) and h.pair_idx == idx:
                h.blockSignals(True)
                h.setPos(QPointF(
                    (pair.left_x + pair.right_x) / 2.0,
                    (pair.left_y + pair.right_y) / 2.0,
                ))
                h.blockSignals(False)
                break
        self._redraw_polygon_only()
        self.pair_moved.emit(idx, pair)

    def _on_center_handle_moved(self, idx: int, dx: float, dy: float) -> None:
        """Move both vertices of a pair by the drag delta."""
        if idx >= len(self._pairs):
            return
        pair = self._pairs[idx]
        pair.left_x  += dx;  pair.left_y  += dy
        pair.right_x += dx;  pair.right_y += dy
        # Keep the corresponding vertex handles in sync without re-triggering
        for h in self._handles:
            if isinstance(h, PointHandle) and h.pair_idx == idx:
                h.blockSignals(True)
                if h.side == "left":
                    h.setPos(QPointF(pair.left_x, pair.left_y))
                else:
                    h.setPos(QPointF(pair.right_x, pair.right_y))
                h.blockSignals(False)
        self._redraw_polygon_only()
        self.pair_moved.emit(idx, pair)


# ── Module-level helpers ───────────────────────────────────────────────────────

def _load_pixmap(path: str) -> Optional[QPixmap]:
    """Load an image file, using OpenCV for wide format support."""
    try:
        import cv2
        import numpy as np
        bgr = cv2.imread(path)
        if bgr is not None:
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb.shape
            qimg = QImage(rgb.data.tobytes(), w, h, ch * w,
                          QImage.Format.Format_RGB888)
            return QPixmap.fromImage(qimg)
    except Exception:
        pass
    # Fallback: Qt native loader
    return QPixmap(path)


def _make_polygon_item(
    scene: QGraphicsScene,
    pairs: List[AbsolutePair],
    edge_color: QColor,
    fill_color: QColor,
    z: float = 1.0,
) -> Optional[QGraphicsPolygonItem]:
    """Build a filled polygon from pairs, sorted top-to-bottom."""
    if not pairs:
        return None
    sorted_p = sorted(pairs, key=lambda p: p.midpoint_y())
    # Clockwise wound: left column top→bottom, then right column bottom→top
    pts = (
        [QPointF(p.left_x,  p.left_y)  for p in sorted_p] +
        [QPointF(p.right_x, p.right_y) for p in reversed(sorted_p)]
    )
    if len(pts) < 3:
        return None
    item = scene.addPolygon(
        QPolygonF(pts),
        QPen(edge_color, 1.5),
        QBrush(fill_color),
    )
    item.setZValue(z)
    return item
