"""
ResultsPanel — QTreeWidget showing analysis results at three levels.

Level 1 (top):   step_id  [✓ ok | ⚠ warning | ✗ error]
Level 2:         Tip N    [status]  confidence %
Level 3:         individual detail fields

TipIdentification fields shown:
    tip_present, perceived_tip_type, tip_type_match,
    seating_depth_px, tip_angle_deg, confidence

LiquidMeasurement fields shown:
    total_volume_ul, confidence, slug count, bubble count,
    per-slug breakdown (order, type, volume)
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QPushButton,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from pa_gui.analysis.models import StepDef

# ── Status helpers ─────────────────────────────────────────────────────────────

def _simplify(status: str) -> str:
    """Map detailed status strings to ok / warning / error."""
    if status == "ok":
        return "ok"
    if status in ("warning", "warning_angle", "warning_height", "low_confidence"):
        return "warning"
    return "error"   # missing, unexpected, wrong_type, error, failed, pending


_COLOR: Dict[str, str] = {
    "ok":      "#2ecc71",
    "warning": "#f39c12",
    "error":   "#e74c3c",
    "pending": "#888888",
}
_ICON: Dict[str, str] = {
    "ok":      "✓",
    "warning": "⚠",
    "error":   "✗",
    "pending": "…",
}


def _cat(status: str) -> str:
    return _simplify(status)


def _color(status: str) -> QColor:
    return QColor(_COLOR.get(_cat(status), "#ffffff"))


def _icon(status: str) -> str:
    return _ICON.get(_cat(status), "?")


def _item(label: str, value: str, status: Optional[str] = None) -> QTreeWidgetItem:
    it = QTreeWidgetItem([label, value])
    if status:
        it.setForeground(0, _color(status))
        it.setForeground(1, _color(status))
    return it


# ── Widget ─────────────────────────────────────────────────────────────────────

class ResultsPanel(QWidget):
    # Emitted when the user clicks a top-level step row; carries the StepDef.
    step_selected = Signal(object)   # StepDef

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._steps: List[StepDef] = []
        self._build_ui()

    def populate(self, steps: List[StepDef]) -> None:
        """Replace the entire tree with the current step list."""
        self._steps = list(steps)
        self._tree.clear()
        for step in steps:
            self._tree.addTopLevelItem(self._make_step_item(step))
        self._tree.expandToDepth(1)

    def update_step(self, step: StepDef) -> None:
        """Replace just one top-level step item (used during live run)."""
        for i, s in enumerate(self._steps):
            if s.step_id == step.step_id:
                self._steps[i] = step
                break
        else:
            self._steps.append(step)
        for i in range(self._tree.topLevelItemCount()):
            item = self._tree.topLevelItem(i)
            if item.data(0, Qt.ItemDataRole.UserRole) == step.step_id:
                self._tree.takeTopLevelItem(i)
                new_item = self._make_step_item(step)
                self._tree.insertTopLevelItem(i, new_item)
                new_item.setExpanded(True)
                return
        # Not found — append
        self._tree.addTopLevelItem(self._make_step_item(step))

    def _on_item_clicked(self, item: QTreeWidgetItem, column: int) -> None:
        """Emit step_selected when the user clicks a top-level (step) row."""
        if self._tree.indexOfTopLevelItem(item) >= 0:
            step_id = item.data(0, Qt.ItemDataRole.UserRole)
            for step in self._steps:
                if step.step_id == step_id:
                    self.step_selected.emit(step)
                    break

    # ── UI construction ────────────────────────────────────────────────────────

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(4)

        # Tree widget created first so button slots can reference it.
        self._tree = QTreeWidget()
        self._tree.setColumnCount(2)
        self._tree.setHeaderLabels(["Step / Pipette / Detail", "Value"])
        hdr = self._tree.header()
        hdr.setStretchLastSection(False)
        hdr.setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        hdr.setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        self._tree.setAlternatingRowColors(True)
        self._tree.setUniformRowHeights(True)
        self._tree.itemClicked.connect(self._on_item_clicked)

        # Header row with expand/collapse buttons.
        hdr_row = QHBoxLayout()
        hdr_row.addWidget(QLabel("<b>Results</b>"))
        hdr_row.addStretch()
        btn_exp = QPushButton("Expand all")
        btn_exp.setFixedHeight(24)
        btn_exp.clicked.connect(self._tree.expandAll)
        btn_col = QPushButton("Collapse all")
        btn_col.setFixedHeight(24)
        btn_col.clicked.connect(self._tree.collapseAll)
        hdr_row.addWidget(btn_exp)
        hdr_row.addWidget(btn_col)
        layout.addLayout(hdr_row)
        layout.addWidget(self._tree, 1)

    # ── Item builders ──────────────────────────────────────────────────────────

    def _make_step_item(self, step: StepDef) -> QTreeWidgetItem:
        st = step.status
        top = QTreeWidgetItem([
            f"{_icon(st)}  {step.step_id}",
            st,
        ])
        top.setData(0, Qt.ItemDataRole.UserRole, step.step_id)
        top.setForeground(0, _color(st))
        top.setForeground(1, _color(st))

        if step.error:
            # Show first line in the tree; full traceback in the tooltip.
            first_line = step.error.strip().splitlines()[-1]
            err_item = _item("Error", first_line, "error")
            err_item.setToolTip(1, step.error)
            err_item.setToolTip(0, step.error)
            top.addChild(err_item)
        elif step.result is not None:
            for pip in step.result.get("pipettes", []):
                top.addChild(self._make_pip_item(step.analysis_type, pip))

        return top

    def _make_pip_item(
        self, analysis_type: str, pip: Dict[str, Any]
    ) -> QTreeWidgetItem:
        idx  = pip.get("index", 0)
        conf = pip.get("confidence", 0.0)

        if analysis_type == "TipIdentification":
            seat_st = pip.get("seating_status", "ok")
            label   = f"{_icon(seat_st)}  Tip {idx + 1}"
            value   = f"{seat_st}  ({conf:.0f}%)"
            pip_item = QTreeWidgetItem([label, value])
            pip_item.setForeground(0, _color(seat_st))
            pip_item.setForeground(1, _color(seat_st))

            for key, display in [
                ("tip_present",        "Tip present"),
                ("perceived_tip_type", "Perceived type"),
                ("tip_type_match",     "Type match"),
                ("seating_depth_px",   "Seating depth (px)"),
                ("tip_angle_deg",      "Tip angle (°)"),
                ("confidence",         "Confidence (%)"),
            ]:
                val = pip.get(key)
                if val is not None:
                    pip_item.addChild(_item(display, str(val)))

        else:  # LiquidMeasurement
            total    = pip.get("total_volume_ul")
            vol_str  = f"{total:.1f} µL" if total is not None else "—"
            n_slugs  = len(pip.get("slugs",   []))
            n_bub    = len(pip.get("bubbles", []))
            pip_st   = "ok" if conf >= 70 else "warning"

            label    = f"{_icon(pip_st)}  Tip {idx + 1}"
            value    = f"{vol_str}  ({conf:.0f}%)"
            pip_item = QTreeWidgetItem([label, value])
            pip_item.setForeground(0, _color(pip_st))
            pip_item.setForeground(1, _color(pip_st))

            for display, val in [
                ("Total volume",   vol_str),
                ("Confidence (%)", f"{conf:.0f}"),
                ("Slugs",          str(n_slugs)),
                ("Bubbles",        str(n_bub)),
            ]:
                pip_item.addChild(_item(display, val))

            for slug in pip.get("slugs", []):
                order   = slug.get("order", "?")
                stype   = slug.get("type", "")
                svol    = slug.get("volume_ul", 0.0)
                s_item  = _item(
                    f"  Slug {order}",
                    f"{stype}  {svol:.1f} µL",
                )
                pip_item.addChild(s_item)

        return pip_item
