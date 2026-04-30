"""
Editable table showing the current polygon pairs.

Columns: Row | Left Y | Left Z | Right Y | Right Z

Each numeric cell is directly editable — typing a value and pressing Enter
propagates the change back to the canvas via the pair_changed signal.
The Row column is display-only and auto-renumbered.

Signals:
    pair_changed(int, AbsolutePair)  — user edited a cell value
    pair_deleted(int)                — user clicked "Delete Row"
    all_cleared()                    — user clicked "Clear All"
    row_selected(int)                — user selected a row (for canvas highlight)
"""

from __future__ import annotations

from typing import List

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QHBoxLayout,
    QHeaderView,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from pa_gui.roi_calibration.models import AbsolutePair


class PolygonEditor(QWidget):
    pair_changed = Signal(int, object)   # (pair_index, AbsolutePair)
    pair_deleted = Signal(int)           # pair_index
    all_cleared  = Signal()
    row_selected = Signal(int)           # pair_index

    _HEADERS = ("Row", "Left Y", "Left Z", "Right Y", "Right Z")

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._updating = False   # re-entrancy guard

        # ── Table ──────────────────────────────────────────────────────────────
        self._table = QTableWidget(0, len(self._HEADERS), self)
        self._table.setHorizontalHeaderLabels(self._HEADERS)
        self._table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.Stretch)
        self._table.horizontalHeader().setMinimumSectionSize(40)
        self._table.setMinimumWidth(0)
        self._table.verticalHeader().setVisible(False)
        self._table.setSelectionBehavior(
            QTableWidget.SelectionBehavior.SelectRows)
        self._table.setAlternatingRowColors(True)
        self._table.itemChanged.connect(self._on_cell_changed)
        self._table.selectionModel().selectionChanged.connect(
            self._on_selection_changed)

        # ── Buttons ────────────────────────────────────────────────────────────
        btn_row = QHBoxLayout()
        self._del_btn = QPushButton("Delete Row")
        self._clr_btn = QPushButton("Clear All")
        self._del_btn.clicked.connect(self._on_delete_selected)
        self._clr_btn.clicked.connect(self._on_clear_all)
        btn_row.addWidget(self._del_btn)
        btn_row.addWidget(self._clr_btn)
        btn_row.addStretch()

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._table)
        layout.addLayout(btn_row)

    # ── Public API ─────────────────────────────────────────────────────────────

    def set_pairs(self, pairs: List[AbsolutePair]) -> None:
        """Populate the table from a list of AbsolutePairs."""
        self._updating = True
        self._table.setRowCount(len(pairs))
        for i, p in enumerate(pairs):
            self._set_row(i, p)
        self._updating = False

    def update_pair(self, idx: int, pair: AbsolutePair) -> None:
        """Update one row without triggering pair_changed (e.g. from canvas drag)."""
        if idx < 0 or idx >= self._table.rowCount():
            return
        self._updating = True
        self._set_row(idx, pair)
        self._updating = False

    def add_pair(self, pair: AbsolutePair) -> None:
        """Append a new row."""
        self._updating = True
        row = self._table.rowCount()
        self._table.setRowCount(row + 1)
        self._set_row(row, pair)
        self._updating = False

    def remove_pair(self, idx: int) -> None:
        """Remove a row and renumber remaining rows."""
        self._updating = True
        self._table.removeRow(idx)
        self._renumber()
        self._updating = False

    def clear(self) -> None:
        self._updating = True
        self._table.setRowCount(0)
        self._updating = False

    # ── Private helpers ────────────────────────────────────────────────────────

    def _set_row(self, row: int, pair: AbsolutePair) -> None:
        values = (row, pair.left_x, pair.left_y, pair.right_x, pair.right_y)
        for col, val in enumerate(values):
            text = str(row) if col == 0 else f"{val:.1f}"
            item = QTableWidgetItem(text)
            if col == 0:
                item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            self._table.setItem(row, col, item)

    def _renumber(self) -> None:
        for i in range(self._table.rowCount()):
            item = self._table.item(i, 0)
            if item:
                item.setText(str(i))

    def _read_row(self, row: int) -> AbsolutePair:
        def f(col: int) -> float:
            item = self._table.item(row, col)
            try:
                return float(item.text()) if item else 0.0
            except ValueError:
                return 0.0
        return AbsolutePair(
            left_x=f(1), left_y=f(2),
            right_x=f(3), right_y=f(4),
        )

    # ── Slots ──────────────────────────────────────────────────────────────────

    def _on_cell_changed(self, item: QTableWidgetItem) -> None:
        if self._updating:
            return
        col = item.column()
        if col == 0:
            return
        try:
            float(item.text())
        except ValueError:
            return
        row = item.row()
        self.pair_changed.emit(row, self._read_row(row))

    def _on_selection_changed(self) -> None:
        rows = {idx.row() for idx in self._table.selectedIndexes()}
        if rows:
            self.row_selected.emit(min(rows))

    def _on_delete_selected(self) -> None:
        rows = sorted(
            {idx.row() for idx in self._table.selectedIndexes()},
            reverse=True,
        )
        for row in rows:
            self._updating = True
            self._table.removeRow(row)
            self._renumber()
            self._updating = False
            self.pair_deleted.emit(row)

    def _on_clear_all(self) -> None:
        self._updating = True
        self._table.setRowCount(0)
        self._updating = False
        self.all_cleared.emit()
