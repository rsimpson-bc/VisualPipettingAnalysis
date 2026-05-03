"""Parameter grid widget for the Parameter Sweep tab.

Displays one row per IntensityDetection schema parameter with:
  - Parameter name (read-only)
  - Current value (read-only, shown from baseline params)
  - Sweep spec (editable): blank = fixed, "0,5,10" = explicit list,
    "3:15:5" = linspace(3,15,5), "true,false" = boolean list
  - # values (computed live)

Below the table: "Total combinations: N" label (coloured by size) and
estimated time.

Usage
-----
    widget = ParamGridWidget(schema, baseline_params)
    widget.sweep_changed.connect(lambda: print(widget.get_combos()))
    ...
    combos = widget.get_combos()  # list of dicts
"""

from __future__ import annotations

import itertools
import json
import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QFrame, QHBoxLayout, QHeaderView, QLabel,
    QSizePolicy, QTableWidget, QTableWidgetItem,
    QVBoxLayout, QWidget,
)


# ── Sweep spec parser ─────────────────────────────────────────────────────────

def parse_sweep_spec(spec: str, current_value: Any) -> Tuple[List[Any], Optional[str]]:
    """Parse a sweep spec string into a list of values.

    Returns ``(values, error_message)``.
    If *spec* is blank, returns ``([current_value], None)``.
    """
    spec = spec.strip()
    if not spec:
        return [current_value], None

    # Boolean shorthand
    lower = spec.lower()
    if lower in ("true", "false"):
        return [lower == "true"], None
    if "," in spec:
        # Check for boolean list
        parts = [p.strip().lower() for p in spec.split(",")]
        if all(p in ("true", "false") for p in parts):
            return [p == "true" for p in parts], None
        # Numeric list
        try:
            values = [_coerce(p, current_value) for p in parts]
            return values, None
        except Exception as exc:
            return [], f"Bad list: {exc}"

    if ":" in spec:
        # linspace syntax:  start:stop:N
        parts = spec.split(":")
        if len(parts) != 3:
            return [], "linspace format is start:stop:N"
        try:
            start = float(parts[0])
            stop  = float(parts[1])
            n     = int(parts[2])
            if n < 2:
                return [], "linspace N must be ≥ 2"
            arr = np.linspace(start, stop, n)
            values = [_coerce(str(v), current_value) for v in arr]
            return values, None
        except Exception as exc:
            return [], f"Bad linspace: {exc}"

    # Single value
    try:
        return [_coerce(spec, current_value)], None
    except Exception as exc:
        return [], f"Cannot parse '{spec}': {exc}"


def _validate_value(value: Any, prop: Dict[str, Any]) -> Optional[str]:
    """Check a parsed value against schema constraints (enum, min/max).
    Returns an error string, or None if valid.
    """
    enum = prop.get("enum")
    if enum is not None and value not in enum:
        options = ", ".join(str(e) for e in enum)
        return f"'{value}' not in allowed values: {options}"
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        mn = prop.get("minimum")
        if mn is not None and value < mn:
            return f"{value} < minimum {mn}"
        ex_mn = prop.get("exclusiveMinimum")
        if ex_mn is not None and value <= ex_mn:
            return f"{value} \u2264 exclusiveMinimum {ex_mn}"
        mx = prop.get("maximum")
        if mx is not None and value > mx:
            return f"{value} > maximum {mx}"
        ex_mx = prop.get("exclusiveMaximum")
        if ex_mx is not None and value >= ex_mx:
            return f"{value} \u2265 exclusiveMaximum {ex_mx}"
    return None


def _coerce(text: str, reference: Any) -> Any:
    """Cast a string value to match the type of *reference*."""
    text = text.strip()
    if isinstance(reference, bool):
        return text.lower() == "true"
    if isinstance(reference, int):
        v = float(text)
        return int(round(v))
    if isinstance(reference, float):
        return float(text)
    # string (enum)
    return text


# ── Widget ────────────────────────────────────────────────────────────────────

_SKIP_PARAMS = frozenset({"_group_blur", "_group_gradient", "_group_contrast"})


class ParamGridWidget(QWidget):
    """Parameter sweep grid.

    sweep_changed is emitted whenever any sweep spec is edited.
    """

    sweep_changed = Signal()

    def __init__(
        self,
        schema: Dict[str, Any],
        baseline_params: Dict[str, Any],
        parent=None,
    ) -> None:
        super().__init__(parent)
        self._schema   = schema
        self._baseline = dict(baseline_params)
        self._param_order: List[str] = []   # schema param keys, group headers excluded
        self._build_ui()
        self._populate()

    # ── Public API ────────────────────────────────────────────────────────

    def get_combos(self) -> Tuple[List[Dict[str, Any]], Optional[str]]:
        """Return ``(combos, error_message)`` where ``combos`` is the cartesian
        product of all sweep specs.  ``error_message`` is None if all specs
        are valid."""
        param_values: Dict[str, List[Any]] = {}
        for key in self._param_order:
            row = self._row_for_key(key)
            spec_item = self._table.item(row, 2)
            spec = spec_item.text() if spec_item else ""
            current = self._baseline.get(key)
            values, err = parse_sweep_spec(spec, current)
            if err:
                return [], f"{key}: {err}"
            prop = self._schema.get("properties", {}).get(key, {})
            for v in values:
                verr = _validate_value(v, prop)
                if verr:
                    return [], f"{key}: {verr}"
            param_values[key] = values

        keys   = list(param_values.keys())
        combos = [
            dict(zip(keys, combo))
            for combo in itertools.product(*param_values.values())
        ]
        return combos, None

    def get_n_combos(self) -> int:
        n = 1
        for key in self._param_order:
            row  = self._row_for_key(key)
            spec_item = self._table.item(row, 2)
            spec = spec_item.text() if spec_item else ""
            current = self._baseline.get(key)
            values, err = parse_sweep_spec(spec, current)
            if err:
                return 0
            prop = self._schema.get("properties", {}).get(key, {})
            if any(_validate_value(v, prop) for v in values):
                return 0
            n *= len(values)
        return n

    def update_baseline(self, params: Dict[str, Any]) -> None:
        """Refresh the "Current" column after the user edits baseline params."""
        self._baseline = dict(params)
        for key in self._param_order:
            row = self._row_for_key(key)
            val = params.get(key, "")
            item = self._table.item(row, 1)
            if item:
                item.setText(str(val))
        self._refresh_summary()

    def get_specs(self) -> Dict[str, str]:
        """Return a dict of {param_key: sweep_spec_string} for all params."""
        specs: Dict[str, str] = {}
        for key in self._param_order:
            row = self._row_for_key(key)
            item = self._table.item(row, 2)
            specs[key] = item.text() if item else ""
        return specs

    def set_specs(self, specs: Dict[str, str]) -> None:
        """Restore sweep specs from a previously saved dict."""
        self._table.itemChanged.disconnect(self._on_spec_changed)
        for key, spec in specs.items():
            row = self._row_for_key(key)
            if row < 0:
                continue
            item = self._table.item(row, 2)
            if item is not None:
                item.setText(spec)
        self._table.itemChanged.connect(self._on_spec_changed)
        # Refresh all # values and summary (using full validation) after bulk set
        for key in self._param_order:
            row = self._row_for_key(key)
            if row >= 0:
                self._refresh_row(row, key)
        self._refresh_summary()
        self.sweep_changed.emit()

    # ── UI construction ───────────────────────────────────────────────────

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(4)

        root.addWidget(QLabel(
            "<b>Parameter Sweep Grid</b>"
            "  <span style='color:#888;font-size:10px;'>"
            "Blank = fixed; 0,5,10 = list; 1:10:5 = linspace; true,false = bool</span>"
        ))

        self._table = QTableWidget(0, 4)
        self._table.setHorizontalHeaderLabels(
            ["Parameter", "Current", "Sweep Spec", "# Values"]
        )
        hh = self._table.horizontalHeader()
        hh.setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        hh.setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        hh.setSectionResizeMode(2, QHeaderView.ResizeMode.Stretch)
        hh.setSectionResizeMode(3, QHeaderView.ResizeMode.ResizeToContents)
        self._table.verticalHeader().setVisible(False)
        self._table.itemChanged.connect(self._on_spec_changed)
        root.addWidget(self._table, 1)

        sep = QFrame()
        sep.setFrameShape(QFrame.Shape.HLine)
        sep.setStyleSheet("color:#444;")
        root.addWidget(sep)

        self._summary_lbl = QLabel("Total combinations: —")
        self._summary_lbl.setStyleSheet("font-weight:bold;")
        root.addWidget(self._summary_lbl)

        self._time_lbl = QLabel("")
        self._time_lbl.setStyleSheet("color:#888; font-size:10px;")
        root.addWidget(self._time_lbl)

    def _populate(self) -> None:
        props = self._schema.get("properties", {})
        self._table.setRowCount(0)
        self._param_order.clear()
        row = 0
        for key, prop in props.items():
            if key in _SKIP_PARAMS:
                # Group header — render as a section separator row
                self._table.insertRow(row)
                label = prop.get("const", key)
                header_item = QTableWidgetItem(label)
                header_item.setFlags(Qt.ItemFlag.ItemIsEnabled)
                header_item.setBackground(QColor(40, 40, 55))
                header_item.setForeground(QColor(160, 160, 200))
                self._table.setItem(row, 0, header_item)
                for col in (1, 2, 3):
                    filler = QTableWidgetItem("")
                    filler.setFlags(Qt.ItemFlag.ItemIsEnabled)
                    filler.setBackground(QColor(40, 40, 55))
                    self._table.setItem(row, col, filler)
                self._table.setSpan(row, 0, 1, 4)
                row += 1
                continue

            current = self._baseline.get(key, prop.get("default", ""))
            self._param_order.append(key)

            self._table.insertRow(row)

            # Column 0: param name (read-only)
            name_item = QTableWidgetItem(key)
            name_item.setFlags(Qt.ItemFlag.ItemIsEnabled)
            name_item.setToolTip(prop.get("description", ""))
            self._table.setItem(row, 0, name_item)

            # Column 1: current value (read-only)
            cur_item = QTableWidgetItem(str(current))
            cur_item.setFlags(Qt.ItemFlag.ItemIsEnabled)
            cur_item.setForeground(QColor(160, 200, 160))
            enum_vals = prop.get("enum")
            if enum_vals:
                enum_str = ", ".join(str(v) for v in enum_vals)
                cur_item.setToolTip(f"Valid options: {enum_str}")
            self._table.setItem(row, 1, cur_item)

            # Column 2: sweep spec (editable)
            spec_item = QTableWidgetItem("")
            if enum_vals:
                enum_str = ", ".join(str(v) for v in enum_vals)
                spec_item.setToolTip(
                    f"Valid options: {enum_str}\n"
                    f"Enter one or more separated by commas, e.g.: {enum_str}"
                )
            self._table.setItem(row, 2, spec_item)

            # Column 3: # values (read-only)
            n_item = QTableWidgetItem("1")
            n_item.setFlags(Qt.ItemFlag.ItemIsEnabled)
            n_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            self._table.setItem(row, 3, n_item)

            row += 1

        self._refresh_summary()

    # ── Slots / helpers ───────────────────────────────────────────────────

    def _refresh_row(self, row: int, key: str) -> None:
        """Recompute the '# Values' cell for *row* / *key* from the current spec.
        Marks '!' (red) for both parse errors and schema validation errors.
        """
        spec_item = self._table.item(row, 2)
        n_item    = self._table.item(row, 3)
        if spec_item is None or n_item is None:
            return
        spec    = spec_item.text()
        current = self._baseline.get(key)
        values, err = parse_sweep_spec(spec, current)
        if err:
            n_item.setText("!")
            n_item.setForeground(QColor(255, 80, 80))
            n_item.setToolTip(err)
            return
        prop = self._schema.get("properties", {}).get(key, {})
        val_errors = [e for e in (_validate_value(v, prop) for v in values) if e]
        if val_errors:
            n_item.setText("!")
            n_item.setForeground(QColor(255, 80, 80))
            n_item.setToolTip(val_errors[0])
        else:
            n_item.setText(str(len(values)))
            n_item.setForeground(QColor(200, 200, 200))
            n_item.setToolTip("")

    def _on_spec_changed(self, item: QTableWidgetItem) -> None:
        if item.column() != 2:
            return
        name_item = self._table.item(item.row(), 0)
        if name_item is None:
            return
        key = name_item.text()
        if key not in self._baseline:
            return  # group header row
        self._refresh_row(item.row(), key)
        self._refresh_summary()
        self.sweep_changed.emit()

    def _refresh_summary(self) -> None:
        n = self.get_n_combos()
        if n == 0:
            self._summary_lbl.setText("Total combinations: (parse error)")
            self._summary_lbl.setStyleSheet("font-weight:bold; color:#ff6060;")
            self._time_lbl.setText("")
            return

        if n > 5000:
            colour = "#ff6060"
        elif n > 500:
            colour = "#f5c542"
        else:
            colour = "#80e080"
        self._summary_lbl.setText(f"Total combinations: {n:,}")
        self._summary_lbl.setStyleSheet(f"font-weight:bold; color:{colour};")
        n_images = 1  # updated externally if caller knows
        est = n * n_images * 0.25
        self._time_lbl.setText(
            f"Estimated time (1 image, no cache): ~{est:.0f} s"
        )

    def _row_for_key(self, key: str) -> int:
        for r in range(self._table.rowCount()):
            item = self._table.item(r, 0)
            if item and item.text() == key:
                return r
        return -1

    def update_time_estimate(self, n_images: int, n_unique_signal: int) -> None:
        """Update the time estimate label given image count and unique signal runs."""
        n = self.get_n_combos()
        est_signal = n_unique_signal * n_images * 0.25
        self._time_lbl.setText(
            f"Estimated time ({n_images} images, cache optimised): ~{est_signal:.0f} s "
            f"({n:,} combos, {n_unique_signal} unique signal sets)"
        )
