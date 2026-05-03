"""
Schema-driven parameter form widget.

Given a JSON Schema dict (from schema_loader) and a current params dict,
renders an appropriate Qt input widget for each property:

    bool        → QCheckBox
    integer     → QSpinBox    (respects minimum / maximum)
    number      → QDoubleSpinBox (respects minimum / maximum)
    string/enum → QComboBox   (choices from "enum") or QLineEdit (free text)
    string key starting with "_group_" → section divider label (separator row)

The ``weight`` pseudo-param (not in the schema but injected at the mode level
in analysis_config.json) is always rendered as a QDoubleSpinBox 0–1 at the
top of the form.

Public API
----------
ParamFormWidget(schema, params, parent=None)
    .get_params() → dict   — current values, ready to write back to JSON
    .set_params(params)    — programmatic update of all fields
    .params_changed        — signal emitted whenever any field changes

Optional A/B diff mode
----------------------
Pass ``reference_params`` to highlight rows whose current value differs from
the reference.  For each changed row the row-label turns gold and a small
"A:<value>" annotation appears to the right of the input widget.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)


class _CollapsibleSection(QWidget):
    """A labelled section that can be expanded/collapsed by clicking its header."""

    def __init__(self, title: str, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # Header button
        self._btn = QPushButton()
        self._btn.setCheckable(True)
        self._btn.setChecked(True)   # expanded by default
        self._btn.setFlat(True)
        self._btn.setStyleSheet(
            "QPushButton { text-align:left; padding:3px 4px;"
            " font-weight:bold; color:#aaa;"
            " border:none; border-bottom:1px solid #555; }"
            "QPushButton:hover { color:#ccc; }"
        )
        self._btn.clicked.connect(self._on_toggle)
        self._set_title(title)
        layout.addWidget(self._btn)

        # Body: holds the QFormLayout
        self._body = QWidget()
        self._body.setContentsMargins(8, 2, 0, 2)
        self._form = QFormLayout(self._body)
        self._form.setLabelAlignment(Qt.AlignmentFlag.AlignRight)
        self._form.setFieldGrowthPolicy(
            QFormLayout.FieldGrowthPolicy.ExpandingFieldsGrow
        )
        self._form.setContentsMargins(0, 2, 0, 2)
        self._form.setSpacing(3)
        layout.addWidget(self._body)

    @property
    def form(self) -> QFormLayout:
        return self._form

    def set_expanded(self, expanded: bool) -> None:
        self._btn.setChecked(expanded)
        self._body.setVisible(expanded)
        self._set_title(self._title)

    def is_expanded(self) -> bool:
        return self._btn.isChecked()

    def _set_title(self, title: str) -> None:
        self._title = title
        arrow = "▼" if self._btn.isChecked() else "▶"
        self._btn.setText(f" {arrow}  {title}")

    def _on_toggle(self, checked: bool) -> None:
        self._body.setVisible(checked)
        self._set_title(self._title)



class ParamFormWidget(QWidget):
    """
    Renders a JSON Schema as an editable form.

    Parameters
    ----------
    schema : dict
        Resolved JSON Schema (properties already merged if allOf was present).
    params : dict
        Current parameter values (will be used as initial widget values).
    show_weight : bool
        When True, a 'weight' field (0–1 float) is prepended to the form.
    weight : float
        Initial weight value.
    """

    params_changed = Signal()

    def __init__(
        self,
        schema: Dict[str, Any],
        params: Dict[str, Any],
        *,
        show_weight: bool = False,
        weight: float = 1.0,
        reference_params: Optional[Dict[str, Any]] = None,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self._schema = schema
        self._widgets: Dict[str, Any] = {}    # param_name → input widget
        self._labels: Dict[str, QLabel] = {}  # param_name → row label
        self._ref_labels: Dict[str, QLabel] = {}  # param_name → "A:val" annotation
        self._reference_params = reference_params
        self._show_weight = show_weight
        self._sections: List[_CollapsibleSection] = []

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)

        inner = QWidget()
        self._root_layout = QVBoxLayout(inner)
        self._root_layout.setContentsMargins(4, 4, 4, 4)
        self._root_layout.setSpacing(2)

        # Top-level form layout for weight + any params before first group
        self._top_form = QFormLayout()
        self._top_form.setLabelAlignment(Qt.AlignmentFlag.AlignRight)
        self._top_form.setFieldGrowthPolicy(
            QFormLayout.FieldGrowthPolicy.ExpandingFieldsGrow
        )
        self._top_form.setContentsMargins(0, 0, 0, 0)
        self._top_form.setSpacing(3)
        self._root_layout.addLayout(self._top_form)

        # _form points to the *active* QFormLayout (top-level or current section)
        self._form = self._top_form

        self._root_layout.addStretch(1)

        scroll.setWidget(inner)

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.addWidget(scroll)

        self._build_form(params, weight)

        if reference_params is not None:
            self.params_changed.connect(self._refresh_ref_annotations)
            self._refresh_ref_annotations()

    # ── Form construction ──────────────────────────────────────────────────────

    def _build_form(self, params: Dict[str, Any], weight: float) -> None:
        if self._show_weight:
            self._add_weight_row(weight)

        props: Dict[str, Any] = self._schema.get("properties", {})
        for key, prop in props.items():
            if key.startswith("_group_"):
                self._add_section_header(prop.get("const", key[7:]))
                continue
            self._add_param_row(key, prop, params.get(key, prop.get("default")))

    def _add_weight_row(self, weight: float) -> None:
        sb = QDoubleSpinBox()
        sb.setRange(0.0, 1.0)
        sb.setDecimals(3)
        sb.setSingleStep(0.05)
        sb.setValue(float(weight))
        sb.setToolTip("Relative weight of this mode in the integrator vote.")
        sb.valueChanged.connect(self.params_changed)
        self._widgets["_weight"] = sb
        lbl = QLabel("<b>weight</b>")
        lbl.setToolTip("Relative weight of this mode in the integrator vote.")
        self._form.addRow(lbl, sb)

    def _add_section_header(self, title: str) -> None:
        text = title.strip("- ").strip()
        # Remove the stretch added in __init__ temporarily, insert section, re-add stretch
        # (stretch is always last; sections are inserted before it)
        stretch_item = self._root_layout.takeAt(self._root_layout.count() - 1)
        section = _CollapsibleSection(text, self)
        self._root_layout.addWidget(section)
        self._root_layout.addItem(stretch_item)
        self._sections.append(section)
        self._form = section.form  # subsequent params go into this section's form

    def _add_separator(self) -> None:
        line = QFrame()
        line.setFrameShape(QFrame.Shape.HLine)
        line.setStyleSheet("color: #aaa;")
        self._form.addRow(line)

    def _add_param_row(self, key: str, prop: Dict[str, Any], value: Any) -> None:
        prop_type = prop.get("type", "string")
        description = prop.get("description", "")
        label = QLabel(key)
        label.setToolTip(description)
        label.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed)
        self._labels[key] = label

        widget = self._make_widget(key, prop, prop_type, value)
        if widget is None:
            return
        widget.setToolTip(description)
        self._widgets[key] = widget

        if self._reference_params is not None:
            ref_lbl = QLabel()
            ref_lbl.setFixedWidth(58)
            ref_lbl.setStyleSheet("color: #888; font-size: 9px;")
            ref_lbl.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
            self._ref_labels[key] = ref_lbl
            row_w = QWidget()
            hbox = QHBoxLayout(row_w)
            hbox.setContentsMargins(0, 0, 0, 0)
            hbox.setSpacing(3)
            hbox.addWidget(widget, 1)
            hbox.addWidget(ref_lbl)
            self._form.addRow(label, row_w)
        else:
            self._form.addRow(label, widget)

    def _make_widget(
        self, key: str, prop: Dict[str, Any], prop_type: str, value: Any
    ) -> Optional[QWidget]:
        # Enum → QComboBox
        if "enum" in prop:
            cb = QComboBox()
            for choice in prop["enum"]:
                cb.addItem(str(choice))
            if value is not None:
                idx = cb.findText(str(value))
                if idx >= 0:
                    cb.setCurrentIndex(idx)
            cb.currentIndexChanged.connect(self.params_changed)
            return cb

        if prop_type == "boolean":
            chk = QCheckBox()
            chk.setChecked(bool(value) if value is not None else prop.get("default", False))
            chk.checkStateChanged.connect(self.params_changed)
            return chk

        if prop_type == "integer":
            sb = QSpinBox()
            sb.setMinimum(int(prop.get("minimum", -99999)))
            sb.setMaximum(int(prop.get("maximum", 99999)))
            sb.setValue(int(value) if value is not None else int(prop.get("default", 0)))
            sb.valueChanged.connect(self.params_changed)
            return sb

        if prop_type == "number":
            sb = QDoubleSpinBox()
            minimum = prop.get("minimum", None)
            maximum = prop.get("maximum", None)
            sb.setMinimum(float(minimum) if minimum is not None else -1e9)
            sb.setMaximum(float(maximum) if maximum is not None else 1e9)
            # Decide step / decimal precision from default magnitude
            default_val = float(value) if value is not None else float(prop.get("default", 0.0))
            if abs(default_val) < 2.0:
                sb.setDecimals(3)
                sb.setSingleStep(0.05)
            else:
                sb.setDecimals(1)
                sb.setSingleStep(1.0)
            sb.setValue(default_val)
            sb.valueChanged.connect(self.params_changed)
            return sb

        if prop_type == "string":
            le = QLineEdit()
            le.setText(str(value) if value is not None else str(prop.get("default", "")))
            le.textChanged.connect(self.params_changed)
            return le

        return None  # Unknown type — skip

    # ── Public API ─────────────────────────────────────────────────────────────

    def get_params(self) -> Dict[str, Any]:
        """Return current widget values as a params dict, ready for JSON serialisation."""
        props = self._schema.get("properties", {})
        out: Dict[str, Any] = {}
        for key, widget in self._widgets.items():
            if key == "_weight":
                continue  # weight is returned separately via get_weight()
            prop = props.get(key, {})
            out[key] = self._read_widget(widget, prop.get("type", "string"))
        return out

    def get_weight(self) -> Optional[float]:
        """Return current weight value, or None if show_weight=False."""
        w = self._widgets.get("_weight")
        if w is None:
            return None
        return float(w.value())

    def set_params(self, params: Dict[str, Any], weight: Optional[float] = None) -> None:
        """Update all widgets from a new params dict. Silently ignores unknown keys."""
        props = self._schema.get("properties", {})
        for key, value in params.items():
            widget = self._widgets.get(key)
            if widget is None:
                continue
            prop = props.get(key, {})
            self._write_widget(widget, prop.get("type", "string"), prop, value)
        if weight is not None:
            w = self._widgets.get("_weight")
            if w is not None:
                w.blockSignals(True)
                w.setValue(float(weight))
                w.blockSignals(False)
        if self._reference_params is not None:
            self._refresh_ref_annotations()

    def reset_to_defaults(self) -> None:
        """Reset all widgets to the default values specified in the schema."""
        props = self._schema.get("properties", {})
        for key, prop in props.items():
            if key.startswith("_group_"):
                continue
            default = prop.get("default")
            if default is None:
                continue
            widget = self._widgets.get(key)
            if widget is None:
                continue
            self._write_widget(widget, prop.get("type", "string"), prop, default)

    # ── Reference annotations ──────────────────────────────────────────────────

    def _refresh_ref_annotations(self) -> None:
        """Update 'A: val' labels and row-label colours for all params."""
        if not self._reference_params:
            return
        props = self._schema.get("properties", {})
        for key, widget in self._widgets.items():
            if key == "_weight":
                continue
            ref_lbl = self._ref_labels.get(key)
            row_lbl = self._labels.get(key)
            if ref_lbl is None:
                continue
            ref_val = self._reference_params.get(key)
            if ref_val is None:
                continue
            prop = props.get(key, {})
            cur_val = self._read_widget(widget, prop.get("type", "string"))
            changed = self._vals_differ(cur_val, ref_val)
            if changed:
                ref_lbl.setText(f"A:{ref_val}")
                if row_lbl:
                    row_lbl.setStyleSheet("color: #f5c842;")
            else:
                ref_lbl.setText("")
                if row_lbl:
                    row_lbl.setStyleSheet("")

    @staticmethod
    def _vals_differ(cur: Any, ref: Any) -> bool:
        """Return True if cur and ref represent meaningfully different values."""
        try:
            return abs(float(cur) - float(ref)) > 1e-9
        except (TypeError, ValueError):
            return str(cur) != str(ref)

    # ── Internal helpers ───────────────────────────────────────────────────────

    @staticmethod
    def _read_widget(widget: QWidget, prop_type: str) -> Any:
        if isinstance(widget, QCheckBox):
            return widget.isChecked()
        if isinstance(widget, QSpinBox):
            return widget.value()
        if isinstance(widget, QDoubleSpinBox):
            return widget.value()
        if isinstance(widget, QComboBox):
            return widget.currentText()
        if isinstance(widget, QLineEdit):
            return widget.text()
        return None

    @staticmethod
    def _write_widget(
        widget: QWidget, prop_type: str, prop: Dict[str, Any], value: Any
    ) -> None:
        widget.blockSignals(True)
        try:
            if isinstance(widget, QCheckBox):
                widget.setChecked(bool(value))
            elif isinstance(widget, QSpinBox):
                widget.setValue(int(value))
            elif isinstance(widget, QDoubleSpinBox):
                widget.setValue(float(value))
            elif isinstance(widget, QComboBox):
                idx = widget.findText(str(value))
                if idx >= 0:
                    widget.setCurrentIndex(idx)
            elif isinstance(widget, QLineEdit):
                widget.setText(str(value))
        finally:
            widget.blockSignals(False)
