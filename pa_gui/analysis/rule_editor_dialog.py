"""Behavior Rules Editor Dialog.

Opens from Mode Compare → right-click a preset → "Edit behavior rules (GUI)…".

Design
------
* Left panel: list of rules (trigger + target label), with Add / Delete /
  Move-Up / Move-Down buttons.
* Right panel: form editor for the selected rule.  Trigger-specific param
  rows appear/hide based on the selected trigger.  Gating groups
  (low_above_*, low_below_*, high_above_*, high_below_*, value_*, pre_low_*)
  are collapsible sections — the section is collapsed when the window_px (or
  value_min_strength) field is zero, and the group is zeroed out when it is
  collapsed.
* Changes are **live**: every edit signals ``rules_changed(list)`` which the
  caller hooks to re-run interpretations in Mode Compare (debounced).
* "Save & Close": writes to disk via ``ab_presets.update_behavior_rules``.
* "Discard & Close": reverts rules to the state when the dialog was opened,
  emits ``rules_changed`` once with the original rules, then closes.
"""

from __future__ import annotations

import copy
from typing import Any, Callable, Dict, List, Optional

from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtWidgets import (
    QCheckBox, QComboBox, QDialog, QDoubleSpinBox, QFormLayout,
    QGroupBox, QHBoxLayout, QLabel, QListWidget, QListWidgetItem,
    QPushButton, QScrollArea, QSizePolicy, QSpinBox,
    QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget,
)

# ---------------------------------------------------------------------------
# Schema data (mirrors rule_optimizer._TRIGGER_PARAMS / _DEFAULTS)
# ---------------------------------------------------------------------------

_TRIGGERS = ["peak", "high_signal", "low_signal", "rising_edge"]
_TARGETS  = ["tip_bottom", "meniscus", "liquid_in_tip", "gas_in_tip", "below_tip"]
_PEAK_SIDES = ["(none)", "top", "bottom"]

# (param_name, lo, hi, is_int)
_TRIGGER_PARAMS: Dict[str, List] = {
    "peak": [
        ("min_prominence_frac",       0.01, 0.95,  False),
        ("smoothing_px",              1.0,  30.0,  True),
        ("spread_px",                 0.5,  50.0,  False),
    ],
    "high_signal": [
        ("threshold_frac",  0.01, 0.95, False),
        ("smoothing_px",    1.0,  30.0, True),
        ("spread_px",       0.5,  50.0, False),
    ],
    "low_signal": [
        ("threshold_frac",  0.01, 0.95, False),
        ("smoothing_px",    1.0,  30.0, True),
        ("spread_px",       0.5,  50.0, False),
    ],
    "rising_edge": [
        ("threshold_frac",  0.01, 0.95,  False),
        ("smoothing_px",    1.0,  30.0,  True),
        ("spread_px",       0.5,  50.0,  False),
    ],
}

# Collapsible gating groups — only present for "peak"
# Each entry: (group_label, enable_key, enable_lo, enable_hi, params)
# enable_key is the window_px param that gates the group.
_PEAK_GATE_GROUPS = [
    ("value gate (signal minimum)",
     "value_min_strength",
     [("value_min_strength", 0.0, 1.0, False)],
     ),
    ("low-above gate  (low signal above peak required)",
     "low_above_window_px",
     [("low_above_window_px",      0,   200, True),
      ("low_above_threshold_frac", 0.01,0.99,False),
      ("low_above_min_strength",   0.0, 1.0, False)],
     ),
    ("low-below gate  (low signal below peak required)",
     "low_below_window_px",
     [("low_below_window_px",      0,   200, True),
      ("low_below_threshold_frac", 0.01,0.99,False),
      ("low_below_min_strength",   0.0, 1.0, False)],
     ),
    ("high-above gate  (high signal above peak required)",
     "high_above_window_px",
     [("high_above_window_px",      0,   200, True),
      ("high_above_threshold_frac", 0.01,0.99,False),
      ("high_above_min_strength",   0.0, 1.0, False)],
     ),
    ("high-below gate  (high signal below peak required)",
     "high_below_window_px",
     [("high_below_window_px",      0,   200, True),
      ("high_below_threshold_frac", 0.01,0.99,False),
      ("high_below_min_strength",   0.0, 1.0, False)],
     ),
]

# For rising_edge: pre-low gate
_RISING_EDGE_GATE_GROUPS = [
    ("pre-low gate  (signal must be low before the crossing)",
     "pre_low_window_px",
     [("pre_low_window_px", 0, 200, True),
      ("pre_low_frac",      0.0, 1.0, False)],
     ),
]

_DEFAULTS: Dict[str, Any] = {
    "trigger":                  "peak",
    "target":                   "tip_bottom",
    "direction":                1,
    "strength":                 1.0,
    "spread_px":                5.0,
    "min_prominence_frac":      0.2,
    "smoothing_px":             3,
    "threshold_frac":           0.3,
    "value_min_strength":       0.0,
    "low_above_window_px":      0,
    "low_above_threshold_frac": 0.3,
    "low_above_min_strength":   0.0,
    "low_below_window_px":       0,
    "low_below_threshold_frac":  0.3,
    "low_below_min_strength":    0.0,
    "high_above_window_px":      0,
    "high_above_threshold_frac": 0.7,
    "high_above_min_strength":   1.0,
    "high_below_window_px":      0,
    "high_below_threshold_frac": 0.7,
    "high_below_min_strength":   1.0,
    "pre_low_window_px":         0,
    "pre_low_frac":              0.5,
    "range_px":                  0,
}


def _default_rule() -> Dict[str, Any]:
    return {
        "trigger":           "peak",
        "target":            "tip_bottom",
        "direction":         1,
        "strength":          1.0,
        "min_prominence_frac": 0.2,
        "smoothing_px":      3,
        "spread_px":         5.0,
    }


def _rule_label(rule: Dict[str, Any]) -> str:
    trig   = rule.get("trigger", "?")
    target = rule.get("target", "?")
    d      = rule.get("direction", 1)
    d_str  = "▲" if d > 0 else "▼"
    return f"{d_str}  {trig}  →  {target}"


# ---------------------------------------------------------------------------
# _CollapseSection — a collapsible QGroupBox with an expand/collapse toggle
# ---------------------------------------------------------------------------

class _CollapseSection(QGroupBox):
    """A QGroupBox with a toggle button that shows/hides its contents."""

    def __init__(self, title: str, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._expanded = False

        self._toggle_btn = QPushButton(f"▶  {title}")
        self._toggle_btn.setFlat(True)
        self._toggle_btn.setStyleSheet("text-align:left; font-weight:bold;")
        self._toggle_btn.clicked.connect(self._toggle)

        self._content = QWidget()
        self._content.setVisible(False)
        self._content_layout = QFormLayout(self._content)
        self._content_layout.setContentsMargins(12, 4, 4, 4)
        self._content_layout.setSpacing(4)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)
        outer.addWidget(self._toggle_btn)
        outer.addWidget(self._content)

        self.setFlat(True)
        self.setStyleSheet("QGroupBox { border: none; }")

    def form_layout(self) -> QFormLayout:
        return self._content_layout

    def set_expanded(self, expanded: bool) -> None:
        self._expanded = expanded
        self._content.setVisible(expanded)
        prefix = "▼" if expanded else "▶"
        text = self._toggle_btn.text()
        # Replace leading arrow
        for arrow in ("▼", "▶"):
            if text.startswith(arrow):
                text = prefix + text[1:]
                break
        self._toggle_btn.setText(text)

    def is_expanded(self) -> bool:
        return self._expanded

    def _toggle(self) -> None:
        self.set_expanded(not self._expanded)


# ---------------------------------------------------------------------------
# _RuleWidget — editor for one rule
# ---------------------------------------------------------------------------

class _RuleWidget(QWidget):
    """Form widget for editing a single behavior rule dict.

    Emits ``changed()`` whenever any field is edited.
    """

    changed = Signal()

    def __init__(self, rule: Dict[str, Any], parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._rule: Dict[str, Any] = {}          # filled by _populate
        self._building = False                   # suppress change signals during build

        # -- Top-level form --------------------------------------------------
        self._outer = QFormLayout(self)
        self._outer.setContentsMargins(8, 8, 8, 8)
        self._outer.setSpacing(6)

        # Trigger
        self._trigger_cb = QComboBox()
        self._trigger_cb.addItems(_TRIGGERS)
        self._outer.addRow("Trigger:", self._trigger_cb)

        # Target
        self._target_cb = QComboBox()
        self._target_cb.addItems(_TARGETS)
        self._outer.addRow("Target:", self._target_cb)

        # Direction
        self._dir_cb = QComboBox()
        self._dir_cb.addItems(["▲  +1  (push up / forward)", "▼  -1  (push down / backward)"])
        self._outer.addRow("Direction:", self._dir_cb)

        # Strength
        self._strength_sb = QDoubleSpinBox()
        self._strength_sb.setRange(0.0, 10.0)
        self._strength_sb.setSingleStep(0.1)
        self._strength_sb.setDecimals(3)
        self._outer.addRow("Strength:", self._strength_sb)

        # peak_side (peak only, categorical)
        self._peak_side_cb = QComboBox()
        self._peak_side_cb.addItems(_PEAK_SIDES)
        self._peak_side_row_label = QLabel("Peak side:")
        self._outer.addRow(self._peak_side_row_label, self._peak_side_cb)

        # ── Trigger-specific numeric params ─────────────────────────────────
        self._trig_param_widgets: Dict[str, QWidget] = {}
        self._trig_param_rows: Dict[str, tuple] = {}  # name → (label_widget, spinbox)

        # pre-create param rows for all triggers (they'll be shown/hidden)
        all_params: Dict[str, tuple] = {}
        for params in _TRIGGER_PARAMS.values():
            for (pname, lo, hi, is_int) in params:
                if pname not in all_params:
                    all_params[pname] = (lo, hi, is_int)

        self._param_labels: Dict[str, QLabel] = {}
        self._param_spins: Dict[str, QWidget] = {}
        for pname, (lo, hi, is_int) in all_params.items():
            lbl = QLabel(_nice_label(pname) + ":")
            if is_int:
                sb: QWidget = QSpinBox()
                sb.setRange(int(lo), int(hi))  # type: ignore[attr-defined]
            else:
                sb = QDoubleSpinBox()
                sb.setRange(lo, hi)           # type: ignore[attr-defined]
                sb.setSingleStep(0.01)        # type: ignore[attr-defined]
                sb.setDecimals(4)             # type: ignore[attr-defined]
            self._param_labels[pname] = lbl
            self._param_spins[pname]  = sb
            self._outer.addRow(lbl, sb)

        # ── Gating groups (collapsible) ──────────────────────────────────────
        self._gate_sections: List[tuple] = []  # (section, enable_key, param_spins)
        for group_label, enable_key, params in _PEAK_GATE_GROUPS + _RISING_EDGE_GATE_GROUPS:
            sec = _CollapseSection(group_label)
            g_spins: Dict[str, QWidget] = {}
            for (pname, lo, hi, is_int) in params:
                lbl2 = QLabel(_nice_label(pname) + ":")
                if is_int:
                    gsb: QWidget = QSpinBox()
                    gsb.setRange(int(lo), int(hi))  # type: ignore[attr-defined]
                else:
                    gsb = QDoubleSpinBox()
                    gsb.setRange(lo, hi)            # type: ignore[attr-defined]
                    gsb.setSingleStep(0.01)         # type: ignore[attr-defined]
                    gsb.setDecimals(4)              # type: ignore[attr-defined]
                sec.form_layout().addRow(lbl2, gsb)
                g_spins[pname] = gsb
                _connect_spinbox(gsb, self._on_any_change)
            sec.set_expanded(False)
            # Collapse toggle also triggers change (zeros the enable_key if collapsing)
            sec._toggle_btn.clicked.connect(self._on_gate_toggled)
            self._gate_sections.append((sec, enable_key, g_spins))
            self._outer.addRow(sec)

        # ── Spatial modifier ─────────────────────────────────────────────────
        self._spatial_chk = QCheckBox("enabled")
        self._spatial_type_cb = QComboBox()
        self._spatial_type_cb.addItems(["offset_below", "offset_above"])
        self._spatial_range_sb = QSpinBox()
        self._spatial_range_sb.setRange(0, 500)
        spatial_w = QWidget()
        spatial_h = QHBoxLayout(spatial_w)
        spatial_h.setContentsMargins(0, 0, 0, 0)
        spatial_h.addWidget(self._spatial_chk)
        spatial_h.addWidget(QLabel("type:"))
        spatial_h.addWidget(self._spatial_type_cb)
        spatial_h.addWidget(QLabel("range_px:"))
        spatial_h.addWidget(self._spatial_range_sb)
        spatial_h.addStretch()
        self._outer.addRow("Spatial modifier:", spatial_w)

        # ── Location prior table ─────────────────────────────────────────────
        self._prior_table = QTableWidget(0, 2)
        self._prior_table.setHorizontalHeaderLabels(["z_frac", "weight"])
        self._prior_table.horizontalHeader().setStretchLastSection(True)
        self._prior_table.setFixedHeight(120)
        prior_btns = QWidget()
        pb_h = QHBoxLayout(prior_btns)
        pb_h.setContentsMargins(0, 0, 0, 0)
        self._prior_add_btn    = QPushButton("+ Add anchor")
        self._prior_remove_btn = QPushButton("− Remove")
        pb_h.addWidget(self._prior_add_btn)
        pb_h.addWidget(self._prior_remove_btn)
        pb_h.addStretch()
        prior_w = QWidget()
        pv = QVBoxLayout(prior_w)
        pv.setContentsMargins(0, 0, 0, 0)
        pv.addWidget(self._prior_table)
        pv.addWidget(prior_btns)
        self._outer.addRow("Location prior:", prior_w)

        # -- Wire signals ----------------------------------------------------
        _connect_spinbox(self._strength_sb, self._on_any_change)
        self._trigger_cb.currentIndexChanged.connect(self._on_trigger_changed)
        self._target_cb.currentIndexChanged.connect(self._on_any_change)
        self._dir_cb.currentIndexChanged.connect(self._on_any_change)
        self._peak_side_cb.currentIndexChanged.connect(self._on_any_change)
        for sb in self._param_spins.values():
            _connect_spinbox(sb, self._on_any_change)
        self._spatial_chk.stateChanged.connect(self._on_any_change)
        self._spatial_type_cb.currentIndexChanged.connect(self._on_any_change)
        _connect_spinbox(self._spatial_range_sb, self._on_any_change)
        self._prior_add_btn.clicked.connect(self._add_prior_row)
        self._prior_remove_btn.clicked.connect(self._remove_prior_row)
        self._prior_table.itemChanged.connect(self._on_any_change)

        # -- Populate --------------------------------------------------------
        self._populate(rule)

    # ------------------------------------------------------------------
    # Population
    # ------------------------------------------------------------------

    def _populate(self, rule: Dict[str, Any]) -> None:
        self._building = True
        try:
            self._rule = copy.deepcopy(rule)

            trig = rule.get("trigger", "peak")
            _set_combo(self._trigger_cb, trig)

            _set_combo(self._target_cb, rule.get("target", "tip_bottom"))

            d = rule.get("direction", 1)
            self._dir_cb.setCurrentIndex(0 if d >= 0 else 1)

            self._strength_sb.setValue(float(rule.get("strength", 1.0)))

            ps = rule.get("peak_side", None)
            if ps:
                _set_combo(self._peak_side_cb, ps)
            else:
                self._peak_side_cb.setCurrentIndex(0)

            # Trigger params
            for pname, sb in self._param_spins.items():
                val = rule.get(pname, _DEFAULTS.get(pname, 0))
                _set_spinbox(sb, val)

            # Gate groups
            for (sec, enable_key, g_spins) in self._gate_sections:
                for pname, sb in g_spins.items():
                    val = rule.get(pname, _DEFAULTS.get(pname, 0))
                    _set_spinbox(sb, val)
                # Expand if the enable_key is non-zero
                en_val = float(rule.get(enable_key, 0))
                sec.set_expanded(en_val != 0.0)

            # Spatial modifier
            sm = rule.get("spatial_modifier")
            has_sm = isinstance(sm, dict)
            self._spatial_chk.setChecked(has_sm)
            if has_sm:
                _set_combo(self._spatial_type_cb, sm.get("type", "offset_below"))
                self._spatial_range_sb.setValue(int(sm.get("range_px", 0)))
            else:
                self._spatial_type_cb.setCurrentIndex(0)
                self._spatial_range_sb.setValue(0)

            # Location prior
            self._prior_table.setRowCount(0)
            prior = rule.get("location_prior") or []
            for pair in prior:
                if len(pair) >= 2:
                    self._append_prior_row(float(pair[0]), float(pair[1]))

            # Show/hide correct param rows
            self._update_visibility(trig)
        finally:
            self._building = False

    # ------------------------------------------------------------------
    # Visibility
    # ------------------------------------------------------------------

    def _update_visibility(self, trig: str) -> None:
        """Show only the param rows applicable to the current trigger."""
        visible_params = {p[0] for p in _TRIGGER_PARAMS.get(trig, [])}
        for pname, lbl in self._param_labels.items():
            visible = pname in visible_params
            lbl.setVisible(visible)
            self._param_spins[pname].setVisible(visible)

        # peak_side only for peak
        is_peak = (trig == "peak")
        self._peak_side_row_label.setVisible(is_peak)
        self._peak_side_cb.setVisible(is_peak)

        # Gate groups: show peak groups for peak, pre_low for rising_edge
        for (sec, enable_key, g_spins) in self._gate_sections:
            # Identify which kind of group this is
            is_peak_gate   = enable_key in ("value_min_strength", "low_above_window_px",
                                             "low_below_window_px",  "high_above_window_px",
                                             "high_below_window_px")
            is_rising_gate = enable_key == "pre_low_window_px"
            if is_peak_gate:
                sec.setVisible(is_peak)
            elif is_rising_gate:
                sec.setVisible(trig == "rising_edge")
            else:
                sec.setVisible(False)

    # ------------------------------------------------------------------
    # Change handling
    # ------------------------------------------------------------------

    def _on_trigger_changed(self) -> None:
        trig = self._trigger_cb.currentText()
        self._update_visibility(trig)
        if not self._building:
            self._on_any_change()

    def _on_gate_toggled(self) -> None:
        """Called after any collapse-section toggle button is clicked.
        If a group is collapsed, zero out its enable_key in the rule."""
        if not self._building:
            self._on_any_change()

    def _on_any_change(self, *_args: Any) -> None:
        if self._building:
            return
        self.changed.emit()

    # ------------------------------------------------------------------
    # Prior table helpers
    # ------------------------------------------------------------------

    def _append_prior_row(self, z_frac: float, weight: float) -> None:
        row = self._prior_table.rowCount()
        self._prior_table.insertRow(row)
        self._prior_table.setItem(row, 0, QTableWidgetItem(f"{z_frac:.3f}"))
        self._prior_table.setItem(row, 1, QTableWidgetItem(f"{weight:.3f}"))

    def _add_prior_row(self) -> None:
        self._append_prior_row(0.0, 1.0)
        self._on_any_change()

    def _remove_prior_row(self) -> None:
        rows = self._prior_table.selectedItems()
        if rows:
            self._prior_table.removeRow(self._prior_table.currentRow())
        elif self._prior_table.rowCount() > 0:
            self._prior_table.removeRow(self._prior_table.rowCount() - 1)
        self._on_any_change()

    # ------------------------------------------------------------------
    # Read back rule dict
    # ------------------------------------------------------------------

    def read_rule(self) -> Dict[str, Any]:
        """Build and return a rule dict from the current form values."""
        trig = self._trigger_cb.currentText()
        rule: Dict[str, Any] = {
            "trigger":   trig,
            "target":    self._target_cb.currentText(),
            "direction": 1 if self._dir_cb.currentIndex() == 0 else -1,
            "strength":  self._strength_sb.value(),
        }

        if trig == "peak":
            ps_idx = self._peak_side_cb.currentIndex()
            if ps_idx > 0:
                rule["peak_side"] = _PEAK_SIDES[ps_idx]

        # Trigger-specific numeric params (visible only)
        visible_params = {p[0] for p in _TRIGGER_PARAMS.get(trig, [])}
        for pname, sb in self._param_spins.items():
            if pname in visible_params:
                rule[pname] = _read_spinbox(sb)

        # Gate groups (if expanded and visible)
        for (sec, enable_key, g_spins) in self._gate_sections:
            if not sec.isVisible():
                continue
            if sec.is_expanded():
                for pname, sb in g_spins.items():
                    rule[pname] = _read_spinbox(sb)
            else:
                # Collapsed → zero out the enable key so the gate is inactive
                rule[enable_key] = 0

        # Spatial modifier
        if self._spatial_chk.isChecked():
            rule["spatial_modifier"] = {
                "type":     self._spatial_type_cb.currentText(),
                "range_px": self._spatial_range_sb.value(),
            }

        # Location prior
        prior = []
        for row in range(self._prior_table.rowCount()):
            try:
                z = float((self._prior_table.item(row, 0) or QTableWidgetItem("0")).text())
                w = float((self._prior_table.item(row, 1) or QTableWidgetItem("0")).text())
                prior.append([z, w])
            except ValueError:
                pass
        if prior:
            rule["location_prior"] = prior

        return rule


# ---------------------------------------------------------------------------
# BehaviorRulesEditorDialog
# ---------------------------------------------------------------------------

class BehaviorRulesEditorDialog(QDialog):
    """Modal dialog for editing the behavior_rules of a preset.

    Parameters
    ----------
    entry:
        The preset dict (must have "id", "description", "behavior_rules").
    on_rules_changed:
        Callable invoked (debounced ~250 ms) with the new list of rule dicts
        each time any field changes — for live preview in Mode Compare.
    parent:
        Parent widget.
    """

    def __init__(
        self,
        entry: Dict[str, Any],
        on_rules_changed: Callable[[List[Dict[str, Any]]], None],
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self._entry          = entry
        self._on_rules_changed = on_rules_changed
        self._original_rules  = copy.deepcopy(entry.get("behavior_rules") or [])
        self._current_rule_widget: Optional[_RuleWidget] = None

        desc = entry.get("description", "")
        self.setWindowTitle(f"Edit behavior rules — {desc}")
        self.resize(760, 600)

        # -- Debounce timer --------------------------------------------------
        self._change_timer = QTimer(self)
        self._change_timer.setSingleShot(True)
        self._change_timer.setInterval(250)
        self._change_timer.timeout.connect(self._emit_rules_changed)

        # -- Main layout: list on left, editor on right ----------------------
        root = QHBoxLayout(self)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(8)

        # ── Left: rule list + buttons ────────────────────────────────────────
        left = QVBoxLayout()
        left.setSpacing(4)

        self._rule_list = QListWidget()
        self._rule_list.setFixedWidth(210)
        self._rule_list.currentRowChanged.connect(self._on_rule_selected)
        left.addWidget(QLabel("Rules:"))
        left.addWidget(self._rule_list, 1)

        btn_row = QHBoxLayout()
        self._add_btn = QPushButton("+ Add")
        self._del_btn = QPushButton("− Del")
        self._up_btn  = QPushButton("↑")
        self._dn_btn  = QPushButton("↓")
        for b in (self._add_btn, self._del_btn, self._up_btn, self._dn_btn):
            btn_row.addWidget(b)
        left.addLayout(btn_row)

        root.addLayout(left)

        # ── Right: scrollable rule editor ────────────────────────────────────
        self._editor_scroll = QScrollArea()
        self._editor_scroll.setWidgetResizable(True)
        self._editor_scroll.setSizePolicy(QSizePolicy.Policy.Expanding,
                                          QSizePolicy.Policy.Expanding)
        self._editor_placeholder = QLabel("Select a rule to edit, or add a new one.")
        self._editor_placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._editor_scroll.setWidget(self._editor_placeholder)
        root.addWidget(self._editor_scroll, 1)

        # ── Bottom buttons ───────────────────────────────────────────────────
        bottom = QHBoxLayout()
        self._discard_btn = QPushButton("Discard & Close")
        self._save_btn    = QPushButton("Save && Close")
        self._save_btn.setDefault(True)
        bottom.addStretch()
        bottom.addWidget(self._discard_btn)
        bottom.addWidget(self._save_btn)

        outer_v = QVBoxLayout()
        outer_v.addLayout(root, 1)
        outer_v.addLayout(bottom)
        # Replace the root QHBoxLayout with a QVBoxLayout wrapper
        # We need to restructure: use a proper outer VBox
        # Redo: clear and use a proper outer layout
        # (Qt doesn't allow changing the layout after construction, so we use a container)

        # Actually restructure using a container widget approach:
        # The root layout is the dialog layout — set to VBox
        self.setLayout(None)  # type: ignore[arg-type]

        outer_vbox = QVBoxLayout(self)
        content_h  = QHBoxLayout()
        content_h.setContentsMargins(0, 0, 0, 0)
        content_h.setSpacing(8)
        content_h.addLayout(left)
        content_h.addWidget(self._editor_scroll, 1)
        outer_vbox.addLayout(content_h, 1)
        outer_vbox.addLayout(bottom)

        # -- Wire buttons ----------------------------------------------------
        self._add_btn.clicked.connect(self._add_rule)
        self._del_btn.clicked.connect(self._delete_rule)
        self._up_btn.clicked.connect(self._move_up)
        self._dn_btn.clicked.connect(self._move_down)
        self._save_btn.clicked.connect(self._save_and_close)
        self._discard_btn.clicked.connect(self._discard_and_close)

        # -- Populate --------------------------------------------------------
        rules = list(entry.get("behavior_rules") or [])
        self._rules: List[Dict[str, Any]] = [copy.deepcopy(r) for r in rules]
        self._rebuild_list(select_row=0)

    # ------------------------------------------------------------------
    # List management
    # ------------------------------------------------------------------

    def _rebuild_list(self, select_row: int = -1) -> None:
        self._rule_list.blockSignals(True)
        self._rule_list.clear()
        for rule in self._rules:
            item = QListWidgetItem(_rule_label(rule))
            self._rule_list.addItem(item)
        self._rule_list.blockSignals(False)
        if self._rule_list.count() > 0:
            row = max(0, min(select_row, self._rule_list.count() - 1))
            self._rule_list.setCurrentRow(row)
        else:
            self._show_placeholder()

    def _show_placeholder(self) -> None:
        self._current_rule_widget = None
        self._editor_scroll.setWidget(self._editor_placeholder)

    def _on_rule_selected(self, row: int) -> None:
        if row < 0 or row >= len(self._rules):
            self._show_placeholder()
            return
        # Flush current editor state back before switching
        self._flush_current()
        # Build new editor
        widget = _RuleWidget(self._rules[row])
        widget.changed.connect(self._on_rule_changed)
        self._current_rule_widget = widget
        self._editor_scroll.setWidget(widget)

    def _flush_current(self) -> None:
        """Commit current editor values back to self._rules."""
        row = self._rule_list.currentRow()
        if self._current_rule_widget is not None and 0 <= row < len(self._rules):
            self._rules[row] = self._current_rule_widget.read_rule()

    def _on_rule_changed(self) -> None:
        """Called on any field change; flush and schedule debounced emit."""
        self._flush_current()
        # Update list label for current rule
        row = self._rule_list.currentRow()
        if 0 <= row < len(self._rules):
            item = self._rule_list.item(row)
            if item:
                item.setText(_rule_label(self._rules[row]))
        self._change_timer.start()

    def _emit_rules_changed(self) -> None:
        self._on_rules_changed(copy.deepcopy(self._rules))

    # ------------------------------------------------------------------
    # Add / delete / reorder
    # ------------------------------------------------------------------

    def _add_rule(self) -> None:
        self._flush_current()
        new_rule = _default_rule()
        self._rules.append(new_rule)
        self._rebuild_list(select_row=len(self._rules) - 1)
        self._change_timer.start()

    def _delete_rule(self) -> None:
        row = self._rule_list.currentRow()
        if 0 <= row < len(self._rules):
            self._rules.pop(row)
            self._current_rule_widget = None
            self._rebuild_list(select_row=max(0, row - 1))
            self._change_timer.start()

    def _move_up(self) -> None:
        self._flush_current()
        row = self._rule_list.currentRow()
        if row > 0:
            self._rules[row - 1], self._rules[row] = self._rules[row], self._rules[row - 1]
            self._rebuild_list(select_row=row - 1)
            self._change_timer.start()

    def _move_down(self) -> None:
        self._flush_current()
        row = self._rule_list.currentRow()
        if row < len(self._rules) - 1:
            self._rules[row + 1], self._rules[row] = self._rules[row], self._rules[row + 1]
            self._rebuild_list(select_row=row + 1)
            self._change_timer.start()

    # ------------------------------------------------------------------
    # Save / discard
    # ------------------------------------------------------------------

    def _save_and_close(self) -> None:
        self._flush_current()
        from pa_gui.analysis import ab_presets
        ab_presets.update_behavior_rules(self._entry["id"], self._rules)
        # Emit one final synchronous update (bypassing debounce)
        self._change_timer.stop()
        self._on_rules_changed(copy.deepcopy(self._rules))
        self.accept()

    def _discard_and_close(self) -> None:
        self._change_timer.stop()
        self._on_rules_changed(copy.deepcopy(self._original_rules))
        self.reject()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _nice_label(name: str) -> str:
    """Convert snake_case param name to a human-readable label."""
    return name.replace("_", " ")


def _set_combo(cb: QComboBox, value: str) -> None:
    idx = cb.findText(value)
    if idx >= 0:
        cb.setCurrentIndex(idx)


def _set_spinbox(sb: QWidget, value: Any) -> None:
    if isinstance(sb, QSpinBox):
        sb.setValue(int(value))
    elif isinstance(sb, QDoubleSpinBox):
        sb.setValue(float(value))


def _read_spinbox(sb: QWidget) -> Any:
    if isinstance(sb, QSpinBox):
        return sb.value()
    elif isinstance(sb, QDoubleSpinBox):
        return sb.value()
    return 0


def _connect_spinbox(sb: QWidget, slot: Callable) -> None:
    if isinstance(sb, QSpinBox):
        sb.valueChanged.connect(slot)
    elif isinstance(sb, QDoubleSpinBox):
        sb.valueChanged.connect(slot)
