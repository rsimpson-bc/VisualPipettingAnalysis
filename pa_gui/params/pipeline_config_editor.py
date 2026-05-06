"""
Pipeline Config Editor widget.

Displays the full contents of an analysis_config.json in an editable tree
structure.  Selecting a pipeline mode or integrator in the tree loads its
schema-driven ParamFormWidget on the right.  Changes can be saved back to
the JSON file.

Layout:
    ┌──────────────────┬────────────────────────────────────────────────┐
    │ Pipelines  [+][-]│  [mode name / integrator]                      │
    │  ▶ full_liquid   │                                                │
    │    IntensityDet… │  [ParamFormWidget for selected mode/integrator] │
    │    LineCont_Term │                                                │
    │    …             │                                                │
    │    [integrator]  │                                                │
    │  ▶ fast_liquid   │                                                │
    │  ▶ full_tip      │                                                │
    ├──────────────────┴────────────────────────────────────────────────┤
    │  [Revert]   [enabled ✓]                          [Save to file]  │
    └────────────────────────────────────────────────────────────────────┘
"""

from __future__ import annotations

import copy
import json
import os
from typing import Any, Dict, List, Optional

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QSplitter,
    QTabWidget,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
    QCheckBox,
)

from pa_gui.params.schema_loader import load_mode_schema, load_integrator_schema
from pa_gui.params.param_form import ParamFormWidget
from pa_gui.params.mode_docs import ModeDocsView


# Tree item roles
_ROLE_KIND      = Qt.ItemDataRole.UserRole       # "mode" | "integrator" | "pipeline"
_ROLE_PIPELINE  = Qt.ItemDataRole.UserRole + 1   # pipeline name string
_ROLE_MODE_IDX  = Qt.ItemDataRole.UserRole + 2   # int index within pipeline.modes list


class PipelineConfigEditor(QWidget):
    """
    Full analysis_config.json editor.

    Signals
    -------
    config_saved(path: str)
        Emitted after the config is successfully written to disk.
    """

    config_saved = Signal(str)

    def __init__(self, config_path: Optional[str] = None, instrument_config_path: Optional[str] = None, parent=None) -> None:
        super().__init__(parent)
        self._config_path: Optional[str] = config_path
        self._instrument_config_path: Optional[str] = instrument_config_path
        self._config: Dict[str, Any] = {}          # live in-memory copy
        self._original: Dict[str, Any] = {}        # snapshot for revert
        self._current_form: Optional[ParamFormWidget] = None
        self._current_item_key: Optional[tuple] = None  # (pipeline_name, mode_idx|"integrator")
        self._ab_dialogs: list = []  # keep references to open A/B compare windows

        self._build_ui()

        if config_path and os.path.isfile(config_path):
            self._load_file(config_path)

    # ── UI construction ────────────────────────────────────────────────────────

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(4)

        # ── Top toolbar: file path ─────────────────────────────────────────
        toolbar = QHBoxLayout()
        toolbar.addWidget(QLabel("Config:"))
        self._path_label = QLabel("<i>no file loaded</i>")
        self._path_label.setWordWrap(False)
        toolbar.addWidget(self._path_label, 1)
        btn_open = QPushButton("Open…")
        btn_open.setFixedWidth(70)
        btn_open.clicked.connect(self._on_open)
        toolbar.addWidget(btn_open)
        root.addLayout(toolbar)

        # ── Main splitter: tree (left) + form (right) ──────────────────────
        splitter = QSplitter(Qt.Orientation.Horizontal)

        self._tree = QTreeWidget()
        self._tree.setHeaderLabel("Pipeline / Mode")
        self._tree.setMinimumWidth(180)
        self._tree.currentItemChanged.connect(self._on_item_changed)
        self._tree.itemDoubleClicked.connect(self._on_item_double_clicked)
        splitter.addWidget(self._tree)

        self._form_container = QWidget()
        self._form_layout = QVBoxLayout(self._form_container)
        self._form_layout.setContentsMargins(4, 0, 4, 0)
        self._form_layout.setSpacing(4)
        self._form_header = QLabel()
        self._form_header.setWordWrap(False)
        self._form_header.setFixedHeight(20)
        self._form_layout.addWidget(self._form_header)

        # Tabs: Parameters | How it works
        self._tabs = QTabWidget()

        # Parameters tab — hosts placeholder + ParamFormWidget
        self._params_tab = QWidget()
        self._params_tab_layout = QVBoxLayout(self._params_tab)
        self._params_tab_layout.setContentsMargins(0, 0, 0, 0)
        self._params_tab_layout.setSpacing(4)
        self._form_placeholder = QLabel(
            "<i>Select a mode or integrator to edit its parameters.</i>"
        )
        self._form_placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._params_tab_layout.addWidget(self._form_placeholder, 1)
        self._tabs.addTab(self._params_tab, "Parameters")

        # How-it-works tab — markdown viewer
        self._docs_view = ModeDocsView()
        self._tabs.addTab(self._docs_view, "How it works")

        self._form_layout.addWidget(self._tabs, 1)
        splitter.addWidget(self._form_container)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 3)
        root.addWidget(splitter, 1)

        # ── Bottom toolbar: enabled toggle + revert + save ─────────────────
        bottom = QHBoxLayout()
        self._enabled_chk = QCheckBox("Mode enabled")
        self._enabled_chk.setVisible(False)
        self._enabled_chk.checkStateChanged.connect(self._on_enabled_changed)
        bottom.addWidget(self._enabled_chk)
        bottom.addStretch()
        self._ab_btn = QPushButton("A/B Compare…")
        self._ab_btn.setToolTip("Compare this mode's current parameters against modified values side by side.")
        self._ab_btn.setEnabled(False)
        self._ab_btn.setVisible(False)
        self._ab_btn.clicked.connect(self._on_open_ab_compare)
        bottom.addWidget(self._ab_btn)
        self._mode_compare_btn = QPushButton("Mode Compare…")
        self._mode_compare_btn.setToolTip("Compare multiple saved presets for this mode side by side.")
        self._mode_compare_btn.setEnabled(False)
        self._mode_compare_btn.setVisible(False)
        self._mode_compare_btn.clicked.connect(self._on_open_mode_compare)
        bottom.addWidget(self._mode_compare_btn)
        btn_revert = QPushButton("Revert")
        btn_revert.setToolTip("Discard unsaved changes and reload from file.")
        btn_revert.clicked.connect(self._on_revert)
        bottom.addWidget(btn_revert)
        btn_save = QPushButton("Save to file")
        btn_save.setStyleSheet(
            "QPushButton { background:#2a5f9e; color:white; font-weight:bold; }"
            "QPushButton:disabled { background:#444; color:#888; }"
        )
        btn_save.clicked.connect(self._on_save)
        bottom.addWidget(btn_save)
        root.addLayout(bottom)

    # ── File I/O ───────────────────────────────────────────────────────────────

    def _on_open(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Open analysis_config.json",
            self._config_path or "",
            "JSON files (*.json);;All files (*)",
        )
        if path:
            self._load_file(path)

    def _load_file(self, path: str) -> None:
        try:
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
        except Exception as exc:
            QMessageBox.critical(self, "Load error", str(exc))
            return
        self._config_path = path
        self._config = data
        self._original = copy.deepcopy(data)
        self._path_label.setText(os.path.basename(path))
        self._path_label.setToolTip(path)
        self._populate_tree()

    def _on_save(self) -> None:
        if not self._config_path:
            path, _ = QFileDialog.getSaveFileName(
                self, "Save analysis_config.json",
                "", "JSON files (*.json)"
            )
            if not path:
                return
            self._config_path = path
        # Commit current form values before saving
        self._commit_current_form()
        try:
            with open(self._config_path, "w", encoding="utf-8") as f:
                json.dump(self._config, f, indent=2)
        except Exception as exc:
            QMessageBox.critical(self, "Save error", str(exc))
            return
        self._original = copy.deepcopy(self._config)
        self.config_saved.emit(self._config_path)

    def _on_revert(self) -> None:
        if QMessageBox.question(
            self, "Revert changes",
            "Discard all unsaved changes and reload from file?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        ) == QMessageBox.StandardButton.Yes:
            self._config = copy.deepcopy(self._original)
            self._populate_tree()

    # ── Tree ───────────────────────────────────────────────────────────────────

    def _populate_tree(self) -> None:
        self._tree.blockSignals(True)
        self._tree.clear()
        pipelines: Dict[str, Any] = self._config.get("pipelines", {})
        for pipeline_name, pipeline in pipelines.items():
            p_item = QTreeWidgetItem([pipeline_name])
            p_item.setData(0, _ROLE_KIND, "pipeline")
            p_item.setData(0, _ROLE_PIPELINE, pipeline_name)
            p_item.setFont(0, _bold_font(p_item.font(0)))
            for idx, mode_cfg in enumerate(pipeline.get("modes", [])):
                m_item = QTreeWidgetItem([mode_cfg.get("mode_name", f"mode_{idx}")])
                enabled = mode_cfg.get("enabled", True)
                if not enabled:
                    m_item.setForeground(0, _grey_brush())
                m_item.setData(0, _ROLE_KIND, "mode")
                m_item.setData(0, _ROLE_PIPELINE, pipeline_name)
                m_item.setData(0, _ROLE_MODE_IDX, idx)
                p_item.addChild(m_item)
            i_item = QTreeWidgetItem(["[integrator]"])
            i_item.setData(0, _ROLE_KIND, "integrator")
            i_item.setData(0, _ROLE_PIPELINE, pipeline_name)
            i_item.setForeground(0, _grey_brush())
            p_item.addChild(i_item)
            self._tree.addTopLevelItem(p_item)
            p_item.setExpanded(True)
        self._tree.blockSignals(False)
        # Clear form
        self._clear_form()
        self._current_item_key = None

    def _on_item_changed(
        self, current: QTreeWidgetItem, previous: QTreeWidgetItem
    ) -> None:
        if current is None:
            return
        # Commit any pending edits in the previous form before switching
        self._commit_current_form()
        kind = current.data(0, _ROLE_KIND)
        if kind == "pipeline":
            self._clear_form()
            return
        pipeline_name = current.data(0, _ROLE_PIPELINE)
        if kind == "mode":
            idx = current.data(0, _ROLE_MODE_IDX)
            self._load_mode_form(pipeline_name, idx, current)
        elif kind == "integrator":
            self._load_integrator_form(pipeline_name, current)

    def _load_mode_form(
        self, pipeline_name: str, idx: int, item: QTreeWidgetItem
    ) -> None:
        pipeline = self._config["pipelines"][pipeline_name]
        mode_cfg = pipeline["modes"][idx]
        mode_name = mode_cfg.get("mode_name", "")
        schema = load_mode_schema(mode_name)

        self._clear_form()
        self._form_placeholder.setVisible(False)
        self._current_item_key = (pipeline_name, idx)

        # Determine analysis type for the integrator context (cosmetic only)
        self._form_header.setText(f"<b>{mode_name}</b>  —  {pipeline_name}")
        self._form_header.setToolTip(
            schema.get("description", "") if schema else ""
        )

        self._enabled_chk.setVisible(True)
        self._enabled_chk.blockSignals(True)
        self._enabled_chk.setChecked(mode_cfg.get("enabled", True))
        self._enabled_chk.blockSignals(False)

        self._ab_btn.setVisible(True)
        self._ab_btn.setEnabled(True)
        self._mode_compare_btn.setVisible(True)
        self._mode_compare_btn.setEnabled(True)

        # Update the docs tab to match the selected mode
        self._docs_view.show_mode(mode_name)

        if schema is None:
            self._params_tab_layout.addWidget(
                QLabel(f"<i>No schema registered for mode: {mode_name}</i>"), 1
            )
            return

        form = ParamFormWidget(
            schema,
            mode_cfg.get("params", {}),
            show_weight=True,
            weight=mode_cfg.get("weight", 1.0),
        )
        form.params_changed.connect(lambda: self._on_form_changed(pipeline_name, idx))
        self._current_form = form
        self._params_tab_layout.addWidget(form, 1)

    def _load_integrator_form(self, pipeline_name: str, item: QTreeWidgetItem) -> None:
        pipeline = self._config["pipelines"][pipeline_name]

        # Detect analysis type from mode names
        modes = pipeline.get("modes", [])
        analysis_type = _guess_analysis_type(modes)
        schema = load_integrator_schema(analysis_type)

        self._clear_form()
        self._form_placeholder.setVisible(False)
        self._current_item_key = (pipeline_name, "integrator")

        self._form_header.setText(f"<b>Integrator</b>  —  {pipeline_name}")
        self._form_header.setToolTip(
            schema.get("description", "") if schema else ""
        )
        self._enabled_chk.setVisible(False)

        # Integrators don't have per-mode docs; clear the tab.
        self._docs_view.show_mode("")

        if schema is None:
            self._params_tab_layout.addWidget(
                QLabel(f"<i>No integrator schema for analysis type: {analysis_type}</i>"), 1
            )
            return

        form = ParamFormWidget(
            schema,
            pipeline.get("integrator_params", {}),
            show_weight=False,
        )
        form.params_changed.connect(
            lambda: self._on_integrator_form_changed(pipeline_name)
        )
        self._current_form = form
        self._params_tab_layout.addWidget(form, 1)

    # ── Form helpers ───────────────────────────────────────────────────────────

    def _clear_form(self) -> None:
        """Remove the current ParamFormWidget (or no-schema label) from the
        Parameters tab and re-show the placeholder. Leaves the docs tab
        contents alone — callers update it explicitly."""
        self._current_form = None
        self._enabled_chk.setVisible(False)
        self._ab_btn.setVisible(False)
        self._ab_btn.setEnabled(False)
        self._mode_compare_btn.setVisible(False)
        self._mode_compare_btn.setEnabled(False)
        # Remove every widget in the Parameters tab except the placeholder.
        for i in reversed(range(self._params_tab_layout.count())):
            item = self._params_tab_layout.itemAt(i)
            w = item.widget() if item else None
            if w is not None and w is not self._form_placeholder:
                self._params_tab_layout.takeAt(i)
                w.deleteLater()
        self._form_placeholder.setVisible(True)

    def _commit_current_form(self) -> None:
        """Write the current form's values back into self._config."""
        if self._current_form is None or self._current_item_key is None:
            return
        pipeline_name, key = self._current_item_key
        pipeline = self._config["pipelines"].get(pipeline_name)
        if pipeline is None:
            return
        if key == "integrator":
            pipeline["integrator_params"] = self._current_form.get_params()
        else:
            idx = int(key)
            mode_cfg = pipeline["modes"][idx]
            mode_cfg["params"] = self._current_form.get_params()
            w = self._current_form.get_weight()
            if w is not None:
                mode_cfg["weight"] = round(w, 4)

    def _on_form_changed(self, pipeline_name: str, idx: int) -> None:
        """Live update when a field changes (doesn't auto-save, just marks dirty)."""
        pass  # Commit happens on item switch or explicit save.

    def _on_integrator_form_changed(self, pipeline_name: str) -> None:
        pass

    def _on_open_ab_compare(self) -> None:
        """Open a non-modal A/B comparison window for the currently selected mode."""
        if self._current_item_key is None:
            return
        pipeline_name, key = self._current_item_key
        if key == "integrator":
            return
        self._commit_current_form()
        mode_cfg  = self._config["pipelines"][pipeline_name]["modes"][int(key)]
        mode_name = mode_cfg.get("mode_name", "")
        params    = copy.deepcopy(mode_cfg.get("params", {}))
        schema    = load_mode_schema(mode_name)
        if schema is None:
            from PySide6.QtWidgets import QMessageBox
            QMessageBox.warning(
                self, "No schema",
                f"No schema registered for mode '{mode_name}'.\n"
                "Cannot open A/B comparison without a schema.",
            )
            return
        from pa_gui.analysis.ab_compare_dialog import ABCompareDialog
        dlg = ABCompareDialog(
            mode_name, schema, params,
            instrument_config_path=self._instrument_config_path or "",
            parent=self,
        )
        # Capture pipeline_name / mode_idx for the apply callback.
        _pipeline_name = pipeline_name
        _mode_idx = int(key)
        dlg.params_applied.connect(
            lambda new_params, pn=_pipeline_name, mi=_mode_idx:
                self._apply_ab_params(pn, mi, new_params)
        )
        dlg.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)
        dlg.destroyed.connect(
            lambda obj=dlg: self._ab_dialogs.remove(obj)
            if obj in self._ab_dialogs else None
        )
        self._ab_dialogs.append(dlg)
        dlg.show()

    def _on_open_mode_compare(self) -> None:
        """Open a non-modal Mode Compare window for the currently selected mode."""
        if self._current_item_key is None:
            return
        pipeline_name, key = self._current_item_key
        if key == "integrator":
            return
        self._commit_current_form()
        mode_cfg  = self._config["pipelines"][pipeline_name]["modes"][int(key)]
        mode_name = mode_cfg.get("mode_name", "")
        from pa_gui.analysis.mode_compare_dialog import ModeCompareDialog
        dlg = ModeCompareDialog(
            mode_name,
            instrument_config_path=self._instrument_config_path or "",
            parent=self,
        )
        dlg.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)
        dlg.show()

    def _apply_ab_params(
        self, pipeline_name: str, mode_idx: int, new_params: dict
    ) -> None:
        """
        Write B params back into the config, then refresh the form if this
        mode is currently selected.
        """
        pipeline = self._config["pipelines"].get(pipeline_name)
        if pipeline is None:
            return
        pipeline["modes"][mode_idx]["params"] = new_params

        # If the user is currently viewing this same mode, reload the form so
        # the widgets reflect the new values immediately.
        if (self._current_item_key is not None
                and self._current_item_key == (pipeline_name, mode_idx)):
            self._current_form.set_params(new_params)

    def _on_item_double_clicked(self, item: QTreeWidgetItem, column: int) -> None:
        """Double-clicking a mode row toggles its enabled state."""
        if item is None:
            return
        if item.data(0, _ROLE_KIND) != "mode":
            return
        pipeline_name = item.data(0, _ROLE_PIPELINE)
        idx = item.data(0, _ROLE_MODE_IDX)
        pipeline = self._config["pipelines"].get(pipeline_name)
        if pipeline is None:
            return
        mode_cfg = pipeline["modes"][int(idx)]
        new_enabled = not mode_cfg.get("enabled", True)
        mode_cfg["enabled"] = new_enabled
        item.setForeground(0, _grey_brush() if not new_enabled else _default_brush())
        # Sync the checkbox if this is also the currently selected item
        if (self._current_item_key is not None
                and self._current_item_key == (pipeline_name, idx)):
            self._enabled_chk.blockSignals(True)
            self._enabled_chk.setChecked(new_enabled)
            self._enabled_chk.blockSignals(False)

    def _on_enabled_changed(self, state) -> None:
        if self._current_item_key is None:
            return
        pipeline_name, idx = self._current_item_key
        if idx == "integrator":
            return
        pipeline = self._config["pipelines"].get(pipeline_name)
        if pipeline is None:
            return
        mode_cfg = pipeline["modes"][int(idx)]
        enabled = self._enabled_chk.isChecked()
        mode_cfg["enabled"] = enabled
        # Update tree item colour
        item = self._find_tree_item(pipeline_name, int(idx))
        if item:
            item.setForeground(0, _grey_brush() if not enabled else _default_brush())

    # ── Tree lookup ────────────────────────────────────────────────────────────

    def _find_tree_item(
        self, pipeline_name: str, mode_idx: int
    ) -> Optional[QTreeWidgetItem]:
        for i in range(self._tree.topLevelItemCount()):
            p_item = self._tree.topLevelItem(i)
            if p_item.data(0, _ROLE_PIPELINE) == pipeline_name:
                for j in range(p_item.childCount()):
                    child = p_item.child(j)
                    if (child.data(0, _ROLE_KIND) == "mode"
                            and child.data(0, _ROLE_MODE_IDX) == mode_idx):
                        return child
        return None

    # ── Public API ─────────────────────────────────────────────────────────────

    def load_config(self, path: str) -> None:
        """Programmatic file load."""
        self._load_file(path)

    def get_config(self) -> Dict[str, Any]:
        """Return the current in-memory config (after committing pending edits)."""
        self._commit_current_form()
        return copy.deepcopy(self._config)

    def get_pipeline_config(self, pipeline_name: str) -> Optional[Dict[str, Any]]:
        """Return a single pipeline config dict (copy), or None."""
        self._commit_current_form()
        return copy.deepcopy(self._config.get("pipelines", {}).get(pipeline_name))


# ── Helpers ────────────────────────────────────────────────────────────────────

def _guess_analysis_type(modes: List[Dict[str, Any]]) -> str:
    tip_modes = {"InwardEdgeScan", "WindowContrastAlign"}
    for m in modes:
        if m.get("mode_name") in tip_modes:
            return "TipIdentification"
    return "LiquidMeasurement"


def _bold_font(font):
    from PySide6.QtGui import QFont
    f = QFont(font)
    f.setBold(True)
    return f


def _grey_brush():
    from PySide6.QtGui import QBrush, QColor
    return QBrush(QColor("#888888"))


def _default_brush():
    from PySide6.QtGui import QBrush
    return QBrush()
