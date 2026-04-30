"""
CalibrationTab — top-level widget for the "Camera Calibration" tab.

Hosts CheckerboardPanel (left) and RotationPanel (right) side-by-side.
Owns the instrument_config path and handles saving calibration results.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QSizePolicy,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from pa_gui.camera_calibration.calibration_data import CameraCalibrationData
from pa_gui.camera_calibration.checkerboard_panel import CheckerboardPanel
from pa_gui.camera_calibration.rotation_panel import RotationPanel


class CalibrationTab(QWidget):
    """Camera calibration tab: checkerboard (left) + rotation (right)."""

    def __init__(
        self,
        instrument_config_path: Optional[str] = None,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self._config_path: Optional[str] = None
        self._ic_data: Dict[str, Any] = {}

        self._build_ui()

        if instrument_config_path:
            self.set_config_path(instrument_config_path)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def set_config_path(self, path: str) -> None:
        """Load instrument_config.json and refresh child panels."""
        self._config_path = path
        self._ic_path_edit.setText(path)
        self._load_ic(path)

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(8, 8, 8, 8)

        # ── Warning banner ────────────────────────────────────────────
        warn = QLabel(
            "⚠  Camera calibration should be completed <b>before</b> ROI calibration.  "
            "ROI polygon positions are measured in pixels on the undistorted image."
        )
        warn.setWordWrap(True)
        warn.setStyleSheet(
            "background: #3a3000; color: #f0c040; border: 1px solid #7a6000; "
            "padding: 6px; border-radius: 4px;"
        )
        root.addWidget(warn)

        # ── Instrument config toolbar ─────────────────────────────────
        ic_row = QHBoxLayout()
        ic_row.addWidget(QLabel("Instrument Config:"))
        self._ic_path_edit = QLineEdit()
        self._ic_path_edit.setPlaceholderText("instrument_config.json")
        ic_row.addWidget(self._ic_path_edit, 1)

        browse_btn = QPushButton("Browse…")
        browse_btn.clicked.connect(self._on_browse_ic)
        ic_row.addWidget(browse_btn)

        load_btn = QPushButton("Load")
        load_btn.clicked.connect(lambda: self._load_ic(self._ic_path_edit.text()))
        ic_row.addWidget(load_btn)

        root.addLayout(ic_row)

        # ── Status bar ────────────────────────────────────────────────
        self._status = QLabel("No instrument config loaded.")
        self._status.setStyleSheet("color: #888;")
        root.addWidget(self._status)

        # ── Main splitter ─────────────────────────────────────────────
        splitter = QSplitter(Qt.Orientation.Horizontal)

        self._checkerboard_panel = CheckerboardPanel()
        self._checkerboard_panel.calibration_accepted.connect(self._on_calibration_accepted)
        splitter.addWidget(self._checkerboard_panel)

        self._rotation_panel = RotationPanel()
        self._rotation_panel.rotation_saved.connect(self._on_rotation_saved)
        splitter.addWidget(self._rotation_panel)

        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 2)

        root.addWidget(splitter, 1)

    # ------------------------------------------------------------------
    # Slots
    # ------------------------------------------------------------------

    def _on_browse_ic(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Open Instrument Config",
            self._ic_path_edit.text(),
            "JSON files (*.json);;All files (*)",
        )
        if path:
            self._ic_path_edit.setText(path)
            self._load_ic(path)

    def _load_ic(self, path: str) -> None:
        if not path:
            return
        p = Path(path)
        if not p.exists():
            self._status.setText(f"File not found: {path}")
            return
        try:
            with p.open("r", encoding="utf-8") as fh:
                self._ic_data = json.load(fh)
        except (json.JSONDecodeError, OSError) as exc:
            self._status.setText(f"Error loading config: {exc}")
            return

        self._config_path = path

        # Pre-populate checkerboard panel with calibration folder if present
        cal_folder = self._ic_data.get("default_calibration_folder", "")
        if cal_folder:
            self._checkerboard_panel.set_folder(cal_folder)

        # Populate rotation panel
        mandrels = self._ic_data.get("mandrels", [])
        self._rotation_panel.set_mandrels(mandrels)

        existing_cal = self._ic_data.get("camera_calibration")
        if existing_cal:
            rot = existing_cal.get("rotation_deg", 0.0)
            self._rotation_panel.set_rotation_deg(rot)
            self._rotation_panel.set_calibration(
                existing_cal.get("camera_matrix"),
                existing_cal.get("dist_coeffs"),
            )
            rms = existing_cal.get("rms_reprojection_error", 0.0)
            n = existing_cal.get("n_images_used", 0)
            ts = existing_cal.get("calibrated_at", "")
            self._status.setText(
                f"Existing calibration loaded — RMS {rms:.4f} px, "
                f"{n} images, calibrated {ts}"
            )
        else:
            self._rotation_panel.set_calibration(None, None)
            self._status.setText("Instrument config loaded. No camera calibration saved yet.")

    def _on_calibration_accepted(self, cal: CameraCalibrationData) -> None:
        if not self._config_path:
            self._status.setText("No instrument config path set — cannot save.")
            return
        try:
            cal.save_to_config(self._config_path)
        except (OSError, json.JSONDecodeError) as exc:
            self._status.setText(f"Save failed: {exc}")
            return

        self._status.setText(
            f"Calibration saved — RMS {cal.rms:.4f} px, {cal.n_images_used} images used.  "
            "Re-do ROI calibration on undistorted images."
        )
        # Refresh the rotation panel in case it needs to know the calibration was saved
        self._load_ic(self._config_path)

    def _on_rotation_saved(self, angle_deg: float) -> None:
        if not self._config_path:
            self._status.setText("No instrument config path set — cannot save.")
            return

        # We need an existing calibration record to attach rotation to.
        # If none exists yet, we create a minimal placeholder so the key is present.
        p = Path(self._config_path)
        try:
            with p.open("r", encoding="utf-8") as fh:
                data = json.load(fh)
        except (OSError, json.JSONDecodeError) as exc:
            self._status.setText(f"Load failed: {exc}")
            return

        cal_section = data.setdefault("camera_calibration", {})
        cal_section["rotation_deg"] = angle_deg

        try:
            with p.open("w", encoding="utf-8") as fh:
                json.dump(data, fh, indent=2)
        except OSError as exc:
            self._status.setText(f"Save failed: {exc}")
            return

        self._status.setText(f"Rotation correction saved: {angle_deg:.4f}°")
        self._rotation_panel.set_rotation_deg(angle_deg)
