"""
CheckerboardPanel — widget for running OpenCV checkerboard calibration and
reviewing the results before accepting them into instrument_config.json.
"""

from __future__ import annotations

from typing import Dict, Optional

import cv2
import numpy as np
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QColor, QImage, QPixmap
from PySide6.QtWidgets import (
    QDoubleSpinBox,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QProgressBar,
    QPushButton,
    QSizePolicy,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from pa_gui.camera_calibration.calibration_data import CameraCalibrationData
from pa_gui.camera_calibration.calibration_worker import CalibrationWorker


class CheckerboardPanel(QWidget):
    """
    Left side of the Camera Calibration tab.

    Signals
    -------
    calibration_accepted(CameraCalibrationData)
        Emitted when the user clicks "Accept & Save".
    """

    calibration_accepted: Signal = Signal(object)

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._worker: Optional[CalibrationWorker] = None
        self._last_result: Optional[Dict] = None
        self._build_ui()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(8, 8, 8, 8)

        # ── Folder row ────────────────────────────────────────────────
        folder_row = QHBoxLayout()
        folder_row.addWidget(QLabel("Calibration images folder:"))
        self._folder_edit = QLineEdit()
        self._folder_edit.setPlaceholderText("Select folder containing checkerboard images…")
        folder_row.addWidget(self._folder_edit, 1)
        browse_btn = QPushButton("Browse…")
        browse_btn.clicked.connect(self._on_browse)
        folder_row.addWidget(browse_btn)
        root.addLayout(folder_row)

        # ── Board settings ────────────────────────────────────────────
        board_group = QGroupBox("Checkerboard settings")
        board_layout = QHBoxLayout(board_group)

        board_layout.addWidget(QLabel("Inner corners X:"))
        self._corners_x = QSpinBox()
        self._corners_x.setRange(2, 30)
        self._corners_x.setValue(9)
        self._corners_x.setToolTip(
            "Number of inner corners along the horizontal direction.\n"
            "Must match the physical board exactly."
        )
        board_layout.addWidget(self._corners_x)

        board_layout.addWidget(QLabel("  Y:"))
        self._corners_y = QSpinBox()
        self._corners_y.setRange(2, 30)
        self._corners_y.setValue(6)
        self._corners_y.setToolTip(
            "Number of inner corners along the vertical direction.\n"
            "Must match the physical board exactly."
        )
        board_layout.addWidget(self._corners_y)

        board_layout.addSpacing(16)
        board_layout.addWidget(QLabel("Square size (mm):"))
        self._square_mm = QDoubleSpinBox()
        self._square_mm.setRange(0.1, 500.0)
        self._square_mm.setValue(1.0)
        self._square_mm.setDecimals(2)
        self._square_mm.setToolTip(
            "Physical size of one square in millimetres.\n"
            "Only affects the physical-unit scale of distortion coefficients;\n"
            "pixel-space remapping works correctly with the default value of 1.0."
        )
        board_layout.addWidget(self._square_mm)

        board_layout.addStretch()

        self._run_btn = QPushButton("Run Calibration")
        self._run_btn.setFixedHeight(32)
        self._run_btn.clicked.connect(self._on_run)
        board_layout.addWidget(self._run_btn)

        root.addWidget(board_group)

        # ── Progress bar ──────────────────────────────────────────────
        self._progress = QProgressBar()
        self._progress.setVisible(False)
        self._progress.setTextVisible(True)
        root.addWidget(self._progress)

        # ── RMS display ───────────────────────────────────────────────
        rms_row = QHBoxLayout()
        rms_label = QLabel("RMS reprojection error:")
        rms_label.setStyleSheet("font-weight: bold;")
        rms_row.addWidget(rms_label)
        self._rms_value = QLabel("—")
        self._rms_value.setStyleSheet("font-size: 18px; font-weight: bold; min-width: 100px;")
        rms_row.addWidget(self._rms_value)
        self._rms_note = QLabel("(< 0.5 px = excellent  |  0.5–1.0 px = acceptable  |  > 1.0 px = retake)")
        self._rms_note.setStyleSheet("color: #888;")
        rms_row.addWidget(self._rms_note, 1)
        root.addLayout(rms_row)

        # ── Results table + preview ────────────────────────────────────
        results_row = QHBoxLayout()

        self._table = QTableWidget(0, 3)
        self._table.setHorizontalHeaderLabels(["Filename", "Detected", "Residual (px)"])
        self._table.horizontalHeader().setStretchLastSection(False)
        self._table.horizontalHeader().setSectionResizeMode(
            0, self._table.horizontalHeader().ResizeMode.Stretch
        )
        self._table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self._table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        results_row.addWidget(self._table, 2)

        preview_layout = QVBoxLayout()
        preview_layout.addWidget(QLabel("Corner preview:"))
        self._preview_label = QLabel("(no preview)")
        self._preview_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._preview_label.setMinimumSize(240, 180)
        self._preview_label.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )
        self._preview_label.setStyleSheet("background: #111; color: #555; border: 1px solid #333;")
        preview_layout.addWidget(self._preview_label, 1)
        results_row.addLayout(preview_layout, 1)

        root.addLayout(results_row, 1)

        # ── Accept button ─────────────────────────────────────────────
        accept_row = QHBoxLayout()
        accept_row.addStretch()
        self._accept_btn = QPushButton("Accept && Save")
        self._accept_btn.setEnabled(False)
        self._accept_btn.setFixedHeight(32)
        self._accept_btn.setStyleSheet(
            "QPushButton:enabled { background: #2a7a2a; color: white; font-weight: bold; }"
        )
        self._accept_btn.clicked.connect(self._on_accept)
        accept_row.addWidget(self._accept_btn)
        root.addLayout(accept_row)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def set_folder(self, path: str) -> None:
        """Pre-populate the folder path (e.g. from instrument_config)."""
        self._folder_edit.setText(path)

    # ------------------------------------------------------------------
    # Slots
    # ------------------------------------------------------------------

    def _on_browse(self) -> None:
        folder = QFileDialog.getExistingDirectory(
            self, "Select Calibration Images Folder", self._folder_edit.text()
        )
        if folder:
            self._folder_edit.setText(folder)

    def _on_run(self) -> None:
        folder = self._folder_edit.text().strip()
        if not folder:
            return

        # Reset UI
        self._last_result = None
        self._accept_btn.setEnabled(False)
        self._table.setRowCount(0)
        self._rms_value.setText("—")
        self._rms_value.setStyleSheet("font-size: 18px; font-weight: bold;")
        self._preview_label.clear()
        self._preview_label.setText("(running…)")

        self._progress.setVisible(True)
        self._progress.setValue(0)
        self._run_btn.setEnabled(False)

        self._worker = CalibrationWorker(
            folder=folder,
            board_w=self._corners_x.value(),
            board_h=self._corners_y.value(),
            square_mm=self._square_mm.value(),
            parent=self,
        )
        self._worker.progress.connect(self._on_progress)
        self._worker.finished.connect(self._on_worker_finished)
        self._worker.error.connect(self._on_worker_error)
        self._worker.start()

    def _on_progress(self, current: int, total: int, filename: str, found: bool) -> None:
        self._progress.setMaximum(total)
        self._progress.setValue(current)
        status = "✓" if found else "✗"
        self._progress.setFormat(f"{current}/{total}  {status}  {filename}")

    def _on_worker_finished(self, result: Dict) -> None:
        self._run_btn.setEnabled(True)
        self._progress.setVisible(False)
        self._last_result = result
        self._populate_results(result)
        self._accept_btn.setEnabled(True)

    def _on_worker_error(self, message: str) -> None:
        self._run_btn.setEnabled(True)
        self._progress.setVisible(False)
        self._rms_value.setText("Error")
        self._rms_value.setStyleSheet("font-size: 18px; font-weight: bold; color: red;")
        self._preview_label.setText(message)

    def _on_accept(self) -> None:
        if self._last_result is None:
            return
        r = self._last_result
        cal = CameraCalibrationData(
            camera_matrix=r["camera_matrix"],
            dist_coeffs=r["dist_coeffs"],
            image_size=tuple(r["image_size"]),  # type: ignore[arg-type]
            rms=r["rms"],
            n_images_used=r["n_valid"],
        )
        self.calibration_accepted.emit(cal)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _populate_results(self, result: Dict) -> None:
        rms = result["rms"]

        # Color-code RMS
        if rms < 0.5:
            color = "#2ecc71"    # green
            quality = "excellent"
        elif rms < 1.0:
            color = "#f39c12"    # amber
            quality = "acceptable"
        else:
            color = "#e74c3c"    # red
            quality = "poor — consider retaking images"

        self._rms_value.setText(f"{rms:.4f} px  ({quality})")
        self._rms_value.setStyleSheet(
            f"font-size: 18px; font-weight: bold; color: {color};"
        )

        # Populate table
        per_image = result["per_image"]
        self._table.setRowCount(len(per_image))
        for row, entry in enumerate(per_image):
            self._table.setItem(row, 0, QTableWidgetItem(entry["filename"]))

            detected_item = QTableWidgetItem("✓" if entry["detected"] else "✗")
            detected_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            if entry["detected"]:
                detected_item.setForeground(QColor("#2ecc71"))
            else:
                detected_item.setForeground(QColor("#e74c3c"))
            self._table.setItem(row, 1, detected_item)

            residual = entry.get("residual_px")
            if residual is not None:
                res_item = QTableWidgetItem(f"{residual:.4f}")
                res_item.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
                self._table.setItem(row, 2, res_item)
            else:
                self._table.setItem(row, 2, QTableWidgetItem("—"))

        # Corner preview
        preview_bgr: Optional[np.ndarray] = result.get("preview_image")
        if preview_bgr is not None:
            self._set_preview(preview_bgr)
        else:
            self._preview_label.setText("(no preview available)")

    def _set_preview(self, bgr: np.ndarray) -> None:
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        qimg = QImage(rgb.data, w, h, ch * w, QImage.Format.Format_RGB888)
        pixmap = QPixmap.fromImage(qimg).scaled(
            self._preview_label.width(),
            self._preview_label.height(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self._preview_label.setPixmap(pixmap)

    def resizeEvent(self, event) -> None:  # noqa: N802
        super().resizeEvent(event)
        # Re-scale preview when panel is resized
        if self._last_result is not None and self._last_result.get("preview_image") is not None:
            self._set_preview(self._last_result["preview_image"])
