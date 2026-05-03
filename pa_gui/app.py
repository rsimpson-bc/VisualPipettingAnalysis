"""PA GUI — top-level MainWindow."""

from __future__ import annotations

from typing import Optional

from PySide6.QtWidgets import QMainWindow, QTabWidget

from pa_gui.camera_calibration.calibration_tab import CalibrationTab
from pa_gui.roi_calibration.calibration_panel import CalibrationPanel
from pa_gui.analysis.analysis_tab import AnalysisTab
from pa_gui.sweep.sweep_tab import SweepTab


class MainWindow(QMainWindow):
    def __init__(self, instrument_config_path: Optional[str] = None) -> None:
        super().__init__()
        self.setWindowTitle("PA — Pipette Analysis")
        self.resize(1400, 900)

        self._tabs = QTabWidget()
        self.setCentralWidget(self._tabs)

        # Camera Calibration comes first — ROIs must be drawn on undistorted images.
        self._cam_cal_tab = CalibrationTab(instrument_config_path)
        self._tabs.addTab(self._cam_cal_tab, "Camera Calibration")

        self._roi_tab = CalibrationPanel(instrument_config_path)
        self._tabs.addTab(self._roi_tab, "ROI Calibration")

        self._analysis_tab = AnalysisTab(instrument_config_path)
        self._tabs.addTab(self._analysis_tab, "Run Analysis")

        self._sweep_tab = SweepTab(instrument_config_path)
        self._tabs.addTab(self._sweep_tab, "Parameter Sweep")

        # Future tabs (Live View) will be added here.
