"""
Layer 3 — Liquid interpreters.

One interpreter class per liquid analysis mode (or mode group where the
logic is genuinely identical).  Each converts a ZProfile into a
List[PointOfInterest] by applying thresholding, detrending, peak/trough
finding, and noise rejection.
"""

from __future__ import annotations

from typing import List
import numpy as np

from pa.pipeline.types import ZProfile, PointOfInterest
from pa.pipeline.interpreters.base import BaseLiquidInterpreter
from pa.pipeline import signal_primitives as sp


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _threshold_peaks(
    signal: np.ndarray,
    z_axis: np.ndarray,
    threshold: float,
    min_distance: int,
    mode_name: str,
    mode_weight: float,
    label: str = "",
) -> List[PointOfInterest]:
    """Find peaks above threshold and return as weighted POIs."""
    indices = sp.find_peaks(signal, threshold=threshold, min_distance=min_distance)
    pois = []
    for idx in indices:
        # Normalise peak height to [0,1] relative to max for weight scaling
        relative_height = float(signal[idx]) / (signal.max() + 1e-6)
        pois.append(PointOfInterest(
            z_px=float(z_axis[idx]),
            weight=relative_height * mode_weight,
            source_mode=mode_name,
            label=label,
        ))
    return pois


# ---------------------------------------------------------------------------
# Intensity Detection interpreter
# ---------------------------------------------------------------------------

class IntensityInterpreter(BaseLiquidInterpreter):
    """
    Params:
        intensity_threshold (float): minimum normalised intensity to consider (0–1)
        min_distance (int):          minimum px between POIs
        mode_weight (float):         overall weight for this mode's POIs
    """

    def interpret(self, profile: ZProfile) -> List[PointOfInterest]:
        params = self.params
        signal = sp.normalize_signal(profile.signal)
        threshold = params.get("intensity_threshold", 0.3)
        min_dist = params.get("min_distance", 5)
        weight = params.get("mode_weight", 1.0)
        return _threshold_peaks(
            signal, profile.z_axis_px, threshold, min_dist,
            profile.mode, weight, label="intensity_peak",
        )


class RowContrastInterpreter(BaseLiquidInterpreter):
    """
    Same threshold/peak-finding logic as IntensityInterpreter, applied to the
    per-row std/variance signal produced by RowContrastDetection.

    Params:
        intensity_threshold (float): minimum normalised contrast to consider (0–1)
        min_distance (int):          minimum px between POIs
        mode_weight (float):         overall weight for this mode's POIs
    """

    def interpret(self, profile: ZProfile) -> List[PointOfInterest]:
        params = self.params
        signal = sp.normalize_signal(profile.signal)
        threshold = params.get("intensity_threshold", 0.3)
        min_dist = params.get("min_distance", 5)
        weight = params.get("mode_weight", 1.0)
        return _threshold_peaks(
            signal, profile.z_axis_px, threshold, min_dist,
            profile.mode, weight, label="contrast_peak",
        )


# ---------------------------------------------------------------------------
# Line Continuity interpreters  (shared logic, different signal semantics)
# ---------------------------------------------------------------------------

class LineContinuityThresholdInterpreter(BaseLiquidInterpreter):
    """
    Works for Terminations, PatternChange, and Density sub-modes.
    Normalises the signal, optionally smooths, then finds peaks above threshold.

    Params:
        threshold (float):       0–1 after normalisation
        smooth_window (int):     moving average window (1 = no smoothing)
        min_distance (int)
        mode_weight (float)
    """

    def interpret(self, profile: ZProfile) -> List[PointOfInterest]:
        params = self.params
        signal = sp.normalize_signal(profile.signal)
        smooth = params.get("smooth_window", 3)
        if smooth > 1:
            signal = sp.smooth_moving_average(signal, smooth)
        threshold = params.get("threshold", 0.3)
        min_dist = params.get("min_distance", 5)
        weight = params.get("mode_weight", 1.0)
        return _threshold_peaks(
            signal, profile.z_axis_px, threshold, min_dist,
            profile.mode, weight, label="ridge_change",
        )


class LineContinuityCorrelationInterpreter(BaseLiquidInterpreter):
    """
    Low correlation = candidate transition → find troughs (inverted peaks).

    Params:
        threshold (float):   upper bound for trough to count (0–1 normalised)
        min_distance (int)
        mode_weight (float)
    """

    def interpret(self, profile: ZProfile) -> List[PointOfInterest]:
        params = self.params
        signal = sp.normalize_signal(profile.signal)
        threshold = params.get("threshold", 0.5)
        min_dist = params.get("min_distance", 5)
        weight = params.get("mode_weight", 1.0)
        indices = sp.find_troughs(signal, threshold=threshold, min_distance=min_dist)
        pois = []
        for idx in indices:
            relative_depth = 1.0 - float(signal[idx]) / (signal.max() + 1e-6)
            pois.append(PointOfInterest(
                z_px=float(profile.z_axis_px[idx]),
                weight=relative_depth * weight,
                source_mode=profile.mode,
                label="low_correlation",
            ))
        return pois


# ---------------------------------------------------------------------------
# Extrema Lines interpreters
# ---------------------------------------------------------------------------

class ExtremaLinesDensityInterpreter(BaseLiquidInterpreter):
    """
    Detrended density signal: find local minima (troughs) that represent
    sudden dips in line count at a liquid transition.

    Params:
        trough_threshold (float):   upper bound for residual trough (signal already detrended)
        min_distance (int)
        mode_weight (float)
    """

    def interpret(self, profile: ZProfile) -> List[PointOfInterest]:
        params = self.params
        signal = profile.signal   # already detrended by mode
        threshold = params.get("trough_threshold", -0.1)
        min_dist = params.get("min_distance", 5)
        weight = params.get("mode_weight", 1.0)
        indices = sp.find_troughs(signal, threshold=threshold, min_distance=min_dist)
        pois = []
        for idx in indices:
            depth = abs(float(signal[idx]))
            pois.append(PointOfInterest(
                z_px=float(profile.z_axis_px[idx]),
                weight=min(depth * weight, 1.0),
                source_mode=profile.mode,
                label="density_dip",
            ))
        return pois


class ExtremaLinesTerminationInterpreter(BaseLiquidInterpreter):
    """
    Detrended termination signal: find local maxima (peaks) that represent
    sudden spikes in terminations at a liquid transition.

    Params:
        peak_threshold (float):   minimum residual value to count as a peak
        min_distance (int)
        mode_weight (float)
    """

    def interpret(self, profile: ZProfile) -> List[PointOfInterest]:
        params = self.params
        signal = profile.signal   # already detrended by mode
        threshold = params.get("peak_threshold", 0.1)
        min_dist = params.get("min_distance", 5)
        weight = params.get("mode_weight", 1.0)
        return _threshold_peaks(
            signal, profile.z_axis_px, threshold, min_dist,
            profile.mode, weight, label="termination_spike",
        )
