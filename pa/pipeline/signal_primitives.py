"""
Layer 0 — Signal Primitives.

Stateless functions that operate on 1-D numpy arrays (Z-axis signals).
No analysis logic here — only reusable signal processing operations.
"""

from __future__ import annotations

import numpy as np
from typing import Literal, List, Tuple


# ---------------------------------------------------------------------------
# Detrending
# ---------------------------------------------------------------------------

def detrend_linear(signal: np.ndarray) -> np.ndarray:
    """
    Remove a best-fit linear trend from a signal.
    Returns the residual (signal - linear_fit), float32.

    Use this when a signal is expected to increase or decrease monotonically
    (e.g. ridge count increases with Z as the tip gets wider), and you want
    to see local deviations from that trend.
    """
    x = np.arange(len(signal), dtype=np.float32)
    y = signal.astype(np.float32)
    # Least-squares linear fit
    coeffs = np.polyfit(x, y, deg=1)
    trend = np.polyval(coeffs, x)
    return y - trend


def detrend_polynomial(signal: np.ndarray, degree: int = 2) -> np.ndarray:
    """
    Remove a best-fit polynomial trend of the given degree.
    Returns the residual, float32.
    """
    x = np.arange(len(signal), dtype=np.float32)
    y = signal.astype(np.float32)
    coeffs = np.polyfit(x, y, deg=degree)
    trend = np.polyval(coeffs, x)
    return y - trend


def detrend(
    signal: np.ndarray,
    method: Literal["linear", "polynomial"] = "linear",
    **kwargs,
) -> np.ndarray:
    """Dispatch to detrend_linear or detrend_polynomial by name."""
    if method == "linear":
        return detrend_linear(signal)
    if method == "polynomial":
        return detrend_polynomial(signal, **kwargs)
    raise ValueError(f"Unknown detrend method: {method!r}.")


# ---------------------------------------------------------------------------
# Smoothing
# ---------------------------------------------------------------------------

def smooth_moving_average(signal: np.ndarray, window: int) -> np.ndarray:
    """Simple moving average along a 1-D signal. Returns float32."""
    if window < 2:
        return signal.astype(np.float32)
    kernel = np.ones(window, dtype=np.float32) / window
    return np.convolve(signal.astype(np.float32), kernel, mode="same")


# ---------------------------------------------------------------------------
# Derivative
# ---------------------------------------------------------------------------

def derivative(signal: np.ndarray) -> np.ndarray:
    """
    First discrete derivative (np.diff with boundary padding).
    Returns array of same length as input, float32.
    A peak in the derivative indicates a rapid increase in the original signal.
    A trough indicates a rapid decrease.
    """
    d = np.diff(signal.astype(np.float32), prepend=signal[0])
    return d


# ---------------------------------------------------------------------------
# Normalisation
# ---------------------------------------------------------------------------

def normalize_signal(
    signal: np.ndarray,
    out_min: float = 0.0,
    out_max: float = 1.0,
) -> np.ndarray:
    """Min-max normalise a 1-D signal to [out_min, out_max]. Returns float32."""
    s = signal.astype(np.float32)
    lo, hi = s.min(), s.max()
    if hi == lo:
        return np.full_like(s, out_min)
    return (s - lo) / (hi - lo) * (out_max - out_min) + out_min


# ---------------------------------------------------------------------------
# Peak / trough finding
# ---------------------------------------------------------------------------

def find_peaks(
    signal: np.ndarray,
    threshold: float = 0.0,
    min_distance: int = 1,
) -> np.ndarray:
    """
    Find local maxima above a threshold.
    Returns array of indices where peaks occur.
    min_distance: minimum number of samples between returned peaks.
    """
    from scipy.signal import find_peaks as _sp_find_peaks
    indices, _ = _sp_find_peaks(signal, height=threshold, distance=min_distance)
    return indices


def find_troughs(
    signal: np.ndarray,
    threshold: float = 0.0,
    min_distance: int = 1,
) -> np.ndarray:
    """
    Find local minima below a threshold (i.e. peaks in the negated signal).
    Returns array of indices.
    threshold here is an *upper* bound — only troughs below this value are returned.
    """
    from scipy.signal import find_peaks as _sp_find_peaks
    indices, _ = _sp_find_peaks(-signal, height=-threshold, distance=min_distance)
    return indices


# ---------------------------------------------------------------------------
# Edge masking
# ---------------------------------------------------------------------------

def mask_signal_edges(
    signal: np.ndarray,
    ignore_top: int = 0,
    ignore_bottom: int = 0,
) -> np.ndarray:
    """
    Return a copy of *signal* with the first *ignore_top* and last
    *ignore_bottom* elements forced to zero.

    This is applied after all per-row feature extraction and before POI
    detection so that the signal chart shows 0 at the masked positions and
    no peaks can be found there.
    """
    if ignore_top <= 0 and ignore_bottom <= 0:
        return signal
    out = signal.copy()
    n = len(out)
    if ignore_top > 0:
        out[:min(ignore_top, n)] = 0.0
    if ignore_bottom > 0:
        out[max(0, n - ignore_bottom):] = 0.0
    return out


# ---------------------------------------------------------------------------
# Multi-frame statistics
# ---------------------------------------------------------------------------

def stack_mean_variance(
    profiles: List[np.ndarray],
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Given N 1-D signals of the same length, compute per-element mean and variance.
    Returns (mean, variance) as float32 arrays.
    """
    arr = np.stack([p.astype(np.float32) for p in profiles], axis=0)
    return arr.mean(axis=0), arr.var(axis=0)
