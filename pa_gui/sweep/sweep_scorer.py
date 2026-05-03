"""Signal scorer for the Parameter Sweep tab — no Qt dependency.

Given an IntensityDetection signal and GT annotations for one image, this
module computes how well the signal "highlights" the annotated transitions.

Coordinate convention
---------------------
The signal ``z_axis_px`` contains ROI-relative row indices (0 = top of ROI).
GT annotations are stored as full-image rows.
The caller supplies ``roi_y0`` (the ROI bounding-box top edge in image space)
so this module can map: ``signal_idx = clamp(y_px - roi_y0, 0, len(signal)-1)``.

Score definition
----------------
For each GT annotation at image-row ``y_px``:

1. Map to signal index: ``idx = clamp(y_px - roi_y0, 0, N-1)``
2. Search window: ``signal[max(0, idx-tol) : idx+tol+1]``
3. Find the local extremum (peak or trough — whichever has the highest
   absolute deviation from the signal median).
4. ``prominence = |extremum_value - signal_median|``

Scale-invariant SNR score
--------------------------
The aggregate score is a signal-to-noise ratio bounded in ``[0, 1]``::

    spurious_excess = max(0, p95(background) − median)
    score = gt_mean_prom / (gt_mean_prom + spurious_excess + ε)

where ``background`` is the signal *outside* all GT windows and
``gt_mean_prom`` is the mean of all per-annotation prominences.

Using the 95th-percentile of the background (rather than its maximum) makes
the metric robust to single-sample noise spikes while still catching any
broad hump or systematic elevated region between annotations.

This formula is purely ratio-based: a parameter set that produces strong GT
peaks relative to the background wins regardless of the absolute signal scale
(important for comparing e.g. gradient vs raw-intensity extractors).

``n_spurious_peaks`` and ``spurious_peak_mass`` are still computed via
``find_peaks`` and stored in ``ScoreResult`` as diagnostics; they are **not**
used in the score calculation.

Returns
-------
``ScoreResult`` with:
  - ``per_annotation``:   list of per-GT prominence values (same order as annotations)
  - ``background_std``:   std dev of non-GT signal (diagnostic only)
  - ``n_spurious_peaks``: count of Z-score-exceeding peaks outside GT windows (diagnostic)
  - ``spurious_peak_mass``: sum of those peak heights above median (diagnostic)
  - ``score``:            SNR ratio in [0, 1]; higher is better
  - ``out_of_range``:     number of GT points whose mapped index hit the clamp boundary
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.signal import find_peaks

from pa_gui.sweep.gt_data import GTAnnotation


@dataclass
class ScoreResult:
    per_annotation:    List[float]      = field(default_factory=list)
    background_std:    float            = 0.0
    n_spurious_peaks:  int              = 0
    spurious_peak_mass: float           = 0.0
    score:             float            = 0.0
    out_of_range:      int              = 0
    per_type:          Dict[str, float] = field(default_factory=dict)
    skipped:           bool             = False   # True = no GT annotations


def score_signal(
    signal: np.ndarray,
    z_axis_px: np.ndarray,
    annotations: List[GTAnnotation],
    roi_y0: int,
    tolerance_px: int = 10,
    spurious_peak_k: float = 2.0,
    intensity_threshold: float = 0.0,
) -> ScoreResult:
    """Score one image's signal against its GT annotations.

    Parameters
    ----------
    signal : 1-D float array (length N), expected to be normalized to [0, 1]
             (same normalization the interpreter applies before threshold gating).
    z_axis_px : 1-D int array (length N), ROI-relative row indices
    annotations : list of GTAnnotation for this image
    roi_y0 : top edge of the ROI in full-image pixel rows
    tolerance_px : search radius around each GT point (full-image px)
    spurious_peak_k : Z-score multiplier for spurious peak height threshold
        (threshold = signal_mean + k * signal_std).  Default 2.0.
    intensity_threshold : the same ``intensity_threshold`` param the interpreter
        uses to gate peak detection.  Any GT window whose best peak value is
        below this threshold is treated as a missed detection (prominence = 0),
        because the interpreter would not have reported it.
    """
    result = ScoreResult()
    if signal is None or len(signal) == 0:
        return result

    N = len(signal)
    signal_f = signal.astype(np.float64)
    median = float(np.median(signal_f))

    # Global stats used for the spurious-peak Z-score threshold
    sig_mean = float(np.mean(signal_f))
    sig_std  = float(np.std(signal_f))
    height_threshold = sig_mean + spurious_peak_k * sig_std

    # Build a boolean mask of "GT-window" rows (for background + spurious calc)
    gt_mask = np.zeros(N, dtype=bool)

    prominences: List[float] = []
    for ann in annotations:
        # Map full-image Y to signal index
        roi_rel = ann.y_px - roi_y0
        idx = int(np.clip(roi_rel, 0, N - 1))
        if roi_rel < 0 or roi_rel >= N:
            result.out_of_range += 1

        lo = max(0, idx - tolerance_px)
        hi = min(N, idx + tolerance_px + 1)
        gt_mask[lo:hi] = True

        window = signal_f[lo:hi]
        if len(window) == 0:
            prominences.append(0.0)
            continue

        # Find extremum with highest absolute deviation from median
        max_idx = int(np.argmax(np.abs(window - median)))
        peak_val = window[max_idx]
        # If the interpreter's threshold would suppress this peak, it counts
        # as a missed detection — prominence is 0.
        if intensity_threshold > 0.0 and float(peak_val) < intensity_threshold:
            prominences.append(0.0)
            continue
        prominence = abs(peak_val - median)
        prominences.append(prominence)

    result.per_annotation = prominences

    # Background signal — all values outside GT windows
    background = signal_f[~gt_mask]
    result.background_std = float(np.std(background)) if len(background) > 1 else 0.0

    # Spurious peak diagnostics (not used in score — stored for inspection only)
    if len(background) > 0 and height_threshold > median:
        peak_local_idxs, _ = find_peaks(background, height=height_threshold)
        spurious_heights = background[peak_local_idxs]
        result.n_spurious_peaks   = int(len(peak_local_idxs))
        result.spurious_peak_mass = float(np.sum(np.abs(spurious_heights - median))) if len(spurious_heights) > 0 else 0.0

    # Scale-invariant SNR score ─────────────────────────────────────────────
    # spurious_excess = how far the 95th-percentile of the background exceeds
    # the signal median.  Using p95 (not max) makes the metric robust to single
    # outlier noise spikes while still detecting any broad hump between GT points.
    if len(background) > 0:
        bg_p95 = float(np.percentile(background, 95))
        spurious_excess = max(0.0, bg_p95 - median)
    else:
        spurious_excess = 0.0

    gt_mean_prom = float(np.mean(prominences)) if prominences else 0.0
    # score ∈ [0, 1]:  1 = perfect (all GT signal, no background); 0 = no GT signal
    denom = gt_mean_prom + spurious_excess + 1e-9
    result.score = gt_mean_prom / denom

    # Per-type sub-scores using the same spurious_excess denominator
    type_groups: Dict[str, List[float]] = defaultdict(list)
    for ann, prom in zip(annotations, prominences):
        type_groups[ann.type].append(prom)
    result.per_type = {
        t: float(np.mean(proms)) / (float(np.mean(proms)) + spurious_excess + 1e-9)
        for t, proms in type_groups.items()
        if proms
    }

    return result


def aggregate_scores(
    per_image: List[ScoreResult],
) -> Tuple[float, List[float], Dict[str, float], List[Optional[Dict[str, float]]]]:
    """Return (mean_score, per_image_scores, type_means, per_image_type_scores).

    Skipped images (no GT annotations) are represented as NaN in
    ``per_image_scores`` and as ``None`` in ``per_image_type_scores``.
    Both are excluded from all mean calculations.
    ``type_means`` maps annotation type name to mean sub-score across
    images that had at least one annotation of that type.
    """
    per_scores = [
        float("nan") if r.skipped else r.score
        for r in per_image
    ]
    per_image_type_scores: List[Optional[Dict[str, float]]] = [
        None if r.skipped else dict(r.per_type)
        for r in per_image
    ]
    active = [r for r in per_image if not r.skipped]
    mean = float(np.mean([r.score for r in active])) if active else 0.0

    all_types: set = set(t for r in active for t in r.per_type)
    type_means: Dict[str, float] = {
        t: float(np.mean([r.per_type[t] for r in active if t in r.per_type]))
        for t in all_types
    }
    return mean, per_scores, type_means, per_image_type_scores
