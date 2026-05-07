"""
Signal-level interpretation engine (Phase A).

Takes one or more DebugStage signal arrays plus a list of BehaviorRules
(stored in the preset's ``behavior_rules`` key) and produces an
InterpretationResult containing:

  - per-z state scores (below_tip / gas_in_tip / liquid_in_tip)
  - the Viterbi-decoded state sequence (globally consistent label per row)
  - KDE densities and point estimates for tip_bottom and meniscus boundaries

Architecture
------------
Signal (1D array, z_axis)
    │
    ▼
_apply_trigger()     – converts signal → activation array (0–1) per row
    │
    ▼
_apply_rule()        – multiplies by strength, smears with Gaussian kernel,
                       accumulates into state_scores or boundary_density
    │
    ▼
viterbi_decode()     – global argmax over (state × row) with transition costs
    │
    ▼
InterpretationResult

State indices
-------------
    STATE_GAS   = 0   (gas_in_tip: above the meniscus, inside tip)
    STATE_LIQ   = 1   (liquid_in_tip)
    STATE_BELOW = 2   (below_tip: shank region, outside liquid)

Valid no-bubble transition sequence (top → bottom):
    gas_in_tip  →  liquid_in_tip  →  below_tip
No backward transitions.  Transition to the same state is free.

Boundary targets
----------------
    "tip_bottom"  – edge between liquid_in_tip and below_tip
    "meniscus"    – edge between gas_in_tip and liquid_in_tip

Rule schema (JSON-compatible dict)
-----------------------------------
Each rule has the following keys:

    trigger (str)        – one of: "peak" | "high_signal" | "low_signal" | "rising_edge"
    target (str)         – state or boundary to vote for:
                           "gas_in_tip" | "liquid_in_tip" | "below_tip" |
                           "tip_bottom" | "meniscus"
    direction (int)      – +1 (evidence for) or -1 (evidence against)
    strength (float)     – multiplier for the activation; default 1.0

    For trigger "peak":
        min_prominence_frac (float) – fraction of signal range required for
                                      peak prominence; default 0.2
        smoothing_px (int)          – pre-smoothing window before peak finding;
                                      default 3
        spread_px (float)           – σ of Gaussian kernel applied to each peak
                                      vote; default 5.0

    For trigger "high_signal" | "low_signal":
        threshold_frac (float)      – fraction of signal range for the threshold;
                                      default 0.3 (high) / 0.3 (low, from top)
        smoothing_px (int)          – rolling average window; default 5
        spread_px (float)           – σ to smear the per-row activation; default 3.0

    For trigger "rising_edge":
        threshold_frac (float)      – threshold as a fraction of signal range.
                                      A crossing is detected when the smoothed
                                      signal rises from below to at or above
                                      sig_min + threshold_frac * sig_range.
                                      default 0.3
        smoothing_px (int)          – pre-smoothing window; default 5
        spread_px (float)           – σ of Gaussian applied to the impulse;
                                      default 5.0
        pre_low_window_px (int)     – if > 0, the N rows immediately before
                                      each crossing are inspected; the crossing
                                      is weighted by the fraction of those rows
                                      that are below the threshold (confirms a
                                      genuine low→high transition); default 0
                                      (disabled)
        pre_low_frac (float)        – minimum fraction of the pre-window that
                                      must be below threshold; crossings below
                                      this fraction are suppressed; default 0.5
                                      (only used when pre_low_window_px > 0)

    spatial_modifier (dict | None)  – optional shift applied to the activation:
        {"type": "offset_below", "range_px": N}
            shifts the activation array downward by up to N rows
            (useful when the signal appears N rows above its physical cause)
        {"type": "offset_above", "range_px": N}
            shifts upward

    location_prior (list[list] | None) – optional spatial prior multiplied
        against the activation after trigger + spatial_modifier.  Expressed as
        a list of [z_frac, weight] anchor pairs where z_frac ∈ [0, 1] (0 = top
        of z-range, 1 = bottom) and weight ∈ [0, 1].  Linear interpolation is
        used between anchors; values outside the anchor range are clamped to
        the nearest endpoint weight.  Anchors need not be sorted — they are
        sorted internally.  Example for a tip-bottom prior that confines
        evidence to the lower third of the image::

            "location_prior": [[0.0, 0.0], [0.6, 0.0], [0.8, 1.0], [1.0, 1.0]]

        The anchor weights are optimizable by the rule optimizer.

Example rules list
------------------
::

    [
      {
        "trigger": "peak",
        "min_prominence_frac": 0.35,
        "smoothing_px": 3,
        "spread_px": 8,
        "target": "tip_bottom",
        "direction": 1,
        "strength": 1.2,
        "location_prior": [[0.0, 0.0], [0.6, 0.0], [0.8, 1.0], [1.0, 1.0]]
      },
      {
        "trigger": "high_signal",
        "threshold_frac": 0.25,
        "smoothing_px": 5,
        "spread_px": 3,
        "target": "gas_in_tip",
        "direction": 1,
        "strength": 0.7,
        "spatial_modifier": {"type": "offset_above", "range_px": 15}
      }
    ]
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# State constants
# ---------------------------------------------------------------------------

STATE_GAS   = 0   # gas_in_tip  (above meniscus)
STATE_LIQ   = 1   # liquid_in_tip
STATE_BELOW = 2   # below_tip

_STATE_NAMES = {STATE_GAS: "gas_in_tip", STATE_LIQ: "liquid_in_tip", STATE_BELOW: "below_tip"}

# Transition cost matrix: cost[from_state][to_state].
# 0   = free (same state or valid forward transition)
# inf = forbidden (backward / invalid)
_INF = 1e9
_TRANSITION_COST: List[List[float]] = [
    #           to_gas   to_liq   to_below
    [0.0,       0.0,     _INF  ],  # from gas
    [_INF,      0.0,     0.0   ],  # from liquid
    [_INF,      _INF,    0.0   ],  # from below
]


# ---------------------------------------------------------------------------
# Output dataclass
# ---------------------------------------------------------------------------

@dataclass
class InterpretationResult:
    """
    Per-image interpretation output from the signal interpreter.

    All arrays are indexed by the *global* z_axis (full-image row coordinates).

    Attributes
    ----------
    z_axis : np.ndarray
        Global z axis in full-image pixel coordinates (same as used for
        alignment in Mode Compare).  Length N.

    state_scores : np.ndarray
        Shape (N, 3).  Log-likelihood accumulator for each state at each z.
        Column order: [gas_in_tip, liquid_in_tip, below_tip].
        Positive = evidence for; negative = evidence against.

    state_sequence : np.ndarray
        Shape (N,).  Integer in {0, 1, 2} — Viterbi-decoded state per row.
        Guaranteed to follow a valid transition sequence.

    tip_bottom_density : np.ndarray
        Shape (N,).  KDE density for the tip_bottom boundary.  Unnormalised.

    meniscus_density : np.ndarray
        Shape (N,).  KDE density for the meniscus boundary.  Unnormalised.

    tip_bottom_z : Optional[float]
        Full-image z estimate for the tip-bottom boundary (peak of KDE).
        None if the density is flat (no evidence).

    meniscus_z : Optional[float]
        Full-image z estimate for the meniscus boundary (highest KDE peak).
        None if no evidence.
    """
    z_axis:             np.ndarray
    state_scores:       np.ndarray                 # (N, 3)
    state_sequence:     np.ndarray                 # (N,) int
    tip_bottom_density: np.ndarray                 # (N,)
    meniscus_density:   np.ndarray                 # (N,)
    tip_bottom_z:       Optional[float]  = None
    meniscus_z:         Optional[float]  = None


# ---------------------------------------------------------------------------
# Trigger helpers
# ---------------------------------------------------------------------------

def _smooth(signal: np.ndarray, window: int) -> np.ndarray:
    """Rolling average, returned as float32 same length as input."""
    if window < 2 or len(signal) < 2:
        return signal.astype(np.float32)
    kernel = np.ones(window, dtype=np.float32) / window
    return np.convolve(signal.astype(np.float32), kernel, mode="same")


def _normalize(signal: np.ndarray) -> np.ndarray:
    """Min-max to [0, 1]. Returns float32."""
    s = signal.astype(np.float32)
    lo, hi = float(s.min()), float(s.max())
    if hi == lo:
        return np.zeros_like(s)
    return (s - lo) / (hi - lo)


def _gaussian_kernel(n: int, sigma: float) -> np.ndarray:
    """Unnormalised 1-D Gaussian kernel of length n, centred, float32."""
    x = np.arange(n, dtype=np.float32) - n // 2
    g = np.exp(-0.5 * (x / max(sigma, 1e-6)) ** 2)
    return g / g.sum()


def _smear(activation: np.ndarray, spread_px: float) -> np.ndarray:
    """Convolve a per-row activation array with a Gaussian of σ=spread_px."""
    if spread_px < 0.5:
        return activation.astype(np.float32)
    n = max(3, int(spread_px * 6) | 1)   # odd kernel width ~6σ
    kernel = _gaussian_kernel(n, spread_px)
    return np.convolve(activation.astype(np.float32), kernel, mode="same")


def _apply_spatial_modifier(
    activation: np.ndarray,
    modifier: Optional[Dict[str, Any]],
) -> np.ndarray:
    """Shift the activation array up or down by range_px rows."""
    if modifier is None:
        return activation
    kind = modifier.get("type", "")
    shift = int(modifier.get("range_px", 0))
    if shift <= 0 or kind not in ("offset_below", "offset_above"):
        return activation
    n = len(activation)
    out = np.zeros(n, dtype=np.float32)
    if kind == "offset_below":   # shift content downward
        src_end = max(0, n - shift)
        out[shift:shift + src_end] = activation[:src_end]
    else:                        # offset_above — shift content upward
        src_start = min(shift, n)
        out[:n - src_start] = activation[src_start:]
    return out


def _trigger_peak(
    signal: np.ndarray,
    rule: Dict[str, Any],
) -> np.ndarray:
    """
    Returns a per-row activation: Gaussian-smeared impulse at each peak.
    Peak prominence ≥ min_prominence_frac × signal_range required.

    ``peak_side``
        ``"both"`` (default): classic symmetric local maximum.
        ``"top"``: fires at the leading edge of a rise coming from above
        (low row indices); the signal may stay high/noisy below.
        ``"bottom"``: fires at the trailing edge of a rise coming from below
        (high row indices); the signal may stay high/noisy above.

    ``low_above_window_px`` / ``low_above_threshold_frac`` / ``low_above_min_strength``
        Downweight peaks that are *not* immediately below a quiescent zone.
        For each candidate peak at row *i*, the fraction of the
        ``low_above_window_px`` rows immediately above it (rows
        ``i − window … i − 1``) whose smoothed value is ≤
        ``low_above_threshold_frac × sig_range`` is computed as *low_frac*.
        The peak weight is then multiplied by:

            low_frac × (1 − low_above_min_strength) + low_above_min_strength

        ``low_above_min_strength = 0.0`` (default): hard gate — a peak with
        zero low rows above it is zeroed out entirely.
        ``low_above_min_strength = 1.0``: modifier is disabled.
        Intermediate values give a soft preference.
        Setting ``low_above_window_px = 0`` (default) disables the modifier.

    ``low_below_window_px`` / ``low_below_threshold_frac`` / ``low_below_min_strength``
        Mirror of the above, looking at the ``low_below_window_px`` rows
        *below* the peak (rows ``i + 1 … i + window``).  Use this when the
        target boundary sits immediately above a quiescent zone (e.g. a peak
        just before the signal drops to zero at the tip bottom).

    ``high_above_window_px`` / ``high_above_threshold_frac`` / ``high_above_min_strength``
        Boost peaks that *are* immediately below a high-signal zone.
        For each candidate peak at row *i*, the fraction of the
        ``high_above_window_px`` rows immediately above it whose smoothed
        value is ≥ ``sig_min + high_above_threshold_frac × sig_range`` is
        computed as *high_frac*.  The weight is multiplied by:

            high_frac × (1 − high_above_min_strength) + high_above_min_strength

        ``high_above_min_strength = 0.0``: hard gate — only peaks beneath a
        high-signal zone survive.  ``= 1.0``: disabled (default).
        Setting ``high_above_window_px = 0`` (default) disables the modifier.

    ``high_below_window_px`` / ``high_below_threshold_frac`` / ``high_below_min_strength``
        Mirror of ``high_above_*``, looking at rows *below* the peak.
        Boosts peaks that sit immediately above a high-signal zone.
    """
    from scipy.signal import find_peaks as _sp_peaks

    smoothing             = int(rule.get("smoothing_px", 3))
    min_prom_frac         = float(rule.get("min_prominence_frac", 0.2))
    spread_px             = float(rule.get("spread_px", 5.0))
    value_min_str         = float(rule.get("value_min_strength", 1.0))
    peak_side             = str(rule.get("peak_side", "both")).lower()
    low_above_window      = int(rule.get("low_above_window_px", 0))
    low_above_thresh_frac = float(rule.get("low_above_threshold_frac", 0.3))
    low_above_min_str     = float(rule.get("low_above_min_strength", 0.0))
    low_below_window      = int(rule.get("low_below_window_px", 0))
    low_below_thresh_frac = float(rule.get("low_below_threshold_frac", 0.3))
    low_below_min_str     = float(rule.get("low_below_min_strength", 0.0))
    high_above_window     = int(rule.get("high_above_window_px", 0))
    high_above_thresh_frac = float(rule.get("high_above_threshold_frac", 0.7))
    high_above_min_str    = float(rule.get("high_above_min_strength", 1.0))
    high_below_window     = int(rule.get("high_below_window_px", 0))
    high_below_thresh_frac = float(rule.get("high_below_threshold_frac", 0.7))
    high_below_min_str    = float(rule.get("high_below_min_strength", 1.0))

    # Guard: if the original signal is constant there are no real peaks
    orig_range = float(signal.max()) - float(signal.min())
    if orig_range < 1e-9:
        return np.zeros(len(signal), dtype=np.float32)

    s = _smooth(signal, smoothing).astype(np.float32)
    sig_range = float(s.max() - s.min())
    if sig_range < 1e-9:
        return np.zeros(len(signal), dtype=np.float32)

    sig_max = float(s.max())
    sig_min = float(s.min())
    n = len(s)
    min_prom_abs    = min_prom_frac * orig_range
    threshold_level = sig_min + min_prom_abs
    val_denom       = max(sig_max - threshold_level, 1e-9)
    use_val_scale   = (value_min_str != 1.0)

    low_above_abs  = sig_min + low_above_thresh_frac * sig_range
    use_low_above  = (low_above_window > 0 and low_above_min_str < 1.0)
    low_below_abs  = sig_min + low_below_thresh_frac * sig_range
    use_low_below  = (low_below_window > 0 and low_below_min_str < 1.0)
    high_above_abs = sig_min + high_above_thresh_frac * sig_range
    use_high_above = (high_above_window > 0 and high_above_min_str < 1.0)
    high_below_abs = sig_min + high_below_thresh_frac * sig_range
    use_high_below = (high_below_window > 0 and high_below_min_str < 1.0)

    def _peak_weight(idx: int, raw_score: float) -> float:
        """Chain all per-peak multipliers and return the final weight."""
        w = raw_score / orig_range  # normalise raw prominence / edge score

        # value_min_strength: de-emphasise peaks whose value is near threshold
        if use_val_scale:
            t = min(1.0, max(0.0, (sig_max - float(s[idx])) / val_denom))
            w *= 1.0 - t * (1.0 - value_min_str)

        # low_above: prefer peaks immediately below a quiescent (low-signal) zone
        if use_low_above:
            start = max(0, idx - low_above_window)
            above = s[start:idx]
            low_frac = float(np.sum(above <= low_above_abs)) / max(1, len(above))
            w *= low_frac * (1.0 - low_above_min_str) + low_above_min_str

        # low_below: prefer peaks immediately above a quiescent (low-signal) zone
        if use_low_below:
            end = min(n, idx + low_below_window + 1)
            below = s[idx + 1:end]
            low_frac = float(np.sum(below <= low_below_abs)) / max(1, len(below))
            w *= low_frac * (1.0 - low_below_min_str) + low_below_min_str

        # high_above: prefer peaks that are immediately below a high-signal zone
        if use_high_above:
            start = max(0, idx - high_above_window)
            above = s[start:idx]
            high_frac = float(np.sum(above >= high_above_abs)) / max(1, len(above))
            w *= high_frac * (1.0 - high_above_min_str) + high_above_min_str

        # high_below: prefer peaks that are immediately above a high-signal zone
        if use_high_below:
            end = min(n, idx + high_below_window + 1)
            below = s[idx + 1:end]
            high_frac = float(np.sum(below >= high_below_abs)) / max(1, len(below))
            w *= high_frac * (1.0 - high_below_min_str) + high_below_min_str

        return w

    activation = np.zeros(len(signal), dtype=np.float32)

    if peak_side in ("top", "bottom"):
        window  = max(1, int(round(spread_px)))
        idx_arr = np.arange(n, dtype=np.int32)
        if peak_side == "top":
            ref_idx = np.maximum(idx_arr - window, 0)
        else:  # "bottom"
            ref_idx = np.minimum(idx_arr + window, n - 1)
        edge_score = np.maximum(s - s[ref_idx], 0.0)

        indices, props = _sp_peaks(edge_score, height=min_prom_abs)
        if len(indices) == 0:
            return activation
        for idx, eh in zip(indices, props["peak_heights"]):
            activation[int(idx)] = _peak_weight(int(idx), float(eh))

    else:
        # "both" — classic symmetric prominence-based peak detection
        indices, props = _sp_peaks(s, prominence=min_prom_abs)
        if len(indices) == 0:
            return activation
        for idx, prom in zip(indices, props["prominences"]):
            activation[int(idx)] = _peak_weight(int(idx), float(prom))

    return _smear(activation, spread_px)


def _trigger_high_signal(
    signal: np.ndarray,
    rule: Dict[str, Any],
) -> np.ndarray:
    """
    Returns per-row activation = normalised signal value where it exceeds
    threshold_frac × signal_range, else 0.
    """
    smoothing = int(rule.get("smoothing_px", 5))
    threshold_frac = float(rule.get("threshold_frac", 0.3))
    spread_px = float(rule.get("spread_px", 3.0))

    # Guard: constant original signal → no meaningful "high" region
    orig_range = float(signal.max()) - float(signal.min())
    if orig_range < 1e-9:
        return np.zeros(len(signal), dtype=np.float32)

    s = _smooth(signal, smoothing).astype(np.float32)
    sig_range = float(s.max() - s.min())
    if sig_range < 1e-9:
        return np.zeros(len(signal), dtype=np.float32)

    threshold = float(s.min()) + threshold_frac * sig_range
    norm = _normalize(s)
    activation = np.where(s >= threshold, norm, 0.0).astype(np.float32)
    return _smear(activation, spread_px)


def _trigger_low_signal(
    signal: np.ndarray,
    rule: Dict[str, Any],
) -> np.ndarray:
    """
    Returns per-row activation = 1 − normalised_signal where signal is below
    threshold_frac × signal_range from the top (i.e. close to the minimum).
    """
    smoothing = int(rule.get("smoothing_px", 5))
    threshold_frac = float(rule.get("threshold_frac", 0.3))
    spread_px = float(rule.get("spread_px", 3.0))

    s = _smooth(signal, smoothing).astype(np.float32)
    sig_range = float(s.max() - s.min())
    if sig_range < 1e-9:
        return np.ones(len(signal), dtype=np.float32)

    # Threshold from the bottom: rows below min + threshold_frac*range
    threshold = float(s.min()) + threshold_frac * sig_range
    inv_norm = 1.0 - _normalize(s)
    activation = np.where(s <= threshold, inv_norm, 0.0).astype(np.float32)
    return _smear(activation, spread_px)


def _trigger_rising_edge(
    signal: np.ndarray,
    rule: Dict[str, Any],
) -> np.ndarray:
    """
    Detects the leading edge of an upward threshold crossing.

    Rather than locating the peak of a spike (which shifts with spike
    amplitude), this places a focused impulse at the row where the smoothed
    signal first crosses from below to at-or-above
    ``sig_min + threshold_frac * sig_range``.  The impulse weight is
    proportional to the magnitude of the crossing (how far above the
    threshold the signal lands), so strong spikes score higher than weak ones.

    Parameters
    ----------
    threshold_frac : float
        Fraction of signal range that defines the crossing level.
        Default 0.3.
    smoothing_px : int
        Pre-smoothing window (rolling average).  Default 5.
    spread_px : float
        σ of Gaussian applied to each crossing impulse.  Default 5.0.
    pre_low_window_px : int
        If > 0, the ``pre_low_window_px`` rows immediately before each
        crossing are inspected.  The crossing weight is multiplied by the
        fraction of those rows that are below the threshold, so crossings
        that did not come from a genuine low region are down-weighted.
        Default 0 (disabled).
    pre_low_frac : float
        Minimum fraction of the pre-window that must be below threshold for
        the crossing to survive (others are zeroed out).
        Only used when ``pre_low_window_px > 0``.  Default 0.5.
    """
    smoothing          = int(rule.get("smoothing_px", 5))
    threshold_frac     = float(rule.get("threshold_frac", 0.3))
    spread_px          = float(rule.get("spread_px", 5.0))
    pre_low_window     = int(rule.get("pre_low_window_px", 0))
    pre_low_frac_min   = float(rule.get("pre_low_frac", 0.5))

    orig_range = float(signal.max()) - float(signal.min())
    if orig_range < 1e-9:
        return np.zeros(len(signal), dtype=np.float32)

    s = _smooth(signal, smoothing).astype(np.float32)
    sig_range = float(s.max() - s.min())
    if sig_range < 1e-9:
        return np.zeros(len(signal), dtype=np.float32)

    threshold = float(s.min()) + threshold_frac * sig_range
    n = len(s)
    below = s < threshold

    activation = np.zeros(n, dtype=np.float32)
    for i in range(1, n):
        if below[i - 1] and not below[i]:
            # Crossing found at row i
            # Weight = how far above the threshold the signal lands
            weight = (float(s[i]) - threshold) / sig_range

            # Optional pre-window gate: must have been genuinely low before
            if pre_low_window > 0:
                start = max(0, i - pre_low_window)
                pre_slice = below[start:i]
                frac_low = float(np.sum(pre_slice)) / max(1, len(pre_slice))
                if frac_low < pre_low_frac_min:
                    continue
                weight *= frac_low

            activation[i] = max(weight, 0.0)

    return _smear(activation, spread_px)


_TRIGGER_FNS = {
    "peak":         _trigger_peak,
    "high_signal":  _trigger_high_signal,
    "low_signal":   _trigger_low_signal,
    "rising_edge":  _trigger_rising_edge,
}


# ---------------------------------------------------------------------------
# Location prior
# ---------------------------------------------------------------------------

def _apply_location_prior(
    activation: np.ndarray,
    prior_anchors: list,
    global_z_axis: np.ndarray,
) -> np.ndarray:
    """
    Multiply *activation* element-wise by a weight curve defined by
    *prior_anchors* — a list of [z_frac, weight] pairs.

    z_frac = 0.0 corresponds to the minimum z in *global_z_axis* (top of
    the display); z_frac = 1.0 corresponds to the maximum z (bottom).
    Linear interpolation is used; values outside the anchor range are
    clamped to the nearest endpoint weight.
    """
    if not prior_anchors:
        return activation

    # Sort anchors by z_frac and unpack
    sorted_anchors = sorted(prior_anchors, key=lambda a: a[0])
    anchor_fracs   = np.array([a[0] for a in sorted_anchors], dtype=np.float64)
    anchor_weights = np.array([a[1] for a in sorted_anchors], dtype=np.float64)

    z_min = float(global_z_axis[0])
    z_max = float(global_z_axis[-1])
    z_span = max(z_max - z_min, 1e-9)

    # Convert global_z_axis to fractions
    z_fracs = (global_z_axis.astype(np.float64) - z_min) / z_span

    # Interpolate weight at every row, clamping outside the anchor range
    weights = np.interp(
        z_fracs,
        anchor_fracs,
        anchor_weights,
        left=anchor_weights[0],
        right=anchor_weights[-1],
    ).astype(np.float32)

    return activation * weights


# ---------------------------------------------------------------------------
# Rule application
# ---------------------------------------------------------------------------

_STATE_TARGETS = {
    "gas_in_tip":    STATE_GAS,
    "liquid_in_tip": STATE_LIQ,
    "below_tip":     STATE_BELOW,
}

_BOUNDARY_TARGETS = {"tip_bottom", "meniscus"}


def _apply_rule(
    signal: np.ndarray,
    rule: Dict[str, Any],
    state_scores: np.ndarray,       # (N, 3) — mutated in place
    tip_bottom_density: np.ndarray, # (N,)   — mutated in place
    meniscus_density: np.ndarray,   # (N,)   — mutated in place
    z_axis_full: Optional[np.ndarray],
    global_z_axis: np.ndarray,
) -> None:
    """
    Apply one rule to accumulate evidence into state_scores or boundary densities.

    signal and z_axis_full are in the mode's coordinate system (may be a
    sub-range of global_z_axis).  We interpolate the activation onto
    global_z_axis before accumulating.
    """
    trigger_name = rule.get("trigger", "")
    trigger_fn = _TRIGGER_FNS.get(trigger_name)
    if trigger_fn is None:
        return   # unknown trigger — silently skip

    activation = trigger_fn(signal, rule)   # per-row in mode-local coords

    # Apply spatial modifier (shift up/down) before projecting to global axis
    modifier = rule.get("spatial_modifier")
    activation = _apply_spatial_modifier(activation, modifier)

    # Project mode-local activation onto global z axis by interpolation.
    # If z_axis_full is None, fall back to treating indices as z values.
    if z_axis_full is not None and len(z_axis_full) == len(signal):
        activation_global = np.interp(
            global_z_axis,
            z_axis_full.astype(np.float64),
            activation.astype(np.float64),
            left=0.0, right=0.0,
        ).astype(np.float32)
    else:
        # No z_axis: assume signal spans the full global range, resample
        x_local = np.linspace(
            float(global_z_axis[0]), float(global_z_axis[-1]), len(signal)
        )
        activation_global = np.interp(
            global_z_axis.astype(np.float64),
            x_local,
            activation.astype(np.float64),
            left=0.0, right=0.0,
        ).astype(np.float32)

    # Apply location prior (after projection so it operates in global z-space)
    prior = rule.get("location_prior")
    if prior:
        activation_global = _apply_location_prior(
            activation_global, prior, global_z_axis
        )

    direction = float(rule.get("direction", 1))
    strength  = float(rule.get("strength",  1.0))
    contribution = direction * strength * activation_global

    target = rule.get("target", "")
    if target in _STATE_TARGETS:
        col = _STATE_TARGETS[target]
        state_scores[:, col] += contribution
    elif target == "tip_bottom":
        tip_bottom_density += np.maximum(0.0, contribution)
    elif target == "meniscus":
        meniscus_density += np.maximum(0.0, contribution)
    # Unknown target — silently skip


# ---------------------------------------------------------------------------
# Viterbi decoder
# ---------------------------------------------------------------------------

def viterbi_decode(
    state_scores: np.ndarray,     # (N, 3) log-likelihood per row per state
    transition_cost: Optional[List[List[float]]] = None,
) -> np.ndarray:
    """
    Viterbi algorithm over a linear chain of N observations with 3 states.

    Returns the maximum-likelihood state sequence as an integer array (N,)
    with values in {STATE_GAS=0, STATE_LIQ=1, STATE_BELOW=2}.

    transition_cost[s_from][s_to] is the penalty subtracted when transitioning
    from state s_from to s_to.  The default enforces the valid physical order:
        gas_in_tip → liquid_in_tip → below_tip (no backward transitions).
    """
    if transition_cost is None:
        transition_cost = _TRANSITION_COST

    n, s = state_scores.shape
    tc = np.array(transition_cost, dtype=np.float64)

    # dp[i, s] = best score ending at row i in state s
    dp   = np.full((n, s), -np.inf, dtype=np.float64)
    back = np.zeros((n, s), dtype=np.int32)

    # Initialise: all starting states are feasible from the top
    dp[0, :] = state_scores[0, :]
    back[0, :] = np.arange(s, dtype=np.int32)

    for i in range(1, n):
        for t in range(s):
            best_score = -np.inf
            best_from  = 0
            for f in range(s):
                cost = tc[f, t]
                if cost >= _INF:
                    continue
                score = dp[i - 1, f] - cost + state_scores[i, t]
                if score > best_score:
                    best_score = score
                    best_from  = f
            dp[i, t]   = best_score
            back[i, t] = best_from

    # Backtrack
    sequence = np.empty(n, dtype=np.int32)
    sequence[n - 1] = int(np.argmax(dp[n - 1]))
    for i in range(n - 2, -1, -1):
        sequence[i] = back[i + 1, sequence[i + 1]]

    return sequence


# ---------------------------------------------------------------------------
# KDE boundary estimator
# ---------------------------------------------------------------------------

def _kde_peak(density: np.ndarray, z_axis: np.ndarray) -> Optional[float]:
    """Return z of the highest peak in density, or None if density is flat."""
    if density.max() < 1e-9:
        return None
    idx = int(np.argmax(density))
    return float(z_axis[idx])


def _state_transition_weights(
    state_sequence: np.ndarray,
    from_state: int,
    to_state: int,
    spread_px: float = 15.0,
) -> np.ndarray:
    """Return per-row weights (0–1) peaking at rows where the Viterbi state
    transitions from *from_state* to *to_state*.

    If no such transition exists the array is all-ones (no weighting applied).
    Multiple transitions produce overlapping Gaussian bumps; after smearing the
    result is normalised to [0, 1] so the weight never exceeds 1.

    Used to refine boundary densities: e.g. tip_bottom evidence should be
    strongest where the LIQ→BELOW transition actually occurs.
    """
    n = len(state_sequence)
    # Transition at row i means seq[i-1]==from_state and seq[i]==to_state.
    if n < 2:
        return np.ones(n, dtype=np.float32)

    transitions = np.where(
        (state_sequence[1:] == to_state) & (state_sequence[:-1] == from_state)
    )[0] + 1   # +1: index of first row *in* the new state

    if len(transitions) == 0:
        return np.ones(n, dtype=np.float32)

    impulse = np.zeros(n, dtype=np.float32)
    impulse[transitions] = 1.0
    weights = _smear(impulse, spread_px)
    w_max = float(weights.max())
    if w_max > 1e-9:
        weights /= w_max
    return weights.clip(0.0, 1.0)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def interpret_signals(
    signals: List[Dict[str, Any]],
    rules_per_signal: List[List[Dict[str, Any]]],
    global_z_axis: np.ndarray,
) -> InterpretationResult:
    """
    Interpret one or more signals using their associated behavior rules.

    Parameters
    ----------
    signals : list of dicts
        Each dict must have:
            "signal"   : np.ndarray (1-D)
            "z_axis"   : np.ndarray (same length, full-image coords) or None
        Extra keys are ignored.

    rules_per_signal : list of lists
        ``rules_per_signal[i]`` is the list of behavior rule dicts for
        ``signals[i]``.  Must be the same length as *signals*.

    global_z_axis : np.ndarray
        The common z axis (full-image row coordinates) onto which all
        activations are projected before accumulation.  Typically the
        union of all signal z-ranges from Mode Compare's _render_all.

    Returns
    -------
    InterpretationResult
    """
    n = len(global_z_axis)
    state_scores       = np.zeros((n, 3), dtype=np.float32)
    tip_bottom_density = np.zeros(n, dtype=np.float32)
    meniscus_density   = np.zeros(n, dtype=np.float32)

    for sig_dict, rules in zip(signals, rules_per_signal):
        # Accept either a dict {"signal": ..., "z_axis": ...}
        # or a plain tuple/list (signal_array, z_axis_array).
        if isinstance(sig_dict, dict):
            raw_signal  = np.asarray(sig_dict["signal"], dtype=np.float32)
            z_axis_full = sig_dict.get("z_axis")
        else:
            raw_signal  = np.asarray(sig_dict[0], dtype=np.float32)
            z_axis_full = sig_dict[1] if len(sig_dict) > 1 else None

        if z_axis_full is not None:
            z_axis_full = np.asarray(z_axis_full, dtype=np.float32)

        if raw_signal.size < 2:
            continue

        for rule in rules:
            if not rule.get("enabled", True):  # skip disabled rules
                continue
            _apply_rule(
                raw_signal, rule,
                state_scores, tip_bottom_density, meniscus_density,
                z_axis_full, global_z_axis,
            )

    state_sequence = viterbi_decode(state_scores.astype(np.float64))

    # Refine boundary densities by the Viterbi transition location.
    # tip_bottom evidence is strongest where LIQ→BELOW transitions occur;
    # meniscus evidence is strongest where GAS→LIQ transitions occur.
    # If the Viterbi sequence has no matching transition the weights are all-ones
    # (no effect), preserving the unweighted behaviour as a fallback.
    tb_weights = _state_transition_weights(state_sequence, STATE_LIQ, STATE_BELOW)
    mn_weights = _state_transition_weights(state_sequence, STATE_GAS, STATE_LIQ)
    tip_bottom_density = tip_bottom_density * tb_weights
    meniscus_density   = meniscus_density   * mn_weights

    tip_bottom_z = _kde_peak(tip_bottom_density, global_z_axis)
    meniscus_z   = _kde_peak(meniscus_density,   global_z_axis)

    return InterpretationResult(
        z_axis             = global_z_axis.copy(),
        state_scores       = state_scores,
        state_sequence     = state_sequence,
        tip_bottom_density = tip_bottom_density,
        meniscus_density   = meniscus_density,
        tip_bottom_z       = tip_bottom_z,
        meniscus_z         = meniscus_z,
    )
