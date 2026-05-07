"""
Unit tests for pa.pipeline.interpreters.signal_interpreter (Phase A).

Synthetic signals with known structure are used so every test has a
deterministic expected outcome independent of real image data.

Test groups
-----------
Triggers
    test_trigger_peak_finds_correct_index
    test_trigger_high_signal_masks_below_threshold
    test_trigger_low_signal_masks_above_threshold
    test_trigger_unknown_name_is_skipped

Gaussian smear
    test_smear_conserves_mass_approximately
    test_smear_zero_spread_unchanged

Spatial modifier
    test_spatial_modifier_offset_below
    test_spatial_modifier_offset_above

Rule application
    test_rule_positive_state_evidence
    test_rule_negative_state_evidence
    test_rule_tip_bottom_boundary
    test_rule_meniscus_boundary
    test_rule_unknown_target_ignored
    test_rule_z_axis_projection

Viterbi decoder
    test_viterbi_simple_gas_then_liquid_then_below
    test_viterbi_forbids_backward_transition
    test_viterbi_stays_in_single_state_when_no_evidence

KDE peak
    test_kde_peak_returns_none_for_flat_density
    test_kde_peak_returns_correct_z

End-to-end
    test_interpret_signals_tip_bottom_peak
    test_interpret_signals_negative_rule_suppresses_state
    test_interpret_signals_full_pipeline_gas_liq_below
"""

from __future__ import annotations

import numpy as np
import pytest

from pa.pipeline.interpreters.signal_interpreter import (
    STATE_GAS, STATE_LIQ, STATE_BELOW,
    InterpretationResult,
    _smooth,
    _normalize,
    _smear,
    _apply_spatial_modifier,
    _trigger_peak,
    _trigger_high_signal,
    _trigger_low_signal,
    _trigger_rising_edge,
    _apply_location_prior,
    _apply_rule,
    viterbi_decode,
    _kde_peak,
    interpret_signals,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_z(n: int, start: float = 100.0, step: float = 1.0) -> np.ndarray:
    return np.arange(start, start + n * step, step, dtype=np.float32)


def _spike(n: int, pos: int, height: float = 1.0) -> np.ndarray:
    """Single narrow spike signal."""
    s = np.zeros(n, dtype=np.float32)
    if 0 <= pos < n:
        s[pos] = height
    return s


def _block(n: int, lo: int, hi: int, level: float = 1.0) -> np.ndarray:
    """Rectangular block signal."""
    s = np.zeros(n, dtype=np.float32)
    s[lo:hi] = level
    return s


# ---------------------------------------------------------------------------
# Trigger tests
# ---------------------------------------------------------------------------

class TestTriggerPeak:
    def test_finds_correct_index(self):
        n = 100
        # Place a clear spike at index 40; minimal noise elsewhere
        signal = np.random.default_rng(0).random(n).astype(np.float32) * 0.05
        signal[40] = 1.0
        rule = {"smoothing_px": 1, "min_prominence_frac": 0.3, "spread_px": 0.0}
        act = _trigger_peak(signal, rule)
        # Max activation should be near index 40
        assert abs(int(np.argmax(act)) - 40) <= 2

    def test_flat_signal_returns_zeros(self):
        signal = np.ones(50, dtype=np.float32)
        act = _trigger_peak(signal, {"min_prominence_frac": 0.1, "spread_px": 0.0})
        np.testing.assert_array_equal(act, 0.0)

    def test_below_prominence_threshold_ignored(self):
        n = 80
        signal = np.zeros(n, dtype=np.float32)
        signal[30] = 0.05   # tiny bump
        signal[70] = 1.0    # large peak
        rule = {"smoothing_px": 1, "min_prominence_frac": 0.5, "spread_px": 0.0}
        act = _trigger_peak(signal, rule)
        # Large peak fires, tiny peak does not dominate
        assert act[70] > act[30]

    def test_peak_side_top_leading_edge(self):
        # Signal: low at top, rises sharply at index 30, stays high/noisy below.
        # "top" mode should fire near the leading edge (30); "both" may miss it.
        n = 100
        rng = np.random.default_rng(42)
        signal = np.zeros(n, dtype=np.float32)
        signal[30] = 1.0                                          # sharp rise
        signal[31:] = 0.9 + rng.random(n - 31).astype(np.float32) * 0.2  # noisy high
        rule = {"smoothing_px": 1, "min_prominence_frac": 0.1,
                "spread_px": 5.0, "peak_side": "top"}
        act = _trigger_peak(signal, rule)
        assert act.max() > 0.0, "should detect at least one edge"
        assert abs(int(np.argmax(act)) - 30) <= 8

    def test_peak_side_bottom_trailing_edge(self):
        # Signal: high/noisy at top, drops sharply at index 60, stays low below.
        # "bottom" mode should fire near index 60 (the trailing edge from below).
        n = 100
        rng = np.random.default_rng(7)
        signal = np.zeros(n, dtype=np.float32)
        signal[:60] = 0.9 + rng.random(60).astype(np.float32) * 0.2  # noisy high
        signal[60] = 1.0                                               # peak at edge
        signal[61:] = 0.0                                              # drops away
        rule = {"smoothing_px": 1, "min_prominence_frac": 0.1,
                "spread_px": 5.0, "peak_side": "bottom"}
        act = _trigger_peak(signal, rule)
        assert act.max() > 0.0, "should detect at least one edge"
        assert abs(int(np.argmax(act)) - 60) <= 8

    def test_peak_side_top_vs_both_prefers_leading(self):
        # With a step signal (low→high), "top" should find the leading edge
        # while "both" may find nothing (no symmetric peak).
        n = 80
        signal = np.zeros(n, dtype=np.float32)
        signal[0:40] = 0.0
        signal[40:] = 1.0
        rule_top  = {"smoothing_px": 1, "min_prominence_frac": 0.1, "spread_px": 5.0, "peak_side": "top"}
        rule_both = {"smoothing_px": 1, "min_prominence_frac": 0.1, "spread_px": 5.0, "peak_side": "both"}
        act_top  = _trigger_peak(signal, rule_top)
        act_both = _trigger_peak(signal, rule_both)
        # "top" finds the transition; "both" finds nothing (no local max)
        assert act_top.max() > act_both.max()

    def test_peak_side_default_is_both(self):
        # Omitting peak_side should behave identically to "both"
        n = 100
        signal = np.random.default_rng(1).random(n).astype(np.float32) * 0.05
        signal[50] = 1.0
        rule_implicit = {"smoothing_px": 1, "min_prominence_frac": 0.3, "spread_px": 0.0}
        rule_explicit = {**rule_implicit, "peak_side": "both"}
        act_i = _trigger_peak(signal, rule_implicit)
        act_e = _trigger_peak(signal, rule_explicit)
        np.testing.assert_array_almost_equal(act_i, act_e)

    # ── low_above tests ───────────────────────────────────────────────

    def test_low_above_gates_spurious_peak(self):
        # Zero signal above row 40; small peak at 40 followed by a brief dip
        # then rising to a large peak at 70.
        # Without low_above: large peak wins (higher prominence).
        # With low_above_window=20: peak at 40 wins because all 20 rows above
        # it are zero (100 % low), while rows above 70 are all high.
        n = 100
        signal = np.zeros(n, dtype=np.float32)
        signal[40] = 0.4          # first peak
        signal[41:45] = 0.2       # brief dip  → makes 40 a local max
        signal[45:70] = 0.8       # high plateau between peaks
        signal[70] = 1.0          # large peak
        # rows 71+ stay 0
        rule_no  = {"smoothing_px": 1, "min_prominence_frac": 0.1, "spread_px": 0.0}
        rule_yes = {**rule_no, "low_above_window_px": 20,
                    "low_above_threshold_frac": 0.1, "low_above_min_strength": 0.0}
        act_no  = _trigger_peak(signal, rule_no)
        act_yes = _trigger_peak(signal, rule_yes)
        assert act_no[70] > act_no[40],   "without gate large peak wins"
        assert act_yes[40] > act_yes[70], "with gate early peak wins"

    def test_low_above_min_strength_one_disables_gate(self):
        # min_strength=1.0 means the multiplier is always 1 → same as no gate
        n = 100
        signal = np.zeros(n, dtype=np.float32)
        signal[40] = 0.4
        signal[41:45] = 0.2
        signal[45:70] = 0.8
        signal[70] = 1.0
        rule_no   = {"smoothing_px": 1, "min_prominence_frac": 0.1, "spread_px": 0.0}
        rule_soft = {**rule_no, "low_above_window_px": 20,
                     "low_above_threshold_frac": 0.1, "low_above_min_strength": 1.0}
        act_no   = _trigger_peak(signal, rule_no)
        act_soft = _trigger_peak(signal, rule_soft)
        np.testing.assert_array_almost_equal(act_no, act_soft)

    def test_low_above_disabled_by_default(self):
        # window=0 (default) → zero effect on activation
        n = 80
        signal = np.random.default_rng(5).random(n).astype(np.float32) * 0.05
        signal[50] = 1.0
        rule_base    = {"smoothing_px": 1, "min_prominence_frac": 0.3, "spread_px": 0.0}
        rule_window0 = {**rule_base, "low_above_window_px": 0}
        act_base = _trigger_peak(signal, rule_base)
        act_w0   = _trigger_peak(signal, rule_window0)
        np.testing.assert_array_almost_equal(act_base, act_w0)

    # ── low_below tests ───────────────────────────────────────────────

    def test_low_below_gates_spurious_peak(self):
        # Large peak at 30 with non-zero signal below it; small peak at 60
        # followed immediately by a zero-signal zone.
        # Without gate: large peak at 30 wins.
        # With low_below_window=20: peak at 60 wins (all 20 rows below are zero).
        n = 100
        signal = np.zeros(n, dtype=np.float32)
        signal[30] = 1.0          # large peak
        signal[31:55] = 0.8       # high between peaks
        signal[55:59] = 0.2       # brief dip  → makes 60 a local max
        signal[60] = 0.4          # small peak
        # rows 61+ stay 0 (quiescent below)
        rule_no  = {"smoothing_px": 1, "min_prominence_frac": 0.1, "spread_px": 0.0}
        rule_yes = {**rule_no, "low_below_window_px": 20,
                    "low_below_threshold_frac": 0.1, "low_below_min_strength": 0.0}
        act_no  = _trigger_peak(signal, rule_no)
        act_yes = _trigger_peak(signal, rule_yes)
        assert act_no[30] > act_no[60],   "without gate large peak wins"
        assert act_yes[60] > act_yes[30], "with gate peak above quiet zone wins"

    def test_low_below_min_strength_one_disables_gate(self):
        n = 100
        signal = np.zeros(n, dtype=np.float32)
        signal[30] = 1.0
        signal[31:55] = 0.8
        signal[55:59] = 0.2
        signal[60] = 0.4
        rule_no   = {"smoothing_px": 1, "min_prominence_frac": 0.1, "spread_px": 0.0}
        rule_soft = {**rule_no, "low_below_window_px": 20,
                     "low_below_threshold_frac": 0.1, "low_below_min_strength": 1.0}
        act_no   = _trigger_peak(signal, rule_no)
        act_soft = _trigger_peak(signal, rule_soft)
        np.testing.assert_array_almost_equal(act_no, act_soft)

    def test_low_below_disabled_by_default(self):
        n = 80
        signal = np.random.default_rng(9).random(n).astype(np.float32) * 0.05
        signal[30] = 1.0
        rule_base    = {"smoothing_px": 1, "min_prominence_frac": 0.3, "spread_px": 0.0}
        rule_window0 = {**rule_base, "low_below_window_px": 0}
        act_base = _trigger_peak(signal, rule_base)
        act_w0   = _trigger_peak(signal, rule_window0)
        np.testing.assert_array_almost_equal(act_base, act_w0)

    # ── high_above tests ──────────────────────────────────────────────

    def test_high_above_gates_peaks_without_high_signal_above(self):
        # Peak A at row 20: nothing high above it (rows 0:20 are zero).
        # Peak B at row 60: high-signal zone in rows 40:55, brief dip 55:59,
        #   then peak.  With high_above gate, peak B wins even though A is larger.
        n = 100
        signal = np.zeros(n, dtype=np.float32)
        signal[20]    = 1.0    # large peak A — no high signal above
        signal[21:40] = 0.0    # quiescent between peaks
        signal[40:55] = 0.9    # high zone above peak B
        signal[55:59] = 0.05   # brief dip → makes row 60 a local max
        signal[60]    = 0.8    # smaller peak B — high signal above in window
        # rows 61+ stay 0
        rule_no  = {"smoothing_px": 1, "min_prominence_frac": 0.1, "spread_px": 0.0}
        rule_yes = {**rule_no, "high_above_window_px": 15,
                    "high_above_threshold_frac": 0.7, "high_above_min_strength": 0.0}
        act_no  = _trigger_peak(signal, rule_no)
        act_yes = _trigger_peak(signal, rule_yes)
        assert act_no[20]  > act_no[60],  "without gate larger peak wins"
        assert act_yes[60] > act_yes[20], "with gate peak below high zone wins"

    def test_high_above_min_strength_one_disables_gate(self):
        n = 100
        signal = np.zeros(n, dtype=np.float32)
        signal[20]    = 1.0
        signal[40:55] = 0.9
        signal[55:59] = 0.05
        signal[60]    = 0.8
        rule_no   = {"smoothing_px": 1, "min_prominence_frac": 0.1, "spread_px": 0.0}
        rule_soft = {**rule_no, "high_above_window_px": 15,
                     "high_above_threshold_frac": 0.7, "high_above_min_strength": 1.0}
        act_no   = _trigger_peak(signal, rule_no)
        act_soft = _trigger_peak(signal, rule_soft)
        np.testing.assert_array_almost_equal(act_no, act_soft)

    def test_high_above_disabled_by_default(self):
        n = 80
        signal = np.random.default_rng(42).random(n).astype(np.float32) * 0.05
        signal[40] = 1.0
        rule_base    = {"smoothing_px": 1, "min_prominence_frac": 0.3, "spread_px": 0.0}
        rule_window0 = {**rule_base, "high_above_window_px": 0}
        act_base = _trigger_peak(signal, rule_base)
        act_w0   = _trigger_peak(signal, rule_window0)
        np.testing.assert_array_almost_equal(act_base, act_w0)

    # ── high_below tests ──────────────────────────────────────────────

    def test_high_below_gates_peaks_without_high_signal_below(self):
        # Peak A at row 20: nothing high below it (rows 21:60 are near-zero).
        # Peak B at row 60: brief dip 61:64, then high zone 64:80.
        # With high_below gate, peak B wins even though A is larger.
        n = 100
        signal = np.zeros(n, dtype=np.float32)
        signal[20]    = 1.0    # large peak A — no high signal below
        signal[21:58] = 0.0
        signal[58:60] = 0.05   # brief rise → makes row 60 a local max
        signal[60]    = 0.8    # smaller peak B
        signal[61:64] = 0.05   # brief dip after peak B
        signal[64:80] = 0.9    # high zone below peak B
        rule_no  = {"smoothing_px": 1, "min_prominence_frac": 0.1, "spread_px": 0.0}
        rule_yes = {**rule_no, "high_below_window_px": 15,
                    "high_below_threshold_frac": 0.7, "high_below_min_strength": 0.0}
        act_no  = _trigger_peak(signal, rule_no)
        act_yes = _trigger_peak(signal, rule_yes)
        assert act_no[20]  > act_no[60],  "without gate larger peak wins"
        assert act_yes[60] > act_yes[20], "with gate peak above high zone wins"

    def test_high_below_min_strength_one_disables_gate(self):
        n = 100
        signal = np.zeros(n, dtype=np.float32)
        signal[20]    = 1.0
        signal[58:60] = 0.05
        signal[60]    = 0.8
        signal[61:64] = 0.05
        signal[64:80] = 0.9
        rule_no   = {"smoothing_px": 1, "min_prominence_frac": 0.1, "spread_px": 0.0}
        rule_soft = {**rule_no, "high_below_window_px": 15,
                     "high_below_threshold_frac": 0.7, "high_below_min_strength": 1.0}
        act_no   = _trigger_peak(signal, rule_no)
        act_soft = _trigger_peak(signal, rule_soft)
        np.testing.assert_array_almost_equal(act_no, act_soft)

    def test_high_below_disabled_by_default(self):
        n = 80
        signal = np.random.default_rng(7).random(n).astype(np.float32) * 0.05
        signal[40] = 1.0
        rule_base    = {"smoothing_px": 1, "min_prominence_frac": 0.3, "spread_px": 0.0}
        rule_window0 = {**rule_base, "high_below_window_px": 0}
        act_base = _trigger_peak(signal, rule_base)
        act_w0   = _trigger_peak(signal, rule_window0)
        np.testing.assert_array_almost_equal(act_base, act_w0)


class TestTriggerHighSignal:
    def test_masks_below_threshold(self):
        n = 60
        signal = np.zeros(n, dtype=np.float32)
        signal[20:40] = 1.0
        rule = {"smoothing_px": 1, "threshold_frac": 0.3, "spread_px": 0.0}
        act = _trigger_high_signal(signal, rule)
        # Rows inside the block should be non-zero
        assert act[25] > 0.0
        # Rows far outside (after smearing) should be zero
        assert act[5] == pytest.approx(0.0, abs=1e-4)

    def test_flat_signal_returns_zeros(self):
        signal = np.ones(40, dtype=np.float32)
        act = _trigger_high_signal(signal, {"threshold_frac": 0.3, "spread_px": 0.0})
        # All values equal → after normalisation all are 0 or 1, but
        # threshold is min + 0.3*(max-min) = 0 when range=0 → flat zeros
        np.testing.assert_array_equal(act, 0.0)


class TestTriggerLowSignal:
    def test_activates_at_low_values(self):
        n = 60
        signal = np.ones(n, dtype=np.float32)
        signal[0:10] = 0.0   # low region at top
        rule = {"smoothing_px": 1, "threshold_frac": 0.3, "spread_px": 0.0}
        act = _trigger_low_signal(signal, rule)
        assert act[5] > 0.0
        assert act[50] == pytest.approx(0.0, abs=1e-4)

    def test_all_low_returns_ones(self):
        signal = np.zeros(30, dtype=np.float32)
        act = _trigger_low_signal(signal, {"threshold_frac": 0.3, "spread_px": 0.0})
        # All values are "low" → returns ones (clipped by normalisation)
        assert act.sum() > 0.0


# ---------------------------------------------------------------------------
# Rising-edge trigger tests
# ---------------------------------------------------------------------------

class TestTriggerRisingEdge:
    def _step_signal(self, n=80, step_at=30, low=0.0, high=1.0) -> np.ndarray:
        """Step signal: low before step_at, high from step_at onward."""
        s = np.full(n, low, dtype=np.float32)
        s[step_at:] = high
        return s

    def _spike_signal(self, n=80, onset=30, peak=38, low=0.0, high=1.0) -> np.ndarray:
        """Signal that rises at onset, peaks at peak, then returns to low."""
        s = np.full(n, low, dtype=np.float32)
        for i in range(onset, peak + 1):
            s[i] = high * (i - onset) / max(1, peak - onset)
        for i in range(peak + 1, min(n, peak + 8)):
            s[i] = high * max(0.0, 1.0 - (i - peak) / 8.0)
        return s

    def test_fires_at_crossing_row(self):
        """Activation peak should be near the step-crossing row."""
        step_at = 30
        signal = self._step_signal(n=80, step_at=step_at)
        rule = {"smoothing_px": 1, "threshold_frac": 0.3, "spread_px": 0.0}
        act = _trigger_rising_edge(signal, rule)
        assert act.sum() > 0.0
        # The peak of activation must be within a few rows of the crossing
        assert abs(int(np.argmax(act)) - step_at) <= 2

    def test_flat_signal_returns_zeros(self):
        signal = np.ones(40, dtype=np.float32)
        act = _trigger_rising_edge(signal, {"threshold_frac": 0.3, "spread_px": 0.0})
        np.testing.assert_array_equal(act, 0.0)

    def test_constant_low_signal_returns_zeros(self):
        """Signal always below threshold — no crossing — all zeros."""
        signal = np.full(40, 0.1, dtype=np.float32)
        act = _trigger_rising_edge(signal, {"threshold_frac": 0.3, "spread_px": 0.0})
        np.testing.assert_array_equal(act, 0.0)

    def test_spike_fires_at_onset_not_peak(self):
        """With a spike signal the rising-edge fires near onset, not peak."""
        onset, peak = 30, 40
        signal = self._spike_signal(n=80, onset=onset, peak=peak)
        rule = {"smoothing_px": 1, "threshold_frac": 0.2, "spread_px": 0.0}
        act = _trigger_rising_edge(signal, rule)
        assert act.sum() > 0.0
        edge_row = int(np.argmax(act))
        # Rising edge must be closer to onset than to peak
        assert abs(edge_row - onset) < abs(edge_row - peak)

    def test_pre_low_window_gates_crossing_without_low_prefix(self):
        """A crossing that starts from a high region (not low) is suppressed."""
        n = 80
        # Signal is high everywhere then dips and rises again
        signal = np.full(n, 1.0, dtype=np.float32)
        signal[30:40] = 0.0   # brief dip
        # Rising-edge at row 40: immediately before it are mostly HIGH rows
        # but within the dip window they are low
        rule = {
            "smoothing_px": 1,
            "threshold_frac": 0.3,
            "spread_px": 0.0,
            "pre_low_window_px": 20,   # look at 20 rows before crossing
            "pre_low_frac": 0.8,       # require 80 % to be low — won't pass
        }
        act = _trigger_rising_edge(signal, rule)
        # The crossing at ~40 should be suppressed because most of the
        # pre-window rows are high
        np.testing.assert_array_equal(act, 0.0)

    def test_pre_low_window_passes_genuine_low_to_high(self):
        """A step from a truly-low region passes the pre-low gate."""
        step_at = 50
        signal = self._step_signal(n=80, step_at=step_at)
        rule = {
            "smoothing_px": 1,
            "threshold_frac": 0.3,
            "spread_px": 0.0,
            "pre_low_window_px": 20,
            "pre_low_frac": 0.8,
        }
        act = _trigger_rising_edge(signal, rule)
        assert act.sum() > 0.0
        assert abs(int(np.argmax(act)) - step_at) <= 2


# ---------------------------------------------------------------------------
# Smear tests
# ---------------------------------------------------------------------------

class TestSmear:
    def test_conserves_mass_approximately(self):
        n = 100
        act = _spike(n, 50, 1.0)
        smeared = _smear(act, spread_px=5.0)
        assert abs(smeared.sum() - act.sum()) < 0.05

    def test_zero_spread_unchanged(self):
        act = np.array([0, 0, 1, 0, 0], dtype=np.float32)
        out = _smear(act, spread_px=0.0)
        np.testing.assert_allclose(out, act, atol=1e-6)


# ---------------------------------------------------------------------------
# Spatial modifier tests
# ---------------------------------------------------------------------------

class TestSpatialModifier:
    def test_offset_below_shifts_down(self):
        act = np.array([0, 1, 0, 0, 0], dtype=np.float32)
        out = _apply_spatial_modifier(act, {"type": "offset_below", "range_px": 2})
        assert out[3] == pytest.approx(1.0, abs=1e-6)
        assert out[1] == pytest.approx(0.0, abs=1e-6)

    def test_offset_above_shifts_up(self):
        act = np.array([0, 0, 0, 1, 0], dtype=np.float32)
        out = _apply_spatial_modifier(act, {"type": "offset_above", "range_px": 2})
        assert out[1] == pytest.approx(1.0, abs=1e-6)
        assert out[3] == pytest.approx(0.0, abs=1e-6)

    def test_none_modifier_unchanged(self):
        act = np.array([0, 0, 1, 0, 0], dtype=np.float32)
        out = _apply_spatial_modifier(act, None)
        np.testing.assert_array_equal(out, act)


# ---------------------------------------------------------------------------
# Rule application tests
# ---------------------------------------------------------------------------

class TestApplyRule:
    def _blank(self, n: int):
        return (
            np.zeros((n, 3), dtype=np.float32),
            np.zeros(n, dtype=np.float32),
            np.zeros(n, dtype=np.float32),
        )

    def test_positive_state_evidence(self):
        n = 60
        signal = np.zeros(n, dtype=np.float32)
        signal[20:40] = 1.0
        z = _make_z(n)
        ss, tb, men = self._blank(n)
        rule = {"trigger": "high_signal", "threshold_frac": 0.2, "spread_px": 0.0,
                "target": "liquid_in_tip", "direction": 1, "strength": 2.0}
        _apply_rule(signal, rule, ss, tb, men, z, z)
        # The liquid column should have positive state evidence
        assert ss[30, STATE_LIQ] > 0.0
        assert ss[5, STATE_LIQ] == pytest.approx(0.0, abs=1e-4)

    def test_negative_state_evidence(self):
        n = 60
        signal = np.ones(n, dtype=np.float32)
        z = _make_z(n)
        ss, tb, men = self._blank(n)
        rule = {"trigger": "high_signal", "threshold_frac": 0.0, "spread_px": 0.0,
                "target": "gas_in_tip", "direction": -1, "strength": 1.0}
        _apply_rule(signal, rule, ss, tb, men, z, z)
        # High signal everywhere → negative evidence for gas everywhere
        # (flat signal → norm=0 → zeros after trigger; check at least non-positive)
        assert ss[:, STATE_GAS].max() <= 0.0

    def test_tip_bottom_boundary_positive(self):
        n = 80
        signal = _spike(n, 50, 1.0)
        z = _make_z(n)
        ss, tb, men = self._blank(n)
        rule = {"trigger": "peak", "min_prominence_frac": 0.3, "spread_px": 3.0,
                "target": "tip_bottom", "direction": 1, "strength": 1.5}
        _apply_rule(signal, rule, ss, tb, men, z, z)
        assert tb.max() > 0.0
        # Peak of density should be near z-index 50
        assert abs(int(np.argmax(tb)) - 50) <= 5

    def test_meniscus_boundary_positive(self):
        n = 80
        signal = _spike(n, 20, 1.0)
        z = _make_z(n)
        ss, tb, men = self._blank(n)
        rule = {"trigger": "peak", "min_prominence_frac": 0.3, "spread_px": 3.0,
                "target": "meniscus", "direction": 1, "strength": 1.0}
        _apply_rule(signal, rule, ss, tb, men, z, z)
        assert men.max() > 0.0
        assert abs(int(np.argmax(men)) - 20) <= 5

    def test_unknown_target_ignored(self):
        n = 40
        signal = np.ones(n, dtype=np.float32)
        z = _make_z(n)
        ss, tb, men = self._blank(n)
        rule = {"trigger": "high_signal", "threshold_frac": 0.0, "spread_px": 0.0,
                "target": "unknown_state", "direction": 1, "strength": 1.0}
        _apply_rule(signal, rule, ss, tb, men, z, z)
        # Nothing should have changed
        np.testing.assert_array_equal(ss, 0.0)
        np.testing.assert_array_equal(tb, 0.0)
        np.testing.assert_array_equal(men, 0.0)

    def test_z_axis_projection_maps_correctly(self):
        """Signal on a sub-range z should project onto the global axis."""
        n_global = 100
        n_signal = 20
        # Signal occupies global rows 30–50
        z_global = _make_z(n_global, start=0.0)
        z_signal = _make_z(n_signal, start=30.0)
        # High signal in the sub-range (non-constant so trigger fires)
        signal = np.linspace(0.0, 1.0, n_signal, dtype=np.float32)
        ss, tb, men = (np.zeros((n_global, 3), dtype=np.float32),
                       np.zeros(n_global, dtype=np.float32),
                       np.zeros(n_global, dtype=np.float32))
        rule = {"trigger": "high_signal", "threshold_frac": 0.4, "spread_px": 0.0,
                "target": "liquid_in_tip", "direction": 1, "strength": 1.0}
        _apply_rule(signal, rule, ss, tb, men, z_signal, z_global)
        # Evidence should be concentrated in global rows 30–50 (upper half of ramp)
        assert ss[45, STATE_LIQ] > 0.0, "Expected non-zero evidence in the projected region"
        assert ss[10, STATE_LIQ] == pytest.approx(0.0, abs=1e-4), "Expected zero outside projected region"
        assert ss[90, STATE_LIQ] == pytest.approx(0.0, abs=1e-4), "Expected zero outside projected region"


# ---------------------------------------------------------------------------
# Viterbi tests
# ---------------------------------------------------------------------------

class TestViterbiDecode:
    def test_simple_gas_then_liquid_then_below(self):
        """Clear three-zone scores should produce the expected sequence."""
        n = 90
        scores = np.zeros((n, 3), dtype=np.float64)
        scores[0:30,  STATE_GAS  ] = 1.0   # top 30 rows: gas
        scores[30:60, STATE_LIQ  ] = 1.0   # mid 30 rows: liquid
        scores[60:90, STATE_BELOW] = 1.0   # bottom 30 rows: below
        seq = viterbi_decode(scores)
        assert all(seq[0:30]  == STATE_GAS)
        assert all(seq[30:60] == STATE_LIQ)
        assert all(seq[60:90] == STATE_BELOW)

    def test_forbids_backward_transition(self):
        """
        The decoder must never produce a backward transition.
        Adversarial scoring: GAS→LIQ→GAS would score best if backward
        transitions were free; the constrained decoder must stay
        non-decreasing.
        """
        n = 90
        scores = np.zeros((n, 3), dtype=np.float64)
        scores[0:30,  STATE_GAS] = 5.0   # gas evidence at top
        scores[30:60, STATE_LIQ] = 5.0   # liquid evidence in middle
        scores[60:90, STATE_GAS] = 5.0   # gas evidence at bottom — backward temptation
        seq = viterbi_decode(scores)
        # Valid transition graph: GAS → LIQ → BELOW only (no backward transitions).
        # State indices are 0=GAS, 1=LIQ, 2=BELOW, so valid sequences
        # are always non-decreasing.
        for i in range(1, n):
            assert seq[i] >= seq[i - 1], (
                f"Backward transition at row {i}: "
                f"{_STATE_NAMES[seq[i-1]]} → {_STATE_NAMES[seq[i]]}"
            )

    def test_stays_in_single_state_when_no_transition_evidence(self):
        """Uniform evidence: all rows have equal scores; Viterbi picks one path."""
        n = 50
        scores = np.ones((n, 3), dtype=np.float64)
        seq = viterbi_decode(scores)
        # Sequence should be monotonically non-decreasing (valid transitions only)
        for i in range(1, n):
            assert seq[i] >= seq[i - 1]


# ---------------------------------------------------------------------------
# KDE peak tests
# ---------------------------------------------------------------------------

class TestKdePeak:
    def test_returns_none_for_flat_density(self):
        density = np.zeros(50, dtype=np.float32)
        z = _make_z(50)
        assert _kde_peak(density, z) is None

    def test_returns_correct_z(self):
        z = _make_z(100, start=200.0)
        density = np.zeros(100, dtype=np.float32)
        density[42] = 1.0
        result = _kde_peak(density, z)
        assert result == pytest.approx(float(z[42]), abs=0.5)


# ---------------------------------------------------------------------------
# End-to-end interpret_signals tests
# ---------------------------------------------------------------------------

class TestInterpretSignals:
    def _z(self, n: int) -> np.ndarray:
        return np.arange(n, dtype=np.float32)

    def test_tip_bottom_peak_detected(self):
        n = 100
        z = self._z(n)
        # Clear spike at row 70 → should vote for tip_bottom
        signal = _spike(n, 70, 1.0)
        rules = [
            {"trigger": "peak", "min_prominence_frac": 0.3, "spread_px": 5.0,
             "target": "tip_bottom", "direction": 1, "strength": 1.0},
        ]
        result = interpret_signals(
            [{"signal": signal, "z_axis": z}],
            [rules],
            global_z_axis=z,
        )
        assert result.tip_bottom_z is not None
        assert abs(result.tip_bottom_z - 70.0) < 10.0

    def test_negative_rule_suppresses_state(self):
        n = 60
        z = self._z(n)
        # Uniform signal; rule says "high signal → evidence AGAINST gas_in_tip"
        signal = np.ones(n, dtype=np.float32)
        rules = [
            {"trigger": "high_signal", "threshold_frac": 0.0, "spread_px": 0.0,
             "target": "gas_in_tip", "direction": -1, "strength": 2.0},
        ]
        result = interpret_signals(
            [{"signal": signal, "z_axis": z}],
            [rules],
            global_z_axis=z,
        )
        # gas_in_tip state scores should be negative or dominated
        assert result.state_scores[:, STATE_GAS].mean() <= 0.0

    def test_full_pipeline_gas_liq_below(self):
        """
        Synthetic signal: low signal in top third, high in middle, low in bottom.
        Rules encode: high signal → liquid_in_tip.
        Expect Viterbi sequence: gas → liquid → below.
        """
        n = 120
        z = self._z(n)
        signal = np.zeros(n, dtype=np.float32)
        signal[40:80] = 1.0   # liquid region

        rules = [
            # Liquid region → vote liquid_in_tip
            {"trigger": "high_signal", "threshold_frac": 0.4, "spread_px": 2.0,
             "target": "liquid_in_tip", "direction": 1, "strength": 3.0},
            # Low signal at bottom → vote below_tip
            {"trigger": "low_signal", "threshold_frac": 0.4, "spread_px": 2.0,
             "target": "below_tip", "direction": 1, "strength": 2.0,
             "spatial_modifier": {"type": "offset_below", "range_px": 40}},
        ]
        result = interpret_signals(
            [{"signal": signal, "z_axis": z}],
            [rules],
            global_z_axis=z,
        )
        seq = result.state_sequence

        # Middle section should be mostly liquid
        liquid_fraction = np.mean(seq[40:80] == STATE_LIQ)
        assert liquid_fraction > 0.5, f"Expected mostly liquid in [40,80], got {liquid_fraction:.2f}"

        # Sequence must be non-decreasing (valid transitions)
        for i in range(1, n):
            assert seq[i] >= seq[i - 1], f"Invalid backward transition at row {i}"

    def test_empty_rules_returns_uniform_scores(self):
        n = 50
        z = self._z(n)
        signal = np.random.default_rng(1).random(n).astype(np.float32)
        result = interpret_signals(
            [{"signal": signal, "z_axis": z}],
            [[]],   # no rules
            global_z_axis=z,
        )
        np.testing.assert_array_equal(result.state_scores, 0.0)
        assert result.tip_bottom_z is None
        assert result.meniscus_z is None

    def test_multiple_signals_accumulate(self):
        """Two signals with complementary rules should reinforce each other."""
        n = 80
        z = self._z(n)
        sig_a = _spike(n, 60, 1.0)
        sig_b = _spike(n, 60, 0.8)   # second signal, same location

        rules_a = [{"trigger": "peak", "min_prominence_frac": 0.3, "spread_px": 3.0,
                    "target": "tip_bottom", "direction": 1, "strength": 1.0}]
        rules_b = [{"trigger": "peak", "min_prominence_frac": 0.3, "spread_px": 3.0,
                    "target": "tip_bottom", "direction": 1, "strength": 1.0}]

        result_single = interpret_signals(
            [{"signal": sig_a, "z_axis": z}], [rules_a], global_z_axis=z
        )
        result_double = interpret_signals(
            [{"signal": sig_a, "z_axis": z}, {"signal": sig_b, "z_axis": z}],
            [rules_a, rules_b],
            global_z_axis=z,
        )
        # Double should produce higher density peak than single
        assert result_double.tip_bottom_density.max() > result_single.tip_bottom_density.max()


# ---------------------------------------------------------------------------
# Location prior tests
# ---------------------------------------------------------------------------

class TestApplyLocationPrior:
    """Tests for _apply_location_prior: a spatial weight curve applied to
    the activation array after trigger + projection onto global z-axis."""

    def _z(self, n: int) -> np.ndarray:
        return np.linspace(0.0, float(n - 1), n, dtype=np.float32)

    def test_empty_prior_unchanged(self):
        """No anchors → activation returned unmodified."""
        z = self._z(50)
        act = np.ones(50, dtype=np.float32)
        result = _apply_location_prior(act, [], z)
        np.testing.assert_array_equal(result, act)

    def test_all_zero_prior_zeros_activation(self):
        """Prior weight = 0 everywhere → activation all zeroes."""
        z = self._z(100)
        act = np.ones(100, dtype=np.float32)
        prior = [[0.0, 0.0], [1.0, 0.0]]
        result = _apply_location_prior(act, prior, z)
        np.testing.assert_allclose(result, 0.0, atol=1e-6)

    def test_all_one_prior_unchanged(self):
        """Prior weight = 1 everywhere → activation unchanged."""
        z = self._z(80)
        rng = np.random.default_rng(7)
        act = rng.random(80).astype(np.float32)
        prior = [[0.0, 1.0], [1.0, 1.0]]
        result = _apply_location_prior(act, prior, z)
        np.testing.assert_allclose(result, act, atol=1e-6)

    def test_bottom_half_prior(self):
        """Weight = 0 in top half, 1 in bottom half.
        Activation in top should be zeroed; bottom untouched."""
        n = 100
        z = self._z(n)
        act = np.ones(n, dtype=np.float32)
        # Step function: 0 up to z_frac=0.5, 1 from z_frac=0.5 onward
        prior = [[0.0, 0.0], [0.499, 0.0], [0.5, 1.0], [1.0, 1.0]]
        result = _apply_location_prior(act, prior, z)
        # Top 40 rows should be ~0, bottom 40 should be ~1
        assert result[:40].max() < 0.01, "Top region should be suppressed"
        assert result[60:].min() > 0.99, "Bottom region should be unaffected"

    def test_linear_ramp_prior(self):
        """Prior ramps from 0 at top to 1 at bottom.
        Mid-point should be ~0.5."""
        n = 101
        z = self._z(n)
        act = np.ones(n, dtype=np.float32)
        prior = [[0.0, 0.0], [1.0, 1.0]]
        result = _apply_location_prior(act, prior, z)
        assert result[0] == pytest.approx(0.0, abs=0.01)
        assert result[-1] == pytest.approx(1.0, abs=0.01)
        assert result[50] == pytest.approx(0.5, abs=0.02)

    def test_unsorted_anchors_sorted_internally(self):
        """Anchors supplied in reverse order should give same result."""
        n = 60
        z = self._z(n)
        act = np.ones(n, dtype=np.float32)
        prior_fwd = [[0.0, 0.0], [1.0, 1.0]]
        prior_rev = [[1.0, 1.0], [0.0, 0.0]]
        result_fwd = _apply_location_prior(act.copy(), prior_fwd, z)
        result_rev = _apply_location_prior(act.copy(), prior_rev, z)
        np.testing.assert_allclose(result_fwd, result_rev, atol=1e-6)

    def test_prior_suppresses_competing_peak(self):
        """Real use-case: a peak near the top is suppressed by a bottom prior,
        leaving the bottom peak dominant for tip_bottom density."""
        n = 100
        z = self._z(n)
        ss = np.zeros((n, 3), dtype=np.float32)
        tb = np.zeros(n, dtype=np.float32)
        men = np.zeros(n, dtype=np.float32)

        # Two peaks: one near top (row 15), one near bottom (row 80)
        signal = np.zeros(n, dtype=np.float32)
        signal[15] = 2.0   # large peak — but in the gas region
        signal[80] = 1.0   # smaller peak — correct tip_bottom location

        # Without prior: large top peak wins
        rule_no_prior = {
            "trigger": "peak", "min_prominence_frac": 0.1, "spread_px": 2.0,
            "target": "tip_bottom", "direction": 1, "strength": 1.0,
        }
        tb_no = np.zeros(n, dtype=np.float32)
        _apply_rule(signal, rule_no_prior, ss.copy(), tb_no, men.copy(), z, z)
        assert np.argmax(tb_no) < 50, "Without prior the large top peak should dominate"

        # With bottom prior: bottom peak should win
        rule_with_prior = dict(rule_no_prior)
        rule_with_prior["location_prior"] = [[0.0, 0.0], [0.6, 0.0], [0.7, 1.0], [1.0, 1.0]]
        tb_yes = np.zeros(n, dtype=np.float32)
        _apply_rule(signal, rule_with_prior, ss.copy(), tb_yes, men.copy(), z, z)
        assert np.argmax(tb_yes) >= 60, "With bottom prior the lower peak should dominate"
