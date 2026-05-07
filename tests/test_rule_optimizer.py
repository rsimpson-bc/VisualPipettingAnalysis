"""
Tests for pa.optimization.rule_optimizer

Coverage
--------
TestEncodeDecode      (7 tests) – roundtrip, multiple rules, empty rules, immutability
TestComputeLoss       (3 tests) – empty annotations, empty rules, loss ordering
TestOptimizeRules     (3 tests) – empty rules, no annotations, result structure
"""

from __future__ import annotations

import numpy as np
import pytest

from pa.optimization.rule_optimizer import (
    OptimizationResult,
    _MISSING_PENALTY,
    compute_loss,
    decode_rules,
    encode_rules,
    optimize_rules,
)


# ---------------------------------------------------------------------------
# Helper factories
# ---------------------------------------------------------------------------

def _peak_rule(
    strength: float = 1.2,
    spread_px: float = 8.0,
    min_prom: float = 0.3,
    smoothing: int = 3,
) -> dict:
    return {
        "trigger": "peak",
        "target": "tip_bottom",
        "direction": 1,
        "strength": strength,
        "spread_px": spread_px,
        "min_prominence_frac": min_prom,
        "smoothing_px": smoothing,
    }


def _high_rule(
    strength: float = 0.8,
    spread_px: float = 5.0,
    threshold: float = 0.3,
    smoothing: int = 5,
) -> dict:
    return {
        "trigger": "high_signal",
        "target": "gas_in_tip",
        "direction": 1,
        "strength": strength,
        "spread_px": spread_px,
        "threshold_frac": threshold,
        "smoothing_px": smoothing,
    }


def _rule_with_modifier(range_px: int = 20) -> dict:
    r = _peak_rule()
    r["spatial_modifier"] = {"type": "offset_below", "range_px": range_px}
    return r


def _signal_with_peak(peak_z: int, n: int = 100):
    """Return (signal, z_axis) with a single smooth peak at *peak_z*."""
    z = np.linspace(0.0, n - 1, n, dtype=np.float32)
    sig = np.zeros(n, dtype=np.float32)
    sig[peak_z] = 1.0
    sig = np.convolve(sig, np.ones(5, dtype=np.float32) / 5, mode="same")
    return sig, z


# ===========================================================================
# encode / decode
# ===========================================================================

class TestEncodeDecode:

    def test_roundtrip_peak_rule(self):
        rules = [_peak_rule()]
        x0, specs, bounds = encode_rules(rules)
        restored = decode_rules(x0, rules, specs)
        assert restored[0]["strength"]            == pytest.approx(rules[0]["strength"])
        assert restored[0]["spread_px"]           == pytest.approx(rules[0]["spread_px"])
        assert restored[0]["min_prominence_frac"] == pytest.approx(rules[0]["min_prominence_frac"])
        assert restored[0]["smoothing_px"]        == rules[0]["smoothing_px"]

    def test_roundtrip_high_signal_rule(self):
        rules = [_high_rule()]
        x0, specs, bounds = encode_rules(rules)
        restored = decode_rules(x0, rules, specs)
        assert restored[0]["threshold_frac"] == pytest.approx(rules[0]["threshold_frac"])
        assert restored[0]["smoothing_px"]   == rules[0]["smoothing_px"]

    def test_roundtrip_spatial_modifier(self):
        rules = [_rule_with_modifier(range_px=15)]
        x0, specs, bounds = encode_rules(rules)
        restored = decode_rules(x0, rules, specs)
        assert restored[0]["spatial_modifier"]["range_px"] == 15

    def test_multiple_rules(self):
        rules = [_peak_rule(), _high_rule()]
        x0, specs, bounds = encode_rules(rules)
        assert len(x0) == len(bounds) == len(specs)
        restored = decode_rules(x0, rules, specs)
        assert len(restored) == 2
        assert restored[0]["trigger"] == "peak"
        assert restored[1]["trigger"] == "high_signal"
        # Structural fields preserved
        assert restored[0]["target"]    == rules[0]["target"]
        assert restored[0]["direction"] == rules[0]["direction"]

    def test_empty_rules(self):
        x0, specs, bounds = encode_rules([])
        assert len(x0) == 0
        assert decode_rules(x0, [], specs) == []

    def test_bounds_cover_all_specs(self):
        rules = [_peak_rule(), _rule_with_modifier()]
        x0, specs, bounds = encode_rules(rules)
        assert len(bounds) == len(specs) == len(x0)
        for lo, hi in bounds:
            assert lo < hi

    def test_decode_does_not_mutate_original(self):
        rules = [_peak_rule()]
        x0, specs, _ = encode_rules(rules)
        original_strength = rules[0]["strength"]
        tweaked = x0.copy()
        # Strength is the first element (first common param)
        tweaked[0] = original_strength * 3.0
        decode_rules(tweaked, rules, specs)
        assert rules[0]["strength"] == original_strength   # original unchanged


# ===========================================================================
# compute_loss
# ===========================================================================

class TestComputeLoss:

    def test_empty_annotations_returns_zero(self):
        sig, z = _signal_with_peak(50)
        rules = [_peak_rule(spread_px=4, min_prom=0.1)]
        loss = compute_loss(rules, [(sig, z)], z, annotations=[])
        assert loss == 0.0

    def test_empty_rules_returns_penalty(self):
        sig, z = _signal_with_peak(50)
        annotations = [{"tip_bottom_z": 50.0}]
        loss = compute_loss([], [(sig, z)], z, annotations=annotations)
        assert loss >= _MISSING_PENALTY

    def test_loss_non_negative(self):
        sig, z = _signal_with_peak(50)
        annotations = [{"tip_bottom_z": 50.0}]
        rules = [_peak_rule(spread_px=4, min_prom=0.1)]
        loss = compute_loss(rules, [(sig, z)], z, annotations=annotations)
        assert loss >= 0.0


# ===========================================================================
# optimize_rules
# ===========================================================================

class TestOptimizeRules:

    def test_empty_rules_returns_immediately(self):
        z = np.linspace(0, 99, 100, dtype=np.float32)
        res = optimize_rules([], [(np.zeros(100, dtype=np.float32), z)], z, [])
        assert isinstance(res, OptimizationResult)
        assert res.success
        assert res.optimized_rules == []
        assert res.iterations == 0

    def test_no_annotations_returns_unchanged(self):
        rules = [_peak_rule()]
        z = np.linspace(0, 99, 100, dtype=np.float32)
        sig = np.zeros(100, dtype=np.float32)
        res = optimize_rules(rules, [(sig, z)], z, annotations=[])
        assert res.success
        assert res.final_loss == 0.0

    def test_result_has_correct_structure(self):
        rules = [_peak_rule()]
        sig, z = _signal_with_peak(50)
        annotations = [{"tip_bottom_z": 50.0}]
        res = optimize_rules(rules, [(sig, z)], z, annotations, max_iter=5)
        assert isinstance(res, OptimizationResult)
        assert len(res.optimized_rules) == len(rules)
        assert res.initial_loss >= 0.0
        assert res.final_loss >= 0.0
        # Structural fields must survive optimization
        assert res.optimized_rules[0]["trigger"] == "peak"
        assert res.optimized_rules[0]["target"]  == "tip_bottom"
        assert res.optimized_rules[0]["direction"] == 1

# ===========================================================================
# location_prior encode / decode
# ===========================================================================

class TestLocationPriorEncodeDecode:
    """Verify that location_prior anchor weights are correctly encoded,
    bounded at [0, 1], and restored by decode_rules."""

    def _rule_with_prior(self, anchors):
        r = _peak_rule()
        r["location_prior"] = [[frac, w] for frac, w in anchors]
        return r

    def test_prior_weights_encoded(self):
        """Each anchor weight should appear as a separate parameter."""
        anchors = [(0.0, 0.0), (0.6, 0.0), (0.8, 1.0), (1.0, 1.0)]
        rules = [self._rule_with_prior(anchors)]
        x0, specs, bounds = encode_rules(rules)
        # Count how many specs are for location_prior weights
        prior_specs = [s for s in specs if
                       len(s["path"]) == 3 and s["path"][0] == "location_prior"]
        assert len(prior_specs) == len(anchors)

    def test_prior_weights_bounds_are_zero_to_one(self):
        anchors = [(0.0, 0.3), (1.0, 0.7)]
        rules = [self._rule_with_prior(anchors)]
        x0, specs, bounds = encode_rules(rules)
        prior_specs = [s for s in specs if
                       len(s["path"]) == 3 and s["path"][0] == "location_prior"]
        prior_indices = [
            i for i, s in enumerate(specs)
            if len(s["path"]) == 3 and s["path"][0] == "location_prior"
        ]
        for i in prior_indices:
            assert bounds[i] == (0.0, 1.0)

    def test_prior_weights_roundtrip(self):
        """encode then decode must reproduce the original anchor weights."""
        anchors = [(0.0, 0.0), (0.5, 0.5), (1.0, 1.0)]
        rules = [self._rule_with_prior(anchors)]
        x0, specs, bounds = encode_rules(rules)
        decoded = decode_rules(x0, rules, specs)
        for i, (frac, w) in enumerate(anchors):
            assert decoded[0]["location_prior"][i][0] == pytest.approx(frac, abs=1e-6)
            assert decoded[0]["location_prior"][i][1] == pytest.approx(w,    abs=1e-6)

    def test_modified_weight_decoded_correctly(self):
        """Manually modifying x0 for a prior weight should be decoded correctly."""
        anchors = [(0.0, 0.0), (1.0, 0.0)]
        rules = [self._rule_with_prior(anchors)]
        x0, specs, bounds = encode_rules(rules)
        # Find the index of the second anchor weight and set it to 0.75
        prior_indices = [
            i for i, s in enumerate(specs)
            if len(s["path"]) == 3 and s["path"][0] == "location_prior"
        ]
        x_mod = x0.copy()
        x_mod[prior_indices[1]] = 0.75
        decoded = decode_rules(x_mod, rules, specs)
        assert decoded[0]["location_prior"][1][1] == pytest.approx(0.75, abs=1e-6)

    def test_no_prior_rule_unaffected(self):
        """A rule without location_prior should produce no prior-related specs."""
        rules = [_peak_rule()]   # no location_prior key
        x0, specs, bounds = encode_rules(rules)
        prior_specs = [s for s in specs if
                       len(s["path"]) == 3 and s["path"][0] == "location_prior"]
        assert len(prior_specs) == 0

    def test_prior_does_not_mutate_original_rules(self):
        anchors = [(0.0, 0.2), (1.0, 0.8)]
        rules = [self._rule_with_prior(anchors)]
        import copy
        original = copy.deepcopy(rules)
        x0, specs, bounds = encode_rules(rules)
        x_mod = x0.copy()
        # Corrupt all prior weights in x_mod
        for i, s in enumerate(specs):
            if len(s["path"]) == 3 and s["path"][0] == "location_prior":
                x_mod[i] = 0.99
        decode_rules(x_mod, rules, specs)
        # Original rules must be unchanged
        assert rules[0]["location_prior"] == original[0]["location_prior"]