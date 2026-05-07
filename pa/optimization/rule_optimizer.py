"""
Behavior-rule optimizer for the PA signal interpreter.

Encodes the *numeric* parameters of a preset's ``behavior_rules`` list into a
flat array, optimises them against ground-truth z annotations using
``scipy.optimize``, then decodes the result back into a rules list.

Structural fields — ``trigger``, ``target``, ``direction``, and
``spatial_modifier.type`` — are kept fixed; only the numeric knobs
(strength, spread_px, smoothing_px, min_prominence_frac, threshold_frac,
range_px) are tuned.

Public API
----------
``encode_rules(rules)``
    → ``(x0, param_specs, bounds)``

    *x0*          : np.ndarray  – initial parameter vector
    *param_specs* : list[dict]  – metadata used by decode_rules
    *bounds*      : list[tuple] – (lo, hi) per element, for scipy bounds

``decode_rules(x, rules, param_specs)``
    → list[dict] – rules with numeric params replaced from *x*

``compute_loss(rules, signals, global_z_axis, annotations)``
    → float – mean absolute z error in pixels across all annotated boundaries

``optimize_rules(rules, signals, global_z_axis, annotations, …)``
    → ``OptimizationResult``
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
from scipy.optimize import minimize, differential_evolution

from pa.pipeline.interpreters.signal_interpreter import interpret_signals


# ---------------------------------------------------------------------------
# Parameter registry
# ---------------------------------------------------------------------------

# Each entry: (param_name, lo_bound, hi_bound, is_integer)
# is_integer = True → value is rounded to int when decoded.
_COMMON_PARAMS: List[Tuple[str, float, float, bool]] = [
    ("strength",  0.01, 10.0, False),
    ("spread_px", 0.5,  50.0, False),
]

_TRIGGER_PARAMS: Dict[str, List[Tuple[str, float, float, bool]]] = {
    "peak": [
        ("min_prominence_frac",      0.01, 0.95,  False),
        ("smoothing_px",             1.0,  30.0,  True),
        ("value_min_strength",       0.0,  1.0,   False),
        ("low_above_window_px",      0.0,  200.0, True),
        ("low_above_threshold_frac", 0.01, 0.99,  False),
        ("low_above_min_strength",   0.0,  1.0,   False),
        ("low_below_window_px",      0.0,  200.0, True),
        ("low_below_threshold_frac", 0.01, 0.99,  False),
        ("low_below_min_strength",   0.0,  1.0,   False),
        ("high_above_window_px",     0.0,  200.0, True),
        ("high_above_threshold_frac",0.01, 0.99,  False),
        ("high_above_min_strength",  0.0,  1.0,   False),
        ("high_below_window_px",     0.0,  200.0, True),
        ("high_below_threshold_frac",0.01, 0.99,  False),
        ("high_below_min_strength",  0.0,  1.0,   False),
    ],
    # Note: peak_side is a categorical string — not optimizable.
    "high_signal": [
        ("threshold_frac", 0.01, 0.95, False),
        ("smoothing_px",   1.0,  30.0, True),
    ],
    "low_signal": [
        ("threshold_frac", 0.01, 0.95, False),
        ("smoothing_px",   1.0,  30.0, True),
    ],
    "rising_edge": [
        ("threshold_frac",   0.01, 0.95,  False),
        ("smoothing_px",     1.0,  30.0,  True),
        ("pre_low_window_px",0.0,  200.0, True),
        ("pre_low_frac",     0.0,  1.0,   False),
    ],
}

_SPATIAL_PARAM: Tuple[str, float, float, bool] = ("range_px", 0.0, 200.0, True)

_DEFAULTS: Dict[str, float] = {
    "strength":                 1.0,
    "spread_px":                5.0,
    "min_prominence_frac":      0.2,
    "smoothing_px":             3.0,
    "threshold_frac":           0.3,
    "range_px":                 0.0,
    "value_min_strength":       1.0,
    "low_above_window_px":      0,
    "low_above_threshold_frac": 0.3,
    "low_above_min_strength":   0.0,
    "low_below_window_px":       0,
    "low_below_threshold_frac":  0.3,
    "low_below_min_strength":    0.0,
    "high_above_window_px":      0,
    "high_above_threshold_frac": 0.7,
    "high_above_min_strength":   1.0,
    "high_below_window_px":      0,
    "high_below_threshold_frac": 0.7,
    "high_below_min_strength":   1.0,
    "pre_low_window_px":         0,
    "pre_low_frac":              0.5,
}

# Penalty added to the loss when a required boundary has no prediction.
_MISSING_PENALTY: float = 1000.0


# ---------------------------------------------------------------------------
# Encode / decode
# ---------------------------------------------------------------------------

def encode_rules(
    rules: List[Dict[str, Any]],
) -> Tuple[np.ndarray, List[Dict[str, Any]], List[Tuple[float, float]]]:
    """Flatten numeric parameters from *rules* into a 1-D array.

    Returns
    -------
    x0 : np.ndarray
        Initial parameter vector (current values, clamped to bounds).
    param_specs : list[dict]
        One dict per element with keys:
            ``rule_idx``  – which rule in the list this element belongs to
            ``path``      – key path list for setting the value (e.g.
                            ``["spatial_modifier", "range_px"]``)
            ``is_int``    – whether to round to int on decode
    bounds : list[tuple[float, float]]
        ``(lo, hi)`` per element, suitable for ``scipy.optimize``.
    """
    x0_list: List[float] = []
    specs:   List[Dict[str, Any]] = []
    bounds:  List[Tuple[float, float]] = []

    for rule_idx, rule in enumerate(rules):
        trigger = rule.get("trigger", "")

        def _add(pname: str, lo: float, hi: float, is_int: bool, path: List[str]) -> None:
            val = float(rule.get(pname, _DEFAULTS.get(pname, 0.0)))
            x0_list.append(float(np.clip(val, lo, hi)))
            specs.append({"rule_idx": rule_idx, "path": path, "is_int": is_int})
            bounds.append((lo, hi))

        for pname, lo, hi, is_int in _COMMON_PARAMS:
            _add(pname, lo, hi, is_int, [pname])

        for pname, lo, hi, is_int in _TRIGGER_PARAMS.get(trigger, []):
            _add(pname, lo, hi, is_int, [pname])

        # Spatial modifier range_px (only if modifier is present with a type)
        mod = rule.get("spatial_modifier")
        if mod is not None and mod.get("type") in ("offset_below", "offset_above"):
            pname, lo, hi, is_int = _SPATIAL_PARAM
            val = float(mod.get(pname, _DEFAULTS[pname]))
            x0_list.append(float(np.clip(val, lo, hi)))
            specs.append({"rule_idx": rule_idx,
                          "path": ["spatial_modifier", pname],
                          "is_int": is_int})
            bounds.append((lo, hi))

        # Location prior anchor weights (positions fixed; only weights tuned)
        prior = rule.get("location_prior")
        if prior:
            for anchor_idx, anchor in enumerate(prior):
                weight = float(anchor[1]) if len(anchor) > 1 else 1.0
                x0_list.append(float(np.clip(weight, 0.0, 1.0)))
                specs.append({
                    "rule_idx":   rule_idx,
                    "path":       ["location_prior", anchor_idx, 1],
                    "is_int":     False,
                })
                bounds.append((0.0, 1.0))

    return np.array(x0_list, dtype=np.float64), specs, bounds


def decode_rules(
    x: np.ndarray,
    rules: List[Dict[str, Any]],
    param_specs: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Reconstruct a rules list by substituting values from *x*.

    Values are clamped to their registered bounds and integer params are
    rounded.  The original *rules* list is not modified (deep copy is made).
    """
    result = copy.deepcopy(rules)
    for val, spec in zip(x, param_specs):
        rule_idx = spec["rule_idx"]
        path     = spec["path"]
        is_int   = spec["is_int"]

        final_val: Any = float(val)
        if is_int:
            final_val = max(1, int(round(final_val)))

        target = result[rule_idx]
        for key in path[:-1]:
            target = target[key]
        target[path[-1]] = final_val

    return result


# ---------------------------------------------------------------------------
# Loss function
# ---------------------------------------------------------------------------

def compute_loss(
    rules: List[Dict[str, Any]],
    signals: List[Tuple[np.ndarray, np.ndarray]],
    global_z_axis: np.ndarray,
    annotations: List[Dict[str, Any]],
) -> float:
    """Run ``interpret_signals`` and compute mean absolute z error.

    Parameters
    ----------
    rules : list[dict]
        Current behavior_rules to evaluate.
    signals : list[(signal_1d, z_axis_1d)]
        Pre-extracted signal arrays from the pipeline run.  Each tuple is
        ``(signal_array, z_axis_array)`` in the same coordinate system as
        *global_z_axis*.
    global_z_axis : np.ndarray
        Global z axis passed to ``interpret_signals`` for alignment.
    annotations : list[dict]
        Ground-truth dicts with optional keys ``"tip_bottom_z"`` and
        ``"meniscus_z"``.  Annotations without either key are ignored.

    Returns
    -------
    float
        Mean absolute pixel error.  Returns 0.0 when *annotations* is empty.
    """
    if not annotations:
        return 0.0

    if not rules or not signals:
        n_terms = sum(
            (1 if a.get("tip_bottom_z") is not None else 0)
            + (1 if a.get("meniscus_z") is not None else 0)
            for a in annotations
        )
        return _MISSING_PENALTY * max(1, n_terms)

    try:
        result = interpret_signals(
            signals=signals,
            rules_per_signal=[rules] * len(signals),
            global_z_axis=global_z_axis,
        )
    except Exception:
        return _MISSING_PENALTY * len(annotations)

    total_error = 0.0
    n_terms = 0

    for ann in annotations:
        gt_tb = ann.get("tip_bottom_z")
        gt_mn = ann.get("meniscus_z")

        if gt_tb is not None:
            pred = result.tip_bottom_z
            total_error += abs(pred - float(gt_tb)) if pred is not None else _MISSING_PENALTY
            n_terms += 1

        if gt_mn is not None:
            pred = result.meniscus_z
            total_error += abs(pred - float(gt_mn)) if pred is not None else _MISSING_PENALTY
            n_terms += 1

    return total_error / max(1, n_terms)


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class OptimizationResult:
    """Return value from :func:`optimize_rules`."""
    optimized_rules: List[Dict[str, Any]]
    initial_loss:    float
    final_loss:      float
    iterations:      int
    success:         bool
    message:         str


# ---------------------------------------------------------------------------
# Optimizer
# ---------------------------------------------------------------------------

def optimize_rules(
    rules: List[Dict[str, Any]],
    signals: List[Tuple[np.ndarray, np.ndarray]],
    global_z_axis: np.ndarray,
    annotations: List[Dict[str, Any]],
    method: str = "Nelder-Mead",
    max_iter: int = 400,
    callback: Optional[Callable[[np.ndarray], None]] = None,
) -> OptimizationResult:
    """Optimise numeric parameters in *rules* to minimise z error.

    Parameters
    ----------
    rules : list[dict]
        Starting behavior_rules (structure is kept fixed).
    signals : list[(signal_1d, z_axis_1d)]
        Signal data from the current pipeline run.
    global_z_axis : np.ndarray
        Global z axis for ``interpret_signals``.
    annotations : list[dict]
        Ground truth dicts.  If empty the function returns immediately
        with the unmodified rules and ``final_loss = 0.0``.
    method : str
        ``"Nelder-Mead"`` (default, gradient-free local search),
        ``"Powell"`` (bound-respecting local search), or
        ``"differential_evolution"`` (global stochastic search — slower
        but avoids local minima).
    max_iter : int
        Maximum number of optimizer iterations.
    callback : callable or None
        Called each iteration with the current parameter vector.

    Returns
    -------
    OptimizationResult
    """
    if not rules:
        return OptimizationResult(
            optimized_rules=rules,
            initial_loss=0.0,
            final_loss=0.0,
            iterations=0,
            success=True,
            message="No rules to optimize.",
        )

    x0, param_specs, bounds = encode_rules(rules)

    if len(x0) == 0:
        return OptimizationResult(
            optimized_rules=rules,
            initial_loss=0.0,
            final_loss=0.0,
            iterations=0,
            success=True,
            message="No numeric parameters found in rules.",
        )

    if not annotations:
        return OptimizationResult(
            optimized_rules=rules,
            initial_loss=0.0,
            final_loss=0.0,
            iterations=0,
            success=True,
            message="No annotations — nothing to optimize against.",
        )

    initial_loss = compute_loss(rules, signals, global_z_axis, annotations)

    def _objective(x: np.ndarray) -> float:
        candidate = decode_rules(x, rules, param_specs)
        return compute_loss(candidate, signals, global_z_axis, annotations)

    if method == "differential_evolution":
        opt_result = differential_evolution(
            _objective,
            bounds=bounds,
            maxiter=max_iter,
            seed=42,
            tol=0.1,
            callback=callback,
        )
    elif method == "Powell":
        opt_result = minimize(
            _objective,
            x0,
            method="Powell",
            bounds=bounds,
            options={"maxiter": max_iter, "ftol": 0.1},
            callback=callback,
        )
    else:
        # Nelder-Mead (default) — ignores bounds but is robust for this domain
        opt_result = minimize(
            _objective,
            x0,
            method="Nelder-Mead",
            options={"maxiter": max_iter, "xatol": 0.05, "fatol": 0.1},
            callback=callback,
        )

    optimized_rules = decode_rules(opt_result.x, rules, param_specs)

    return OptimizationResult(
        optimized_rules=optimized_rules,
        initial_loss=initial_loss,
        final_loss=float(opt_result.fun),
        iterations=int(getattr(opt_result, "nit", max_iter)),
        success=bool(getattr(opt_result, "success", True)),
        message=str(getattr(opt_result, "message", "")),
    )
