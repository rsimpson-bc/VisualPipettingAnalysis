"""
Optimization harness — scorer.

Compares PA pipeline output against expected values stored alongside the run.

Expected values file format (Analysis/expected_<step_id>.json):
{
  "pipettes": [
    {
      "index": 1,
      "expected_liquid_ul": 50.0,      // null if not applicable
      "tip_present": true,
      "expected_tip_type": "T200"
    }
  ]
}

Scores returned per step:
  - For liquid: volume_error_ul, volume_error_pct, correct_status (bool)
  - For tip:    tip_present_correct (bool), tip_type_correct (bool),
                seating_status_correct (bool)
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional


def score_results(
    pa_results: List[Any],
    expected: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Score a list of per-pipette results against expected values.
    Returns a flat dict of scalar scores for this step.
    """
    if expected is None:
        return {"scored": False, "reason": "no expected file"}

    scores: Dict[str, Any] = {"scored": True}
    expected_by_index = {p["index"]: p for p in expected.get("pipettes", [])}

    for result in pa_results:
        idx = result.pipette_index
        exp = expected_by_index.get(idx)
        if exp is None:
            continue

        prefix = f"pip{idx}"

        # Liquid scoring
        if hasattr(result, "total_volume_ul") and "expected_liquid_ul" in exp:
            exp_vol = exp["expected_liquid_ul"]
            got_vol = result.total_volume_ul
            if exp_vol is not None and got_vol is not None:
                err = abs(got_vol - exp_vol)
                scores[f"{prefix}_volume_error_ul"] = round(err, 3)
                scores[f"{prefix}_volume_error_pct"] = (
                    round(err / exp_vol * 100, 2) if exp_vol != 0 else None
                )
            else:
                scores[f"{prefix}_volume_error_ul"] = None
                scores[f"{prefix}_volume_error_pct"] = None

        # Tip scoring
        if hasattr(result, "tip_present") and "tip_present" in exp:
            scores[f"{prefix}_tip_present_correct"] = (
                result.tip_present == exp["tip_present"]
            )
        if hasattr(result, "perceived_tip_type") and "expected_tip_type" in exp:
            scores[f"{prefix}_tip_type_correct"] = (
                result.perceived_tip_type == exp["expected_tip_type"]
            )

        # Confidence
        scores[f"{prefix}_confidence"] = round(result.confidence, 1)

    return scores
