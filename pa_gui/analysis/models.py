"""
Data models for the Analysis tab.

StepDef     — one step command (analysis type + subfolders + pipettes).
PipetteDef  — one pipette entry within a step.

Index convention: internally 0-based (matches the pipeline / JSON schemas).
The UI always displays tip numbers 1-based.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


ANALYSIS_TYPES: List[str] = ["TipIdentification", "LiquidMeasurement"]


@dataclass
class PipetteDef:
    index: int               # 0-based, matches schema "index" field
    tip_type: str
    expected_liquid_ul: Optional[float] = None

    def to_dict(self) -> dict:
        return {
            "index": self.index,
            "tip_type": self.tip_type,
            "expected_liquid_ul": self.expected_liquid_ul,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "PipetteDef":
        return cls(
            index=int(d["index"]),
            tip_type=str(d.get("tip_type", "")),
            expected_liquid_ul=d.get("expected_liquid_ul"),
        )


@dataclass
class StepDef:
    step_id: str
    analysis_type: str       # "TipIdentification" | "LiquidMeasurement"
    subfolders: List[str] = field(default_factory=list)
    pipettes: List[PipetteDef] = field(default_factory=list)

    # Populated after running; None = not yet run.
    result: Optional[Dict[str, Any]] = field(default=None, compare=False)
    error:  Optional[str]            = field(default=None, compare=False)
    # Debug data — list of DebugData objects, one per pipette.  None until run with debug=True.
    debug_data: Optional[List[Any]]  = field(default=None, compare=False)

    def to_dict(self) -> dict:
        return {
            "step_id": self.step_id,
            "analysis_type": self.analysis_type,
            "subfolders": list(self.subfolders),
            "pipettes": [p.to_dict() for p in self.pipettes],
        }

    @classmethod
    def from_dict(cls, d: dict) -> "StepDef":
        return cls(
            step_id=str(d["step_id"]),
            analysis_type=str(d.get("analysis_type", "TipIdentification")),
            subfolders=list(d.get("subfolders", [])),
            pipettes=[PipetteDef.from_dict(p) for p in d.get("pipettes", [])],
        )

    @property
    def status(self) -> str:
        """
        'pending'         — not yet run
        'running'         — currently being processed
        'ok'              — completed, overall_status == ok
        'warning'         — completed, overall_status == warning
        'low_confidence'  — completed, overall_status == low_confidence
        'error'           — completed, overall_status == error
        'failed'          — exception during run
        """
        if self.error:
            return "failed"
        if self.result is None:
            return "pending"
        return self.result.get("overall_status", "ok")
