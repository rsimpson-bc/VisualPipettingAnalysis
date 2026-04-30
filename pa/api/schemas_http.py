"""
Pydantic models for HTTP request/response validation.
These mirror the JSON schemas in /schemas/ but are used at runtime.
"""

from __future__ import annotations
from typing import Literal, List, Optional
from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Session
# ---------------------------------------------------------------------------

class StartSessionRequest(BaseModel):
    method_context_path: str


# ---------------------------------------------------------------------------
# Step command (Tier 3)
# ---------------------------------------------------------------------------

class PipetteCommand(BaseModel):
    index: int
    expected_tip_type: str
    expected_liquid_ul: Optional[float] = None


class AnalyzeStepRequest(BaseModel):
    step_id: str
    analysis_type: Literal["TipIdentification", "LiquidMeasurement"]
    subfolders: List[str] = Field(default_factory=list)
    pipettes: List[PipetteCommand] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Step response (summary only — full result is written to filesystem)
# ---------------------------------------------------------------------------

class AnalyzeStepResponse(BaseModel):
    step_id: str
    overall_status: str
    result_path: str
