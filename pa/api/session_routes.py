"""
HTTP routes for PA session lifecycle and step dispatch.
"""

from fastapi import APIRouter, HTTPException
from pa.api.schemas_http import StartSessionRequest, AnalyzeStepRequest, AnalyzeStepResponse
from pa.state import session_state

router = APIRouter()


@router.get("/health")
def health():
    """
    Piomek2 polls this after launching PA until it receives {"status": "ok"}.
    Confirms Tier 1 (instrument config) is loaded and the server is ready.
    """
    ready = session_state._startup is not None
    return {"status": "ok" if ready else "starting"}


@router.post("/session/start")
def start_session(req: StartSessionRequest):
    """
    Load Tier 2 (method context) for this method run.
    Tier 1 is already in memory from process startup.
    """
    session_state.initialize(req.method_context_path)
    return {"status": "ok"}


@router.post("/session/stop")
def stop_session():
    """Flush any remaining state and shut down the session."""
    session_state.clear()
    return {"status": "ok"}


@router.post("/step/analyze", response_model=AnalyzeStepResponse)
def analyze_step(req: AnalyzeStepRequest):
    """Dispatch a single step command to the appropriate analysis module."""
    if not session_state.is_initialized():
        raise HTTPException(status_code=400, detail="No active session. Call /session/start first.")

    from pa.api import _dispatch
    result = _dispatch.run(req)
    return result
