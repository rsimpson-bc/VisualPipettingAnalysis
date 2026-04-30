"""
PA FastAPI server entry point.

Start with:
    uvicorn pa.api.server:app --host 127.0.0.1 --port 8765
"""

from fastapi import FastAPI
from pa.api import session_routes

app = FastAPI(title="PipetteAnalysis", version="0.1.0")

app.include_router(session_routes.router)
