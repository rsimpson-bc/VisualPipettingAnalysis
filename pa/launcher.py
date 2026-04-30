"""
PA process launcher — used by Piomek2 to start and stop the PA server.

Usage in Piomek2:
    from pa.launcher import PALauncher

    launcher = PALauncher(instrument_config_path="C:/configs/instrument_config.json")
    launcher.start()       # called when Piomek2 opens
    ...
    launcher.stop()        # called when Piomek2 closes

Alternatively, use as a context manager:
    with PALauncher(instrument_config_path="...") as pa:
        pa.send_start_session("C:/runs/run_xxx/method_context.json")
        pa.send_analyze_step({...})
        pa.send_stop_session()
"""

from __future__ import annotations

import subprocess
import sys
import time
import json
import urllib.request
import urllib.error
from typing import Any, Dict, Optional


_DEFAULT_HOST = "127.0.0.1"
_DEFAULT_PORT = 8765
_HEALTH_POLL_INTERVAL_S = 0.25
_HEALTH_TIMEOUT_S = 30.0


class PALauncher:
    def __init__(
        self,
        instrument_config_path: str,
        analysis_config_path: str,
        host: str = _DEFAULT_HOST,
        port: int = _DEFAULT_PORT,
        python_executable: Optional[str] = None,
    ):
        self.instrument_config_path = instrument_config_path
        self.analysis_config_path = analysis_config_path
        self.host = host
        self.port = port
        self.python = python_executable or sys.executable
        self._process: Optional[subprocess.Popen] = None

    # ------------------------------------------------------------------
    # Process management
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Launch the PA server subprocess and wait until it is ready."""
        if self._process and self._process.poll() is None:
            return  # already running

        cmd = [
            self.python, "-m", "pa",
            "--instrument-config", self.instrument_config_path,
            "--analysis-config", self.analysis_config_path,
            "--host", self.host,
            "--port", str(self.port),
        ]
        self._process = subprocess.Popen(cmd)
        self._wait_until_ready()

    def stop(self) -> None:
        """Terminate the PA server subprocess."""
        if self._process and self._process.poll() is None:
            self._process.terminate()
            try:
                self._process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._process.kill()
        self._process = None

    def is_running(self) -> bool:
        return self._process is not None and self._process.poll() is None

    def _wait_until_ready(self) -> None:
        """Poll GET /health until the server responds with status 'ok'."""
        url = f"http://{self.host}:{self.port}/health"
        deadline = time.monotonic() + _HEALTH_TIMEOUT_S
        while time.monotonic() < deadline:
            try:
                with urllib.request.urlopen(url, timeout=2) as resp:
                    body = json.loads(resp.read())
                    if body.get("status") == "ok":
                        return
            except (urllib.error.URLError, OSError):
                pass
            time.sleep(_HEALTH_POLL_INTERVAL_S)
        raise TimeoutError(
            f"PA server did not become ready within {_HEALTH_TIMEOUT_S}s. "
            f"Check that 'python -m pa' runs correctly."
        )

    # ------------------------------------------------------------------
    # Context manager
    # ------------------------------------------------------------------

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *_):
        self.stop()

    # ------------------------------------------------------------------
    # HTTP helpers (thin wrappers — no dependency on requests/httpx)
    # ------------------------------------------------------------------

    def _post(self, path: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        url = f"http://{self.host}:{self.port}{path}"
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            url, data=data, headers={"Content-Type": "application/json"}, method="POST"
        )
        with urllib.request.urlopen(req, timeout=60) as resp:
            return json.loads(resp.read())

    def send_start_session(self, method_context_path: str) -> Dict[str, Any]:
        return self._post("/session/start", {"method_context_path": method_context_path})

    def send_stop_session(self) -> Dict[str, Any]:
        return self._post("/session/stop", {})

    def send_analyze_step(self, step_command: Dict[str, Any]) -> Dict[str, Any]:
        return self._post("/step/analyze", step_command)
