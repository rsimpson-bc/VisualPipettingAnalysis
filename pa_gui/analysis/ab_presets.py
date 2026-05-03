"""Preset manager for A/B Comparison parameter sets.

Presets are stored in a single JSON file (default: ``ab_presets.json`` at
the project root).  The active file path is remembered in QSettings and can
be changed by the user at any time.

File format::

    {
      "version": 1,
      "presets": [
        {
          "id": "20260501-143212-a1b2c3",
          "mode": "IntensityDetection",
          "description": "Wider ROI reduces false edge hits on meniscus glare",
          "created": "2026-05-01T14:32:12",
          "params_b": { ... }
        }
      ]
    }
"""

from __future__ import annotations

import json
import os
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

# ── Default path: project root (two levels above this file's package) ────────
_HERE = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(os.path.dirname(_HERE))
DEFAULT_PRESETS_PATH: str = os.path.join(_PROJECT_ROOT, "ab_presets.json")

_SETTINGS_ORG = "rsimpson-bc"
_SETTINGS_APP = "PA-GUI"
_SETTINGS_FILE_KEY = "ab_presets/file_path"


# ── Settings helpers ─────────────────────────────────────────────────────────

def get_presets_path() -> str:
    """Return the currently configured presets file path from QSettings."""
    from PySide6.QtCore import QSettings
    s = QSettings(_SETTINGS_ORG, _SETTINGS_APP)
    return s.value(_SETTINGS_FILE_KEY, DEFAULT_PRESETS_PATH)


def set_presets_path(path: str) -> None:
    """Persist a new presets file path to QSettings."""
    from PySide6.QtCore import QSettings
    s = QSettings(_SETTINGS_ORG, _SETTINGS_APP)
    s.setValue(_SETTINGS_FILE_KEY, path)


# ── File I/O ─────────────────────────────────────────────────────────────────

def load_presets(path: Optional[str] = None) -> Dict[str, Any]:
    """Load and return the presets dict.

    Returns an empty ``{"version": 1, "presets": []}`` structure if the file
    is absent, unreadable, or malformed — never raises.
    """
    if path is None:
        path = get_presets_path()
    if not os.path.isfile(path):
        return {"version": 1, "presets": []}
    try:
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        if not isinstance(data, dict) or "presets" not in data:
            return {"version": 1, "presets": []}
        return data
    except Exception:
        return {"version": 1, "presets": []}


def save_presets(data: Dict[str, Any], path: Optional[str] = None) -> None:
    """Write the presets dict to disk atomically (write→rename)."""
    if path is None:
        path = get_presets_path()
    parent = os.path.dirname(os.path.abspath(path))
    if parent:
        os.makedirs(parent, exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2, ensure_ascii=False)
    os.replace(tmp, path)


# ── Preset CRUD ───────────────────────────────────────────────────────────────

def add_preset(
    mode: str,
    description: str,
    params_b: Dict[str, Any],
    path: Optional[str] = None,
) -> Dict[str, Any]:
    """Append a new preset entry, save the file, and return the new entry dict."""
    data = load_presets(path)
    entry: Dict[str, Any] = {
        "id": datetime.now().strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:6],
        "mode": mode,
        "description": description.strip(),
        "created": datetime.now().isoformat(timespec="seconds"),
        "params_b": params_b,
    }
    data.setdefault("presets", []).append(entry)
    save_presets(data, path)
    return entry


def presets_for_mode(mode: str, path: Optional[str] = None) -> List[Dict[str, Any]]:
    """Return all presets for *mode*, newest-first."""
    data = load_presets(path)
    return [p for p in reversed(data.get("presets", [])) if p.get("mode") == mode]


def format_combo_label(entry: Dict[str, Any], max_desc: int = 50) -> str:
    """Return a short human-readable label for a preset combo-box item."""
    created = entry.get("created", "")[:16].replace("T", " ")  # "2026-05-01 14:32"
    desc = entry.get("description", "").replace("\n", " ")
    if len(desc) > max_desc:
        desc = desc[:max_desc - 1] + "…"
    return f"{created}  {desc}" if desc else created


def format_preview(entry: Dict[str, Any]) -> str:
    """Return a multi-line preview string shown in the preview text area."""
    lines = [
        f"Mode:        {entry.get('mode', '')}",
        f"Created:     {entry.get('created', '')}",
        f"Description: {entry.get('description', '')}",
        "",
        json.dumps(entry.get("params_b", {}), indent=2),
    ]
    return "\n".join(lines)
