"""
Ground-truth annotation sidecar I/O.

Annotations capture the known *tip_bottom_z* and/or *meniscus_z* positions
(in full-image pixel coordinates) for each pipette in a single image.
They travel with the image as a JSON sidecar named::

    <image_stem>_annotations.json

placed in the **same directory** as the image file.  This means annotations
survive when a folder of images is moved, copied, or shared — they are never
stored in a central registry.

Sidecar schema
--------------
::

    {
      "image_filename": "IMG_0123.jpg",
      "pipettes": [
        {
          "id":            "20260506-143512-a1b2c3",
          "pipette_index": 0,
          "tip_bottom_z":  234.0,
          "meniscus_z":    189.0,
          "created":       "2026-05-06T14:35:12",
          "note":          ""
        }
      ]
    }

*tip_bottom_z* and *meniscus_z* are in full-image row coordinates (same
coordinate system as the global z axis used by Mode Compare / the signal
interpreter).  Either value may be ``null`` when not yet annotated.

Public API
----------
``get_sidecar_path(image_path)``
``load_sidecar(image_path)``
``upsert_annotation(image_path, pipette_index, ...)``
``get_annotation(image_path, pipette_index)``
``delete_annotation(image_path, annotation_id)``
``clear_annotation_field(image_path, pipette_index, field)``
"""

from __future__ import annotations

import json
import os
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# Path / load / save helpers
# ---------------------------------------------------------------------------

def get_sidecar_path(image_path: str) -> str:
    """Return the sidecar annotation path for *image_path*.

    ``/some/folder/IMG_0123.jpg``  →  ``/some/folder/IMG_0123_annotations.json``
    """
    base, _ = os.path.splitext(image_path)
    return base + "_annotations.json"


def load_sidecar(image_path: str) -> Dict[str, Any]:
    """Load the sidecar for *image_path*.

    Returns a fresh empty structure if the sidecar does not exist yet.
    """
    sidecar = get_sidecar_path(image_path)
    if os.path.exists(sidecar):
        with open(sidecar, encoding="utf-8") as f:
            return json.load(f)
    return {"image_filename": os.path.basename(image_path), "pipettes": []}


def _save_sidecar(image_path: str, data: Dict[str, Any]) -> None:
    sidecar = get_sidecar_path(image_path)
    with open(sidecar, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


# ---------------------------------------------------------------------------
# CRUD
# ---------------------------------------------------------------------------

def upsert_annotation(
    image_path: str,
    pipette_index: int,
    tip_bottom_z: Optional[float] = None,
    meniscus_z: Optional[float] = None,
    note: str = "",
) -> Dict[str, Any]:
    """Create or update the annotation for *(image_path, pipette_index)*.

    Only non-``None`` fields overwrite existing values; the others are
    preserved.  Returns the final entry dict.
    """
    data = load_sidecar(image_path)
    pipettes: List[Dict[str, Any]] = data.setdefault("pipettes", [])

    existing = next(
        (p for p in pipettes if p.get("pipette_index") == pipette_index),
        None,
    )

    if existing is not None:
        if tip_bottom_z is not None:
            existing["tip_bottom_z"] = float(tip_bottom_z)
        if meniscus_z is not None:
            existing["meniscus_z"] = float(meniscus_z)
        if note:
            existing["note"] = note
        _save_sidecar(image_path, data)
        return existing

    entry: Dict[str, Any] = {
        "id": datetime.now().strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:6],
        "pipette_index": pipette_index,
        "tip_bottom_z": float(tip_bottom_z) if tip_bottom_z is not None else None,
        "meniscus_z": float(meniscus_z) if meniscus_z is not None else None,
        "created": datetime.now().isoformat(timespec="seconds"),
        "note": note,
    }
    pipettes.append(entry)
    _save_sidecar(image_path, data)
    return entry


def get_annotation(
    image_path: str,
    pipette_index: int,
) -> Optional[Dict[str, Any]]:
    """Return the annotation entry for *(image_path, pipette_index)*, or ``None``."""
    data = load_sidecar(image_path)
    return next(
        (p for p in data.get("pipettes", []) if p.get("pipette_index") == pipette_index),
        None,
    )


def delete_annotation(
    image_path: str,
    annotation_id: str,
) -> bool:
    """Remove the annotation with *annotation_id* from the sidecar.

    Returns ``True`` if an entry was removed, ``False`` if the id was not
    found (sidecar may not exist yet).
    """
    sidecar = get_sidecar_path(image_path)
    if not os.path.exists(sidecar):
        return False
    data = load_sidecar(image_path)
    before = len(data.get("pipettes", []))
    data["pipettes"] = [
        p for p in data.get("pipettes", []) if p.get("id") != annotation_id
    ]
    if len(data["pipettes"]) == before:
        return False
    _save_sidecar(image_path, data)
    return True


def clear_annotation_field(
    image_path: str,
    pipette_index: int,
    field: str,
) -> bool:
    """Set *field* (``"tip_bottom_z"`` or ``"meniscus_z"``) to ``None`` without
    deleting the whole entry.  Returns ``True`` if the entry was found."""
    if field not in ("tip_bottom_z", "meniscus_z"):
        raise ValueError(f"Unknown annotation field: {field!r}")
    sidecar = get_sidecar_path(image_path)
    if not os.path.exists(sidecar):
        return False
    data = load_sidecar(image_path)
    for entry in data.get("pipettes", []):
        if entry.get("pipette_index") == pipette_index:
            entry[field] = None
            _save_sidecar(image_path, data)
            return True
    return False
