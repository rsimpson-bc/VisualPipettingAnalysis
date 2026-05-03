"""Ground-truth annotation data model and persistence for the Parameter Sweep tab.

An annotation is one horizontal line placed on a full-resolution image,
labelled as ``tip_bottom``, ``liquid``, or ``bubble``.  All Y coordinates
are in **full-image pixel rows** (origin = top of the original camera frame).

Serialisation format (JSON at ``<primary_folder>/sweep_gt.json``)::

    {
      "version": 1,
      "tolerance_px": 10,
      "images": {
        "/abs/path/to/img_001.png": {
          "annotations": [
            {"type": "tip_bottom", "y_px": 312},
            {"type": "liquid",     "y_px": 480},
            {"type": "bubble",     "y_px": 320}
          ]
        }
      }
    }
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional

# ── Constants ────────────────────────────────────────────────────────────────

GT_FILENAME = "sweep_gt.json"

ANN_TYPES = ("tip_bottom", "liquid", "bubble")

# Colours per annotation type (R, G, B) — used in the view
ANN_COLORS: Dict[str, tuple] = {
    "tip_bottom": (255, 220, 0),    # yellow
    "liquid":     (0, 220, 220),    # cyan
    "bubble":     (100, 255, 100),  # green
}


# ── Data classes ─────────────────────────────────────────────────────────────

@dataclass
class GTAnnotation:
    type: str    # one of ANN_TYPES
    y_px: int    # full-image pixel row

    def to_dict(self) -> dict:
        return {"type": self.type, "y_px": int(self.y_px)}

    @classmethod
    def from_dict(cls, d: dict) -> "GTAnnotation":
        return cls(type=d["type"], y_px=int(d["y_px"]))


@dataclass
class ImageGT:
    image_path: str
    annotations: List[GTAnnotation] = field(default_factory=list)

    def sorted_annotations(self) -> List[GTAnnotation]:
        return sorted(self.annotations, key=lambda a: a.y_px)

    def to_dict(self) -> dict:
        return {"annotations": [a.to_dict() for a in self.annotations]}

    @classmethod
    def from_dict(cls, path: str, d: dict) -> "ImageGT":
        return cls(
            image_path=path,
            annotations=[GTAnnotation.from_dict(a) for a in d.get("annotations", [])],
        )


# ── Persistence ───────────────────────────────────────────────────────────────

def gt_file_path(primary_folder: str) -> str:
    return os.path.join(primary_folder, GT_FILENAME)


def load_gt(primary_folder: str) -> tuple[Dict[str, ImageGT], int]:
    """Return ``(image_gt_dict, tolerance_px)`` from the GT file.
    Both are empty defaults if the file is absent or malformed."""
    path = gt_file_path(primary_folder)
    if not os.path.isfile(path):
        return {}, 10
    try:
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        tolerance = int(data.get("tolerance_px", 10))
        result: Dict[str, ImageGT] = {}
        for img_path, img_data in data.get("images", {}).items():
            result[img_path] = ImageGT.from_dict(img_path, img_data)
        return result, tolerance
    except Exception:
        return {}, 10


def save_gt(
    primary_folder: str,
    image_gts: Dict[str, ImageGT],
    tolerance_px: int,
) -> None:
    """Write GT state atomically."""
    path = gt_file_path(primary_folder)
    data = {
        "version": 1,
        "tolerance_px": tolerance_px,
        "images": {img_path: igt.to_dict() for img_path, igt in image_gts.items()},
    }
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2, ensure_ascii=False)
    os.replace(tmp, path)
