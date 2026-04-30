"""
Save / load ROI definitions to / from instrument_config.json.

Schema written under instrument_config["camera_rois"]:

    "camera_rois": {
        "200uL": {
            "tip_type":      "200uL",
            "anchor_offset": [dx, dy],          # offset of top-centre from
                                                 # calibrated_pixel_yz
            "pairs": [
                {"left": [dx, dy], "right": [dx, dy]},
                ...
            ]
        },
        ...
    }

Absolute pairs drawn on the reference tip are converted to relative offsets
before writing; on load they are converted back to absolute coordinates for
the requested reference tip.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional

from pa_gui.roi_calibration.models import AbsolutePair, RoiDefinition, TIP_TYPES


def save_rois(
    instrument_config_path: str,
    roi_map: Dict[str, List[AbsolutePair]],   # {tip_type: [AbsolutePair, ...]}
    mandrels: List[dict],
    ref_tip_idx: int,
) -> None:
    """
    Convert absolute pairs to relative RoiDefinitions and write them into
    the instrument_config file.  Existing keys not in roi_map are left intact.
    Tip types with empty pair lists are removed from camera_rois.
    """
    ref = _get_mandrel(mandrels, ref_tip_idx)
    if ref is None:
        raise ValueError(
            f"Mandrel with index {ref_tip_idx} not found in instrument_config.")
    ref_y, ref_z = _mandrel_yz(ref)

    path = Path(instrument_config_path)
    config: dict = {}
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            config = json.load(f)

    camera_rois: dict = config.get("camera_rois", {})
    for tip_type, pairs in roi_map.items():
        if not pairs:
            camera_rois.pop(tip_type, None)
            continue
        roi_def = RoiDefinition.from_absolute_pairs(
            tip_type=tip_type,
            pairs=pairs,
            ref_tip_y_px=ref_y,
            ref_tip_z_px=ref_z,
        )
        camera_rois[tip_type] = roi_def.to_dict()

    config["camera_rois"] = camera_rois
    with open(path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)


def load_rois(
    instrument_config_path: str,
    mandrels: List[dict],
    ref_tip_idx: int,
) -> Dict[str, List[AbsolutePair]]:
    """
    Read ROI definitions from instrument_config and convert them to absolute
    coordinates for the given reference tip.

    Returns {tip_type: [AbsolutePair, ...]} for every entry in TIP_TYPES
    (empty list for types not yet defined).
    """
    result: Dict[str, List[AbsolutePair]] = {t: [] for t in TIP_TYPES}

    path = Path(instrument_config_path)
    if not path.exists():
        return result

    with open(path, "r", encoding="utf-8") as f:
        config = json.load(f)

    camera_rois: dict = config.get("camera_rois", {})
    if not camera_rois:
        return result

    ref = _get_mandrel(mandrels, ref_tip_idx)
    if ref is None:
        return result
    ref_y, ref_z = _mandrel_yz(ref)

    for tip_type in TIP_TYPES:
        roi_dict = camera_rois.get(tip_type)
        if roi_dict is not None:
            roi_def = RoiDefinition.from_dict(roi_dict)
            result[tip_type] = roi_def.to_absolute_pairs(ref_y, ref_z)

    return result


# ── Helpers ────────────────────────────────────────────────────────────────────

def _get_mandrel(mandrels: List[dict], index: int) -> Optional[dict]:
    for m in mandrels:
        if m.get("index") == index:
            return m
    return None


def _mandrel_yz(mandrel: dict) -> tuple[float, float]:
    yz = mandrel.get("calibrated_pixel_yz", [0.0, 0.0])
    return float(yz[0]), float(yz[1])
