"""
Shared geometry utilities used by both tip_identification and liquid_measurement.
"""

from __future__ import annotations
import math
from typing import Dict, List, Optional, Tuple


def px_to_mm(px: float, scale: float) -> float:
    """Convert pixels to millimetres using a calibrated scale factor."""
    return px * scale


def angle_between_points(p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
    """Return the angle in degrees of the line from p1 to p2 relative to horizontal."""
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    return math.degrees(math.atan2(dy, dx))


def lookup_roi(
    instrument_config: Dict,
    pipette_index: int,
    tip_type: Optional[str],
) -> Tuple[Optional[tuple], Optional[List[tuple]]]:
    """
    Resolve ROI data for a given pipette from the instrument configuration.

    Returns (roi_bbox, roi_points) where:
        roi_bbox   — (x, y, w, h) bounding rectangle, or None
        roi_points — polygon [(x,y), …] for tip analysis, or None

    Reads the ``camera_rois`` section of instrument_config, which stores ROI
    definitions in the format produced by the ROI calibration tool::

        camera_rois:
          <tip_type>:
            tip_type:      <str>
            anchor_offset: [dx, dy]     # offset from mandrel calibrated_pixel_yz
            pairs:
              - left:  [dx, dy]
                right: [dx, dy]
              ...

    The absolute ROI is resolved per-pipette using the mandrel calibration
    data (``instrument_config["mandrels"]``).  If no mandrel entry is found
    for ``pipette_index`` or no ROI definition exists for the requested
    tip type, both return values are None — triggering full-image processing,
    which is a safe fallback.
    """
    if not instrument_config:
        return None, None

    camera_rois = instrument_config.get("camera_rois")
    if not camera_rois:
        return None, None

    # Find the ROI definition for this tip type (exact match → "default" → first)
    roi_dict: Optional[Dict] = None
    if tip_type and tip_type in camera_rois:
        roi_dict = camera_rois[tip_type]
    elif "default" in camera_rois:
        roi_dict = camera_rois["default"]
    elif camera_rois:
        roi_dict = next(iter(camera_rois.values()))

    if not isinstance(roi_dict, dict):
        return None, None

    # Find this pipette's mandrel calibrated position
    tip_y_px: Optional[float] = None
    tip_z_px: Optional[float] = None
    for m in instrument_config.get("mandrels", []):
        if not isinstance(m, dict):
            continue
        if m.get("index") == pipette_index:
            yz = m.get("calibrated_pixel_yz", [0.0, 0.0])
            tip_y_px = float(yz[0])
            tip_z_px = float(yz[1])
            break

    if tip_y_px is None:
        # No mandrel entry found for this pipette index
        return None, None

    # Apply per-tip correction offset if present
    corrections = roi_dict.get("tip_corrections", {})
    correction = corrections.get(str(pipette_index)) or corrections.get(pipette_index)
    if correction is not None:
        try:
            tip_y_px += float(correction[0])
            tip_z_px += float(correction[1])
        except (TypeError, IndexError):
            pass

    # Convert the relative anchor+pairs definition to absolute polygon vertices
    roi_points = _camera_roi_to_roi_points(roi_dict, tip_y_px, tip_z_px)
    if not roi_points:
        return None, None

    # Derive bounding box from the polygon
    xs = [p[0] for p in roi_points]
    ys = [p[1] for p in roi_points]
    roi_bbox = (int(min(xs)), int(min(ys)),
                int(max(xs)) - int(min(xs)), int(max(ys)) - int(min(ys)))

    return roi_bbox, roi_points


def _camera_roi_to_roi_points(
    roi_dict: Dict,
    tip_y_px: float,
    tip_z_px: float,
) -> List[tuple]:
    """
    Convert a ``camera_rois`` entry to a flat list of (x, y) polygon vertices
    compatible with ``InwardEdgeScan._extract_pairs_from_roi``.

    The encoding follows the pattern expected by ``_extract_pairs_from_roi``::

        roi_points = [L0, R0, R1, R2, ..., R(P-1), L(P-1), ..., L1]

    where pairs are ordered top-to-bottom (smallest image-y first).
    Returns an empty list when fewer than two pairs are defined.
    """
    anchor_off = roi_dict.get("anchor_offset", [0.0, 0.0])
    anchor_x = tip_y_px + float(anchor_off[0])
    anchor_y = tip_z_px + float(anchor_off[1])

    raw_pairs = roi_dict.get("pairs", [])
    if not raw_pairs or len(raw_pairs) < 2:
        return []

    abs_pairs = []
    for p in raw_pairs:
        l, r = p["left"], p["right"]
        abs_pairs.append({
            "lx": anchor_x + float(l[0]),
            "ly": anchor_y + float(l[1]),
            "rx": anchor_x + float(r[0]),
            "ry": anchor_y + float(r[1]),
        })

    # Sort top-to-bottom by midpoint y
    abs_pairs.sort(key=lambda p: (p["ly"] + p["ry"]) / 2)

    # Encode as polygon: [L0, R0, R1, …, R(P-1), L(P-1), …, L1]
    points: List[tuple] = [
        (abs_pairs[0]["lx"], abs_pairs[0]["ly"]),
        (abs_pairs[0]["rx"], abs_pairs[0]["ry"]),
    ]
    for p in abs_pairs[1:]:
        points.append((p["rx"], p["ry"]))
    for p in reversed(abs_pairs[1:]):
        points.append((p["lx"], p["ly"]))

    return points
