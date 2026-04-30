"""
Data model for ROI calibration.

Coordinate convention throughout:
    image_x = column  (horizontal, increasing right)   ← corresponds to robot Y
    image_y = row     (vertical,   increasing downward) ← corresponds to robot -Z

All on-canvas editing uses ABSOLUTE image coordinates (AbsolutePair).
The RELATIVE representation (RoiDefinition) is only used for storage and for
translating the polygon to other tips.

Anchor = midpoint of the topmost pair (lowest image_y), stored as an offset
from the tip's calibrated_pixel_yz:
    calibrated_pixel_yz[0] → image_x (column)
    calibrated_pixel_yz[1] → image_y (row)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


TIP_TYPES: List[str] = ["50uL", "200uL", "1000uL"]


@dataclass
class AbsolutePair:
    """One left/right point pair in absolute image-pixel coordinates."""
    left_x:  float
    left_y:  float
    right_x: float
    right_y: float

    def midpoint_y(self) -> float:
        """Average vertical position of this pair (used for top/bottom ordering)."""
        return (self.left_y + self.right_y) / 2.0

    def midpoint_x(self) -> float:
        return (self.left_x + self.right_x) / 2.0


@dataclass
class RoiDefinition:
    """
    ROI definition for one tip type, stored as offsets from an anchor point.

    anchor_dx, anchor_dy : offset (in image pixels) of the polygon's top-centre
                           anchor from the tip's calibrated_pixel_yz.
    pairs                : list of {"left": [dx, dy], "right": [dx, dy]} dicts,
                           where dx/dy are image-pixel offsets from the anchor.
    """
    tip_type:  str
    anchor_dx: float = 0.0
    anchor_dy: float = 0.0
    pairs: List[Dict[str, List[float]]] = field(default_factory=list)

    # ── Serialisation ────────────────────────────────────────────────────────

    def to_dict(self) -> Dict[str, Any]:
        return {
            "tip_type":      self.tip_type,
            "anchor_offset": [self.anchor_dx, self.anchor_dy],
            "pairs":         self.pairs,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "RoiDefinition":
        obj = cls(tip_type=d.get("tip_type", ""))
        off = d.get("anchor_offset", [0.0, 0.0])
        obj.anchor_dx = float(off[0])
        obj.anchor_dy = float(off[1])
        obj.pairs = d.get("pairs", [])
        return obj

    # ── Coordinate conversion ────────────────────────────────────────────────

    def to_absolute_pairs(
        self, tip_y_px: float, tip_z_px: float
    ) -> List[AbsolutePair]:
        """Return absolute-coordinate pairs for a tip at (tip_y_px, tip_z_px)."""
        ax = tip_y_px + self.anchor_dx
        ay = tip_z_px + self.anchor_dy
        result: List[AbsolutePair] = []
        for p in self.pairs:
            l, r = p["left"], p["right"]
            result.append(AbsolutePair(
                left_x=ax + l[0],  left_y=ay + l[1],
                right_x=ax + r[0], right_y=ay + r[1],
            ))
        return result

    @staticmethod
    def from_absolute_pairs(
        tip_type:    str,
        pairs:       List[AbsolutePair],
        ref_tip_y_px: float,
        ref_tip_z_px: float,
    ) -> "RoiDefinition":
        """
        Build a RoiDefinition from absolute pairs drawn on a reference tip.

        Anchor = midpoint of the topmost (smallest-y) pair.
        All pair points are stored as offsets from that anchor.
        anchor_offset = anchor − calibrated_pixel_yz of the reference tip.
        """
        if not pairs:
            return RoiDefinition(tip_type=tip_type)

        topmost = min(pairs, key=lambda p: p.midpoint_y())
        ax = (topmost.left_x + topmost.right_x) / 2.0
        ay = (topmost.left_y + topmost.right_y) / 2.0

        roi = RoiDefinition(
            tip_type=tip_type,
            anchor_dx=ax - ref_tip_y_px,
            anchor_dy=ay - ref_tip_z_px,
        )
        for p in pairs:
            roi.pairs.append({
                "left":  [p.left_x  - ax, p.left_y  - ay],
                "right": [p.right_x - ax, p.right_y - ay],
            })
        return roi
