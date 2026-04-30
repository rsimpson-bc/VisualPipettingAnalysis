"""
CameraCalibrationData — serialisable dataclass representing the result of one
OpenCV checkerboard calibration session.

Handles conversion between the in-memory representation (numpy arrays) and the
JSON form stored under the ``camera_calibration`` key of instrument_config.json.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional, Tuple


@dataclass
class CameraCalibrationData:
    """Full calibration result for a single camera."""

    # 3×3 intrinsic matrix as a Python list-of-lists (JSON-serialisable).
    camera_matrix: List[List[float]]

    # Distortion coefficients [k1, k2, p1, p2, k3] (or up to 14 elements).
    dist_coeffs: List[float]

    # (width_px, height_px) of images used during calibration.
    image_size: Tuple[int, int]

    # CCW-positive rotation correction angle in degrees (0 = no rotation).
    rotation_deg: float = 0.0

    # RMS reprojection error in pixels from cv2.calibrateCamera.
    rms: float = 0.0

    # Number of checkerboard images that contributed to this calibration.
    n_images_used: int = 0

    # ISO-8601 timestamp (filled automatically by save_to_config).
    calibrated_at: str = ""

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def to_dict(self) -> dict:
        return {
            "camera_matrix": self.camera_matrix,
            "dist_coeffs": self.dist_coeffs,
            "image_size": list(self.image_size),
            "rotation_deg": self.rotation_deg,
            "rms_reprojection_error": self.rms,
            "n_images_used": self.n_images_used,
            "calibrated_at": self.calibrated_at,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "CameraCalibrationData":
        return cls(
            camera_matrix=d["camera_matrix"],
            dist_coeffs=d["dist_coeffs"],
            image_size=tuple(d["image_size"]),  # type: ignore[arg-type]
            rotation_deg=d.get("rotation_deg", 0.0),
            rms=d.get("rms_reprojection_error", 0.0),
            n_images_used=d.get("n_images_used", 0),
            calibrated_at=d.get("calibrated_at", ""),
        )

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save_to_config(self, config_path: str) -> None:
        """
        Write this calibration result into the ``camera_calibration`` key of
        an existing instrument_config.json file.  All other keys are preserved.
        """
        path = Path(config_path)
        with path.open("r", encoding="utf-8") as fh:
            data = json.load(fh)

        self.calibrated_at = datetime.now(timezone.utc).isoformat()
        data["camera_calibration"] = self.to_dict()

        with path.open("w", encoding="utf-8") as fh:
            json.dump(data, fh, indent=2)

    @classmethod
    def load_from_config(cls, config_path: str) -> Optional["CameraCalibrationData"]:
        """
        Read the ``camera_calibration`` section from instrument_config.json.
        Returns None if the key is absent or the file cannot be read.
        """
        path = Path(config_path)
        if not path.exists():
            return None
        try:
            with path.open("r", encoding="utf-8") as fh:
                data = json.load(fh)
        except (json.JSONDecodeError, OSError):
            return None

        cal = data.get("camera_calibration")
        if cal is None:
            return None
        return cls.from_dict(cal)

    def save_rotation_to_config(self, config_path: str) -> None:
        """
        Update only the ``rotation_deg`` field in an existing instrument_config.
        Used by RotationPanel when the user saves rotation without re-running
        the full checkerboard calibration.
        """
        path = Path(config_path)
        with path.open("r", encoding="utf-8") as fh:
            data = json.load(fh)

        cal = data.setdefault("camera_calibration", {})
        cal["rotation_deg"] = self.rotation_deg

        with path.open("w", encoding="utf-8") as fh:
            json.dump(data, fh, indent=2)
