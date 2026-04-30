"""
CalibrationWorker — QThread that runs the OpenCV checkerboard calibration pipeline.

Emits per-image progress and a final result dict so the UI stays responsive.
"""

from __future__ import annotations

import glob
import os
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from PySide6.QtCore import QThread, Signal


class CalibrationWorker(QThread):
    """
    Background thread for checkerboard camera calibration.

    Signals
    -------
    progress(current, total, filename, corners_found)
        Emitted after each image is processed.
    finished(result)
        Emitted on success.  ``result`` keys:
            camera_matrix   : list[list[float]]  (3×3)
            dist_coeffs     : list[float]        (5 elements)
            rms             : float              reprojection error in pixels
            per_image       : list[dict]         one entry per image:
                                filename, detected (bool), residual_px (float | None)
            n_valid         : int                images where corners were detected
            n_total         : int
            preview_image   : np.ndarray | None  BGR image with corners drawn
            image_size      : list[int]          [width, height]
    error(message)
        Emitted on failure (exception or no valid images).
    """

    progress: Signal = Signal(int, int, str, bool)
    finished: Signal = Signal(dict)
    error: Signal = Signal(str)

    # Extensions that cv2.imread handles reliably
    _IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")

    def __init__(
        self,
        folder: str,
        board_w: int,
        board_h: int,
        square_mm: float = 1.0,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self._folder = folder
        self._board_w = board_w
        self._board_h = board_h
        self._square_mm = square_mm
        self._cancelled = False

    def cancel(self) -> None:
        """Request cancellation.  The thread will stop after the current image."""
        self._cancelled = True

    # ------------------------------------------------------------------
    # QThread entry point
    # ------------------------------------------------------------------

    def run(self) -> None:
        try:
            self._run()
        except Exception as exc:  # noqa: BLE001
            self.error.emit(str(exc))

    def _run(self) -> None:
        image_paths = self._collect_images()
        if not image_paths:
            self.error.emit(f"No images found in:\n{self._folder}")
            return

        board_size = (self._board_w, self._board_h)

        # 3-D object points for one board image (Z = 0 plane)
        objp = np.zeros((self._board_w * self._board_h, 3), dtype=np.float32)
        objp[:, :2] = np.mgrid[0 : self._board_w, 0 : self._board_h].T.reshape(-1, 2)
        objp *= self._square_mm

        objpoints: List[np.ndarray] = []
        imgpoints: List[np.ndarray] = []
        per_image: List[dict] = []
        image_size: Optional[Tuple[int, int]] = None
        preview_image: Optional[np.ndarray] = None

        total = len(image_paths)

        for idx, fpath in enumerate(image_paths):
            if self._cancelled:
                self.error.emit("Calibration cancelled.")
                return

            fname = os.path.basename(fpath)
            img = cv2.imread(fpath)

            if img is None:
                self.progress.emit(idx + 1, total, fname, False)
                per_image.append({"filename": fname, "detected": False, "residual_px": None})
                continue

            if image_size is None:
                h, w = img.shape[:2]
                image_size = (w, h)

            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            flags = (
                cv2.CALIB_CB_ADAPTIVE_THRESH
                | cv2.CALIB_CB_NORMALIZE_IMAGE
                | cv2.CALIB_CB_FAST_CHECK
            )
            found, corners = cv2.findChessboardCorners(gray, board_size, flags)

            if found:
                criteria = (
                    cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
                    30,
                    0.001,
                )
                corners_refined = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
                objpoints.append(objp)
                imgpoints.append(corners_refined)

                if preview_image is None:
                    preview_image = img.copy()
                    cv2.drawChessboardCorners(preview_image, board_size, corners_refined, found)

            self.progress.emit(idx + 1, total, fname, found)
            per_image.append({"filename": fname, "detected": found, "residual_px": None})

        n_valid = len(objpoints)
        if n_valid < 3:
            self.error.emit(
                f"Only {n_valid} image(s) had detectable corners (minimum 3 required).\n"
                "Check board dimensions, lighting, and that the full board is visible."
            )
            return

        assert image_size is not None
        rms, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
            objpoints, imgpoints, image_size, None, None
        )

        # Compute per-image residuals
        valid_idx = 0
        for entry in per_image:
            if entry["detected"]:
                imgp_proj, _ = cv2.projectPoints(
                    objpoints[valid_idx],
                    rvecs[valid_idx],
                    tvecs[valid_idx],
                    camera_matrix,
                    dist_coeffs,
                )
                residual = float(
                    np.sqrt(
                        np.mean(
                            np.sum((imgpoints[valid_idx] - imgp_proj) ** 2, axis=2)
                        )
                    )
                )
                entry["residual_px"] = residual
                valid_idx += 1

        self.finished.emit(
            {
                "camera_matrix": camera_matrix.tolist(),
                "dist_coeffs": dist_coeffs.flatten().tolist(),
                "rms": float(rms),
                "per_image": per_image,
                "n_valid": n_valid,
                "n_total": total,
                "preview_image": preview_image,
                "image_size": list(image_size),
            }
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _collect_images(self) -> List[str]:
        paths: List[str] = []
        for entry in sorted(os.listdir(self._folder)):
            if any(entry.lower().endswith(ext) for ext in self._IMAGE_EXTS):
                paths.append(os.path.join(self._folder, entry))
        return paths
