"""
Lens undistortion and rotation correction utilities.

All functions are pure (no side effects, no state).  The caller is responsible
for caching the maps returned by compute_maps() — they are expensive to build
but cheap to reuse.
"""

from __future__ import annotations

from typing import Tuple

import cv2
import numpy as np


def compute_maps(
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
    image_size: Tuple[int, int],
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Pre-compute the undistortion lookup tables for a calibrated camera.

    Parameters
    ----------
    camera_matrix : np.ndarray, shape (3, 3)
    dist_coeffs   : np.ndarray, shape (1, N) or (N,)  N = 4, 5, 8, 12, or 14
    image_size    : (width_px, height_px) — must match the images that will be remapped

    Returns
    -------
    map1, map2 : np.ndarray
        Passed directly to apply_maps().  map1 contains x-coordinates,
        map2 contains y-coordinates (cv2.CV_32FC1).
    """
    new_camera_matrix, _ = cv2.getOptimalNewCameraMatrix(
        camera_matrix, dist_coeffs, image_size, alpha=0
    )
    map1, map2 = cv2.initUndistortRectifyMap(
        camera_matrix,
        dist_coeffs,
        None,
        new_camera_matrix,
        image_size,
        cv2.CV_32FC1,
    )
    return map1, map2


def apply_maps(
    frame: np.ndarray,
    map1: np.ndarray,
    map2: np.ndarray,
) -> np.ndarray:
    """
    Apply pre-computed undistortion maps to a single frame.

    Parameters
    ----------
    frame : np.ndarray  BGR uint8 image
    map1, map2 : np.ndarray  output of compute_maps()

    Returns
    -------
    np.ndarray  undistorted BGR uint8 image, same shape as input
    """
    return cv2.remap(frame, map1, map2, cv2.INTER_LINEAR)


def apply_rotation(frame: np.ndarray, angle_deg: float) -> np.ndarray:
    """
    Rotate a frame about its centre without cropping.

    A positive angle_deg rotates counter-clockwise (OpenCV convention with
    the sign flipped — see note below).

    Note: cv2.getRotationMatrix2D uses a clockwise-positive convention
    internally, so we negate the angle to make the public API
    counter-clockwise-positive (matching the rotation_deg field stored in
    instrument_config).

    Parameters
    ----------
    frame     : np.ndarray  BGR (or grayscale) image
    angle_deg : float       rotation angle in degrees, CCW positive

    Returns
    -------
    np.ndarray  rotated image, same shape as input (borders filled with black)
    """
    if angle_deg == 0.0:
        return frame

    h, w = frame.shape[:2]
    cx, cy = w / 2.0, h / 2.0
    M = cv2.getRotationMatrix2D((cx, cy), -angle_deg, scale=1.0)
    return cv2.warpAffine(frame, M, (w, h), flags=cv2.INTER_LINEAR)
