"""
Image loading utilities.
Resolves subfolder paths relative to the run root and loads images via OpenCV.
"""

from __future__ import annotations
import os
from typing import Callable, Dict, List, Optional, Tuple
import cv2
import numpy as np


def load_step_frames(
    subfolders: List[str],
    run_folder: str,
    transform: Optional[Callable[[np.ndarray], np.ndarray]] = None,
) -> Tuple[List[np.ndarray], List[str]]:
    """
    Load all images from the given subfolders (relative to run_folder).

    Parameters
    ----------
    subfolders : list of subfolder names relative to run_folder
    run_folder : absolute path to the run root
    transform  : optional per-frame callable (e.g. undistort+rotate) applied
                 immediately after imread.  Pass session_state.get_frame_transform().

    Returns (frames, source_paths) as parallel lists.
    Frames are BGR uint8 arrays.
    """
    frames: List[np.ndarray] = []
    paths: List[str] = []
    for subfolder in subfolders:
        folder_path = os.path.join(run_folder, subfolder)
        if not os.path.isdir(folder_path):
            continue
        for fname in sorted(os.listdir(folder_path)):
            if fname.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):
                fpath = os.path.join(folder_path, fname)
                img = cv2.imread(fpath)
                if img is not None:
                    if transform is not None:
                        img = transform(img)
                    frames.append(img)
                    paths.append(fpath)
    return frames, paths


def load_step_images(
    subfolders: List[str],
    run_folder: str,
    transform: Optional[Callable[[np.ndarray], np.ndarray]] = None,
) -> Dict[str, np.ndarray]:
    """
    Legacy helper — returns a dict mapping relative path → BGR image array.
    Prefer load_step_frames() for new code.
    """
    frames, paths = load_step_frames(subfolders, run_folder, transform=transform)
    return {os.path.relpath(p, run_folder): f for p, f in zip(paths, frames)}
