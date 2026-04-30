"""
Layer 1 — Extrema Lines Extractor.

Finds per-row local minima and maxima, stitches them into lines,
eliminates short lines, re-stitches (optional stage-5 endpoint-angle merge),
then derives two 1-D signals consumed by the ExtremaLines Layer-2 mode:

  extrema_minima_signal     — per-row count of lines "present" at that row
                              (density signal; used for ZProfile density)
  extrema_termination_signal — per-row count of line endpoints within ±term_window
                               rows (termination signal)
  edge_map                  — 2-D image with extrema points marked (for viz/debug)

Algorithm origin: _analyze_tip_extrema_lines() + _update_el_graphs() in
  VS_Code_Test/VideoDeltaHeatMap1.py (Stages 1-5 pipeline).
GUI / Tkinter / visualisation code removed.  Stage 4 (intensity segmentation)
is skipped here — the two published signals come from Stage 3 + Stage 5 lines,
matching what `_update_el_graphs()` used with the Stage 5 post-stitch output.
"""

from __future__ import annotations

from typing import List, Tuple

import cv2
import numpy as np

from pa.pipeline.types import ImageSet, RawFeatures
from pa.pipeline.cache import ProcessedImageCache
from pa.pipeline.extractors.base import BaseExtractor
from pa.pipeline import image_primitives as ip

# Type alias for a point in a line
_Point = Tuple[int, int]    # (row, col)
_Line  = List[_Point]


class ExtremaLinesExtractor(BaseExtractor):
    """
    Params (analysis_config.json):
        blur_kernel (int, default 5):               Gaussian blur kernel size (must be odd)
        neighborhood (int, default 3):              ±cols for local min/max detection
        stitch_tolerance (int, default 5):          max col distance for stitching points
        angle_tolerance (float, default 90):        degrees from vertical (90 = no constraint)
        angle_mode (str, default "A"):              "A" = from start; "B" = recent-average
        min_line_length (int, default 5):           minimum vertical span to keep a line
        angle_lock_points (int, default 0):         lock angle after N points (0 = disabled)
        intensity_threshold (float, default 10):    stage-4 intensity change threshold (unused
                                                    for published signals but kept for API compat)
        max_gap (int, default 1):                   max skipped rows allowed during stitch
        merge_angle_tol (float, default 0):         post-stitch merge angle tolerance
        merge_row_sep (int, default 0):             post-stitch merge max row gap
        merge_col_sep (int, default 1):             post-stitch merge max col gap
        s5_vgap (int, default 1):                   Stage-5 vertical gap tolerance
        s5_hgap (int, default 1):                   Stage-5 horizontal gap tolerance
        s5_angle (float, default 10):               Stage-5 endpoint angle tolerance (deg)
        s5_pts (int, default 10):                   Stage-5 points used for angle estimation
        term_window (int, default 0):               ±rows to accumulate terminations
        use_ab (bool, default False)
        stitch_gap (int, default 1):                alias for max_gap (for backwards compat)
    """

    def _extract(self, image_set: ImageSet, cache: ProcessedImageCache) -> RawFeatures:
        p = self.params

        # ── 1. Working image ──────────────────────────────────────────────────
        use_ab = p.get("use_ab", False)
        if use_ab and len(image_set.frames) >= 2:
            working = cache.get_or_compute(
                image_set.pipette_index, "accumulate_ab", {},
                lambda: ip.accumulate_ab(image_set.frames),
            )
            gray8 = np.clip(working, 0, 255).astype(np.uint8)
        else:
            gray8 = cache.get_or_compute(
                image_set.pipette_index, "to_grayscale_frame0", {},
                lambda: ip.to_grayscale(image_set.frames[0]),
            )

        # ── 2. ROI crop + polygon mask ────────────────────────────────────────
        roi = image_set.roi           # (x, y, w, h) or None
        roi_pts = image_set.roi_points  # polygon or None

        if roi is not None:
            gray8 = ip.extract_roi(gray8, roi)

        h, w = gray8.shape

        # Build a polygon mask (only needed if roi_points provided)
        if roi_pts and len(roi_pts) >= 3:
            mask = np.zeros((h, w), dtype=np.uint8)
            ox = roi[0] if roi else 0
            oy = roi[1] if roi else 0
            local_pts = np.array([[p[0] - ox, p[1] - oy] for p in roi_pts], dtype=np.int32)
            cv2.fillPoly(mask, [local_pts], 255)
        else:
            mask = None  # No polygon clipping

        # ── 3. Blur ───────────────────────────────────────────────────────────
        blur_params = {
            "method":      p.get("blur_method", "gaussian"),
            "kernel_size": p.get("blur_kernel", 5),
            "d":           p.get("bilateral_d", 9),
            "sigma_color": p.get("bilateral_sigma_color", 75.0),
            "sigma_space": p.get("bilateral_sigma_space", 75.0),
        }
        blurred = cache.get_or_compute(
            image_set.pipette_index, "blur", blur_params,
            lambda: ip.blur(gray8, **blur_params),
        )

        # ── 3.5. Optional gradient step ───────────────────────────────────────────────
        grad_method = p.get("gradient_method", "none")
        grad_ksize  = p.get("gradient_ksize", 3)
        if grad_method != "none":
            proc = cache.get_or_compute(
                image_set.pipette_index, "gradient",
                {"method": grad_method, "ksize": grad_ksize},
                lambda: ip.apply_gradient_step(blurred, grad_method, grad_ksize),
            )
        else:
            proc = blurred

        # ── 4. Find per-row local min/max points ────────────────────────────────────
        neighborhood  = p.get("neighborhood", 3)
        min_pts, max_pts = _find_extrema_points(proc, mask, neighborhood)

        # ── 5. Stitch into lines ──────────────────────────────────────────────
        stitch_tol   = p.get("stitch_tolerance", 5)
        angle_tol    = p.get("angle_tolerance", 90.0)
        angle_mode   = p.get("angle_mode", "A")[0]   # 'A' or 'B'
        max_gap      = p.get("max_gap", p.get("stitch_gap", 1))
        angle_lock_n = p.get("angle_lock_points", 0)

        min_lines = _stitch_lines(min_pts, stitch_tol, angle_tol, angle_mode,
                                   max_gap, angle_lock_n)
        max_lines = _stitch_lines(max_pts, stitch_tol, angle_tol, angle_mode,
                                   max_gap, angle_lock_n)

        # ── 6. Filter by minimum length ───────────────────────────────────────
        min_len = p.get("min_line_length", 5)
        min_lines = [l for l in min_lines if _vertical_span(l) >= min_len]
        max_lines = [l for l in max_lines if _vertical_span(l) >= min_len]

        # ── 7. Post-stitch merge (Stage 3) ────────────────────────────────────
        merge_angle  = p.get("merge_angle_tol", 0.0)
        merge_row    = p.get("merge_row_sep", 0)
        merge_col    = p.get("merge_col_sep", 1)
        min_lines = _merge_segments(min_lines, merge_angle, merge_row, merge_col)
        max_lines = _merge_segments(max_lines, merge_angle, merge_row, merge_col)

        # ── 8. Stage-5 endpoint-angle stitch ─────────────────────────────────
        s5_vgap  = p.get("s5_vgap", 1)
        s5_hgap  = p.get("s5_hgap", 1)
        s5_angle = p.get("s5_angle", 10.0)
        s5_pts   = p.get("s5_pts", 10)
        all_lines = _stage5_stitch(
            min_lines + max_lines, s5_vgap, s5_hgap, s5_angle, s5_pts)

        # ── 9. Compute 1-D signals from the final Stage-5 lines ───────────────
        term_win = p.get("term_window", 0)
        presence     = _compute_presence(all_lines, h)     # density
        terminations = _compute_terminations(all_lines, h, term_win)

        z_axis = np.arange(h, dtype=np.float32)

        # ── 10. Build edge_map for debug / visualisation ───────────────────────
        edge_map = np.zeros((h, w), dtype=np.uint8)
        for line in all_lines:
            for r, c in line:
                if 0 <= r < h and 0 <= c < w:
                    edge_map[r, c] = 255

        return RawFeatures(
            mode="ExtremaLinesExtractor",
            pipette_index=image_set.pipette_index,
            z_axis_px=z_axis,
            extrema_minima_signal=presence.astype(np.float32),
            extrema_termination_signal=terminations.astype(np.float32),
            edge_map=edge_map,
        )


# ---------------------------------------------------------------------------
# Stage 2 — local extrema detection
# ---------------------------------------------------------------------------

def _find_extrema_points(
    blurred: np.ndarray,
    mask: np.ndarray | None,
    neighborhood: int,
) -> tuple[list[_Point], list[_Point]]:
    """Per-row local minima and local maxima, centre-of-plateau tie-breaking."""
    height, width = blurred.shape
    min_pts: list[_Point] = []
    max_pts: list[_Point] = []

    for row in range(height):
        row_data = blurred[row, :]
        for col in range(neighborhood, width - neighborhood):
            if mask is not None and mask[row, col] == 0:
                continue
            window = row_data[col - neighborhood: col + neighborhood + 1]
            # --- minima ---
            if row_data[col] == window.min():
                idxs = np.where(window == window.min())[0]
                if len(idxs) > 0 and idxs[len(idxs) // 2] == neighborhood:
                    min_pts.append((row, col))
            # --- maxima ---
            if row_data[col] == window.max():
                idxs = np.where(window == window.max())[0]
                if len(idxs) > 0 and idxs[len(idxs) // 2] == neighborhood:
                    max_pts.append((row, col))

    return min_pts, max_pts


# ---------------------------------------------------------------------------
# Stage 3 — bottom-to-top stitching
# ---------------------------------------------------------------------------

def _stitch_lines(
    points: list[_Point],
    stitch_tol: int,
    angle_tol: float,
    angle_mode: str,
    max_gap: int,
    angle_lock_n: int,
) -> list[_Line]:
    """Stitch (row,col) points bottom-to-top into continuous lines."""
    if not points:
        return []

    # Group by row
    by_row: dict[int, list[int]] = {}
    for r, c in points:
        by_row.setdefault(r, []).append(c)
    for r in by_row:
        by_row[r].sort()

    rows = sorted(by_row.keys(), reverse=True)  # bottom → top
    if not rows:
        return []

    lines: list[_Line] = [[(rows[0], c)] for c in by_row[rows[0]]]

    def _ok_angle(start_r, start_c, end_r, end_c):
        if angle_tol >= 90:
            return True
        dy = start_r - end_r
        dx = end_c - start_c
        if dy == 0:
            return False
        ang = np.degrees(np.arctan2(dy, abs(dx)))
        return (90.0 - angle_tol) <= ang <= 90.0

    for row in rows[1:]:
        used_cols: set[int] = set()
        for line in lines:
            last_r, last_c = line[-1]
            if last_r - row - 1 > max_gap:
                continue

            use_lock = angle_lock_n > 0 and len(line) >= angle_lock_n
            if use_lock:
                avg_c = sum(p[1] for p in line[:angle_lock_n]) / angle_lock_n
                candidates = [
                    c for c in by_row[row]
                    if c not in used_cols
                    and abs(c - avg_c) <= stitch_tol
                    and abs(c - last_c) <= stitch_tol * 2
                ]
            else:
                if angle_mode == "A":
                    ref_r, ref_c = line[0]
                else:
                    recent = line[-min(3, len(line)):]
                    ref_r = sum(p[0] for p in recent) / len(recent)
                    ref_c = sum(p[1] for p in recent) / len(recent)
                candidates = [
                    c for c in by_row[row]
                    if c not in used_cols
                    and abs(c - last_c) <= stitch_tol
                    and _ok_angle(ref_r, ref_c, row, c)
                ]

            if candidates:
                candidates.sort(key=lambda c: (abs(c - last_c), c))
                chosen = candidates[0]
                line.append((row, chosen))
                used_cols.add(chosen)

        # Start new lines for unattached points
        for c in by_row[row]:
            if c not in used_cols:
                lines.append([(row, c)])

    return lines


# ---------------------------------------------------------------------------
# Stage 3 post-stitch merge
# ---------------------------------------------------------------------------

def _line_angle(line: _Line) -> float:
    if len(line) < 2:
        return 0.0
    dy = float(line[0][0] - line[-1][0])
    dx = float(line[-1][1] - line[0][1])
    if dy == 0:
        return 0.0
    return float(np.degrees(np.arctan2(dx, dy)))


def _merge_segments(
    lines: list[_Line],
    merge_angle_tol: float,
    merge_row_sep: int,
    merge_col_sep: int,
) -> list[_Line]:
    """Iteratively merge pairs of nearly-parallel, spatially close segments."""
    if not lines:
        return lines
    changed = True
    while changed:
        changed = False
        n = len(lines)
        if n < 2:
            break
        angles = [_line_angle(l) for l in lines]
        absorbed = [False] * n
        new_lines: list[_Line] = []
        for i in range(n):
            if absorbed[i]:
                continue
            top_r_i, top_c_i = lines[i][-1]
            best_j = -1
            best_score = (float("inf"), float("inf"))
            for j in range(n):
                if j == i or absorbed[j]:
                    continue
                bot_r_j, bot_c_j = lines[j][0]
                row_gap = top_r_i - bot_r_j - 1
                if row_gap < 0 or row_gap > merge_row_sep:
                    continue
                col_dist = abs(top_c_i - bot_c_j)
                if col_dist > merge_col_sep:
                    continue
                if merge_angle_tol > 0 and abs(angles[i] - angles[j]) > merge_angle_tol:
                    continue
                score = (row_gap, col_dist)
                if score < best_score:
                    best_score = score
                    best_j = j
            if best_j >= 0:
                new_lines.append(lines[i] + lines[best_j])
                absorbed[i] = True
                absorbed[best_j] = True
                changed = True
        for i in range(n):
            if not absorbed[i]:
                new_lines.append(lines[i])
        lines = new_lines
    return lines


# ---------------------------------------------------------------------------
# Stage 5 — endpoint-angle stitch
# ---------------------------------------------------------------------------

def _endpoint_angle(line: _Line, from_bottom: bool, n_pts: int) -> float:
    pts = line[:n_pts] if from_bottom else line[-n_pts:]
    if len(pts) < 2:
        return 0.0
    r0, c0 = pts[0]
    r1, c1 = pts[-1]
    dy = float(abs(r0 - r1))
    dx = float(c1 - c0)
    if dy == 0:
        return 90.0
    return float(np.degrees(np.arctan2(abs(dx), dy)))


def _stage5_stitch(
    lines: list[_Line],
    s5_vgap: int,
    s5_hgap: int,
    s5_angle: float,
    s5_pts: int,
) -> list[_Line]:
    """Endpoint-angle stitching (Stage 5)."""
    if not lines:
        return lines
    lines = [list(seg) for seg in lines]
    changed = True
    while changed:
        changed = False
        n = len(lines)
        if n < 2:
            break
        absorbed = [False] * n
        new_lines: list[_Line] = []
        for i in range(n):
            if absorbed[i]:
                continue
            top_r_i, top_c_i = lines[i][-1]
            ang_i = _endpoint_angle(lines[i], from_bottom=False, n_pts=s5_pts)
            best_j = -1
            best_score = (float("inf"), float("inf"))
            for j in range(n):
                if j == i or absorbed[j]:
                    continue
                bot_r_j, bot_c_j = lines[j][0]
                row_gap = top_r_i - bot_r_j - 1
                if row_gap < 0 or row_gap > s5_vgap:
                    continue
                col_dist = abs(top_c_i - bot_c_j)
                if col_dist > s5_hgap:
                    continue
                if s5_angle > 0:
                    ang_j = _endpoint_angle(lines[j], from_bottom=True, n_pts=s5_pts)
                    if abs(ang_i - ang_j) > s5_angle:
                        continue
                score = (row_gap, col_dist)
                if score < best_score:
                    best_score = score
                    best_j = j
            if best_j >= 0:
                new_lines.append(lines[i] + lines[best_j])
                absorbed[i] = True
                absorbed[best_j] = True
                changed = True
        for i in range(n):
            if not absorbed[i]:
                new_lines.append(lines[i])
        lines = new_lines
    return lines


# ---------------------------------------------------------------------------
# Signal computation
# ---------------------------------------------------------------------------

def _vertical_span(line: _Line) -> int:
    if len(line) < 2:
        return 0
    rows = [p[0] for p in line]
    return max(rows) - min(rows) + 1


def _compute_presence(lines: list[_Line], height: int) -> np.ndarray:
    """
    Per-row count of lines whose row range spans that row (Stage-5 fill semantics:
    gap rows are counted as present, matching _update_el_graphs Stage-5 logic).
    """
    presence = np.zeros(height, dtype=np.int32)
    for line in lines:
        if not line:
            continue
        rows = [p[0] for p in line]
        for r in range(min(rows), max(rows) + 1):
            if 0 <= r < height:
                presence[r] += 1
    return presence


def _compute_terminations(
    lines: list[_Line], height: int, term_window: int
) -> np.ndarray:
    """
    Per-row count of line endpoints within ±term_window rows.
    Each line contributes its top AND bottom endpoints.
    """
    terminations = np.zeros(height, dtype=np.int32)
    for line in lines:
        if not line:
            continue
        rows = [p[0] for p in line]
        for endpoint_row in (min(rows), max(rows)):
            lo = max(0, endpoint_row - term_window)
            hi = min(height - 1, endpoint_row + term_window)
            terminations[lo: hi + 1] += 1
    return terminations
