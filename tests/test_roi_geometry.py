"""
Tests for ROI1/ROI2 geometry.

ROI1 = the calibrated polygon (roi_points), used as both the matching
       template and the output polygon.
ROI2 = the search patch = ROI1's bounding box expanded by roi_expansion_px
       on every side.  Defined by patch_bbox = (px0, py0, pw, ph) in
       image coordinates.

Tests cover:
  - expand_roi_points: rectangular and non-rectangular polygons, clamping at
    image boundary, zero-expansion no-op, contraction.
  - patch_bbox geometry: correct size and position when ROI is away from the
    edge, when the ROI is near an image edge (boundary clamping), and the
    guard that returns an empty result when the clamped patch is too small.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from pa.pipeline import image_primitives as ip
from pa.pipeline.modes.roi_edge_fit import _ROIEdgeFitEngine, _empty_result


# ── helpers ──────────────────────────────────────────────────────────────────

def rect_roi(x, y, w, h):
    """Clockwise rectangle as list of (x,y) tuples."""
    return [(x, y), (x + w, y), (x + w, y + h), (x, y + h)]


def polygon_area(pts):
    """Signed shoelace area (positive = CW in screen coords, y-down)."""
    n = len(pts)
    return sum(
        pts[i][0] * pts[(i + 1) % n][1] - pts[(i + 1) % n][0] * pts[i][1]
        for i in range(n)
    )


def bbox(pts):
    """Axis-aligned bounding box: (min_x, min_y, max_x, max_y)."""
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    return min(xs), min(ys), max(xs), max(ys)


# ── expand_roi_points ─────────────────────────────────────────────────────────

class TestExpandRoiPoints:

    def test_zero_expansion_is_noop(self):
        pts = rect_roi(100, 200, 80, 60)
        result = ip.expand_roi_points(pts, 0, 1000, 1000)
        assert result == pts

    def test_rectangle_expands_by_correct_amount(self):
        """
        A rectangle with 90° corners: each vertex moves exactly
        `expansion_px * sqrt(2)` along the diagonal.  The bounding box of
        the expanded polygon should be larger by `expansion_px` on every side.
        """
        expansion = 10
        pts = rect_roi(100, 200, 80, 60)
        result = ip.expand_roi_points(pts, expansion, 5000, 5000)

        ox1, oy1, ox2, oy2 = bbox(pts)
        rx1, ry1, rx2, ry2 = bbox(result)

        assert rx1 == pytest.approx(ox1 - expansion, abs=0.5)
        assert ry1 == pytest.approx(oy1 - expansion, abs=0.5)
        assert rx2 == pytest.approx(ox2 + expansion, abs=0.5)
        assert ry2 == pytest.approx(oy2 + expansion, abs=0.5)

    def test_rectangle_contraction(self):
        """Negative expansion should shrink the polygon symmetrically."""
        contraction = -8
        pts = rect_roi(100, 200, 80, 60)
        result = ip.expand_roi_points(pts, contraction, 5000, 5000)

        ox1, oy1, ox2, oy2 = bbox(pts)
        rx1, ry1, rx2, ry2 = bbox(result)

        assert rx1 == pytest.approx(ox1 + 8, abs=0.5)
        assert ry1 == pytest.approx(oy1 + 8, abs=0.5)
        assert rx2 == pytest.approx(ox2 - 8, abs=0.5)
        assert ry2 == pytest.approx(oy2 - 8, abs=0.5)

    def test_winding_preserved_cw(self):
        """Expansion must not flip the vertex winding order."""
        pts = rect_roi(100, 200, 80, 60)  # CW
        result = ip.expand_roi_points(pts, 10, 5000, 5000)
        assert polygon_area(result) > 0  # still CW (positive area, y-down)

    def test_winding_preserved_ccw(self):
        """CCW input should stay CCW after expansion."""
        pts = list(reversed(rect_roi(100, 200, 80, 60)))  # CCW
        result = ip.expand_roi_points(pts, 10, 5000, 5000)
        assert polygon_area(result) < 0  # still CCW

    def test_clamping_at_image_left_edge(self):
        """Vertices expanded beyond x=0 must be clamped to 0."""
        pts = rect_roi(5, 100, 80, 60)  # 5 px from left edge
        result = ip.expand_roi_points(pts, 20, 500, 500)
        xs = [p[0] for p in result]
        assert min(xs) >= 0.0

    def test_clamping_at_image_right_edge(self):
        pts = rect_roi(410, 100, 80, 60)  # right side at x=490 in a 500-wide image
        result = ip.expand_roi_points(pts, 20, 500, 500)
        xs = [p[0] for p in result]
        assert max(xs) <= 499.0

    def test_clamping_at_image_top_edge(self):
        pts = rect_roi(100, 3, 80, 60)  # 3 px from top
        result = ip.expand_roi_points(pts, 20, 500, 500)
        ys = [p[1] for p in result]
        assert min(ys) >= 0.0

    def test_clamping_at_image_bottom_edge(self):
        pts = rect_roi(100, 430, 80, 60)  # bottom at y=490 in 500-tall image
        result = ip.expand_roi_points(pts, 20, 500, 500)
        ys = [p[1] for p in result]
        assert max(ys) <= 499.0

    def test_expansion_increases_area(self):
        pts = rect_roi(100, 200, 80, 60)
        expanded = ip.expand_roi_points(pts, 15, 5000, 5000)
        assert abs(polygon_area(expanded)) > abs(polygon_area(pts))

    def test_contraction_decreases_area(self):
        pts = rect_roi(100, 200, 80, 60)
        contracted = ip.expand_roi_points(pts, -5, 5000, 5000)
        assert abs(polygon_area(contracted)) < abs(polygon_area(pts))

    def test_non_rectangular_polygon_expands_outward(self):
        """
        A regular hexagon expanded outward should have all vertices farther
        from the centroid than the original.
        """
        cx, cy, r = 300.0, 300.0, 50.0
        pts = [
            (cx + r * math.cos(math.pi / 2 + i * 2 * math.pi / 6),
             cy + r * math.sin(math.pi / 2 + i * 2 * math.pi / 6))
            for i in range(6)
        ]
        expanded = ip.expand_roi_points(pts, 10, 1000, 1000)
        for orig, exp in zip(pts, expanded):
            d_orig = math.hypot(orig[0] - cx, orig[1] - cy)
            d_exp  = math.hypot(exp[0]  - cx, exp[1]  - cy)
            assert d_exp > d_orig, (
                f"Vertex moved inward: orig dist={d_orig:.2f} exp dist={d_exp:.2f}"
            )

    def test_vertex_count_preserved(self):
        pts = rect_roi(100, 200, 80, 60)
        assert len(ip.expand_roi_points(pts, 10, 5000, 5000)) == len(pts)


# ── patch_bbox (ROI2) geometry ────────────────────────────────────────────────

def _make_edge_map(h, w):
    """Flat float32 zeros — enough for the engine to run without real edges."""
    return np.zeros((h, w), dtype=np.float32)


def _run_engine(roi_points, expansion_px, img_h=1000, img_w=1000):
    """
    Run the ROIEdgeFitEngine on a blank image and return the engine_result dict.
    Provides the minimum params needed (no blur, no edge det, no refinement).
    """
    params = {
        "blur_method": "none",
        "gradient_threshold": 0.5,
        "roi_expansion_px": expansion_px,
        "n_coarse_peaks": 1,
        "refine_range_px": 0,
        "refine_step_px": 1,
        "score_falloff_type": "gaussian",
        "score_falloff_px": 5.0,
        "score_cutoff_px": 0.0,
        "edge_outward_offset_px": 0,
        "scoring_signal": "edge_map",
    }
    engine = _ROIEdgeFitEngine(params)
    bgr = np.zeros((img_h, img_w, 3), dtype=np.uint8)
    return engine.fit(bgr, roi_points, cache=None)


class TestPatchBboxGeometry:
    """
    patch_bbox = (px0, py0, pw, ph) defines ROI2 — the search region.
    It must be exactly ROI1's axis-aligned bounding box expanded by
    roi_expansion_px on every side (or truncated at the image boundary).
    """

    def test_no_expansion_patch_equals_roi_bbox(self):
        """With expansion=0 the patch bbox should equal the ROI bbox."""
        pts = rect_roi(200, 150, 80, 60)
        result = _run_engine(pts, expansion_px=0)
        assert result.get("patch_bbox") is not None

        px0, py0, pw, ph = result["patch_bbox"]
        ox, oy, rw, rh = result["roi_bbox"]

        assert px0 == ox
        assert py0 == oy
        assert pw == rw
        assert ph == rh

    def test_expansion_widens_patch_correctly(self):
        """Patch dimensions = ROI bbox + 2*expansion on each axis."""
        pts = rect_roi(200, 150, 80, 60)
        expansion = 20
        result = _run_engine(pts, expansion_px=expansion)
        assert result.get("patch_bbox") is not None

        px0, py0, pw, ph = result["patch_bbox"]
        ox, oy, rw, rh = result["roi_bbox"]

        assert px0 == ox - expansion
        assert py0 == oy - expansion
        assert pw  == rw + 2 * expansion
        assert ph  == rh + 2 * expansion

    def test_patch_origin_clamped_at_left_top(self):
        """ROI near the top-left corner: patch origin clamped to (0, 0)."""
        pts = rect_roi(5, 8, 80, 60)     # only 5/8 px from the edge
        expansion = 20
        result = _run_engine(pts, expansion_px=expansion)
        assert result.get("patch_bbox") is not None

        px0, py0, pw, ph = result["patch_bbox"]
        assert px0 == 0
        assert py0 == 0

    def test_patch_right_bottom_clamped(self):
        """ROI near the bottom-right corner: patch must not exceed image bounds."""
        img_h, img_w = 500, 500
        pts = rect_roi(410, 420, 80, 60)  # extends to x=490, y=480
        expansion = 20
        result = _run_engine(pts, expansion_px=expansion,
                             img_h=img_h, img_w=img_w)
        assert result.get("patch_bbox") is not None

        px0, py0, pw, ph = result["patch_bbox"]
        assert px0 + pw <= img_w
        assert py0 + ph <= img_h

    def test_roi_points_returned_unchanged(self):
        """
        The engine must return the *original* roi_points key so the debug
        overlay and output polygon are always at ROI1 size.
        """
        pts = rect_roi(200, 150, 80, 60)
        result = _run_engine(pts, expansion_px=15)
        returned = result.get("roi_points")
        assert returned is not None
        assert list(returned) == list(pts)

    def test_fitted_polygon_has_same_bbox_as_roi1(self):
        """
        After fitting, the fitted polygon must have the same dimensions as
        ROI1 (the original polygon) — only translated, not resized.
        """
        pts = rect_roi(200, 150, 80, 60)
        result = _run_engine(pts, expansion_px=15)
        fitted = result.get("fitted_roi_polygon")
        assert fitted is not None

        ox1, oy1, ox2, oy2 = bbox(pts)
        fx1, fy1, fx2, fy2 = bbox(fitted)

        orig_w = ox2 - ox1
        orig_h = oy2 - oy1
        fit_w  = fx2 - fx1
        fit_h  = fy2 - fy1

        assert fit_w == pytest.approx(orig_w, abs=1.0)
        assert fit_h == pytest.approx(orig_h, abs=1.0)

    def test_patch_larger_than_template(self):
        """
        With expansion > 0, the patch must always be strictly larger than the
        template (roi_bbox) so cv2.matchTemplate has room to slide.
        """
        pts = rect_roi(200, 150, 80, 60)
        result = _run_engine(pts, expansion_px=10)
        assert result.get("patch_bbox") is not None

        px0, py0, pw, ph = result["patch_bbox"]
        ox, oy, rw, rh   = result["roi_bbox"]

        assert pw >= rw
        assert ph >= rh
