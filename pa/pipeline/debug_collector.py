"""
Debug stage collector for the PA pipeline.

Given the ProcessedImageCache (populated after a normal pipeline run) and
the intermediate data objects (RawFeatures, ZProfile, TipRegion, engine
result dict), this module assembles a list of DebugStage objects that the
inspector GUI can display.

This is deliberately separate from the extractors/modes so that:
  - Extractor/mode signatures remain unchanged.
  - Debug cost is zero when debug=False (nothing is called).
  - The collection code has no impact on correctness of the analysis.

Called only by pa.pipeline.runner when debug=True.
"""

from __future__ import annotations

import cv2
import numpy as np
from typing import Any, Dict, List, Optional

from pa.pipeline.types import (
    DebugStage, DebugData, ImageSet,
    ZProfile, TipRegion, RawFeatures,
    PointOfInterest, TipCandidate,
)
from pa.pipeline.cache import ProcessedImageCache
from pa.pipeline import image_primitives as ip_module


# ---------------------------------------------------------------------------
# Shared pre-processing stage (blur)
# ---------------------------------------------------------------------------

def collect_blur_stage(
    pipette_index: int,
    mode_name: str,
    cache: ProcessedImageCache,
    blur_params: dict,
    source_image: np.ndarray,
) -> Optional[DebugStage]:
    """Return a DebugStage for the blurred image, if it is in the cache."""
    blurred = cache.get(pipette_index, "blur", blur_params)
    if blurred is None:
        return None
    img8 = _to_display(blurred)
    return DebugStage(
        name="Blur",
        mode_name=mode_name,
        pipette_index=pipette_index,
        image=img8,
        description=f"Gaussian blur kernel={blur_params.get('kernel_size', '?')}",
        metadata={"blur_method": blur_params.get("method", "gaussian"),
                  "kernel_size": blur_params.get("kernel_size")},
    )


def collect_ab_stage(
    pipette_index: int,
    mode_name: str,
    cache: ProcessedImageCache,
) -> Optional[DebugStage]:
    """Return a DebugStage for the A:B accumulated image, if present."""
    ab = cache.get(pipette_index, "accumulate_ab", {})
    if ab is None:
        return None
    return DebugStage(
        name="A:B Accumulation",
        mode_name=mode_name,
        pipette_index=pipette_index,
        image=_to_display(ab),
        description="Accumulated frame difference (A–B). Bright = change.",
    )


# ---------------------------------------------------------------------------
# Intensity extractor stages
# ---------------------------------------------------------------------------

def collect_intensity_stages(
    pipette_index: int,
    profile: ZProfile,
    poi_list: List[PointOfInterest],
    cache: ProcessedImageCache,
    params: dict,
    roi: Optional[tuple] = None,
    roi_points: Optional[List[tuple]] = None,
    image_size: Optional[tuple] = None,
    mode_name: str = "IntensityDetection",
    features=None,
) -> List[DebugStage]:
    stages = []

    # --- 1. Compute effective ROI (original or expanded) first so image
    #        stages can be cropped to it and share consistent metadata. ---
    expansion = int(params.get("roi_expansion_px", 0) or 0)
    expanded_points: Optional[List[tuple]] = None
    expanded_bbox: Optional[tuple] = None
    if expansion != 0 and roi_points and image_size is not None:
        img_h, img_w = image_size[0], image_size[1]
        expanded_points = ip_module.expand_roi_points(
            roi_points, expansion, img_h, img_w,
        )
        xs = [p[0] for p in expanded_points]
        ys = [p[1] for p in expanded_points]
        x0 = max(0, int(min(xs)))
        y0 = max(0, int(min(ys)))
        x1 = min(img_w, int(max(xs)) + 1)
        y1 = min(img_h, int(max(ys)) + 1)
        expanded_bbox = (x0, y0, max(1, x1 - x0), max(1, y1 - y0))
    effective_bbox = expanded_bbox if expanded_bbox is not None else roi

    # Helper: crop a full-frame image array to the effective ROI bbox.
    def _crop(arr: np.ndarray) -> np.ndarray:
        if arr is None or effective_bbox is None:
            return arr
        return ip_module.extract_roi(arr, effective_bbox)

    # Shared metadata for image stages (display crop + ROI polygon overlays).
    base_img_meta: dict = {}
    if effective_bbox is not None:
        base_img_meta["display_bbox"] = list(effective_bbox)
    if roi_points:
        base_img_meta["roi1_points"] = [list(p) for p in roi_points]
    if expanded_points:
        base_img_meta["roi3_points"] = [list(p) for p in expanded_points]

    # --- 2. A:B accumulation stage (full-frame; no crop applied here) ---
    if params.get("use_ab", True):
        s2 = collect_ab_stage(pipette_index, mode_name, cache)
        if s2:
            stages.append(s2)

    # --- 3. Contrast Frame (RowContrastDetection, use_contrast=True, no A:B) ---
    if (mode_name == "RowContrastDetection"
            and params.get("use_contrast", False)
            and not params.get("use_ab", False)):
        contrast_img = cache.get(pipette_index, "contrast_frame", {})
        if contrast_img is not None:
            stages.append(DebugStage(
                name="Contrast Frame",
                mode_name=mode_name,
                pipette_index=pipette_index,
                image=_crop(_to_display(contrast_img)),
                description=(
                    f"Input frame after reference subtraction "
                    f"({params.get('contrast_mode', 'absolute')} mode, "
                    f"min={params.get('contrast_min', 0)}, "
                    f"max={params.get('contrast_max', 100)})."
                ),
                metadata={
                    **base_img_meta,
                    "contrast_mode": params.get("contrast_mode", "absolute"),
                    "contrast_min": params.get("contrast_min", 0),
                    "contrast_max": params.get("contrast_max", 100),
                },
            ))

    # --- 4. Blur stage.
    #        Use the FULL blur params dict (5 keys) to match the key under
    #        which the extractor stored the result in the cache.  The old
    #        2-key lookup always missed, so this stage never appeared. ---
    blur_p_full = {
        "method":      params.get("blur_method", "gaussian"),
        "kernel_size": params.get("blur_kernel", 5),
        "d":           params.get("bilateral_d", 9),
        "sigma_color": params.get("bilateral_sigma_color", 75.0),
        "sigma_space": params.get("bilateral_sigma_space", 75.0),
    }
    blurred = cache.get(pipette_index, "blur", blur_p_full)
    if blurred is not None:
        stages.append(DebugStage(
            name="Blur",
            mode_name=mode_name,
            pipette_index=pipette_index,
            image=_crop(_to_display(blurred)),
            description=(
                f"{blur_p_full['method']} blur, "
                f"kernel={blur_p_full['kernel_size']}"
            ),
            metadata={**base_img_meta},
        ))

    # --- 5. Gradient stage (RowContrastDetection only, when method != 'none') ---
    if mode_name == "RowContrastDetection":
        grad_method = params.get("gradient_method", "none")
        if grad_method != "none":
            grad_ksize = params.get("gradient_ksize", 3)
            grad_img = cache.get(pipette_index, "gradient",
                                  {"method": grad_method, "ksize": grad_ksize})
            if grad_img is not None:
                stages.append(DebugStage(
                    name="Gradient",
                    mode_name=mode_name,
                    pipette_index=pipette_index,
                    image=_crop(_to_display(grad_img)),
                    description=(
                        f"Sobel gradient magnitude ({grad_method}, "
                        f"ksize={grad_ksize}) applied after blur."
                    ),
                    metadata={
                        **base_img_meta,
                        "gradient_method": grad_method,
                        "gradient_ksize": grad_ksize,
                    },
                ))

    # --- 6. Signal stage ---
    poi_z = [p.z_px for p in poi_list]
    meta: dict = {
        "threshold": params.get("intensity_threshold", 0.3),
        "candidates": len(poi_z),
    }
    if effective_bbox is not None:
        meta["roi"] = list(effective_bbox)
        meta["display_bbox"] = list(effective_bbox)
    if roi_points:
        meta["roi1_points"] = [list(p) for p in roi_points]
    if expanded_points:
        meta["roi3_points"] = [list(p) for p in expanded_points]
    stages.append(DebugStage(
        name="Intensity Signal" if mode_name == "IntensityDetection" else "Row Contrast Signal",
        mode_name=mode_name,
        pipette_index=pipette_index,
        signal=profile.signal,
        z_axis_px=profile.z_axis_px,
        poi_z_px=poi_z,
        extra_signals=getattr(features, "band_signals", None),
        description="Per-row mean intensity (normalised). Peaks = candidate transitions."
                    if mode_name == "IntensityDetection" else
                    "Per-row cross-sectional std (normalised). Peaks = high-contrast transitions.",
        metadata=meta,
    ))
    return stages


# ---------------------------------------------------------------------------
# Ridge extractor stages (LineContinuity family)
# ---------------------------------------------------------------------------

def collect_ridge_stages(
    pipette_index: int,
    mode_name: str,
    features: RawFeatures,
    profile: ZProfile,
    poi_list: List[PointOfInterest],
    cache: ProcessedImageCache,
    params: dict,
    roi: Optional[tuple] = None,
    roi_points: Optional[List[tuple]] = None,
    image_size: Optional[tuple] = None,
) -> List[DebugStage]:
    stages = []

    # ── Compute effective ROI after any expansion ────────────────────────────
    # NOTE: RidgeExtractor crops gray8 to the ROI *before* blurring, so the
    # blur image in cache and edge_map are both already in ROI-local coords.
    # Do NOT apply a second crop; just use the images as-is.  The display_bbox
    # / roi1_points / roi3_points metadata is still required so the AB dialog
    # knows where to crop the *source overlay* and draw ROI polygons.
    expansion = int(params.get("roi_expansion_px", 0) or 0)
    expanded_points: Optional[List[tuple]] = None
    expanded_bbox: Optional[tuple] = None
    if expansion != 0 and roi_points and image_size is not None:
        img_h, img_w = image_size[0], image_size[1]
        expanded_points = ip_module.expand_roi_points(roi_points, expansion, img_h, img_w)
        xs = [p[0] for p in expanded_points]
        ys = [p[1] for p in expanded_points]
        x0 = max(0, int(min(xs)))
        y0 = max(0, int(min(ys)))
        x1 = min(img_w, int(max(xs)) + 1)
        y1 = min(img_h, int(max(ys)) + 1)
        expanded_bbox = (x0, y0, max(1, x1 - x0), max(1, y1 - y0))
    effective_bbox = expanded_bbox if expanded_bbox is not None else roi

    # Shared metadata for image stages
    base_img_meta: dict = {}
    if effective_bbox is not None:
        base_img_meta["display_bbox"] = list(effective_bbox)
    if roi_points:
        base_img_meta["roi1_points"] = [list(p) for p in roi_points]
    if expanded_points:
        base_img_meta["roi3_points"] = [list(p) for p in expanded_points]

    # ── Blur stage (5-key cache lookup to match what RidgeExtractor stores) ──
    blur_p_full = {
        "method":      params.get("blur_method", "gaussian"),
        "kernel_size": params.get("blur_kernel", 5),
        "d":           params.get("bilateral_d", 9),
        "sigma_color": params.get("bilateral_sigma_color", 75.0),
        "sigma_space": params.get("bilateral_sigma_space", 75.0),
    }
    blurred = cache.get(pipette_index, "blur", blur_p_full)
    if blurred is not None:
        stages.append(DebugStage(
            name="Blur",
            mode_name=mode_name,
            pipette_index=pipette_index,
            image=_to_display(blurred),  # already in ROI-local coords, no re-crop
            description=(
                f"{blur_p_full['method']} blur, "
                f"kernel={blur_p_full['kernel_size']}"
            ),
            metadata={
                **base_img_meta,
                "blur_method":  blur_p_full["method"],
                "kernel_size":  blur_p_full["kernel_size"],
            },
        ))

    # ── Ridge Mask stage ─────────────────────────────────────────────────────
    if features.edge_map is not None:
        stages.append(DebugStage(
            name="Ridge Mask",
            mode_name=mode_name,
            pipette_index=pipette_index,
            image=features.edge_map,  # already in ROI-local coords, no re-crop
            description=(
                f"Binary ridge mask after orientation filter "
                f"(±{params.get('angle_tolerance', 30)}°) and morphology."
            ),
            metadata={
                **base_img_meta,
                "ridge_threshold": params.get("ridge_threshold", 30),
                "morph_close":     params.get("morph_close_enabled", True),
                "morph_open":      params.get("morph_open_enabled", True),
            },
        ))

    # 1D signal
    poi_z = [p.z_px for p in poi_list]
    _signal_labels = {
        "LineContinuity_Terminations": ("Termination Signal",
            "Per-row ridge termination count. Peaks = structural change."),
        "LineContinuity_PatternChange": ("Pattern Change Signal",
            "Per-row ridge spacing change. Peaks = pattern discontinuity."),
        "LineContinuity_Correlation": ("Correlation Signal",
            "Per-row cross-band correlation. Troughs = structural gap."),
        "LineContinuity_Density": ("Density Signal",
            "Per-row ridge density. Peaks = density anomaly."),
    }
    label, desc = _signal_labels.get(
        mode_name, (f"{mode_name} Signal", "Mode 1-D output signal."))
    stages.append(DebugStage(
        name=label,
        mode_name=mode_name,
        pipette_index=pipette_index,
        signal=profile.signal,
        z_axis_px=profile.z_axis_px,
        poi_z_px=poi_z,
        description=desc,
        metadata={"threshold": params.get("threshold", 0.3),
                  "candidates": len(poi_z)},
    ))
    return stages


# ---------------------------------------------------------------------------
# ExtremaLines stages
# ---------------------------------------------------------------------------

def collect_extrema_stages(
    pipette_index: int,
    features: RawFeatures,
    density_profile: ZProfile,
    term_profile: ZProfile,
    density_pois: List[PointOfInterest],
    term_pois: List[PointOfInterest],
    cache: ProcessedImageCache,
    params: dict,
) -> List[DebugStage]:
    stages = []
    blur_p = {"method": "gaussian", "kernel_size": params.get("blur_kernel", 5)}
    s = collect_blur_stage(pipette_index, "ExtremaLines", cache, blur_p, None)
    if s:
        stages.append(s)

    if features.edge_map is not None:
        stages.append(DebugStage(
            name="Extrema Points",
            mode_name="ExtremaLines",
            pipette_index=pipette_index,
            image=features.edge_map,
            description="Per-row local minima and maxima marked. Stitched into lines.",
            metadata={"min_line_length": params.get("min_line_length", 5),
                      "stitch_gap": params.get("stitch_gap", 3)},
        ))

    for profile, pois, label, desc in [
        (density_profile, density_pois,
         "Density Signal (detrended)",
         "Line density detrended residual. Troughs = liquid transition."),
        (term_profile, term_pois,
         "Termination Signal (detrended)",
         "Line termination detrended residual. Peaks = liquid transition."),
    ]:
        stages.append(DebugStage(
            name=label,
            mode_name="ExtremaLines",
            pipette_index=pipette_index,
            signal=profile.signal,
            z_axis_px=profile.z_axis_px,
            poi_z_px=[p.z_px for p in pois],
            description=desc,
            metadata={"detrend_method": params.get("detrend_method", "linear"),
                      "candidates": len(pois)},
        ))
    return stages


# ---------------------------------------------------------------------------
# InwardEdgeScan stages
# ---------------------------------------------------------------------------

def collect_inward_edge_stages(
    pipette_index: int,
    engine_result: Dict[str, Any],
    cache: ProcessedImageCache,
    params: dict,
    candidates: Optional[List["TipCandidate"]] = None,
) -> List[DebugStage]:
    stages = []
    blur_p = {"method": params.get("blur_method", "gaussian"),
               "kernel_size": params.get("blur_kernel", 5)}
    s = collect_blur_stage(pipette_index, "InwardEdgeScan", cache, blur_p, None)
    if s:
        stages.append(s)

    # Gradient magnitude
    gm = engine_result.get("gradient_maps") if engine_result else None
    if gm and gm.get("gradient_mag") is not None:
        gm_img = _normalise_to_uint8(gm["gradient_mag"])
        stages.append(DebugStage(
            name="Gradient Magnitude",
            mode_name="InwardEdgeScan",
            pipette_index=pipette_index,
            image=gm_img,
            description="Sobel gradient magnitude. Bright = strong edge.",
            metadata={"gradient_threshold": params.get("gradient_threshold", 30)},
        ))

    # Edge candidate overlay
    if engine_result:
        overlay = _draw_edge_overlay(engine_result, params)
        if overlay is not None:
            stages.append(DebugStage(
                name="Edge Candidates",
                mode_name="InwardEdgeScan",
                pipette_index=pipette_index,
                image=overlay,
                description="Detected edge points: left=green, right=blue, bottom=red.",
                metadata={
                    "confirmation_n": params.get("confirmation_n", 3),
                    "confirmation_m": params.get("confirmation_m", 5),
                    "quality_score":  engine_result.get("quality_score", 0.0),
                    "quality_rating": engine_result.get("quality_rating", ""),
                },
            ))

    # Interpreter result — tip detection bounding box(es)
    if candidates is not None and engine_result:
        result_img = _draw_tip_candidates_overlay(engine_result, candidates, params)
        if result_img is not None:
            cand_meta: Dict[str, Any] = {
                "n_candidates": len(candidates),
                "min_confidence": params.get("min_confidence", 0.3),
                "quality_score": engine_result.get("quality_score", 0.0),
                "quality_rating": engine_result.get("quality_rating", ""),
            }
            for i, c in enumerate(candidates):
                cand_meta[f"c{i}_confidence"] = round(float(c.confidence), 3)
                if c.region is not None:
                    x, y, w, h = c.region
                    cand_meta[f"c{i}_region"] = f"({x:.0f}, {y:.0f}, {w:.0f}×{h:.0f})"
            stages.append(DebugStage(
                name="Tip Detection Result",
                mode_name="InwardEdgeScan",
                pipette_index=pipette_index,
                image=result_img,
                description=(
                    "Interpreter output: detected tip bounding box (cyan) on edge overlay. "
                    "Edge colours: left=green, right=blue, bottom=red."
                    if candidates else
                    "No tip candidates survived the interpreter (confidence threshold not met)."
                ),
                metadata=cand_meta,
            ))

    return stages


def _draw_edge_overlay(result: Dict[str, Any], params: dict) -> Optional[np.ndarray]:
    """Draw left/right/bottom edge points on a black canvas the size of the ROI."""
    # Determine canvas size from all known points
    all_pts: List[tuple] = []
    for key in ("left_edge", "right_edge", "bottom_edge"):
        e = result.get(key) or {}
        all_pts.extend(e.get("points", []))
    roi_offset = result.get("roi_offset", (0, 0))
    roi_pts = result.get("roi") or []

    if not all_pts and not roi_pts:
        return None

    if roi_pts:
        xs = [p[0] for p in roi_pts]
        ys = [p[1] for p in roi_pts]
        w = max(1, int(max(xs)) - int(min(xs)) + 2)
        h = max(1, int(max(ys)) - int(min(ys)) + 2)
        ox, oy = int(min(xs)), int(min(ys))
    else:
        xs = [p[0] for p in all_pts]
        ys = [p[1] for p in all_pts]
        w = max(1, int(max(xs)) - int(min(xs)) + 20)
        h = max(1, int(max(ys)) - int(min(ys)) + 20)
        ox, oy = max(0, int(min(xs)) - 5), max(0, int(min(ys)) - 5)

    canvas = np.zeros((h, w, 3), dtype=np.uint8)

    colors = {
        "left_edge":   (0, 255, 0),
        "right_edge":  (255, 100, 0),
        "bottom_edge": (0, 0, 255),
    }
    for key, color in colors.items():
        e = result.get(key) or {}
        for pt in e.get("points", []):
            cx = int(pt[0]) - ox
            cy = int(pt[1]) - oy
            if 0 <= cx < w and 0 <= cy < h:
                cv2.circle(canvas, (cx, cy), 2, color, -1)

    return canvas


def _draw_tip_candidates_overlay(
    result: Dict[str, Any],
    candidates: List["TipCandidate"],
    params: dict,
) -> Optional[np.ndarray]:
    """
    Edge candidate overlay with each interpreter-accepted candidate's
    bounding box drawn on top in cyan.  If the candidate carries a
    ``fitted_roi_polygon`` in its metadata, the polygon outline is drawn
    instead of a rectangle.  If no candidates survived the interpreter,
    a dim "No detection" label is rendered instead.
    """
    base = _draw_edge_overlay(result, params)

    # Derive origin so we can map global candidate coords → canvas coords
    roi_pts = result.get("roi") or []
    all_edge_pts: List[tuple] = []
    for key in ("left_edge", "right_edge", "bottom_edge"):
        all_edge_pts.extend((result.get(key) or {}).get("points", []))

    if roi_pts:
        xs = [p[0] for p in roi_pts]
        ys = [p[1] for p in roi_pts]
        ox, oy = int(min(xs)), int(min(ys))
        cw = max(1, int(max(xs)) - ox + 2)
        ch = max(1, int(max(ys)) - oy + 2)
    elif all_edge_pts:
        xs = [p[0] for p in all_edge_pts]
        ys = [p[1] for p in all_edge_pts]
        ox = max(0, int(min(xs)) - 5)
        oy = max(0, int(min(ys)) - 5)
        cw = max(1, int(max(xs)) - int(min(xs)) + 20)
        ch = max(1, int(max(ys)) - int(min(ys)) + 20)
    else:
        return base  # nothing to draw on

    if base is None:
        base = np.zeros((ch, cw, 3), dtype=np.uint8)

    canvas = base.copy()
    h, w = canvas.shape[:2]

    if not candidates:
        cv2.putText(
            canvas, "No detection", (8, 24),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (80, 80, 220), 1, cv2.LINE_AA,
        )
        return canvas

    _CYAN = (220, 220, 0)  # BGR cyan
    for c in candidates:
        # Prefer the fitted polygon outline over a plain rectangle
        poly = c.metadata.get("fitted_roi_polygon") if c.metadata else None
        if poly and len(poly) >= 3:
            local_pts = np.array(
                [[int(p[0]) - ox, int(p[1]) - oy] for p in poly], dtype=np.int32
            )
            cv2.polylines(canvas, [local_pts], isClosed=True, color=_CYAN, thickness=2)
            # Draw center-top crosshair if available
            ct = c.metadata.get("center_top")
            if ct is not None:
                ctx = int(ct[0]) - ox
                cty = int(ct[1]) - oy
                if 0 <= ctx < w and 0 <= cty < h:
                    cv2.drawMarker(
                        canvas, (ctx, cty), _CYAN,
                        cv2.MARKER_CROSS, 12, 2, cv2.LINE_AA,
                    )
        elif c.region is not None:
            rx, ry, rw, rh = c.region
            lx  = max(0, int(rx) - ox)
            ly  = max(0, int(ry) - oy)
            lx2 = min(w - 1, int(rx + rw) - ox)
            ly2 = min(h - 1, int(ry + rh) - oy)
            cv2.rectangle(canvas, (lx, ly), (lx2, ly2), _CYAN, 2)

        label = f"conf={c.confidence:.2f}"
        if c.region is not None:
            lx = max(0, int(c.region[0]) - ox)
            ly = max(0, int(c.region[1]) - oy)
        else:
            lx, ly = 2, 12
        label_y = max(ly - 4, 12)
        cv2.putText(canvas, label, (lx + 2, label_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, _CYAN, 1, cv2.LINE_AA)

    return canvas


# ---------------------------------------------------------------------------
# WindowContrastAlign stages
# ---------------------------------------------------------------------------

def collect_window_contrast_stages(
    pipette_index: int,
    cache: ProcessedImageCache,
    params: dict,
    tip_candidate: Optional[TipCandidate],
) -> List[DebugStage]:
    stages = []
    blur_p = {"method": params.get("blur_method", "gaussian"),
               "kernel_size": params.get("blur_kernel", 5)}
    s = collect_blur_stage(pipette_index, "WindowContrastAlign", cache, blur_p, None)
    if s:
        stages.append(s)

    if tip_candidate:
        y_px, z_px, w_px, h_px = tip_candidate.region
        stages.append(DebugStage(
            name="Refined Boundary",
            mode_name="WindowContrastAlign",
            pipette_index=pipette_index,
            description="Refined tip boundary found by contrast maximisation.",
            metadata={
                "confidence": round(tip_candidate.confidence, 3),
                "region": f"y={y_px:.0f} z={z_px:.0f} w={w_px:.0f} h={h_px:.0f}",
                "scan_range": params.get("scan_range", 20),
                "window_width": params.get("window_width", 10),
            },
        ))
    return stages


# ---------------------------------------------------------------------------
# ROIEdgeFit / ROIContrastFit stages
# ---------------------------------------------------------------------------

def collect_roi_fit_stages(
    pipette_index: int,
    mode_name: str,
    engine_result: Optional[Dict[str, Any]],
    candidates: Optional[List["TipCandidate"]] = None,
) -> List[DebugStage]:
    """
    Debug stages for ROIEdgeFit and ROIContrastFit.

    Stages produced
    ---------------
    1. Gradient Magnitude   — the full gradient map (continuous for
                              ROIContrastFit; thresholded edge map for ROIEdgeFit).
    2. Score Heatmap        — the coarse FFT cross-correlation score map,
                              normalised to 0–255 for display.
    3. Fit Result           — gradient/edge image with the fitted ROI polygon
                              outline (cyan) and center-top crosshair drawn on it.
    """
    stages: List[DebugStage] = []
    if not engine_result:
        return stages

    # All three stages render onto a common canvas = the search patch (= ROI2).
    # ROI1 = calibrated polygon. ROI2 = ROI1 + roi_expansion_px on every side.
    # Using identical dimensions for every stage image guarantees the source
    # overlay crops to the same dimensions (no resize → no stretching).
    patch_bbox = engine_result.get("patch_bbox")  # (px0, py0, pw, ph)
    if patch_bbox is None:
        return stages
    px0, py0, pw, ph = patch_bbox

    roi_bbox = engine_result.get("roi_bbox")        # ROI1 bbox in image coords
    roi_points_orig = engine_result.get("roi_points")
    roi3_points_orig = engine_result.get("roi3_points")
    fitted_polygon = (
        candidates[0].metadata.get("fitted_roi_polygon")
        if candidates and candidates[0].metadata else None
    )

    def _new_patch_canvas() -> np.ndarray:
        return np.zeros((ph, pw, 3), dtype=np.uint8)

    # ── 1. Edge map / gradient magnitude ─────────────────────────────────────
    if mode_name == "ROIEdgeFit":
        raw = engine_result.get("edge_map")
        gmap_name = "Edge Map (thresholded)"
        gmap_desc = "Binary edge map after gradient threshold."
    else:
        raw = engine_result.get("gradient_map")
        gmap_name = "Gradient Magnitude"
        gmap_desc = "Sobel gradient magnitude (not thresholded)."

    if raw is not None:
        raw_h, raw_w = raw.shape[:2]
        cy1 = max(0, py0); cy2 = min(raw_h, py0 + ph)
        cx1 = max(0, px0); cx2 = min(raw_w, px0 + pw)
        crop = raw[cy1:cy2, cx1:cx2]
        crop8 = _normalise_to_uint8(crop)
        crop_bgr = cv2.cvtColor(crop8, cv2.COLOR_GRAY2BGR)
        # Place the crop into the patch-sized canvas at its true offset
        stage_img = _new_patch_canvas()
        py_dst = cy1 - py0
        px_dst = cx1 - px0
        stage_img[py_dst:py_dst + crop_bgr.shape[0],
                  px_dst:px_dst + crop_bgr.shape[1]] = crop_bgr
        stages.append(DebugStage(
            name=gmap_name,
            mode_name=mode_name,
            pipette_index=pipette_index,
            image=stage_img,
            description=(
                gmap_desc + " The full image area = ROI2 (search region). "
                "Toggle ROI1/ROI2 overlays in the viewer."
            ),
            metadata={
                "display_bbox": patch_bbox,
                "roi1_points":  [list(p) for p in roi_points_orig] if roi_points_orig else None,
                "roi3_points":  [list(p) for p in roi3_points_orig] if roi3_points_orig else None,
            },
        ))

    # ── 2. Score heatmap ─────────────────────────────────────────────────────
    score_map = engine_result.get("score_map")
    if score_map is not None and score_map.size > 0:
        best_score = float(engine_result.get("best_score", 0.0))
        norm_score = float(engine_result.get("normalised_score", 0.0))
        best_dx = int(engine_result.get("best_dx", 0))
        best_dy = int(engine_result.get("best_dy", 0))

        heatmap_raw = _normalise_to_uint8(score_map)
        heatmap_color = cv2.applyColorMap(heatmap_raw, cv2.COLORMAP_INFERNO)

        # Embed the heatmap into a patch-sized canvas at its natural offset.
        # score pixel (r, c) corresponds to placing the template top-left at
        # patch coords (r, c). Centre of placement = (r + roi_h/2, c + roi_w/2).
        stage_img = _new_patch_canvas()
        if roi_bbox is not None:
            ox, oy, roi_w, roi_h = roi_bbox
            offset_y = roi_h // 2
            offset_x = roi_w // 2
            sh_sm, sw_sm = heatmap_color.shape[:2]
            y_end = min(ph, offset_y + sh_sm)
            x_end = min(pw, offset_x + sw_sm)
            if y_end > offset_y and x_end > offset_x:
                stage_img[offset_y:y_end, offset_x:x_end] = \
                    heatmap_color[:y_end - offset_y, :x_end - offset_x]

            # Best-position crosshair (in patch coords): top-left of fitted
            # template = (oy + best_dy - py0, ox + best_dx - px0)
            # Centre of fitted template = top-left + (roi_h//2, roi_w//2)
            best_cy = (oy + best_dy - py0) + offset_y
            best_cx = (ox + best_dx - px0) + offset_x
            if 0 <= best_cy < ph and 0 <= best_cx < pw:
                cv2.drawMarker(stage_img, (best_cx, best_cy), (0, 255, 0),
                               cv2.MARKER_CROSS, 12, 2, cv2.LINE_AA)

        score_key = "fit_score" if mode_name == "ROIEdgeFit" else "contrast_score"
        stages.append(DebugStage(
            name="Score Heatmap",
            mode_name=mode_name,
            pipette_index=pipette_index,
            image=stage_img,
            description=(
                "FFT cross-correlation score map (each pixel = score for one "
                "ROI1 placement, marked at the centre of that placement). "
                "Bright = high score. Green crosshair = best position."
            ),
            metadata={
                score_key:       round(best_score, 2),
                "normalised":    round(norm_score, 3),
                "best_dx_px":    best_dx,
                "best_dy_px":    best_dy,
                "display_bbox":  patch_bbox,
                "roi1_points":   [list(p) for p in roi_points_orig] if roi_points_orig else None,
                "roi3_points":   [list(p) for p in roi3_points_orig] if roi3_points_orig else None,
            },
        ))

    # ── 3. Fit result — polygon overlay on patch canvas ──────────────────────
    raw_for_fit = (
        engine_result.get("edge_map")
        if mode_name == "ROIEdgeFit"
        else engine_result.get("gradient_map")
    )
    if raw_for_fit is not None:
        raw_h, raw_w = raw_for_fit.shape[:2]
        cy1 = max(0, py0); cy2 = min(raw_h, py0 + ph)
        cx1 = max(0, px0); cx2 = min(raw_w, px0 + pw)
        crop = raw_for_fit[cy1:cy2, cx1:cx2]
        crop8 = _normalise_to_uint8(crop)
        crop_bgr = cv2.cvtColor(crop8, cv2.COLOR_GRAY2BGR)
        stage_img = _new_patch_canvas()
        py_dst = cy1 - py0
        px_dst = cx1 - px0
        stage_img[py_dst:py_dst + crop_bgr.shape[0],
                  px_dst:px_dst + crop_bgr.shape[1]] = crop_bgr
        # Fitted ROI polygon (ROI1 translated by best_dx/best_dy)
        if fitted_polygon and len(fitted_polygon) >= 3:
            local_fit = np.array(
                [[int(p[0]) - px0, int(p[1]) - py0] for p in fitted_polygon],
                dtype=np.int32,
            )
            cv2.polylines(stage_img, [local_fit], isClosed=True,
                          color=(220, 220, 0), thickness=2, lineType=cv2.LINE_AA)
            ct = candidates[0].metadata.get("center_top") if candidates else None
            if ct is not None:
                ctx = int(ct[0]) - px0
                cty = int(ct[1]) - py0
                if 0 <= ctx < pw and 0 <= cty < ph:
                    cv2.drawMarker(stage_img, (ctx, cty), (220, 220, 0),
                                   cv2.MARKER_CROSS, 14, 2, cv2.LINE_AA)
            label = f"conf={candidates[0].confidence:.2f}"
            cv2.putText(stage_img, label, (4, 16),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 220, 0), 1, cv2.LINE_AA)
        elif not candidates:
            cv2.putText(stage_img, "No detection", (8, 24),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (80, 80, 220), 1, cv2.LINE_AA)

        cand_meta: Dict[str, Any] = {"n_candidates": len(candidates) if candidates else 0}
        if candidates:
            best = candidates[0]
            cand_meta["confidence"] = round(float(best.confidence), 3)
            if best.metadata:
                ct = best.metadata.get("center_top")
                if ct:
                    cand_meta["center_top"] = f"({ct[0]:.1f}, {ct[1]:.1f})"
                cand_meta["offset_dx"] = best.metadata.get("offset_dx", 0)
                cand_meta["offset_dy"] = best.metadata.get("offset_dy", 0)
        stages.append(DebugStage(
            name="Fit Result",
            mode_name=mode_name,
            pipette_index=pipette_index,
            image=stage_img,
            description=(
                "Yellow outline = fitted ROI (ROI1 translated to best match). "
                "Crosshair = center-top output coordinate. "
                "Toggle ROI1/ROI2 overlays in the viewer."
            ),
            metadata={
                **cand_meta,
                "display_bbox": patch_bbox,
                "roi1_points":  [list(p) for p in roi_points_orig] if roi_points_orig else None,
                "roi3_points":  [list(p) for p in roi3_points_orig] if roi3_points_orig else None,
            },
        ))

    return stages


def _draw_roi_fit_result(
    engine_result: Dict[str, Any],
    mode_name: str,
    candidates: Optional[List["TipCandidate"]],
) -> Optional[np.ndarray]:
    """Draw the fitted ROI polygon + center-top on the gradient/edge image."""
    raw = (
        engine_result.get("edge_map")
        if mode_name == "ROIEdgeFit"
        else engine_result.get("gradient_map")
    )
    roi_bbox = engine_result.get("roi_bbox")
    if raw is None or roi_bbox is None:
        return None

    ox, oy, roi_w, roi_h = roi_bbox
    # Crop the gradient/edge image to the ROI bounding box for display
    img8 = _normalise_to_uint8(raw)
    h, w = img8.shape[:2]
    lx0 = max(0, ox)
    ly0 = max(0, oy)
    lx1 = min(w, ox + roi_w)
    ly1 = min(h, oy + roi_h)
    if lx1 <= lx0 or ly1 <= ly0:
        canvas = np.zeros((roi_h, roi_w, 3), dtype=np.uint8)
    else:
        crop = img8[ly0:ly1, lx0:lx1]
        canvas = cv2.cvtColor(crop, cv2.COLOR_GRAY2BGR)

    if not candidates:
        cv2.putText(
            canvas, "No detection", (8, 24),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (80, 80, 220), 1, cv2.LINE_AA,
        )
        return canvas

    _CYAN = (220, 220, 0)
    for cand in candidates:
        poly = cand.metadata.get("fitted_roi_polygon") if cand.metadata else None
        if poly and len(poly) >= 3:
            local_pts = np.array(
                [[int(p[0]) - lx0, int(p[1]) - ly0] for p in poly], dtype=np.int32
            )
            cv2.polylines(canvas, [local_pts], isClosed=True, color=_CYAN, thickness=2)
            ct = cand.metadata.get("center_top")
            if ct is not None:
                ctx = int(ct[0]) - lx0
                cty = int(ct[1]) - ly0
                ch_h, ch_w = canvas.shape[:2]
                if 0 <= ctx < ch_w and 0 <= cty < ch_h:
                    cv2.drawMarker(
                        canvas, (ctx, cty), _CYAN,
                        cv2.MARKER_CROSS, 14, 2, cv2.LINE_AA,
                    )
        label = f"conf={cand.confidence:.2f}"
        cv2.putText(canvas, label, (4, 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, _CYAN, 1, cv2.LINE_AA)
    return canvas


# ---------------------------------------------------------------------------
# Result overlay (drawn on source image)
# ---------------------------------------------------------------------------

def build_tip_result_overlay(
    source: np.ndarray,
    engine_result: Optional[Dict[str, Any]],
    candidates: List[TipCandidate],
    roi_points: Optional[List[tuple]],
) -> np.ndarray:
    """Draw ROI polygon and detected tip edges on a copy of the source image."""
    overlay = source.copy() if len(source.shape) == 3 else cv2.cvtColor(source, cv2.COLOR_GRAY2BGR)

    # Draw ROI polygon
    if roi_points and len(roi_points) >= 3:
        pts = np.array([[int(p[0]), int(p[1])] for p in roi_points], dtype=np.int32)
        cv2.polylines(overlay, [pts], isClosed=True, color=(0, 220, 255), thickness=2)

    # Draw detected edges from engine result
    if engine_result:
        colors = {"left_edge": (0, 255, 0), "right_edge": (255, 80, 0), "bottom_edge": (0, 0, 255)}
        for key, color in colors.items():
            e = engine_result.get(key) or {}
            pts_list = e.get("points", [])
            if pts_list:
                pts = np.array([[int(p[0]), int(p[1])] for p in pts_list], dtype=np.int32)
                for pt in pts:
                    cv2.circle(overlay, tuple(pt), 2, color, -1)

    # Draw best candidate region box
    if candidates:
        best = max(candidates, key=lambda c: c.confidence)
        y_px, z_px, w_px, h_px = best.region
        x1, y1 = int(y_px), int(z_px)
        x2, y2 = int(y_px + w_px), int(z_px + h_px)
        cv2.rectangle(overlay, (x1, y1), (x2, y2), (255, 255, 0), 2)

    return overlay


def build_liquid_result_overlay(
    source: np.ndarray,
    poi_lists: List[List[PointOfInterest]],
    roi: Optional[tuple],
) -> np.ndarray:
    """Draw ROI bounding box and POI horizontal lines on the source image."""
    overlay = source.copy() if len(source.shape) == 3 else cv2.cvtColor(source, cv2.COLOR_GRAY2BGR)
    h, w = overlay.shape[:2]

    # ROI bounding box
    if roi:
        x, y, rw, rh = roi
        cv2.rectangle(overlay, (int(x), int(y)), (int(x+rw), int(y+rh)), (0, 220, 255), 2)

    # POI horizontal lines — colour by weight
    for poi_list in poi_lists:
        for poi in poi_list:
            z = int(roi[1] + poi.z_px) if roi else int(poi.z_px)
            alpha = min(1.0, poi.weight)
            g = int(100 + 155 * alpha)
            cv2.line(overlay, (0, z), (w, z), (0, g, 0), 1)

    return overlay


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _to_display(arr: np.ndarray) -> np.ndarray:
    """Convert any numeric array to uint8 for display."""
    if arr.dtype == np.uint8:
        return arr
    mn, mx = arr.min(), arr.max()
    if mx == mn:
        return np.zeros(arr.shape, dtype=np.uint8)
    return ((arr - mn) / (mx - mn) * 255).astype(np.uint8)


def _normalise_to_uint8(arr: np.ndarray) -> np.ndarray:
    return _to_display(arr)
