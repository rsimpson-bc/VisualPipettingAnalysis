"""<br>Layer 1 — Row Contrast Extractor.

Three signal families are supported:

  **Horizontal spread** — ``stat`` in {"std", "variance"}
    For each ROI row, compute the standard deviation (or variance) of pixel
    values *across the columns* of that row.  A row with many alternating
    bright/dark patches (liquid meniscus, bubbles, texture bands) has a high
    within-row spread even if its mean intensity is unremarkable.
    Formula: ``arr.std(axis=1)`` — high where a single scanline contains
    varied pixel values.

  **Vertical** — ``stat`` in {"std_v", "variance_v"}
    For each row position, compute the local standard deviation of the per-row
    *mean intensity* over a sliding window of ``vertical_window_px`` rows.
    This captures regions where the mean signal changes rapidly in the z
    direction (i.e., sharp vertical transitions).
    Formula: sliding-window std of ``arr.mean(axis=1)``.

  **Horizontal peak analysis** — ``stat`` in {"peak_count_h", "peak_prominence_sum_h"}
    For each ROI row, run ``scipy.signal.find_peaks`` on the 1-D pixel
    intensity profile with configurable ``peak_min_distance`` and
    ``peak_min_prominence`` filters.  This separates rows that are "rocky but
    flat" (many tiny bumps all below the prominence threshold) from rows with
    genuine large-amplitude oscillations (few but tall peaks).

    ``peak_count_h`` — count of qualifying peaks, divided by ``width ** peak_width_exponent``.
    ``peak_prominence_sum_h`` — sum of peak prominences, divided by ``width ** peak_width_exponent``.
    The exponent corrects for wider rows having more opportunity for more peaks
    (0 = no correction, 0.5 = square-root, 1.0 = fully per-pixel).

Used by: RowContrastDetection mode.
"""

from __future__ import annotations
import functools
import cv2
import numpy as np
from typing import Optional
from pa.pipeline.types import ImageSet, RawFeatures
from pa.pipeline.cache import ProcessedImageCache
from pa.pipeline.extractors.base import BaseExtractor
from pa.pipeline import image_primitives as ip


@functools.lru_cache(maxsize=64)
def _roi_row_widths(
    roi_points: tuple,          # tuple of (int_x, int_y) rounded polygon vertices
    bbox_x: int,
    bbox_y: int,
    bbox_w: int,
    bbox_h: int,
) -> np.ndarray:
    """Return float32 array of shape (bbox_h,) — valid pixel count per row inside
    the polygon.  Delegates rasterisation to ``ip.roi_polygon_mask`` (also cached).
    """
    return ip.roi_polygon_mask(roi_points, bbox_x, bbox_y, bbox_w, bbox_h).sum(axis=1).astype(np.float32)


class RowContrastExtractor(BaseExtractor):
    """
    Params:
        blur_method (str):          "none" | "gaussian" | "bilateral"
        blur_kernel (int):          kernel size for gaussian blur
        use_ab (bool):              True = accumulate A:B diffs; False = single frame
        gradient_method (str):      optional pre-gradient step
        gradient_ksize (int):       Sobel kernel size
        stat (str):                 "std" (default, horizontal) | "variance" (horizontal) |
                                    "std_v" (vertical) | "variance_v" (vertical) |
                                    "peak_count_h" | "peak_prominence_sum_h"
        vertical_window_px (int):   window size for std_v / variance_v (default 5)
        peak_min_distance (int):    minimum separation in px between counted peaks (default 5)
        peak_min_prominence (float):minimum prominence (0–255 raw intensity units) for a peak
                                    to count; filters out low-amplitude noise wiggles (default 5.0)
        peak_width_exponent (float):width-normalisation exponent applied as
                                    ``signal / width**exponent``; 0=none, 0.5=sqrt, 1=per-px
                                    (default 0.5)
    """

    def _extract(self, image_set: ImageSet, cache: ProcessedImageCache) -> RawFeatures:
        params = self.params

        # Step 1: build working image (single frame or A:B accumulation)
        use_ab = params.get("use_ab", True)
        if use_ab and len(image_set.frames) >= 2:
            working = cache.get_or_compute(
                image_set.pipette_index, "accumulate_ab", {},
                lambda: ip.accumulate_ab(image_set.frames),
            )
        else:
            # Cache the pre-grayscale (potentially contrast-modified) frame so
            # the debug collector can display it as a separate stage.
            if params.get("use_contrast", False) and image_set.frames:
                _frame0 = image_set.frames[0]
                cache.get_or_compute(
                    image_set.pipette_index, "contrast_frame", {},
                    lambda: _frame0,
                )
            working = cache.get_or_compute(
                image_set.pipette_index, "to_grayscale_frame0", {},
                lambda: ip.to_grayscale(image_set.frames[0]),
            )

        # Step 2: blur
        blur_params = {
            "method":      params.get("blur_method", "gaussian"),
            "kernel_size": params.get("blur_kernel", 5),
            "d":           params.get("bilateral_d", 9),
            "sigma_color": params.get("bilateral_sigma_color", 75.0),
            "sigma_space": params.get("bilateral_sigma_space", 75.0),
        }
        blurred = cache.get_or_compute(
            image_set.pipette_index, "blur", blur_params,
            lambda: ip.blur(working.astype(np.uint8), **blur_params),
        )

        # Step 3: optional gradient step
        grad_method = params.get("gradient_method", "none")
        grad_ksize  = params.get("gradient_ksize", 3)
        grad_params = {"method": grad_method, "ksize": grad_ksize}
        if grad_method != "none":
            processed = cache.get_or_compute(
                image_set.pipette_index, "gradient", grad_params,
                lambda: ip.apply_gradient_step(blurred, grad_method, grad_ksize),
            )
        else:
            processed = blurred

        # Step 4: extract ROI (bounding-box crop)
        roi_img = ip.extract_roi(processed, image_set.roi) if image_set.roi else processed

        # Step 4b: build polygon mask in bbox-local coords so stats are
        # computed only over pixels that are actually inside the ROI polygon.
        # Without this, the bounding-box rectangle includes corner pixels
        # outside the polygon, biasing std/var/mean for non-rectangular ROIs.
        arr = roi_img.astype(np.float32)
        n_rows, n_cols = arr.shape[:2]

        poly_mask: Optional[np.ndarray] = None   # (n_rows, n_cols) bool or None
        pts_key: Optional[tuple] = None
        roi_bbox_ints: Optional[tuple] = None

        if image_set.roi_points and image_set.roi:
            roi_x = int(image_set.roi[0]); roi_y = int(image_set.roi[1])
            roi_w = int(image_set.roi[2]); roi_h = int(image_set.roi[3])
            roi_bbox_ints = (roi_x, roi_y, roi_w, roi_h)
            pts_key = tuple(
                (int(round(x)), int(round(y))) for x, y in image_set.roi_points
            )
            full_mask = ip.roi_polygon_mask(pts_key, roi_x, roi_y, roi_w, roi_h)
            # Clamp to actual arr dimensions (image boundary may truncate)
            mh = min(full_mask.shape[0], n_rows)
            mw = min(full_mask.shape[1], n_cols)
            poly_mask = np.zeros((n_rows, n_cols), dtype=bool)
            poly_mask[:mh, :mw] = full_mask[:mh, :mw]

        def _masked_arr() -> np.ma.MaskedArray:
            """arr as masked array — outside-polygon pixels are excluded."""
            if poly_mask is not None:
                return np.ma.array(arr, mask=~poly_mask)
            return np.ma.array(arr)   # unmasked fallback

        # Step 5: per-row stat → 1-D signal
        stat = params.get("stat", "std")
        _band_signals_out = None  # populated only by peak_count_bands_h

        if stat in ("std_v", "variance_v"):
            # Vertical: sliding-window std of per-row mean along the z-axis.
            # Use polygon-masked row means so edge pixels don't skew the result.
            from scipy.ndimage import uniform_filter1d
            ma = _masked_arr()
            raw_means = ma.mean(axis=1)

            # Identify which rows have any valid (in-polygon) pixels.
            # np.ma.getmaskarray always returns a bool array (never np.ma.nomask).
            row_valid = ~np.ma.getmaskarray(raw_means).ravel()[:n_rows]
            row_means_raw = np.asarray(raw_means.filled(0), dtype=np.float32)

            # Nearest-fill rows with no polygon coverage before the sliding
            # filter so that the window at the polygon boundary sees a flat
            # (constant) signal rather than an abrupt 0 → real-value jump.
            # Without this, the transition creates a huge spike that, after
            # squaring (variance_v) and normalization, dominates the whole
            # signal and makes everything else appear as ~0.
            if not row_valid.all() and row_valid.any():
                valid_idxs = np.where(row_valid)[0]
                all_idxs = np.arange(n_rows)
                ii = np.clip(
                    np.searchsorted(valid_idxs, all_idxs),
                    0, len(valid_idxs) - 1,
                )
                ii_prev = np.maximum(0, ii - 1)
                d_next = np.abs(valid_idxs[ii] - all_idxs)
                d_prev = np.abs(valid_idxs[ii_prev] - all_idxs)
                near = np.where(d_prev < d_next, valid_idxs[ii_prev], valid_idxs[ii])
                row_means_filled = row_means_raw.copy()
                row_means_filled[~row_valid] = row_means_raw[near[~row_valid]]
            else:
                row_means_filled = row_means_raw

            win = max(2, int(params.get("vertical_window_px", 5)))
            m1 = uniform_filter1d(row_means_filled, win, mode='nearest')
            m2 = uniform_filter1d(row_means_filled ** 2, win, mode='nearest')
            vert_std = np.sqrt(np.maximum(0.0, m2 - m1 ** 2))

            # Zero out rows with no polygon coverage (their filled values were
            # only used to prevent the boundary spike, not for output).
            if not row_valid.all():
                vert_std[~row_valid] = 0.0

            signal = vert_std ** 2 if stat == "variance_v" else vert_std
        elif stat in (
            "peak_count_h", "peak_prominence_sum_h",
            "peak_prominence_mean_h", "peak_width_mean_h",
            "peak_count_bands_h",
        ):
            # Horizontal peak analysis: per-row find_peaks on valid pixels only.
            from scipy.signal import find_peaks, peak_widths as _peak_widths
            min_dist  = max(1, int(params.get("peak_min_distance", 5)))
            min_prom  = float(params.get("peak_min_prominence", 5.0))
            max_prom  = float(params.get("peak_max_prominence", 0.0))  # 0 = disabled
            width_exp = float(params.get("peak_width_exponent", 0.5))

            # Per-row valid width from polygon mask (cached by polygon identity)
            if pts_key is not None and roi_bbox_ints is not None:
                roi_x, roi_y, roi_w, roi_h = roi_bbox_ints
                row_widths = _roi_row_widths(pts_key, roi_x, roi_y, roi_w, roi_h)
                row_widths = row_widths[:n_rows]
                if len(row_widths) < n_rows:
                    fallback = float(arr.shape[1])
                    row_widths = np.concatenate([
                        row_widths,
                        np.full(n_rows - len(row_widths), fallback, dtype=np.float32),
                    ])
            else:
                row_widths = np.full(n_rows, float(arr.shape[1]), dtype=np.float32)

            row_widths = np.maximum(1.0, row_widths)
            norm_factors = row_widths ** width_exp if width_exp > 0.0 else np.ones(n_rows, dtype=np.float32)

            # --- Band setup for peak_count_bands_h ---
            _BAND_PALETTE = [
                "#4444aa",  # 0–5   dark blue
                "#2277cc",  # 5–10  blue
                "#22aaaa",  # 10–15 teal
                "#22cc66",  # 15–20 green
                "#99cc22",  # 20–25 yellow-green
                "#ddcc00",  # 25–30 yellow
                "#ffaa00",  # 30–35 amber
                "#ff6600",  # 35–40 orange
                "#ff2200",  # 40–45 red-orange
                "#cc0033",  # 45–50 crimson
                "#8800cc",  # >50   purple
            ]
            band_edges: list = []
            band_arrays: list = []
            band_meta: list = []
            if stat == "peak_count_bands_h":
                raw_edges = params.get(
                    "peak_band_edges",
                    [5, 10, 15, 20, 25, 30, 35, 40, 45, 50],
                )
                if isinstance(raw_edges, (list, tuple)) and len(raw_edges) > 0:
                    band_edges = sorted(float(e) for e in raw_edges)
                else:
                    band_edges = [10.0, 25.0, 50.0]
                # N edges → N+1 bands
                n_bands = len(band_edges) + 1
                band_arrays = [np.zeros(n_rows, dtype=np.float32) for _ in range(n_bands)]
                for bi in range(n_bands):
                    lo = min_prom if bi == 0 else band_edges[bi - 1]
                    hi = band_edges[bi] if bi < len(band_edges) else None
                    if hi is None:
                        label = f"≥{lo:.4g}"
                    else:
                        label = f"{lo:.4g}–{hi:.4g}"
                    color = _BAND_PALETTE[bi % len(_BAND_PALETTE)]
                    band_meta.append({"label": label, "color": color, "lo": lo, "hi": hi})

            out = np.empty(n_rows, dtype=np.float32)
            for i in range(n_rows):
                # Use only pixels inside the polygon for this row
                if poly_mask is not None:
                    row_data = arr[i][poly_mask[i]]
                else:
                    row_data = arr[i]
                if len(row_data) < 2:
                    out[i] = 0.0
                    continue

                # For band stat: find all peaks above the global floor; apply
                # max_prom filter; then slice into bands below.
                find_prom_arg = min_prom
                if stat == "peak_count_bands_h":
                    find_prom_arg = min_prom  # floor only; bands handle upper bound

                peaks, props = find_peaks(
                    row_data.astype(float),
                    distance=min_dist,
                    prominence=find_prom_arg,
                )

                # Apply upper prominence bound if set (non-band stats)
                if max_prom > 0.0 and stat != "peak_count_bands_h" and len(peaks) > 0:
                    keep = props["prominences"] <= max_prom
                    peaks = peaks[keep]
                    props = {k: v[keep] for k, v in props.items()}

                nf = norm_factors[i]

                if stat == "peak_count_h":
                    out[i] = len(peaks) / nf
                elif stat == "peak_prominence_sum_h":
                    out[i] = float(props["prominences"].sum()) / nf
                elif stat == "peak_prominence_mean_h":
                    out[i] = (float(props["prominences"].mean()) / nf
                              if len(peaks) > 0 else 0.0)
                elif stat == "peak_width_mean_h":
                    if len(peaks) > 0:
                        widths, _, _, _ = _peak_widths(
                            row_data.astype(float), peaks,
                            rel_height=0.5,
                            prominence_data=(
                                props["prominences"],
                                props["left_bases"],
                                props["right_bases"],
                            ),
                        )
                        out[i] = float(widths.mean()) / nf
                    else:
                        out[i] = 0.0
                elif stat == "peak_count_bands_h":
                    proms = props["prominences"] if len(peaks) > 0 else np.array([])
                    # Apply max_prom global cap if set
                    if max_prom > 0.0 and len(proms) > 0:
                        keep = proms <= max_prom
                        proms = proms[keep]
                    total = 0.0
                    for bi, bm in enumerate(band_meta):
                        lo, hi = bm["lo"], bm["hi"]
                        if len(proms) > 0:
                            mask = proms >= lo
                            if hi is not None:
                                mask &= proms < hi
                            cnt = float(mask.sum())
                        else:
                            cnt = 0.0
                        band_arrays[bi][i] = cnt / nf
                        total += cnt
                    out[i] = total / nf

            signal = out

            # Attach band data to features via a side-channel attribute
            # (set after RawFeatures is built, below)
            _band_signals_out = None
            if stat == "peak_count_bands_h" and band_meta:
                # Convert raw band counts to proportions of total for the
                # color-distribution visualization.  Both band_arrays[bi][r]
                # and out[r] are divided by the same nf factor, so the fractions
                # are simply band_arrays[bi][r] / out[r] = count_bi / total.
                frac_arrays = [
                    np.where(out > 0,
                             band_arrays[bi] / np.maximum(out, 1e-9),
                             0.0).astype(np.float32)
                    for bi in range(len(band_meta))
                ]
                _band_signals_out = [
                    {"signal": frac_arrays[bi], "label": bm["label"],
                     "color": bm["color"], "viz": "band_distribution"}
                    for bi, bm in enumerate(band_meta)
                ]
        elif stat == "variance":
            signal = _masked_arr().var(axis=1).filled(0).astype(np.float32)
        else:  # "std" (default)
            signal = _masked_arr().std(axis=1).filled(0).astype(np.float32)

        # Optional cap for variance-family stats (prevents one bright spot from
        # compressing all other features toward zero after normalisation).
        if stat in ("variance", "variance_v"):
            variance_cap = float(params.get("variance_cap", 0.0))
            if variance_cap > 0.0:
                signal = np.minimum(signal, variance_cap)

        z_axis = np.arange(len(signal))

        raw = RawFeatures(
            mode="RowContrastExtractor",
            pipette_index=image_set.pipette_index,
            z_axis_px=z_axis,
            intensity_signal=signal,
        )
        if _band_signals_out is not None:
            raw.band_signals = _band_signals_out
        return raw
