# InwardEdgeScan

> **Status:** Stub — content TODO.

## Overview

Tip-identification mode. Starting from the ROI boundary, it scans **inward**
along left/right/bottom rays looking for the first strong gradient — i.e. the
edge of the pipette tip itself.

## Inputs

- Single frame (typically the contrast image when `use_contrast=true`).
- ROI **polygon** (not just a bounding box) — defines the scan boundary.

## Algorithm

1. Build working image.
2. Compute Sobel gradient magnitude.
3. For each ray inward from the ROI polygon perimeter, find the first
   contiguous run of `confirmation_n` pixels with gradient ≥
   `gradient_threshold`.
4. Group hits by side (left / right / bottom) — see `_draw_edge_overlay`.
5. The interpreter fits a tip bounding box to the resulting edge points.

*(Screenshots: `img/InwardEdgeScan_*.png` — the **Edge Candidates** stage in
the A/B dialog is the canonical illustration: green = left, blue = right,
red = bottom.)*

## Parameters

- `gradient_threshold`: how strong an edge must be to count.
- `confirmation_n` / `confirmation_m`: an edge is confirmed if `n` of `m`
  consecutive ray samples exceed threshold.
- `roi_expansion_px`: how far the search may stray outside the calibrated ROI.
- `min_confidence`: threshold for the interpreter to accept a candidate.

## Tuning tips

TODO.

## Failure modes

TODO — specular highlights inside the tip, debris on the wall, etc.

## Related modes

- `WindowContrastAlign` — refines a coarse tip box by maximising local
  contrast.
- `ROIEdgeFit` / `ROIContrastFit` — alternative ROI-fitting tip detectors.
