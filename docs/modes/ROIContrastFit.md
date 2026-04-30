# ROIContrastFit

> **Status:** Stub — content TODO.

## Overview

Tip-identification mode that slides the calibrated ROI polygon around the
working image looking for the offset that **maximises contrast** between the
inside and the immediate outside of the polygon.

## Inputs

- Single frame, optionally a contrast working image
  (`use_contrast=true` + reference frames).
- ROI polygon (template).

## Algorithm

1. Build working image.
2. For each candidate offset `(dx, dy)` within the search range, sample mean
   intensities just inside and just outside the shifted polygon.
3. Score = `mean_outside − mean_inside` (or absolute difference).
4. Pick the offset with the best score.

*(Screenshots: `img/ROIContrastFit_*.png`.)*

## Parameters

- `search_range_x_px`, `search_range_y_px`: how far the polygon may move.
- `edge_outward_offset_px`: how far outside the polygon to sample for the
  outer mean.
- (Planned) `horizontal_only`: restrict the search to dx-only when the
  polygon orientation is fixed.

## Tuning tips

TODO.

## Failure modes

TODO — very dark backgrounds make absolute contrast meaningless; consider
`use_contrast=true` against a reference.

## Related modes

- `ROIEdgeFit` — gradient-based equivalent.
- `WindowContrastAlign` — rectangular special case.
