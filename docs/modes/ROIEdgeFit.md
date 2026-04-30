# ROIEdgeFit

> **Status:** Stub — content TODO.

## Overview

Tip-identification mode that fits the calibrated ROI **polygon** to the
gradient field of the working image, allowing each polygon vertex to slide
a few pixels independently to maximise edge alignment.

## Inputs

- Single frame.
- ROI polygon (template).

## Algorithm

1. Compute Sobel gradient magnitude.
2. For each candidate offset `(dx, dy)` of the polygon, score the sum of
   gradient magnitudes sampled along its edges.
3. Pick the offset with the highest score.

*(Screenshots: `img/ROIEdgeFit_*.png`.)*

## Parameters

- Search range, gradient threshold, edge-outward offset.

## Tuning tips

TODO.

## Failure modes

TODO — strong background edges can pull the fit off the tip.

## Related modes

- `ROIContrastFit` — same idea but scores by contrast instead of gradient.
- `InwardEdgeScan` — coarser, ray-based alternative.
