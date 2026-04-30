# ExtremaLines

> **Status:** Stub — content TODO.

## Overview

Detects per-row liquid transitions by finding **local extrema** in the
column-wise intensity profile and tracking how their density and termination
patterns change with depth.

## Inputs

- Single frame, A:B accumulation, or contrast working image.
- ROI bounding box.

## Algorithm

1. Build working image (blur / A:B / contrast).
2. For each column, find local maxima/minima along Z.
3. Aggregate per row → density profile and termination profile.
4. Interpret each profile separately to produce candidate transitions.

*(Screenshots: `img/ExtremaLines_*.png`.)*

## Parameters

- Blur params (shared across all modes).
- Extremum thresholds and minimum spacing.
- Density / termination interpretation thresholds.

## Tuning tips

TODO.

## Failure modes

TODO.

## Related modes

- `LineContinuity_Density` — a continuity-based analogue of the density
  profile.
- `IntensityDetection` — the simpler global-mean alternative.
