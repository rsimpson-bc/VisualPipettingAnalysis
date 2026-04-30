# LineContinuity_PatternChange

> **Status:** Stub — content TODO.

## Overview

Variant of the `LineContinuity_*` family. Detects rows where the **pattern**
of ridge orientations or spacings changes abruptly — e.g. mostly-vertical
lines above, mostly-horizontal lines below.

## Inputs

- Single frame, A:B accumulation, or contrast working image.
- ROI bounding box.

## Algorithm

1. Build working image.
2. Run the shared ridge extractor.
3. Compute a per-row pattern descriptor (orientation histogram or similar).
4. Difference adjacent rows → 1-D pattern-change profile.
5. Peak-find.

*(Screenshots: `img/LineContinuity_PatternChange_*.png`.)*

## Parameters

- Ridge extractor params (shared with siblings).
- Pattern descriptor bin count, change threshold.

## Tuning tips

TODO.

## Failure modes

TODO.

## Related modes

- See [`LineContinuity_Density.md`](LineContinuity_Density.md) for the family
  overview.
