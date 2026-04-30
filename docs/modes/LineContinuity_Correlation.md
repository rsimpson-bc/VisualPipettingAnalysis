# LineContinuity_Correlation

> **Status:** Stub — content TODO.

## Overview

Variant of the `LineContinuity_*` family. For each row, correlates the ridge
pattern with neighbouring rows. A sharp drop in correlation marks a
discontinuity such as a meniscus.

## Inputs

- Single frame, A:B accumulation, or contrast working image.
- ROI bounding box.

## Algorithm

1. Build working image.
2. Run the shared ridge extractor.
3. For each row, compute correlation of its ridge pattern with rows ±k above
   and below.
4. Build a 1-D correlation profile and look for minima.

*(Screenshots: `img/LineContinuity_Correlation_*.png`.)*

## Parameters

- Ridge extractor params (shared with siblings).
- Correlation window size `k`, minimum correlation drop.

## Tuning tips

TODO.

## Failure modes

TODO.

## Related modes

- See [`LineContinuity_Density.md`](LineContinuity_Density.md) for the family
  overview.
