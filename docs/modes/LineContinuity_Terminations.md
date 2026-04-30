# LineContinuity_Terminations

> **Status:** Stub — content TODO.

## Overview

Variant of the `LineContinuity_*` family. Detects rows where many ridges
**terminate** (i.e. ridges that exist above a row but not below). This is
typical of the liquid surface, where vertical features in the tip are
abruptly hidden by the meniscus.

## Inputs

- Single frame, A:B accumulation, or contrast working image.
- ROI bounding box.

## Algorithm

1. Build working image.
2. Run the shared ridge extractor.
3. For each ridge, find its lowest connected pixel → row.
4. Histogram terminations per row → 1-D termination profile.
5. Peak-find on the profile.

*(Screenshots: `img/LineContinuity_Terminations_*.png`.)*

## Parameters

- Ridge extractor params (shared with siblings).
- Minimum ridge length, termination clustering tolerance.

## Tuning tips

TODO.

## Failure modes

TODO — e.g. very short ridges due to noise produce spurious terminations.

## Related modes

- See [`LineContinuity_Density.md`](LineContinuity_Density.md) for the family
  overview.
