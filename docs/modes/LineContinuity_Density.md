# LineContinuity_Density

> **Status:** Stub — content TODO.

## Overview

Members of the `LineContinuity_*` family share a ridge extractor that traces
bright/dark line structures across the ROI. This variant aggregates **ridge
density per row** — rows containing many ridges score high. The liquid
boundary is detected as a sharp drop in ridge density.

## Inputs

- Single frame, A:B accumulation, or contrast working image.
- ROI bounding box.

## Algorithm

1. Build working image.
2. Run the shared ridge extractor (Frangi-style or oriented Sobel).
3. Per row, count ridge pixels above strength threshold.
4. Threshold the resulting 1-D density profile to find candidates.

*(Screenshots: `img/LineContinuity_Density_*.png`.)*

## Parameters

- Ridge extractor params (kernel size, strength threshold, orientation).
- Density threshold and minimum row separation.

## Tuning tips

TODO.

## Failure modes

TODO.

## Related modes

- `LineContinuity_Terminations` — same extractor, looks at where ridges end.
- `LineContinuity_Correlation` — same extractor, correlates ridge patterns.
- `LineContinuity_PatternChange` — same extractor, detects pattern shifts.
- `ExtremaLines` — non-ridge analogue of the density profile.
