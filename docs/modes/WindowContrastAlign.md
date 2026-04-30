# WindowContrastAlign

> **Status:** Stub — content TODO.

## Overview

Tip-identification mode that refines a coarse tip bounding box by sliding a
window and maximising the contrast between inside and outside.

## Inputs

- Single frame.
- A coarse tip region from a prior mode (or the calibrated ROI as a fallback).

## Algorithm

1. Build working image.
2. For each candidate offset `(dx, dy)` within a search range, compute the
   contrast between the pixels inside the candidate window and the pixels
   immediately outside.
3. Choose the `(dx, dy)` that maximises contrast.

*(Screenshots: `img/WindowContrastAlign_*.png`.)*

## Parameters

- Search range `(±dx, ±dy)`.
- Window inset / outset distances.
- Minimum contrast for a confident match.

## Tuning tips

TODO.

## Failure modes

TODO — symmetric ambiguity when the tip is centred but slightly off in y, etc.

## Related modes

- `InwardEdgeScan` — typically run first to provide the coarse box.
- `ROIContrastFit` — full polygon fit by contrast, of which this is a
  rectangular special case.
