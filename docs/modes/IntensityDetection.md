# IntensityDetection

`IntensityDetection` is the simplest and fastest liquid-measurement mode in
the pipeline. It collapses the working image into a one-dimensional **per-row
mean intensity** signal and treats every row whose intensity exceeds a
threshold as a candidate liquid transition.

> Use this mode as your first attempt. It costs almost nothing, gives you a
> graph that is easy to reason about, and if it works there is no reason to
> reach for anything more elaborate.

## Overview

The mode answers a single question for each row of the ROI:

> _On average, how bright is this row in the working image?_

Bright rows are interpreted as rows where something interesting is happening
&mdash; a meniscus, a liquid surface, an A:B difference between frames, or a
strong signal from reference-contrast subtraction. The 1-D signal that comes
out is what you see plotted in the **Intensity Signal** stage of the A/B
Comparison and Stage Detail dialogs.

A small interpreter then walks the normalised signal, picks every peak that
exceeds `intensity_threshold`, enforces a minimum spacing of `min_distance`
pixels, and emits each surviving peak as a `PointOfInterest` for the liquid
integrator to consume.

## Inputs

The mode is unusually flexible about its input image. Three independent
preprocessing strategies can be combined:

| When... | The working image is... |
| --- | --- |
| `use_ab=true` and ≥ 2 frames are loaded | the **A:B accumulated difference** of all frames (bright = pixels that changed between frames). |
| `use_contrast=true` and at least one reference frame is loaded | the **reference-subtraction contrast image** of `frames[0]` against `reference_frames[0]` (bright = pixels that differ from the empty-tip reference). |
| neither of the above | the **first frame**, converted to grayscale. |

If both `use_ab` and `use_contrast` are enabled, the contrast image replaces
`frames[0]` first, then A:B accumulation is run over the resulting set. In
practice you typically pick one or the other.

The ROI bounding box is applied last, immediately before the per-row mean is
computed. Anything outside the ROI is discarded.

## Algorithm

1. **Build working image.**
   - If `use_contrast=true`, run `build_contrast_working_frame` on
     `frames[0]` and `reference_frames[0]` using all `contrast_*` params.
   - Otherwise, if `use_ab=true` and there are ≥ 2 frames, accumulate A:B
     differences across the whole frame list.
   - Otherwise, take `frames[0]` as-is and convert to grayscale.
2. **Blur.** Apply Gaussian or bilateral smoothing per `blur_method` /
   `blur_kernel`. This kills per-pixel noise that would otherwise dominate
   the row mean. `blur_method=none` skips this step.
3. **Optional gradient.** If `gradient_method` is not `none`, replace the
   blurred image with its Sobel response (`x`, `y`, or combined `xy`
   magnitude). This is useful when the *change* between dark and bright
   regions matters more than absolute brightness &mdash; for example,
   detecting a thin meniscus on a uniformly bright background.
4. **Crop ROI.** Anything outside `image_set.roi` is removed.
5. **Per-row mean.** `mean(roi_img, axis=1)` produces a 1-D float array, one
   value per row. This is the signal you see plotted vertically in the
   inspector.
6. **Normalise + threshold.** The interpreter min-max normalises the signal
   to `[0, 1]`, then picks every peak whose value is ≥ `intensity_threshold`,
   subject to a minimum separation of `min_distance` rows.
7. **Emit POIs.** Each surviving peak becomes a `PointOfInterest`. Its weight
   equals the peak's normalised height multiplied by `mode_weight` from the
   pipeline config.

*(Annotated screenshots will live at `img/IntensityDetection_*.png`. The
**Intensity Signal** stage in the A/B Comparison dialog already shows the
horizontal chart with horizontal blue dashed lines marking every emitted
POI &mdash; capture it from there.)*

## Parameters

The full list with min/max/defaults is in the **Parameters** tab. The
following are the ones you will actually want to touch.

### Pre-processing

- **`blur_method`** &mdash; `gaussian` is the right answer for almost every
  image. Use `bilateral` if the liquid edge is being eaten by the blur (rare).
  Use `none` only when the input is already smoothed (e.g. a contrast image
  with a heavy `contrast_blur_kernel`).
- **`blur_kernel`** &mdash; bigger kernels suppress noise but smear the
  transition. `5` is a good starting point for typical 1.3 MP frames; drop to
  `3` for high-resolution close-ups, raise to `9`–`13` for noisy or
  low-light conditions.
- **`use_ab`** &mdash; turn off when you only have one frame. Turn on when
  you have a sequence and want to highlight whatever moved.
- **`gradient_method`** &mdash; leave at `none` unless the liquid surface is
  defined by a brightness *transition* rather than a bright band. `y` (the
  vertical gradient) is the most common non-`none` choice.

### Detection

- **`intensity_threshold`** &mdash; the most important parameter. After
  normalisation, the brightest row is `1.0` and the darkest is `0.0`. A
  threshold of `0.3` keeps everything in the top 70 % of the dynamic range.
  - **Too low** → spurious peaks from noise or specular highlights.
  - **Too high** → the real meniscus gets missed when it produces only a
    modest peak.
- **`min_distance`** &mdash; rejects clustered peaks that all describe the
  same physical feature. `5` works for most ROIs; raise it if a single
  meniscus is being reported as 2–3 adjacent peaks.

### Reference contrast

When `use_contrast=true`, the mode subtracts a reference frame before doing
anything else. The `contrast_*` parameters control that subtraction
independently of the main blur:

- **`contrast_mode`** &mdash; `absolute` (`|sample − reference|`, useful for
  highlighting any change) or `relative` (signed, centred at grey 127, lets
  you tell *brighter* from *darker*).
- **`contrast_min` / `contrast_max`** &mdash; clip and stretch the difference
  magnitudes. Raise `contrast_min` to suppress background flicker; lower
  `contrast_max` to saturate genuinely large differences.
- **`contrast_blur_*`** &mdash; smoothing applied to *both* the sample and
  reference before subtraction. A kernel of `5–9` is typical. The contrast
  pre-blur is independent of the main `blur_kernel`.

## Tuning tips

- **Start with a graph.** Open the A/B Comparison dialog, select
  `IntensityDetection`, run on a representative frame, and look at the
  **Intensity Signal** stage. Hover over the image &mdash; the orange
  crosshair shows you exactly which row of the chart corresponds to which
  row of the image. The right `intensity_threshold` is usually visually
  obvious.
- **Don't fight noise with thresholds.** If the chart is jagged, raise
  `blur_kernel` first. Only crank `intensity_threshold` after smoothing has
  stopped helping.
- **A:B is for sequences, not noise reduction.** `use_ab` reveals what
  *changed* between frames. If your sequence is two near-identical frames,
  A:B will produce mostly noise; pick a single frame instead.
- **Reference contrast is for stable backgrounds.** It works wonderfully for
  pipettes imaged against a fixed lighting setup and a known empty-tip
  reference. It fails the moment the camera, lighting, or pipette position
  changes between sample and reference.

## Failure modes

- **Specular highlights produce false peaks.** A glint on the wall of the
  pipette is, by definition, a very bright row. Either mask it out via the
  ROI or switch to `gradient_method=y` so the highlight is no longer the
  brightest thing in its row.
- **Faint meniscus on a bright tip.** The peak is real but it never reaches
  `intensity_threshold` because the rest of the tip is also bright. Use
  reference-contrast subtraction (`use_contrast=true`) to rebase the image
  on what's *different* about the sample.
- **Two adjacent peaks for one meniscus.** The signal is bouncing across the
  threshold. Raise `min_distance`, raise `blur_kernel`, or both.
- **Empty-tip reference with subtle colour shift.** `contrast_mode=absolute`
  picks up uniform colour offsets as a faint global glow. Use
  `contrast_mode=relative` so positive and negative differences cancel out.

## Related modes

- **[`ExtremaLines`](ExtremaLines.md)** &mdash; same pre-processing pipeline,
  but instead of a row mean it counts per-row local extrema. Better for
  cloudy or refractive liquids where the *texture* changes more sharply
  than the brightness.
- **[`LineContinuity_Density`](LineContinuity_Density.md)** &mdash; uses a
  ridge extractor for similar reasons. Slower; reach for it when neither
  intensity nor extrema produce a clean signal.
