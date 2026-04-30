# Pipeline Mode Documentation

Each `<ModeName>.md` in this folder is rendered inside the **How it works** tab
of the Pipeline Parameter Editor when a mode is selected.

## Conventions

- File name **must** match the `mode_name` exactly (case-sensitive),
  e.g. `IntensityDetection.md` for the `IntensityDetection` mode.
- Standard sections (in order):
  1. **Overview** — one-paragraph summary of what the mode does and when to use it.
  2. **Inputs** — what kind of image/data the mode expects (single frame, A:B pair, reference contrast, ROI shape).
  3. **Algorithm** — step-by-step description of the processing pipeline. Each
     step ideally has an annotated screenshot.
  4. **Parameters** — narrative explanation of the most important params and
     how they interact. (The form on the left already shows their tooltips.)
  5. **Tuning tips** — common adjustments and what they look like.
  6. **Failure modes** — examples of bad outputs and what causes them.
  7. **Related modes** — cross-references to siblings (e.g. all
     `LineContinuity_*` variants share the same extractor).
- Images live in `docs/modes/img/` and are referenced relative to that folder,
  e.g. `![Step 1](img/intensity_step1.png)`.
- Keep paragraphs short. The viewer is narrow.
- Math: surround inline expressions with `$ ... $` and blocks with `$$ ... $$`
  (rendered as plain text by `QTextBrowser`; ASCII fallbacks like `^2` and
  `sqrt(x)` are usually clearer for now).

## Generating images

Screenshots can be captured directly from the **A/B Comparison** dialog:
right-click on either view → *Save image…*. Save to `docs/modes/img/` using
`<ModeName>_<step>.png`. Annotate freehand or with any image editor.

A future automation pass may render canonical example stages from a fixed set
of test images on demand. Until then, hand-curated screenshots are fine.
