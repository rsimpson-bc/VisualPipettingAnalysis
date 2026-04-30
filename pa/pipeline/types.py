"""
Typed output interfaces for the PA pipeline.

Every layer boundary is defined here. Nothing in the pipeline imports from
another layer's implementation — only from this module.

Layer flow:
    ImageSet
        → [Layer 0] ProcessedImageCache
            → [Layer 1] Feature Extractors  →  RawFeatures
                → [Layer 2] Analysis Modes   →  ZProfile | TipRegion
                    → [Layer 3] Interpreters  →  List[PointOfInterest] | List[TipCandidate]
                        → [Layer 4] Integrators → LiquidMeasurementResult | TipIdentificationResult
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import numpy as np


# ---------------------------------------------------------------------------
# Debug output  (populated only when debug=True is passed to the runner)
# ---------------------------------------------------------------------------

@dataclass
class DebugStage:
    """
    One intermediate processing stage captured during a debug run.

    Exactly one of ``image`` or ``signal`` will be set per stage:
      - image   : 2D/3D numpy array (grayscale uint8, or BGR uint8)
      - signal  : 1D float array (a ZProfile signal)

    ``z_axis_px`` is populated alongside ``signal`` to allow correct
    axis labelling in the inspector.

    ``poi_z_px`` carries the Z-pixel positions of detected POIs for
    overlay on signal charts.

    ``metadata`` is a free-form dict for extra context (e.g. threshold
    values, coverage %, mode weight) shown in the inspector's detail pane.
    """
    name: str                                    # human-readable stage label
    mode_name: str                               # which pipeline mode produced it
    pipette_index: int

    image: Optional[np.ndarray] = None           # 2D/3D uint8
    signal: Optional[np.ndarray] = None          # 1D float
    z_axis_px: Optional[np.ndarray] = None       # axis for signal charts

    poi_z_px: Optional[List[float]] = None       # detected POI positions (signal)
    overlay_points: Optional[List[tuple]] = None # [(x,y)…] drawn on image

    metadata: Dict[str, Any] = field(default_factory=dict)
    description: str = ""


@dataclass
class DebugData:
    """
    All debug stages collected during one step's pipeline run for one pipette.
    Stages are ordered: shared pre-processing first, then per-mode stages.
    """
    step_id: str
    pipette_index: int
    source_image: Optional[np.ndarray] = None    # BGR, first frame (for overlay)
    result_overlay: Optional[np.ndarray] = None  # source_image + final result drawn on
    stages: List[DebugStage] = field(default_factory=list)

    def add(self, stage: DebugStage) -> None:
        self.stages.append(stage)



# ---------------------------------------------------------------------------
# Image source
# ---------------------------------------------------------------------------

@dataclass
class ImageSet:
    """
    All images for one pipette in one step.

    Single-frame use:   frames = [img]
    Multi-frame use:    frames = [img_a, img_b, ...]  (for A:B differencing)
    Future stitching:   pre-stitch before constructing; frames = [stitched_img]

    Reference contrast: populate reference_frames with a matching set of images
    (e.g. tip/well without liquid) taken at the same positions.  When a mode
    has use_contrast=True it subtracts a blurred reference from the blurred
    sample to produce the contrast working image.
    """
    pipette_index: int
    frames: List[np.ndarray]         # BGR, uint8  (sample images)
    source_paths: List[str]          # parallel to frames, for audit trail
    roi: Optional[tuple] = None      # (x, y, w, h) bounding box, None = full image
    roi_points: Optional[List[tuple]] = None  # polygon [(x,y),...] for tip analysis;
                                              # None = fall back to full image boundary
    reference_frames: List[np.ndarray] = field(default_factory=list)  # BGR, uint8
    reference_source_paths: List[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Layer 1 → Layer 2  (raw feature signals)
# ---------------------------------------------------------------------------

@dataclass
class RawFeatures:
    """
    Intermediate signals produced by a Layer 1 extractor.
    Each extractor subclass may use only a subset of these fields.
    """
    mode: str
    pipette_index: int

    # 1-D signals indexed by pixel row (Z axis, top=0)
    z_axis_px: Optional[np.ndarray] = None      # row indices
    intensity_signal: Optional[np.ndarray] = None
    ridge_count_signal: Optional[np.ndarray] = None
    ridge_spacing_signal: Optional[np.ndarray] = None
    termination_signal: Optional[np.ndarray] = None
    correlation_signal: Optional[np.ndarray] = None
    line_density_signal: Optional[np.ndarray] = None
    extrema_minima_signal: Optional[np.ndarray] = None
    extrema_termination_signal: Optional[np.ndarray] = None

    # 2-D outputs (tip edge detection)
    edge_map: Optional[np.ndarray] = None       # same HxW as input ROI


# ---------------------------------------------------------------------------
# Layer 2 outputs
# ---------------------------------------------------------------------------

@dataclass
class ZProfile:
    """
    Raw output from one analysis mode for one pipette.
    Passed to a Layer 3 interpreter to produce PointsOfInterest.
    """
    mode: str
    pipette_index: int
    z_axis_px: np.ndarray       # row indices (length N)
    signal: np.ndarray          # primary signal values (length N)
    variance: Optional[np.ndarray] = None   # per-row variance across multi-frame runs


@dataclass
class TipRegion:
    """
    Raw output from one tip analysis mode for one pipette.
    Passed to a Layer 3 interpreter to produce TipCandidates.
    """
    mode: str
    pipette_index: int
    candidates: List["TipCandidate"] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Layer 3 outputs
# ---------------------------------------------------------------------------

@dataclass
class PointOfInterest:
    """
    A candidate liquid transition location produced by a Layer 3 interpreter.
    """
    z_px: float
    weight: float           # 0–1; post-interpreter, accounts for confidence + mode weight
    source_mode: str
    label: str = ""         # e.g. "liquid_top", "liquid_bottom", "bubble"


@dataclass
class TipCandidate:
    """
    A single candidate tip edge location produced by a Layer 3 interpreter.
    Modes may produce multiple candidates ranked by confidence.
    """
    region: tuple                        # (x_px, y_px, width_px, height_px) in image coords
    angle_deg: float
    confidence: float                    # 0–1
    source_mode: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    # metadata keys used by ROIEdgeFit / ROIContrastFit:
    #   center_top          : (cx, ty) float tuple — the fitted tip top-centre in image coords
    #   offset_dx / dy      : int — best translation offset from nominal ROI position
    #   fit_score           : float — raw boundary alignment score (ROIEdgeFit)
    #   contrast_score      : float — raw summed gradient score (ROIContrastFit)
    #   fitted_roi_polygon  : List[tuple] — translated ROI polygon at best offset
    # TODO (future): multi-candidate support — the mode → interpreter → integrator
    # pipeline is already wired for List[TipCandidate], but InwardEdgeScan and
    # the ROI-fit modes currently emit at most one candidate per run.  When
    # multi-hypothesis search is implemented, emit ranked alternatives here.


# ---------------------------------------------------------------------------
# Layer 4 outputs  (match the JSON schemas in /schemas/)
# ---------------------------------------------------------------------------

@dataclass
class LiquidMeasurementResult:
    step_id: str
    pipette_index: int
    total_volume_ul: Optional[float]
    slugs: List[dict] = field(default_factory=list)
    bubbles: List[dict] = field(default_factory=list)
    droplets: List[dict] = field(default_factory=list)
    confidence: float = 0.0
    comments: str = ""
    # Internal: the merged POI list before volume calculation (for optimization harness)
    _poi_list: List[PointOfInterest] = field(default_factory=list, repr=False)


@dataclass
class TipIdentificationResult:
    step_id: str
    pipette_index: int
    tip_present: Optional[bool]
    perceived_tip_type: Optional[str]
    tip_type_match: Optional[bool]
    seating_depth_px: Optional[float]
    tip_angle_deg: Optional[float]
    seating_status: str     # "ok"|"warning_angle"|"warning_height"|"wrong_type"|"missing"|"unexpected"|"low_confidence"
    confidence: float       # 0–100 (matches schema)
    comments: str = ""
    # Internal: raw candidates before consensus (for optimization harness)
    _candidates: List[TipCandidate] = field(default_factory=list, repr=False)


# ---------------------------------------------------------------------------
# Pipeline configuration  (loaded from analysis_config.json)
# ---------------------------------------------------------------------------

@dataclass
class ModeConfig:
    """Configuration for one analysis mode within a pipeline."""
    mode_name: str
    enabled: bool = True
    weight: float = 1.0
    params: dict = field(default_factory=dict)  # mode-specific knobs


@dataclass
class PipelineConfig:
    """One named pipeline (e.g. 'full_liquid', 'fast_liquid', 'full_tip')."""
    name: str
    analysis_type: str          # "LiquidMeasurement" | "TipIdentification"
    modes: List[ModeConfig] = field(default_factory=list)
    integrator_params: dict = field(default_factory=dict)
