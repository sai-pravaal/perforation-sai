"""IPDS Pipeline Constants and Configuration.

This module centralizes all tunable thresholds, constants, and ordering
definitions used across the 23-step IPDS computational pipeline and usability features.
Per project requirements, no threshold or constant may be hardcoded inline.
"""

from typing import Any, Dict, List, Tuple

from scipy.stats import chi2 as _chi2
from scipy.stats import norm as _norm

# -------------------------------------------------------------------------
# Step 18: Fixed Composite Signature Field Ordering
# -------------------------------------------------------------------------
# Fixed session-wide ordering of the 14-element composite signature vector.
# Index i always corresponds to the exact same physical feature across all holes.
COMPOSITE_FIELD_ORDER: List[str] = [
    "mean_radius_um",          # Step 13: Average radius in microns
    "radius_std_um",           # Step 13: Point-to-point radius standard deviation (microns)
    "fourier_c1",              # Step 13: 1st harmonic Fourier coefficient magnitude (microns)
    "fourier_c2",              # Step 13: 2nd harmonic Fourier coefficient magnitude (microns)
    "fourier_c3",              # Step 13: 3rd harmonic Fourier coefficient magnitude (microns)
    "fourier_c4",              # Step 13: 4th harmonic Fourier coefficient magnitude (microns)
    "roundness_ratio",         # Step 13: 4*pi*area / perimeter^2 (NaN for edge teeth)
    "sobel_mean",              # Step 15: Mean boundary gradient magnitude
    "sobel_std",               # Step 15: Standard deviation of boundary gradient
    "glcm_contrast",           # Step 15: Gray-Level Co-occurrence Matrix contrast
    "glcm_homogeneity",        # Step 15: Gray-Level Co-occurrence Matrix homogeneity
    "residual_wobble",         # Step 15: High-frequency residual from Fourier fit (microns)
    "shadow_strength",         # Step 17: Combined shadow vector magnitude
    "shadow_direction_deg"     # Step 17: Dominant shadow angle (0 to 360 degrees)
]

# -------------------------------------------------------------------------
# Step 2: Lens Distortion & Pixel-to-Physical-Unit Calibration
# -------------------------------------------------------------------------
CALIBRATION_PATTERN_SPACING_UM: float = 2000.0       # Known centre-to-centre spacing of the printed target (spec example: 2 mm)
CALIBRATION_PATTERN_SIZE: Tuple[int, int] = (7, 7)   # Inner points per row, per column of the target grid
CALIBRATION_DETECTION_GAMMA: float = 2.2             # Display gamma applied to the detection copy only (frames stay linear)

# -------------------------------------------------------------------------
# Step 3: Illumination Geometry (Path B ball-bearing detection)
# -------------------------------------------------------------------------
BALL_HOUGH_DP: float = 1.2           # HoughCircles accumulator resolution ratio
BALL_HOUGH_MIN_DIST_PX: int = 50     # Minimum distance between detected circle centres
BALL_HOUGH_PARAM1: int = 50          # Canny upper threshold inside HoughCircles
BALL_HOUGH_PARAM2: int = 30          # Accumulator threshold for circle centres
BALL_MIN_RADIUS_PX: int = 20         # Smallest ball radius accepted, in pixels

# -------------------------------------------------------------------------
# Step 5: Cross-Pass Image Registration
# -------------------------------------------------------------------------
REGISTRATION_MAX_SHIFT_PX: float = 5.0          # Corner displacement above this flags possible mid-session movement
REGISTRATION_MIN_ECC_CORRELATION: float = 0.6   # ECC results below this correlation are rejected (ORB fallback)
REGISTRATION_ECC_MAX_ITERS: int = 100           # ECC iterations on the coarse (downscaled) pass
REGISTRATION_ECC_REFINE_ITERS: int = 20         # ECC iterations on the full-resolution refinement pass
REGISTRATION_ECC_EPS: float = 1e-5              # ECC convergence threshold
REGISTRATION_ECC_MAX_DIM: int = 1500            # Longest side of the coarse ECC pass, in pixels
REGISTRATION_ORB_MAX_FEATURES: int = 1000       # Keypoints for the ORB + homography fallback
REGISTRATION_RANSAC_REPROJ_PX: float = 5.0      # RANSAC inlier reprojection threshold for the fallback homography

# -------------------------------------------------------------------------
# Step 6: Stamp Boundary & Orientation
# -------------------------------------------------------------------------
STAMP_ROTATION_WARN_DEG: float = 5.0           # Tilt above this disagrees with the torch 0-deg alignment; warn
STAMP_THRESHOLD_BLUR_PX: int = 5                # Gaussian pre-blur (odd) before the stamp/backdrop Otsu split

# -------------------------------------------------------------------------
# Step 7: Perforation Candidate Detection
# -------------------------------------------------------------------------
HOLE_MIN_DIAMETER_MM: float = 0.5               # Smallest plausible perforation diameter (spec: roughly 1-2 mm)
HOLE_MAX_DIAMETER_MM: float = 3.0               # Largest plausible perforation diameter
HOLE_AREA_MIN_FRACTION: float = 0.3             # Min area as a fraction of the min circle (teeth show part of a circle)
HOLE_AREA_MAX_FACTOR: float = 2.0               # Max area as a multiple of the max circle
HOLE_OVERSIZE_REVIEW_FACTOR: float = 6.0        # Blobs up to this x the max area are flagged for review, larger ignored
HOLE_THRESHOLD_BLUR_PX: int = 3                 # Gaussian pre-blur (odd) before the backlit Otsu split

# -------------------------------------------------------------------------
# Step 8: Hole-Type Classification
# -------------------------------------------------------------------------
EDGE_TOOTH_MARGIN_PX: float = 8.0               # Outline points closer than this to the stamp boundary count as "touching"
EDGE_TOOTH_OUTLINE_FRACTION: float = 0.15       # Fraction of touching outline points that makes a candidate a tooth

# -------------------------------------------------------------------------
# Step 10: Focus-Stack Fusion
# -------------------------------------------------------------------------
FOCUS_SHARPNESS_BLUR_PX: int = 5                # Gaussian kernel smoothing the Laplacian sharpness map (odd)
FOCUS_PYRAMID_LEVELS: int = 4                   # Laplacian-pyramid levels for seam-free blending of a real stack

# -------------------------------------------------------------------------
# Step 9: Per-Hole Region-of-Interest Extraction
# -------------------------------------------------------------------------
ROI_EXPECTED_HOLE_DIAMETER_UM: float = 1200.0   # Nominal perforation hole diameter (spec: roughly 1-2 mm)
ROI_CROP_SCALE_FACTOR: float = 3.0              # Crop side = factor x expected diameter (spec example: three times)
ROI_MIN_CROP_PX: int = 33                       # Lower bound only, so tiny scales still give a usable crop

# -------------------------------------------------------------------------
# Step 12 & 13: Geometric and Shape Extraction Settings
# -------------------------------------------------------------------------
RADIUS_PROFILE_NUM_ANGLES: int = 72       # 360 degrees sampled at 5-degree increments
FOURIER_FIT_ORDER: int = 4                # 1st through 4th harmonics for shape fitting

# -------------------------------------------------------------------------
# Step 4 & Feature 4: Illumination Matching and Calibration
# -------------------------------------------------------------------------
BRIGHTNESS_MISMATCH_TOLERANCE: float = 0.15   # +/- 15% brightness tolerance band
BRIGHTNESS_PATCH_SIZE_PX: int = 32            # Side of the plain-paper reference patch searched for in Step 4
CAMERA_MATCH_KEYS: List[str] = [
    "aperture",
    "shutter_speed",
    "iso",
    "white_balance"
]

# -------------------------------------------------------------------------
# Step 14 & 15: Boundary Texture Sampling and Roughness Settings
# -------------------------------------------------------------------------
BOUNDARY_SAMPLE_SPACING_UM: float = 20.0   # Fixed physical arc-length spacing between boundary samples
BOUNDARY_PATCH_SIZE_PX: int = 7            # Square boundary patch side length in pixels (odd)
GLCM_LEVELS: int = 32                      # Gray levels for GLCM quantization (Step 15); kept low for 7x7 (49-pixel) patches
GLCM_DISTANCES: List[int] = [1]            # GLCM pixel-pair offset distances (Step 15)
GLCM_ANGLES_DEG: List[float] = [0.0]       # GLCM pixel-pair offset angles in degrees (Step 15)

# -------------------------------------------------------------------------
# Step 16: Angular Brightness Sampling Settings
# -------------------------------------------------------------------------
PAPER_SAMPLE_OFFSET_UM: float = 60.0       # Fixed radial offset outward onto paper beside the hole

# -------------------------------------------------------------------------
# Step 19: Reference Group Construction Settings
# -------------------------------------------------------------------------
MIN_HOLES_PER_EDGE: int = 5               # Minimum hole count on edge before pooling fallback

# -------------------------------------------------------------------------
# Step 20: Robust Statistics and Deviation Scoring
# -------------------------------------------------------------------------
MAD_SCALE_CONSTANT: float = 0.6745         # Consistency factor for normal distribution
COMBINATION_METHOD: str = "summed_z"       # Active combination: "summed_z" or "mahalanobis"
# Spread estimate for edge-grouped teeth: "pooled_edge_factor" pools each field's
# MAD across the specimen's edges and scales it by a per-edge noise factor
# (detected ~7x more altered holes than "group_mad" in simulation at 10 teeth/edge,
# while staying near 1% false flags when edges differ in noise); "group_mad" is
# the per-edge MAD only.
SCALE_METHOD: str = "pooled_edge_factor"
# Pooling is applied only to edges with at least this many teeth; below it a
# 3-hole altered run is a large share of the edge and the per-edge MAD did
# better in simulation (detection at 6 teeth: 3.1% per-edge vs 0.9% pooled;
# at 8: 2.0% vs 5.8%; at 10: 5% vs 19%; all calibrated to ~1% false flags).
POOLED_SCALE_MIN_EDGE_SIZE: int = 8
MEAN_AD_SCALE_CONSTANT: float = 0.7979     # sqrt(2/pi): mean-absolute-deviation consistency factor, used only when MAD = 0
ZERO_SPREAD_RELATIVE_FLOOR: float = 0.01   # Spread floor as a fraction of |median| when the whole group is identical

# -------------------------------------------------------------------------
# Step 21: Outlier Thresholding
# -------------------------------------------------------------------------
# A hole is flagged when a genuine hole would score at least this high with
# probability below OUTLIER_TAIL_PROBABILITY (one-sided tail of sum z^2).
# summed_z: Monte Carlo null calibrated to the reference-group size (Step 21),
# converging to chi-square(dof) for large groups; mahalanobis: chi-square(dof).
OUTLIER_TAIL_PROBABILITY: float = 0.01
MIN_REFERENCE_GROUP_SIZE: int = 3          # Fewer comparison holes than this: no calibrated decision, never flagged
NULL_CALIBRATION_DRAWS: int = 20000        # Monte Carlo draws per (group size, dof) null distribution
NULL_CALIBRATION_SEED: int = 20261025      # Fixed seed so cutoffs are reproducible run to run
# Large-sample reference cutoff on the L2 norm for all 14 fields: sqrt(chi2.ppf(0.99, 14)) ~ 5.40.
OUTLIER_CUTOFF: float = float(_chi2.ppf(1.0 - OUTLIER_TAIL_PROBABILITY, df=14) ** 0.5)
# Per-field "driver" level for the breakdown panel: two-sided Bonferroni over 14 fields at the same alpha (~3.38).
FIELD_DRIVER_Z: float = float(_norm.ppf(1.0 - OUTLIER_TAIL_PROBABILITY / (2 * 14)))
FIELD_NOTABLE_Z: float = 1.96              # Single-field two-sided 5% level: "somewhat unusual" wording in the math blocks

# Circular statistics for shadow_direction_deg (STAT-02)
SHADOW_DIRECTION_MIN_RELATIVE_STRENGTH: float = 0.25  # Directions from shadows weaker than this x group median strength are ignored
SHADOW_DIRECTION_MIN_RESULTANT_LENGTH: float = 0.3    # Group mean resultant length below this: no common direction, field not scored

# -------------------------------------------------------------------------
# Step 22: Spatial Clustering and Candidate Region Localization
# -------------------------------------------------------------------------
REGION_MERGE_MAX_GAP: int = 1              # Bridge single non-flagged hole gaps in run
REGION_MERGE_ENABLED: bool = True          # Toggle for merging runs separated by one unflagged hole
REGION_MIN_RUN_LENGTH: int = 2             # Flagged holes needed for a candidate region (a lone flag is not a cluster)
REGION_PITCH_GAP_FACTOR: float = 1.5       # Spacing above this x median pitch means a missing hole (counts as a gap)

# -------------------------------------------------------------------------
# Step 23: Final Output Generation
# -------------------------------------------------------------------------
OUTPUT_DISPLAY_GAMMA: float = 2.2          # Gamma applied to the annotated overview's drawing copy only
OUTPUT_MARKER_RADIUS_FRACTION: float = 0.005  # Hole marker radius as a fraction of image width (min 3 px)

# -------------------------------------------------------------------------
# Feature 26: Validation Mode ROC Cutoff Sweep
# -------------------------------------------------------------------------
# The sweep varies the tail-probability cutoff (scores are calibrated per group,
# so a single raw-score cutoff would mean different things on different edges).
VALIDATION_ROC_TAIL_PROBABILITIES: List[float] = [0.5, 0.2, 0.1, 0.05, 0.02, 0.01, 0.005, 0.002, 0.001, 0.0005, 0.0001]

# -------------------------------------------------------------------------
# Feature 29: Input-Quality Gating
# -------------------------------------------------------------------------
QUALITY_BACKLIT_SEPARATION_PASS: float = 0.80  # Otsu between-class / total variance at or above this: clear bimodal split
QUALITY_BACKLIT_SEPARATION_WARN: float = 0.60  # Between WARN and PASS: marginal, user must acknowledge; below: fail
QUALITY_TORCH_WARN_MULTIPLIER: float = 2.0     # Torch exposure spread up to this x BRIGHTNESS_MISMATCH_TOLERANCE warns; beyond fails

# -------------------------------------------------------------------------
# Feature 18: Suspicion Bands for Evidence-Framed Results Display
# -------------------------------------------------------------------------
# Bands are ranges of the hole's tail probability (lo, hi]: how often a genuine
# hole would score at least this high. 1 in 20 and 1 in 1000 are the edges, so
# the meaning is identical for every group size and for either method.
# Display labels are Low / Medium / High (Feature 18 avoids verdict language).
SUSPICION_BANDS: Dict[str, Tuple[float, float]] = {
    "genuine": (0.05, 1.0),
    "suspicious": (0.001, 0.05),
    "highly_suspicious": (0.0, 0.001)
}


def get_pipeline_constants_dict() -> Dict[str, Any]:
    """Return a full serializable snapshot of all pipeline constants for Feature 25."""
    return {
        "COMPOSITE_FIELD_ORDER": COMPOSITE_FIELD_ORDER,
        "CALIBRATION_PATTERN_SPACING_UM": CALIBRATION_PATTERN_SPACING_UM,
        "CALIBRATION_PATTERN_SIZE": list(CALIBRATION_PATTERN_SIZE),
        "CALIBRATION_DETECTION_GAMMA": CALIBRATION_DETECTION_GAMMA,
        "BALL_HOUGH_DP": BALL_HOUGH_DP,
        "BALL_HOUGH_MIN_DIST_PX": BALL_HOUGH_MIN_DIST_PX,
        "BALL_HOUGH_PARAM1": BALL_HOUGH_PARAM1,
        "BALL_HOUGH_PARAM2": BALL_HOUGH_PARAM2,
        "BALL_MIN_RADIUS_PX": BALL_MIN_RADIUS_PX,
        "REGISTRATION_MAX_SHIFT_PX": REGISTRATION_MAX_SHIFT_PX,
        "REGISTRATION_MIN_ECC_CORRELATION": REGISTRATION_MIN_ECC_CORRELATION,
        "REGISTRATION_ECC_MAX_ITERS": REGISTRATION_ECC_MAX_ITERS,
        "REGISTRATION_ECC_REFINE_ITERS": REGISTRATION_ECC_REFINE_ITERS,
        "REGISTRATION_ECC_EPS": REGISTRATION_ECC_EPS,
        "REGISTRATION_ECC_MAX_DIM": REGISTRATION_ECC_MAX_DIM,
        "REGISTRATION_ORB_MAX_FEATURES": REGISTRATION_ORB_MAX_FEATURES,
        "STAMP_ROTATION_WARN_DEG": STAMP_ROTATION_WARN_DEG,
        "STAMP_THRESHOLD_BLUR_PX": STAMP_THRESHOLD_BLUR_PX,
        "HOLE_THRESHOLD_BLUR_PX": HOLE_THRESHOLD_BLUR_PX,
        "REGISTRATION_RANSAC_REPROJ_PX": REGISTRATION_RANSAC_REPROJ_PX,
        "OUTPUT_MARKER_RADIUS_FRACTION": OUTPUT_MARKER_RADIUS_FRACTION,
        "HOLE_MIN_DIAMETER_MM": HOLE_MIN_DIAMETER_MM,
        "HOLE_MAX_DIAMETER_MM": HOLE_MAX_DIAMETER_MM,
        "HOLE_AREA_MIN_FRACTION": HOLE_AREA_MIN_FRACTION,
        "HOLE_AREA_MAX_FACTOR": HOLE_AREA_MAX_FACTOR,
        "HOLE_OVERSIZE_REVIEW_FACTOR": HOLE_OVERSIZE_REVIEW_FACTOR,
        "EDGE_TOOTH_MARGIN_PX": EDGE_TOOTH_MARGIN_PX,
        "EDGE_TOOTH_OUTLINE_FRACTION": EDGE_TOOTH_OUTLINE_FRACTION,
        "FOCUS_SHARPNESS_BLUR_PX": FOCUS_SHARPNESS_BLUR_PX,
        "FOCUS_PYRAMID_LEVELS": FOCUS_PYRAMID_LEVELS,
        "ROI_EXPECTED_HOLE_DIAMETER_UM": ROI_EXPECTED_HOLE_DIAMETER_UM,
        "ROI_CROP_SCALE_FACTOR": ROI_CROP_SCALE_FACTOR,
        "ROI_MIN_CROP_PX": ROI_MIN_CROP_PX,
        "RADIUS_PROFILE_NUM_ANGLES": RADIUS_PROFILE_NUM_ANGLES,
        "FOURIER_FIT_ORDER": FOURIER_FIT_ORDER,
        "BRIGHTNESS_MISMATCH_TOLERANCE": BRIGHTNESS_MISMATCH_TOLERANCE,
        "BRIGHTNESS_PATCH_SIZE_PX": BRIGHTNESS_PATCH_SIZE_PX,
        "CAMERA_MATCH_KEYS": CAMERA_MATCH_KEYS,
        "BOUNDARY_SAMPLE_SPACING_UM": BOUNDARY_SAMPLE_SPACING_UM,
        "BOUNDARY_PATCH_SIZE_PX": BOUNDARY_PATCH_SIZE_PX,
        "GLCM_LEVELS": GLCM_LEVELS,
        "GLCM_DISTANCES": GLCM_DISTANCES,
        "GLCM_ANGLES_DEG": GLCM_ANGLES_DEG,
        "PAPER_SAMPLE_OFFSET_UM": PAPER_SAMPLE_OFFSET_UM,
        "MIN_HOLES_PER_EDGE": MIN_HOLES_PER_EDGE,
        "MAD_SCALE_CONSTANT": MAD_SCALE_CONSTANT,
        "COMBINATION_METHOD": COMBINATION_METHOD,
        "SCALE_METHOD": SCALE_METHOD,
        "POOLED_SCALE_MIN_EDGE_SIZE": POOLED_SCALE_MIN_EDGE_SIZE,
        "MEAN_AD_SCALE_CONSTANT": MEAN_AD_SCALE_CONSTANT,
        "ZERO_SPREAD_RELATIVE_FLOOR": ZERO_SPREAD_RELATIVE_FLOOR,
        "OUTLIER_TAIL_PROBABILITY": OUTLIER_TAIL_PROBABILITY,
        "MIN_REFERENCE_GROUP_SIZE": MIN_REFERENCE_GROUP_SIZE,
        "NULL_CALIBRATION_DRAWS": NULL_CALIBRATION_DRAWS,
        "NULL_CALIBRATION_SEED": NULL_CALIBRATION_SEED,
        "OUTLIER_CUTOFF": OUTLIER_CUTOFF,
        "FIELD_DRIVER_Z": FIELD_DRIVER_Z,
        "FIELD_NOTABLE_Z": FIELD_NOTABLE_Z,
        "SHADOW_DIRECTION_MIN_RELATIVE_STRENGTH": SHADOW_DIRECTION_MIN_RELATIVE_STRENGTH,
        "SHADOW_DIRECTION_MIN_RESULTANT_LENGTH": SHADOW_DIRECTION_MIN_RESULTANT_LENGTH,
        "REGION_MERGE_MAX_GAP": REGION_MERGE_MAX_GAP,
        "REGION_MERGE_ENABLED": REGION_MERGE_ENABLED,
        "REGION_MIN_RUN_LENGTH": REGION_MIN_RUN_LENGTH,
        "REGION_PITCH_GAP_FACTOR": REGION_PITCH_GAP_FACTOR,
        "OUTPUT_DISPLAY_GAMMA": OUTPUT_DISPLAY_GAMMA,
        "VALIDATION_ROC_TAIL_PROBABILITIES": VALIDATION_ROC_TAIL_PROBABILITIES,
        "QUALITY_BACKLIT_SEPARATION_PASS": QUALITY_BACKLIT_SEPARATION_PASS,
        "QUALITY_BACKLIT_SEPARATION_WARN": QUALITY_BACKLIT_SEPARATION_WARN,
        "QUALITY_TORCH_WARN_MULTIPLIER": QUALITY_TORCH_WARN_MULTIPLIER,
        "SUSPICION_BANDS": {k: [v[0], v[1]] for k, v in SUSPICION_BANDS.items()}
    }
