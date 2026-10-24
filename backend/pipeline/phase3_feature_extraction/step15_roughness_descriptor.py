"""Step 15: Roughness Descriptor Computation (Feature 2 output).

File: backend/pipeline/phase3_feature_extraction/step15_roughness_descriptor.py
Measures edge bumpiness from the Step 14 boundary patches using Sobel gradient
statistics and GLCM texture properties, plus a geometric residual-wobble term
(raw radius minus the Step 13 Fourier fit), producing the RoughnessDescriptor.
"""

import json
import logging
import math
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from skimage.feature import graycomatrix, graycoprops
from skimage.filters import sobel

from backend.config.pipeline_constants import (
    FOURIER_FIT_ORDER,
    GLCM_ANGLES_DEG,
    GLCM_DISTANCES,
    GLCM_LEVELS,
)
from backend.pipeline.phase1_calibration_registration.step02_lens_and_unit_calibration import CalibrationRecord
from backend.pipeline.phase2_geometry_classification.step07_perforation_candidate_detection import HoleRecord
from backend.pipeline.phase3_feature_extraction.step13_shape_deviation_descriptor import (
    _merge_descriptor,
    evaluate_fourier_series,
    fit_fourier_series,
)
from backend.session.session_manager import SessionManager

logger = logging.getLogger(__name__)


def normalize_patch(patch: np.ndarray) -> np.ndarray:
    """Divide a patch by its own mean so texture measures ignore how brightly it was lit.

    Raking-light intensity varies across the field and between holes; without
    this, a brighter hole would read as "rougher" purely from illumination.
    """
    p = patch.astype(np.float64)
    mean = float(np.mean(p))
    return p / mean if mean > 1e-9 else p


def patch_sobel_stats(patch: np.ndarray) -> tuple:
    """Return (mean, std) of the Sobel edge-magnitude map for one (normalised) patch."""
    mag = sobel(normalize_patch(patch))
    return float(np.mean(mag)), float(np.std(mag))


def patch_glcm_props(patch: np.ndarray) -> tuple:
    """Return (contrast, homogeneity) from the patch GLCM.

    The mean-normalised patch is mapped so its mean sits at mid-scale
    (0..2x mean -> 0..GLCM_LEVELS-1), then quantised to GLCM_LEVELS levels.
    """
    quant = np.clip(normalize_patch(patch) / 2.0, 0.0, 1.0)
    quant = np.rint(quant * (GLCM_LEVELS - 1)).astype(np.uint8)
    angles_rad = [math.radians(a) for a in GLCM_ANGLES_DEG]
    glcm = graycomatrix(
        quant,
        distances=GLCM_DISTANCES,
        angles=angles_rad,
        levels=GLCM_LEVELS,
        symmetric=True,
        normed=True,
    )
    contrast = float(np.mean(graycoprops(glcm, "contrast")))
    homogeneity = float(np.mean(graycoprops(glcm, "homogeneity")))
    return contrast, homogeneity


def compute_residual_wobble(radius_profile: Dict[str, Any]) -> Optional[float]:
    """Std of (raw radius - fitted Fourier curve) over valid angle steps, in pixels
    (run_step15 converts it to micrometres).

    Returns None when there is too little outline data to fit (not measured).
    """
    angles = radius_profile.get("angles_deg", [])
    distances = radius_profile.get("distances_px", [])
    valid = radius_profile.get("valid_mask", [])

    coeffs = fit_fourier_series(angles, distances, valid, FOURIER_FIT_ORDER)
    if coeffs is None:
        return None

    valid_angles = [a for a, v in zip(angles, valid) if v]
    raw = np.array([d for d, v in zip(distances, valid) if v and d is not None], dtype=np.float64)
    if raw.size == 0:
        return None
    fitted = evaluate_fourier_series(coeffs, valid_angles, FOURIER_FIT_ORDER)
    return float(np.std(raw - fitted))


def _microns_per_pixel(session_dir: Path, calibration_record: Optional[CalibrationRecord]) -> float:
    """Session scale from the in-memory record, else from stages/calibration.json."""
    if calibration_record is not None:
        return float(calibration_record.microns_per_pixel)
    path = session_dir / "stages" / "calibration.json"
    with open(path, "r", encoding="utf-8") as f:
        return float(json.load(f)["microns_per_pixel"])


def run_step15(
    session_id: str,
    holes: List[HoleRecord],
    patches: Dict[str, List[np.ndarray]],
    radius_profiles: Dict[str, Dict[str, Any]],
    session_manager: Optional[SessionManager] = None,
    calibration_record: Optional[CalibrationRecord] = None
) -> Dict[str, Dict[str, Any]]:
    """Execute Step 15: Roughness Descriptor Computation."""
    if session_manager is None:
        session_manager = SessionManager()

    session_dir = session_manager.get_session_dir(session_id)
    manifest = session_manager.get_manifest_manager(session_id)
    holes_dir = session_dir / "holes"
    upp = _microns_per_pixel(session_dir, calibration_record)

    roughness_descriptors: Dict[str, Dict[str, Any]] = {}

    for h in holes:
        hole_patches = patches.get(h.hole_id, [])
        sobel_means, sobel_stds, glcm_contrasts, glcm_homs = [], [], [], []

        for patch in hole_patches:
            sm, ss = patch_sobel_stats(patch)
            gc, gh = patch_glcm_props(patch)
            sobel_means.append(sm)
            sobel_stds.append(ss)
            glcm_contrasts.append(gc)
            glcm_homs.append(gh)

        wobble_px = compute_residual_wobble(radius_profiles.get(h.hole_id, {}))
        wobble_um = wobble_px * upp if wobble_px is not None else None
        # None marks "no boundary patches", so Step 20 omits the field rather
        # than comparing a fabricated zero roughness.
        descriptor = {
            "sobel_mean": float(np.mean(sobel_means)) if sobel_means else None,
            "sobel_std": float(np.mean(sobel_stds)) if sobel_stds else None,
            "glcm_contrast": float(np.mean(glcm_contrasts)) if glcm_contrasts else None,
            "glcm_homogeneity": float(np.mean(glcm_homs)) if glcm_homs else None,
            "residual_wobble": wobble_um,
        }
        roughness_descriptors[h.hole_id] = descriptor

        h_dir = holes_dir / h.hole_id
        h_dir.mkdir(parents=True, exist_ok=True)
        _merge_descriptor(h_dir / "descriptors.json", "roughness", descriptor)

    manifest.add_stage_entry(
        step_id=15,
        phase=3,
        stage_name="roughness_descriptor",
        output_path="holes/",
        status="complete"
    )

    return roughness_descriptors
