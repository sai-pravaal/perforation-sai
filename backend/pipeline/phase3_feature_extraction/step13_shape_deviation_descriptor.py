"""Step 13: Shape-Deviation Descriptor Computation (Feature 1 output).

File: backend/pipeline/phase3_feature_extraction/step13_shape_deviation_descriptor.py
Fits a low-order Fourier series (harmonics 1-4) to the valid radius-vs-angle
measurements and derives mean_radius_um, radius_std_um, fourier_coeffs, and
roundness_ratio (interior holes only) as the hole's ShapeDescriptor.
"""

import json
import logging
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from backend.config.pipeline_constants import FOURIER_FIT_ORDER
from backend.pipeline.phase1_calibration_registration.step02_lens_and_unit_calibration import CalibrationRecord
from backend.pipeline.phase2_geometry_classification.step07_perforation_candidate_detection import HoleRecord
from backend.session.session_manager import SessionManager

logger = logging.getLogger(__name__)


def _merge_descriptor(descriptors_path: Path, key: str, payload: Dict[str, Any]) -> None:
    """Read-modify-write a single key into the per-hole descriptors.json (never wholesale)."""
    data: Dict[str, Any] = {}
    if descriptors_path.exists():
        try:
            with open(descriptors_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError):
            data = {}
    data[key] = payload
    with open(descriptors_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def fit_fourier_series(
    angles_deg: List[float],
    distances_px: List[Optional[float]],
    valid_mask: List[bool],
    order: int
) -> Optional[np.ndarray]:
    """Fit distance(theta) = a0 + sum_k[ak cos(k th) + bk sin(k th)] over valid samples.

    Returns the full coefficient vector [a0, a1, b1, ..., a_order, b_order],
    or None when there are too few valid samples for a stable fit.
    """
    theta = np.radians([a for a, v in zip(angles_deg, valid_mask) if v])
    r = np.array([d for d, v in zip(distances_px, valid_mask) if v and d is not None], dtype=np.float64)
    if r.size < (2 * order + 1):
        return None

    cols = [np.ones_like(theta)]
    for k in range(1, order + 1):
        cols.append(np.cos(k * theta))
        cols.append(np.sin(k * theta))
    design = np.column_stack(cols)
    coeffs, *_ = np.linalg.lstsq(design, r, rcond=None)
    return coeffs


def evaluate_fourier_series(coeffs: np.ndarray, angles_deg: List[float], order: int) -> np.ndarray:
    """Reconstruct fitted radius values at the given angles from a coefficient vector."""
    theta = np.radians(np.asarray(angles_deg, dtype=np.float64))
    values = np.full_like(theta, coeffs[0])
    for k in range(1, order + 1):
        ak = coeffs[1 + 2 * (k - 1)]
        bk = coeffs[2 + 2 * (k - 1)]
        values = values + ak * np.cos(k * theta) + bk * np.sin(k * theta)
    return values


def fit_fourier_magnitudes(
    angles_deg: List[float],
    distances_px: List[Optional[float]],
    valid_mask: List[bool],
    order: int
) -> List[Optional[float]]:
    """Return the magnitude sqrt(ak^2 + bk^2) for harmonics 1..order (None if unfittable).

    None (not 0.0) marks "not measured" so Step 20 omits it instead of treating
    it as a genuine zero-magnitude harmonic.
    """
    coeffs = fit_fourier_series(angles_deg, distances_px, valid_mask, order)
    if coeffs is None:
        return [None] * order
    magnitudes: List[float] = []
    for k in range(1, order + 1):
        ak = coeffs[1 + 2 * (k - 1)]
        bk = coeffs[2 + 2 * (k - 1)]
        magnitudes.append(float(math.hypot(ak, bk)))
    return magnitudes


def _arc_parameter(angles_deg: np.ndarray) -> np.ndarray:
    """Map the angles of one contiguous arc to t in [-1, 1] (wrap-safe, centred on the arc)."""
    a = np.radians(np.asarray(angles_deg, dtype=np.float64))
    mid = math.atan2(float(np.mean(np.sin(a))), float(np.mean(np.cos(a))))
    delta = np.angle(np.exp(1j * (a - mid)))
    half = float(np.max(np.abs(delta)))
    return delta / half if half > 1e-9 else np.zeros_like(delta)


def fit_shape_model(
    angles_deg: List[float],
    distances_px: List[Optional[float]],
    valid_mask: List[bool],
    order: int,
    partial_arc: bool
) -> Optional[Tuple[List[float], np.ndarray, np.ndarray]]:
    """Fit the smooth radius model and return (terms 1..order, raw radii, fitted radii).

    Terms are harmonic magnitudes for full outlines and signed Legendre
    coefficients for partial arcs.

    Full outlines: Fourier harmonics (fit_fourier_series).
    Partial arcs (edge teeth): Legendre polynomials in the arc parameter t.
    A Fourier fit over the ~150 degrees a tooth shows is ill-conditioned:
    the harmonics are far from orthogonal there and radius noise is
    amplified up to ~500x in the harmonic magnitudes (0.17x on a full
    circle), so tooth "harmonics" were mostly noise. Legendre polynomials are
    orthogonal on an interval, so their terms 1..4 (tilt, flattening,
    3rd/4th-order polygonal deviation) are the stable arc counterpart.
    """
    pairs = [(a, d) for a, d, v in zip(angles_deg, distances_px, valid_mask) if v and d is not None]
    if len(pairs) < 2 * order + 1:
        return None
    angles = np.array([p[0] for p in pairs], dtype=np.float64)
    radii = np.array([p[1] for p in pairs], dtype=np.float64)
    if partial_arc:
        t = _arc_parameter(angles)
        coeffs = np.polynomial.legendre.legfit(t, radii, order)
        # Signed, not magnitudes: every tooth on an edge has its arc oriented
        # the same way, so a term's sign is meaningful (e.g. which side the
        # arc tilts to), and |c| near zero is a folded, one-sided distribution
        # whose tail the median/MAD z of Step 20 overstates.
        return [float(c) for c in coeffs[1:order + 1]], radii, np.polynomial.legendre.legval(t, coeffs)
    coeffs = fit_fourier_series(angles_deg, distances_px, valid_mask, order)
    if coeffs is None:
        return None
    mags = [float(math.hypot(coeffs[1 + 2 * (k - 1)], coeffs[2 + 2 * (k - 1)])) for k in range(1, order + 1)]
    return mags, radii, evaluate_fourier_series(coeffs, angles.tolist(), order)


def compute_roundness_ratio(outline: List[List[float]]) -> Optional[float]:
    """Compute 4*pi*area / perimeter^2 from a closed outline (None if not computable)."""
    if not outline or len(outline) < 3:
        return None
    cnt = np.asarray(outline, dtype=np.float32).reshape((-1, 1, 2))
    area = float(cv2.contourArea(cnt))
    perimeter = float(cv2.arcLength(cnt, True))
    if perimeter <= 1e-9:
        return None
    return (4.0 * math.pi * area) / (perimeter ** 2)


def run_step13(
    session_id: str,
    holes: List[HoleRecord],
    radius_profiles: Dict[str, Dict[str, Any]],
    outlines: Dict[str, List[List[float]]],
    calibration_record: CalibrationRecord,
    session_manager: Optional[SessionManager] = None
) -> Dict[str, Dict[str, Any]]:
    """Execute Step 13: Shape-Deviation Descriptor Computation."""
    if session_manager is None:
        session_manager = SessionManager()

    session_dir = session_manager.get_session_dir(session_id)
    manifest = session_manager.get_manifest_manager(session_id)
    holes_dir = session_dir / "holes"
    upp = calibration_record.microns_per_pixel

    shape_descriptors: Dict[str, Dict[str, Any]] = {}

    for h in holes:
        profile = radius_profiles.get(h.hole_id, {})
        angles = profile.get("angles_deg", [])
        distances = profile.get("distances_px", [])
        valid = profile.get("valid_mask", [])

        valid_r = [d for d, v in zip(distances, valid) if v and d is not None]
        if valid_r:
            mean_radius_um = float(np.mean(valid_r)) * upp
            radius_std_um = float(np.std(valid_r)) * upp
        else:
            # No valid outline samples: record "not measured", not a zero radius.
            mean_radius_um = None
            radius_std_um = None

        # Term magnitudes are radius amplitudes, so they get the same
        # pixel -> micrometre conversion as the radius (comparable across sessions).
        # Teeth (partial arcs) use Legendre terms in the same four slots; they
        # are only ever compared with other teeth (Step 19).
        model = fit_shape_model(angles, distances, valid, FOURIER_FIT_ORDER, h.type == "edge_tooth")
        fourier_coeffs = [m * upp for m in model[0]] if model else [None] * FOURIER_FIT_ORDER

        if h.type == "interior_hole":
            roundness = compute_roundness_ratio(outlines.get(h.hole_id, []))
        else:
            roundness = None

        descriptor = {
            "mean_radius_um": mean_radius_um,
            "radius_std_um": radius_std_um,
            "fourier_coeffs": fourier_coeffs,
            "roundness_ratio": roundness
        }
        shape_descriptors[h.hole_id] = descriptor

        h_dir = holes_dir / h.hole_id
        h_dir.mkdir(parents=True, exist_ok=True)
        _merge_descriptor(h_dir / "descriptors.json", "shape", descriptor)

    manifest.add_stage_entry(
        step_id=13,
        phase=3,
        stage_name="shape_deviation_descriptor",
        output_path="holes/",
        status="complete"
    )

    return shape_descriptors
