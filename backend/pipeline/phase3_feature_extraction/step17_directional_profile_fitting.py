"""Step 17: Directional Profile Fitting (Feature 3 output - Shadow).

File: backend/pipeline/phase3_feature_extraction/step17_directional_profile_fitting.py
Per boundary point, fits brightness across the known torch azimuths to a single
cosine I(phi) = A + B*cos(phi - psi) via linear least squares, then combines all
points by circular vector averaging into one hole-level shadow direction and
strength (ShadowDescriptor).
"""

import json
import logging
import math
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from backend.pipeline.phase2_geometry_classification.step07_perforation_candidate_detection import HoleRecord
from backend.pipeline.phase3_feature_extraction.step13_shape_deviation_descriptor import _merge_descriptor
from backend.session.session_manager import SessionManager

logger = logging.getLogger(__name__)


def fit_point_cosine(phis_rad: np.ndarray, brightness: np.ndarray):
    """Solve I = A + u*cos(phi) + v*sin(phi); return (u, v, B) for one boundary point."""
    design = np.column_stack([np.ones_like(phis_rad), np.cos(phis_rad), np.sin(phis_rad)])
    sol, *_ = np.linalg.lstsq(design, brightness, rcond=None)
    _a, u, v = float(sol[0]), float(sol[1]), float(sol[2])
    b = math.hypot(u, v)
    return u, v, b


def run_step17(
    session_id: str,
    holes: List[HoleRecord],
    angular_tables: Dict[str, Dict[str, Any]],
    illumination_geometry: List[Dict[str, Any]],
    session_manager: Optional[SessionManager] = None
) -> Dict[str, Dict[str, Any]]:
    """Execute Step 17: Directional Profile Fitting."""
    if session_manager is None:
        session_manager = SessionManager()

    session_dir = session_manager.get_session_dir(session_id)
    manifest = session_manager.get_manifest_manager(session_id)
    holes_dir = session_dir / "holes"

    # Map torch tag -> azimuth (degrees) from IlluminationGeometry. Only the
    # azimuth enters the fit: the spec's model I(phi) = A + B*cos(phi - psi) is
    # a function of torch azimuth alone, so Step 3's elevation is recorded for
    # traceability but intentionally not used here.
    azimuth_by_tag: Dict[str, float] = {}
    for entry in illumination_geometry:
        idx = entry.get("torch_index")
        if idx is not None and entry.get("azimuth_deg") is not None:
            azimuth_by_tag[f"torch_{idx}"] = float(entry["azimuth_deg"])

    shadow_descriptors: Dict[str, Dict[str, Any]] = {}

    for h in holes:
        table = angular_tables.get(h.hole_id, {})
        torch_order = table.get("torch_order", [])
        rows = table.get("rows", [])
        missing = [t for t in torch_order if t not in azimuth_by_tag]
        if missing:
            # A defaulted azimuth would make the cos/sin design matrix singular
            # and the fitted direction meaningless, so stop instead of guessing.
            raise ValueError(f"Step 17: illumination geometry has no azimuth for {', '.join(missing)}")
        phis = np.radians([azimuth_by_tag[t] for t in torch_order])

        sum_u, sum_v = 0.0, 0.0
        b_values: List[float] = []
        for row in rows:
            brightness = np.asarray(row.get("brightness", []), dtype=np.float64)
            if brightness.size < 3 or brightness.size != phis.size:
                continue
            u, v, b = fit_point_cosine(phis, brightness)
            sum_u += u
            sum_v += v
            b_values.append(b)

        if b_values:
            strength: Optional[float] = float(math.hypot(sum_u, sum_v))
            direction_deg: Optional[float] = float(math.degrees(math.atan2(sum_v, sum_u)) % 360.0)
            conf = {"mean": float(np.mean(b_values)), "min": float(np.min(b_values)), "max": float(np.max(b_values))}
        else:
            # No usable boundary rows: shadow not measured (None, not 0 strength at 0 deg).
            strength = None
            direction_deg = None
            conf = {"mean": None, "min": None, "max": None}

        descriptor = {
            "strength": strength,
            "direction_deg": direction_deg,
            "per_point_confidence_summary": conf
        }
        shadow_descriptors[h.hole_id] = descriptor

        h_dir = holes_dir / h.hole_id
        h_dir.mkdir(parents=True, exist_ok=True)
        _merge_descriptor(h_dir / "descriptors.json", "shadow", descriptor)

    manifest.add_stage_entry(
        step_id=17,
        phase=3,
        stage_name="directional_profile_fitting",
        output_path="holes/",
        status="complete"
    )

    return shadow_descriptors
