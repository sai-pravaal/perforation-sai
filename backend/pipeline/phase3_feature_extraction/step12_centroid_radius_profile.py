"""Step 12: Centroid Estimation & Radius-Profile Conversion (Feature 1).

File: backend/pipeline/phase3_feature_extraction/step12_centroid_radius_profile.py
Computes a single center point per hole (area moments for interior holes, a
best-fit circle for edge teeth) and converts the subpixel outline into a fixed
72-step radius profile, explicitly marking angle steps with no real data.
"""

import json
import logging
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
from scipy.optimize import least_squares

from backend.config.pipeline_constants import RADIUS_PROFILE_NUM_ANGLES
from backend.pipeline.phase2_geometry_classification.step07_perforation_candidate_detection import HoleRecord
from backend.session.session_manager import SessionManager

logger = logging.getLogger(__name__)


def _moments_center(outline: np.ndarray) -> Tuple[float, float]:
    """Compute center of a closed outline via image moments (cx=M10/M00, cy=M01/M00)."""
    cnt = outline.reshape((-1, 1, 2)).astype(np.float32)
    m = cv2.moments(cnt)
    if abs(m["m00"]) < 1e-9:
        return float(np.mean(outline[:, 0])), float(np.mean(outline[:, 1]))
    return float(m["m10"] / m["m00"]), float(m["m01"] / m["m00"])


def _kasa_circle_fit(xs: np.ndarray, ys: np.ndarray) -> Tuple[float, float, float]:
    """Algebraic (Kasa) circle fit returning (cx, cy, r) as an initial estimate."""
    a = np.column_stack([xs, ys, np.ones_like(xs)])
    b = xs ** 2 + ys ** 2
    sol, *_ = np.linalg.lstsq(a, b, rcond=None)
    cx = sol[0] / 2.0
    cy = sol[1] / 2.0
    r = math.sqrt(max(0.0, sol[2] + cx ** 2 + cy ** 2))
    return float(cx), float(cy), float(r)


def fit_circle_center(outline: np.ndarray) -> Tuple[float, float]:
    """Fit a circle to arc points (Kasa init refined by least squares); returns (cx, cy)."""
    xs = outline[:, 0]
    ys = outline[:, 1]
    cx0, cy0, r0 = _kasa_circle_fit(xs, ys)

    def residuals(p):
        cx, cy, r = p
        return np.hypot(xs - cx, ys - cy) - r

    try:
        res = least_squares(residuals, x0=[cx0, cy0, r0])
        return float(res.x[0]), float(res.x[1])
    except Exception:
        return cx0, cy0


def _ray_outline_distance(pts: np.ndarray, cx: float, cy: float, theta: float) -> Optional[float]:
    """Distance from the centre to where the ray at angle theta crosses the outline.

    The crossing is interpolated along the polyline segment it hits, so the
    radius belongs to exactly this angle (no nearest-vertex angle error).
    Closed outlines from marching squares repeat their first point, so their
    segment list is already closed; an open arc (tooth) only yields a value
    where the ray actually meets the visible boundary, and None ("no data")
    otherwise. With several crossings the nearest is used.
    """
    p0 = pts[:-1]
    p1 = pts[1:]
    d = np.array([math.cos(theta), math.sin(theta)])
    seg = p1 - p0
    rel = p0 - np.array([cx, cy])
    # Solve c + t*d = p0 + s*seg for t (along ray) and s (along segment).
    denom = d[0] * seg[:, 1] - d[1] * seg[:, 0]
    ok = np.abs(denom) > 1e-12
    t = np.where(ok, (rel[:, 0] * seg[:, 1] - rel[:, 1] * seg[:, 0]) / np.where(ok, denom, 1.0), np.nan)
    s = np.where(ok, (rel[:, 0] * d[1] - rel[:, 1] * d[0]) / np.where(ok, denom, 1.0), np.nan)
    hit = ok & (t > 0) & (s >= 0) & (s <= 1)
    if not np.any(hit):
        return None
    return float(np.min(t[hit]))


def build_radius_profile(
    outline: List[List[float]],
    hole_type: Optional[str]
) -> Dict[str, Any]:
    """Convert a subpixel outline into a fixed-angle RadiusProfile dict."""
    num_angles = RADIUS_PROFILE_NUM_ANGLES
    step_deg = 360.0 / num_angles
    angles_deg = [round(i * step_deg, 4) for i in range(num_angles)]

    if not outline or len(outline) < 3:
        return {
            "center_px": [0.0, 0.0],
            "angles_deg": angles_deg,
            "distances_px": [None] * num_angles,
            "valid_mask": [False] * num_angles
        }

    pts = np.asarray(outline, dtype=np.float64)
    if hole_type == "edge_tooth":
        cx, cy = fit_circle_center(pts)
    else:
        cx, cy = _moments_center(pts)

    distances: List[Optional[float]] = []
    valid: List[bool] = []
    for a in angles_deg:
        r = _ray_outline_distance(pts, cx, cy, math.radians(a))
        distances.append(r)
        valid.append(r is not None)

    return {
        "center_px": [float(cx), float(cy)],
        "angles_deg": angles_deg,
        "distances_px": distances,
        "valid_mask": valid
    }


def run_step12(
    session_id: str,
    holes: List[HoleRecord],
    outlines: Dict[str, List[List[float]]],
    session_manager: Optional[SessionManager] = None
) -> Dict[str, Dict[str, Any]]:
    """Execute Step 12: Centroid Estimation & Radius-Profile Conversion."""
    if session_manager is None:
        session_manager = SessionManager()

    session_dir = session_manager.get_session_dir(session_id)
    manifest = session_manager.get_manifest_manager(session_id)
    holes_dir = session_dir / "holes"

    profiles: Dict[str, Dict[str, Any]] = {}

    for h in holes:
        outline = outlines.get(h.hole_id, [])
        profile = build_radius_profile(outline, h.type)
        profiles[h.hole_id] = profile

        h_dir = holes_dir / h.hole_id
        h_dir.mkdir(parents=True, exist_ok=True)
        out_path = h_dir / "radius_profile.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(profile, f, indent=2)

    manifest.add_stage_entry(
        step_id=12,
        phase=3,
        stage_name="centroid_radius_profile",
        output_path="holes/",
        status="complete"
    )

    return profiles
