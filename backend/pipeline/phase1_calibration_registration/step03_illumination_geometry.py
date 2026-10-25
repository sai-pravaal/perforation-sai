"""Step 3: Illumination Geometry Determination.

File: backend/pipeline/phase1_calibration_registration/step03_illumination_geometry.py
Computes final azimuth and elevation angles for the 4 torches via Path A
(arithmetic rig geometry + correction) or Path B (ball-bearing mirror glint reflection).
"""

import json
import logging
import math
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from backend.config.pipeline_constants import (
    BALL_HOUGH_DP,
    BALL_HOUGH_MIN_DIST_PX,
    BALL_HOUGH_PARAM1,
    BALL_HOUGH_PARAM2,
    BALL_MIN_RADIUS_PX,
)
from backend.pipeline.phase0_ingestion.step01_raw_capture_ingestion import RawFrameRecord
from backend.pipeline.phase1_calibration_registration.step02_lens_and_unit_calibration import _to_detection_u8
from backend.pipeline.phase2_geometry_classification.step06_stamp_boundary_orientation import detect_stamp_orientation
from backend.session.manifest import ManifestManager
from backend.session.session_manager import SessionManager

logger = logging.getLogger(__name__)


# Azimuth convention shared by Path A and Path B (and consumed by Step 17):
# degrees in the stamp frame, 0 along the stamp's row direction (+x after Step 6
# straightening), increasing counter-clockwise as seen from the camera, in [0, 360).
# Path A azimuths are protractor readings set against the stamp's row, so they are
# already in this frame. Path B is measured in raw image coordinates (y down,
# before Step 6), so it is converted here.


def _normalize_azimuth(deg: float) -> float:
    return float(deg) % 360.0


def solve_ball_bearing_reflection(
    diffuse_bb: np.ndarray,
    torch_bb: np.ndarray
) -> Optional[Tuple[float, float]]:
    """Solve 3D light direction via chrome ball bearing reflection geometry (Path B).

    Formula: Light Ray L = 2 * (V . N) * N - V
    where V is viewing ray [0, 0, 1] and N is sphere surface normal at glint.
    Returns (azimuth, elevation) with azimuth counter-clockwise in *image*
    axes (y flipped to point up); the caller rotates it into the stamp frame.
    """
    diffuse_u8 = _to_detection_u8(diffuse_bb)
    torch_u8 = (np.clip(torch_bb, 0.0, 1.0) * 255.0).astype(np.uint8)

    # Detect circular ball boundary
    circles = cv2.HoughCircles(
        diffuse_u8, cv2.HOUGH_GRADIENT, dp=BALL_HOUGH_DP, minDist=BALL_HOUGH_MIN_DIST_PX,
        param1=BALL_HOUGH_PARAM1, param2=BALL_HOUGH_PARAM2, minRadius=BALL_MIN_RADIUS_PX,
        maxRadius=int(min(diffuse_u8.shape) / 2)
    )
    if circles is None:
        # Fallback: thresholding + minEnclosingCircle
        _, thresh = cv2.threshold(diffuse_u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        cnts, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            return None
        largest = max(cnts, key=cv2.contourArea)
        (cx, cy), radius = cv2.minEnclosingCircle(largest)
    else:
        c = circles[0][0]
        cx, cy, radius = float(c[0]), float(c[1]), float(c[2])

    if radius <= 0:
        return None

    # Find brightest glint in torch image within the ball circle
    mask = np.zeros_like(torch_u8)
    cv2.circle(mask, (int(cx), int(cy)), int(radius), 255, -1)
    masked_torch = cv2.bitwise_and(torch_u8, torch_u8, mask=mask)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(masked_torch)
    gx, gy = max_loc

    # Compute normalized glint position relative to ball center
    dx = (gx - cx) / radius
    dy = (gy - cy) / radius
    r_sq = dx * dx + dy * dy
    if r_sq > 1.0:
        return None

    dz = math.sqrt(1.0 - r_sq)
    # Surface normal N at glint
    nx, ny, nz = dx, dy, dz

    # Viewing vector V = [0, 0, 1] pointing to camera
    # L = 2 * (V . N) * N - V = 2 * nz * [nx, ny, nz] - [0, 0, 1]
    lx = 2.0 * nz * nx
    ly = 2.0 * nz * ny
    lz = 2.0 * nz * nz - 1.0

    # Convert L direction to azimuth and elevation in degrees. Image y points
    # down, so it is negated to get a counter-clockwise (y-up) azimuth.
    azimuth = math.degrees(math.atan2(-ly, lx)) % 360.0
    norm_xy = math.sqrt(lx * lx + ly * ly)
    elevation = math.degrees(math.atan2(lz, norm_xy))

    return round(azimuth, 2), round(elevation, 2)


def run_step03(
    session_id: str,
    frames: Dict[str, RawFrameRecord],
    session_manager: Optional[SessionManager] = None
) -> List[Dict[str, Any]]:
    """Execute Step 3: Illumination Geometry Determination."""
    if session_manager is None:
        session_manager = SessionManager()

    session_dir = session_manager.get_session_dir(session_id)
    manifest = session_manager.get_manifest_manager(session_id)
    session_log = session_manager.load_session_log(session_id) or {}
    ball_bearing_used = session_log.get("ball_bearing_used", False)

    illumination_geometry: List[Dict[str, Any]] = []

    # Stamp tilt in the raw frame (the same measurement Step 6 later cancels),
    # used to rotate Path B's image-frame azimuths into the stamp frame.
    stamp_tilt_deg = 0.0
    if ball_bearing_used and ("backlit" in frames or "diffuse" in frames):
        outline_tag = "backlit" if "backlit" in frames else "diffuse"  # same source as Step 6
        stamp_tilt_deg, _, _ = detect_stamp_orientation(frames[outline_tag].brightness_array,
                                                        outline_tag == "backlit")

    for i in range(1, 5):
        tag = f"torch_{i}"
        frame = frames.get(tag)
        meta = frame.torch_meta if frame else None

        # Path A calculation (always available; required inputs, no silent defaults)
        if not meta:
            raise ValueError(f"No session-log geometry for {tag}; enter its height, distance and azimuth.")
        missing = [k for k in ("height_cm", "distance_cm", "azimuth_deg") if meta.get(k) is None]
        if missing:
            raise ValueError(f"Session log for {tag} is missing {', '.join(missing)}.")
        h = float(meta["height_cm"])
        d = float(meta["distance_cm"])
        corr = float(meta.get("manual_correction_deg", 0.0) or 0.0)
        az = _normalize_azimuth(float(meta["azimuth_deg"]))
        elev = SessionManager.compute_elevation(h, d, corr)

        source_path = "calculated"
        final_az = az
        final_elev = elev

        # Path B check
        if ball_bearing_used:
            bb_diff_frame = frames.get("ball_bearing_diffuse")
            bb_torch_frame = frames.get(f"ball_bearing_torch_{i}")
            bb_sol = None
            if bb_diff_frame is not None and bb_torch_frame is not None:
                bb_sol = solve_ball_bearing_reflection(
                    bb_diff_frame.brightness_array,
                    bb_torch_frame.brightness_array
                )
            if bb_sol is not None:
                image_az, final_elev = bb_sol
                # A stamp tilted clockwise by t shows a stamp-frame azimuth a at
                # image azimuth a - t, so a = image azimuth + t.
                final_az = _normalize_azimuth(image_az + stamp_tilt_deg)
                source_path = "ball_bearing"
            else:
                logger.warning(
                    "Step 3: ball bearing marked as used but Path B failed for %s; "
                    "using the calculated (Path A) direction instead", tag
                )

        illumination_geometry.append({
            "torch_index": i,
            "azimuth_deg": round(final_az, 2),
            "elevation_deg": round(final_elev, 2),
            "source_path": source_path
        })

    # Persist illumination geometry
    stages_dir = session_dir / "stages"
    stages_dir.mkdir(parents=True, exist_ok=True)
    out_path = stages_dir / "illumination_geometry.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(illumination_geometry, f, indent=2)

    manifest.add_stage_entry(
        step_id=3,
        phase=1,
        stage_name="illumination_geometry",
        output_path="stages/illumination_geometry.json",
        status="complete"
    )

    return illumination_geometry
