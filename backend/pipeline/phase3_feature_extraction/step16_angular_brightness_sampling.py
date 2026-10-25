"""Step 16: Angular Brightness Sampling Per Torch Direction (Feature 3).

File: backend/pipeline/phase3_feature_extraction/step16_angular_brightness_sampling.py
At each valid angle step from the Step 12 grid, samples paper brightness just
outside the hole edge in each of the four brightness-matched torch crops, using
bilinear interpolation, producing one table per hole (rows=valid angles,
columns=torches).
"""

import json
import logging
import math
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import numpy as np

from backend.config.pipeline_constants import PAPER_SAMPLE_BAND_STEPS, PAPER_SAMPLE_BAND_UM
from backend.pipeline.phase1_calibration_registration.step02_lens_and_unit_calibration import CalibrationRecord
from backend.pipeline.phase2_geometry_classification.step07_perforation_candidate_detection import HoleRecord
from backend.session.session_manager import SessionManager

logger = logging.getLogger(__name__)

TORCH_TAGS = ["torch_1", "torch_2", "torch_3", "torch_4"]


def sample_bilinear(image: np.ndarray, x: float, y: float) -> float:
    """Bilinear-interpolated brightness at fractional (x, y)."""
    val = cv2.getRectSubPix(image.astype(np.float32), (1, 1), (float(x), float(y)))
    return float(val[0, 0])


def run_step16(
    session_id: str,
    holes: List[HoleRecord],
    radius_profiles: Dict[str, Dict[str, Any]],
    crops: Dict[str, Dict[str, np.ndarray]],
    calibration_record: CalibrationRecord,
    session_manager: Optional[SessionManager] = None
) -> Dict[str, Dict[str, Any]]:
    """Execute Step 16: Angular Brightness Sampling."""
    if session_manager is None:
        session_manager = SessionManager()

    session_dir = session_manager.get_session_dir(session_id)
    manifest = session_manager.get_manifest_manager(session_id)
    holes_dir = session_dir / "holes"

    upp = max(1e-6, calibration_record.microns_per_pixel)
    offsets_px = np.linspace(PAPER_SAMPLE_BAND_UM[0], PAPER_SAMPLE_BAND_UM[1], PAPER_SAMPLE_BAND_STEPS) / upp

    tables: Dict[str, Dict[str, Any]] = {}

    for h in holes:
        profile = radius_profiles.get(h.hole_id, {})
        center = profile.get("center_px", [0.0, 0.0])
        angles = profile.get("angles_deg", [])
        distances = profile.get("distances_px", [])
        valid = profile.get("valid_mask", [])

        hole_crops = crops.get(h.hole_id, {})
        torch_imgs = {t: hole_crops[t] for t in TORCH_TAGS if t in hole_crops}
        present_torches = [t for t in TORCH_TAGS if t in torch_imgs]

        rows: List[Dict[str, Any]] = []
        for a, d, v in zip(angles, distances, valid):
            if not v or d is None or not present_torches:
                continue
            rad = math.radians(a)
            pts = [(center[0] + (d + o) * math.cos(rad), center[1] + (d + o) * math.sin(rad)) for o in offsets_px]
            brightness = [float(np.mean([sample_bilinear(torch_imgs[t], px, py) for px, py in pts]))
                          for t in present_torches]
            rows.append({"angle_deg": float(a), "brightness": brightness})

        table = {"torch_order": present_torches, "rows": rows}
        tables[h.hole_id] = table

        h_dir = holes_dir / h.hole_id
        h_dir.mkdir(parents=True, exist_ok=True)
        with open(h_dir / "angular_brightness_table.json", "w", encoding="utf-8") as f:
            json.dump(table, f, indent=2)

    manifest.add_stage_entry(
        step_id=16,
        phase=3,
        stage_name="angular_brightness_sampling",
        output_path="holes/",
        status="complete"
    )

    return tables
