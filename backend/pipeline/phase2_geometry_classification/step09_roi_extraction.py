"""Step 9: Per-Hole Region-of-Interest Extraction.

File: backend/pipeline/phase2_geometry_classification/step09_roi_extraction.py
Extracts subpixel-centered square crops for each classified hole from every
registered frame (diffuse, torch_1..torch_4, backlit), sized consistently in physical microns.
"""

import json
import logging
import math
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import numpy as np

from backend.config.pipeline_constants import (
    ROI_CROP_SCALE_FACTOR,
    ROI_EXPECTED_HOLE_DIAMETER_UM,
    ROI_MIN_CROP_PX,
)
from backend.pipeline.phase0_ingestion.step01_raw_capture_ingestion import RawFrameRecord
from backend.pipeline.phase1_calibration_registration.step02_lens_and_unit_calibration import CalibrationRecord
from backend.pipeline.phase2_geometry_classification.step07_perforation_candidate_detection import HoleRecord
from backend.session.manifest import ManifestManager
from backend.session.session_manager import SessionManager

logger = logging.getLogger(__name__)


def save_linear_png(path: Path, image: np.ndarray) -> None:
    """Persist a linear [0, 1] float plane as a 16-bit PNG (keeps resume precision)."""
    cv2.imwrite(str(path), (np.clip(image, 0.0, 1.0) * 65535.0).round().astype(np.uint16))


def load_linear_png(path: Path) -> Optional[np.ndarray]:
    """Read a PNG written by save_linear_png (or a legacy 8-bit one) back to float32 [0, 1]."""
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        return None
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    scale = 65535.0 if img.dtype == np.uint16 else 255.0
    return img.astype(np.float32) / scale


def extract_subpixel_crop(
    image: np.ndarray,
    center_px: List[float],
    crop_size_px: int
) -> np.ndarray:
    """Extract square crop centered at (cx, cy) using bilinear interpolation."""
    cx, cy = float(center_px[0]), float(center_px[1])
    crop = cv2.getRectSubPix(image, (crop_size_px, crop_size_px), (cx, cy))
    return crop.astype(np.float32)


def run_step09(
    session_id: str,
    holes: List[HoleRecord],
    frames: Dict[str, RawFrameRecord],
    calibration_record: CalibrationRecord,
    session_manager: Optional[SessionManager] = None,
    expected_hole_diameter_um: float = ROI_EXPECTED_HOLE_DIAMETER_UM,
    crop_scale_factor: float = ROI_CROP_SCALE_FACTOR
) -> Dict[str, Dict[str, np.ndarray]]:
    """Execute Step 9: Per-Hole Region-of-Interest Extraction."""
    if session_manager is None:
        session_manager = SessionManager()

    session_dir = session_manager.get_session_dir(session_id)
    manifest = session_manager.get_manifest_manager(session_id)
    holes_dir = session_dir / "holes"
    holes_dir.mkdir(parents=True, exist_ok=True)

    # Compute crop size in pixels (e.g. 3x hole diameter in microns / microns_per_pixel)
    crop_size_um = expected_hole_diameter_um * crop_scale_factor
    crop_size_px = int(round(crop_size_um / max(1e-3, calibration_record.microns_per_pixel)))
    # Ensure odd dimension for symmetric center. Only a lower bound is applied;
    # an upper clamp would break the "same physical size in every session" rule
    # at fine pixel scales (a 1.2 mm hole no longer fits at <= ~5 um/px).
    crop_size_px = max(ROI_MIN_CROP_PX, crop_size_px)
    if crop_size_px % 2 == 0:
        crop_size_px += 1

    frame_keys = [
        "diffuse", "torch_1", "torch_2", "torch_3", "torch_4", "backlit"
    ]
    available_frames = {k: frames[k].brightness_array for k in frame_keys if k in frames}

    all_crops: Dict[str, Dict[str, np.ndarray]] = {}
    image_entries: List[Dict[str, Any]] = []

    for h in holes:
        h_dir = holes_dir / h.hole_id / "crops"
        h_dir.mkdir(parents=True, exist_ok=True)
        hole_crops: Dict[str, np.ndarray] = {}

        for tag, arr in available_frames.items():
            crop = extract_subpixel_crop(arr, h.centroid_px, crop_size_px)
            hole_crops[tag] = crop

            # Save 16-bit linear crop to disk (reloaded verbatim on resume)
            save_linear_png(h_dir / f"{tag}.png", crop)
            image_entries.append({
                "path": f"holes/{h.hole_id}/crops/{tag}.png",
                "stage": "roi_crop",
                "hole_id": h.hole_id,
                "edge": h.edge,
            })

        all_crops[h.hole_id] = hole_crops

    # Feature 6/10: every crop is a displayable, filterable gallery item.
    if image_entries:
        manifest.add_image_entries(image_entries)

    manifest.add_stage_entry(
        step_id=9,
        phase=2,
        stage_name="roi_extraction",
        output_path="holes/",
        status="complete"
    )

    return all_crops
