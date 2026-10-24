"""Step 7: Perforation Candidate Detection.

File: backend/pipeline/phase2_geometry_classification/step07_perforation_candidate_detection.py
Thresholds the backlit transillumination frame, finds connected bright blobs,
filters by physical perforation hole dimensions (via CalibrationRecord),
and generates preliminary HoleRecord candidates.
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
    HOLE_AREA_MAX_FACTOR,
    HOLE_AREA_MIN_FRACTION,
    HOLE_MAX_DIAMETER_MM,
    HOLE_MIN_DIAMETER_MM,
    HOLE_OVERSIZE_REVIEW_FACTOR,
    HOLE_THRESHOLD_BLUR_PX,
)
from backend.pipeline.phase0_ingestion.step01_raw_capture_ingestion import RawFrameRecord
from backend.pipeline.phase1_calibration_registration.step02_lens_and_unit_calibration import CalibrationRecord
from backend.session.manifest import ManifestManager
from backend.session.session_manager import SessionManager

logger = logging.getLogger(__name__)


class HoleRecord:
    """Represents a candidate perforation hole or edge tooth."""

    def __init__(
        self,
        hole_id: str,
        centroid_px: List[float],
        bbox_px: List[float],
        raw_outline_px: List[List[float]],
        hole_type: Optional[str] = None,
        edge: Optional[str] = None,
        position_index: Optional[int] = None
    ) -> None:
        self.hole_id = hole_id
        self.centroid_px = [float(centroid_px[0]), float(centroid_px[1])]
        self.bbox_px = [float(v) for v in bbox_px]  # [x, y, w, h]
        self.raw_outline_px = [[float(pt[0]), float(pt[1])] for pt in raw_outline_px]
        self.type = hole_type  # 'interior_hole' or 'edge_tooth'
        self.edge = edge  # 'top', 'bottom', 'left', 'right'
        self.position_index = int(position_index) if position_index is not None else None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "hole_id": self.hole_id,
            "type": self.type,
            "edge": self.edge,
            "position_index": self.position_index,
            "centroid_px": self.centroid_px,
            "bbox_px": self.bbox_px,
            "raw_outline_px": self.raw_outline_px
        }


def detect_perforation_candidates(
    backlit_image: np.ndarray,
    microns_per_pixel: float,
    min_diameter_mm: float = HOLE_MIN_DIAMETER_MM,
    max_diameter_mm: float = HOLE_MAX_DIAMETER_MM,
    oversize_out: Optional[List[Dict[str, Any]]] = None
) -> List[Dict[str, Any]]:
    """Find bright hole candidates in backlit frame filtered by physical area.

    Blobs larger than the plausible range but within HOLE_OVERSIZE_REVIEW_FACTOR
    of it (likely two merged holes or a paper flaw) are appended to oversize_out
    for manual review instead of being silently dropped (spec Step 7).
    """
    img_u8 = (np.clip(backlit_image, 0.0, 1.0) * 255.0).astype(np.uint8)

    # Otsu thresholding: bright holes vs dark paper
    blur = cv2.GaussianBlur(img_u8, (HOLE_THRESHOLD_BLUR_PX, HOLE_THRESHOLD_BLUR_PX), 0)
    _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Connected components
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(thresh, connectivity=8)

    # Convert physical diameter range to expected pixel area range
    # Area = pi * (d / 2)^2
    r_min_um = (min_diameter_mm * 1000.0) / 2.0
    r_max_um = (max_diameter_mm * 1000.0) / 2.0
    r_min_px = r_min_um / microns_per_pixel
    r_max_px = r_max_um / microns_per_pixel
    area_min_px = math.pi * (r_min_px ** 2) * HOLE_AREA_MIN_FRACTION  # teeth show only part of a circle
    area_max_px = math.pi * (r_max_px ** 2) * HOLE_AREA_MAX_FACTOR    # margin for slightly merged outlines

    candidates = []
    for label in range(1, num_labels):
        area = stats[label, cv2.CC_STAT_AREA]
        if area_max_px < area <= area_max_px * HOLE_OVERSIZE_REVIEW_FACTOR and oversize_out is not None:
            oversize_out.append({
                "centroid_px": [float(centroids[label][0]), float(centroids[label][1])],
                "bbox_px": [float(stats[label, cv2.CC_STAT_LEFT]), float(stats[label, cv2.CC_STAT_TOP]),
                            float(stats[label, cv2.CC_STAT_WIDTH]), float(stats[label, cv2.CC_STAT_HEIGHT])],
                "area_px": int(area),
                "reason": "larger than a single perforation (possible merged holes or paper flaw)",
            })
            continue
        if area_min_px <= area <= area_max_px:
            x = float(stats[label, cv2.CC_STAT_LEFT])
            y = float(stats[label, cv2.CC_STAT_TOP])
            w = float(stats[label, cv2.CC_STAT_WIDTH])
            h = float(stats[label, cv2.CC_STAT_HEIGHT])
            cx, cy = float(centroids[label][0]), float(centroids[label][1])

            # Extract contour of this component
            comp_mask = np.uint8(labels == label) * 255
            cnts, _ = cv2.findContours(comp_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            poly = cnts[0].reshape(-1, 2).tolist() if cnts else [[x, y], [x + w, y], [x + w, y + h], [x, y + h]]

            candidates.append({
                "centroid_px": [cx, cy],
                "bbox_px": [x, y, w, h],
                "raw_outline_px": poly,
                "area": area
            })

    return candidates


def run_step07(
    session_id: str,
    frames: Dict[str, RawFrameRecord],
    calibration_record: CalibrationRecord,
    session_manager: Optional[SessionManager] = None
) -> List[HoleRecord]:
    """Execute Step 7: Perforation Candidate Detection."""
    if session_manager is None:
        session_manager = SessionManager()

    session_dir = session_manager.get_session_dir(session_id)
    manifest = session_manager.get_manifest_manager(session_id)
    stages_dir = session_dir / "stages"
    stages_dir.mkdir(parents=True, exist_ok=True)

    if "backlit" not in frames:
        raise KeyError("Missing required 'backlit' frame for Step 7 candidate detection")

    backlit_frame = frames["backlit"]
    oversize: List[Dict[str, Any]] = []
    raw_candidates = detect_perforation_candidates(
        backlit_frame.brightness_array,
        microns_per_pixel=calibration_record.microns_per_pixel,
        oversize_out=oversize
    )
    # Over-sized blobs are kept for a manual look (spec: "flagged for a manual
    # look before being fully trusted"), not passed on as holes.
    with open(stages_dir / "oversize_candidates.json", "w", encoding="utf-8") as f:
        json.dump(oversize, f, indent=2)
    if oversize:
        logger.warning("Step 7: %d over-sized blob(s) flagged for manual review in stages/oversize_candidates.json",
                       len(oversize))

    hole_records: List[HoleRecord] = []
    for i, c in enumerate(raw_candidates, 1):
        temp_id = f"candidate_{i:03d}"
        hr = HoleRecord(
            hole_id=temp_id,
            centroid_px=c["centroid_px"],
            bbox_px=c["bbox_px"],
            raw_outline_px=c["raw_outline_px"]
        )
        hole_records.append(hr)

    # Persist preliminary holes
    out_file = stages_dir / "holes.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump([h.to_dict() for h in hole_records], f, indent=2)

    manifest.add_stage_entry(
        step_id=7,
        phase=2,
        stage_name="perforation_candidate_detection",
        output_path="stages/holes.json",
        status="complete"
    )

    return hole_records
