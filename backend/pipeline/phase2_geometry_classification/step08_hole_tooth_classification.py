"""Step 8: Hole-Type Classification (Full Hole vs. Edge Tooth) & Edge/Row Assignment.

File: backend/pipeline/phase2_geometry_classification/step08_hole_tooth_classification.py
Determines whether candidate outline is a closed loop (interior_hole) or cut open (edge_tooth),
assigns each to nearest stamp edge (top, bottom, left, right), and sorts 1D along edge.
"""

import json
import logging
import math
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from backend.config.pipeline_constants import EDGE_TOOTH_MARGIN_PX, EDGE_TOOTH_OUTLINE_FRACTION
from backend.pipeline.phase2_geometry_classification.step06_stamp_boundary_orientation import StampFrame
from backend.pipeline.phase2_geometry_classification.step07_perforation_candidate_detection import HoleRecord
from backend.session.manifest import ManifestManager
from backend.session.session_manager import SessionManager

logger = logging.getLogger(__name__)


def classify_hole_type(
    hole_outline: List[List[float]],
    stamp_boundary: List[List[float]],
    margin_px: float = EDGE_TOOTH_MARGIN_PX
) -> str:
    """Classify candidate as 'edge_tooth' or 'interior_hole' based on boundary proximity."""
    stamp_pts = np.array(stamp_boundary, dtype=np.float32).reshape((-1, 1, 2))
    pts_near_edge = 0

    for pt in hole_outline:
        dist = cv2.pointPolygonTest(stamp_pts, (float(pt[0]), float(pt[1])), measureDist=True)
        # Inward distance is positive, outward is negative, on border is 0
        if abs(dist) < margin_px:
            pts_near_edge += 1

    # If substantial portion touches the boundary, it is an edge tooth
    fraction = pts_near_edge / max(1, len(hole_outline))
    return "edge_tooth" if fraction > EDGE_TOOTH_OUTLINE_FRACTION else "interior_hole"


def assign_edge_and_position(
    holes: List[HoleRecord],
    stamp_bbox: List[float]
) -> List[HoleRecord]:
    """Assign candidates to top, bottom, left, or right edge and sort 1D sequentially."""
    sx, sy, sw, sh = stamp_bbox
    left_x = sx
    right_x = sx + sw
    top_y = sy
    bottom_y = sy + sh

    edge_groups: Dict[str, List[HoleRecord]] = {
        "top": [],
        "bottom": [],
        "left": [],
        "right": []
    }
    interior: List[HoleRecord] = []

    for h in holes:
        if h.type == "interior_hole":
            # Interior holes (blocks/panes) belong to no edge (HoleRecord allows
            # edge=None); they are compared among themselves in Step 19.
            h.edge = None
            interior.append(h)
            continue
        cx, cy = h.centroid_px
        d_top = abs(cy - top_y)
        d_bottom = abs(cy - bottom_y)
        d_left = abs(cx - left_x)
        d_right = abs(cx - right_x)

        min_dist = min(d_top, d_bottom, d_left, d_right)
        if min_dist == d_top:
            h.edge = "top"
            edge_groups["top"].append(h)
        elif min_dist == d_bottom:
            h.edge = "bottom"
            edge_groups["bottom"].append(h)
        elif min_dist == d_left:
            h.edge = "left"
            edge_groups["left"].append(h)
        else:
            h.edge = "right"
            edge_groups["right"].append(h)

    # Sort each edge sequentially along its 1D line direction
    # Top and bottom: left to right (increasing X)
    edge_groups["top"].sort(key=lambda h: h.centroid_px[0])
    edge_groups["bottom"].sort(key=lambda h: h.centroid_px[0])
    # Left and right: top to bottom (increasing Y)
    edge_groups["left"].sort(key=lambda h: h.centroid_px[1])
    edge_groups["right"].sort(key=lambda h: h.centroid_px[1])

    final_holes: List[HoleRecord] = []
    for edge_name, group in edge_groups.items():
        for idx, h in enumerate(group, 1):
            h.position_index = idx
            h.hole_id = f"{edge_name}_{idx:02d}"
            final_holes.append(h)

    # Interior holes: stable ids in reading order (top-to-bottom, then left-to-right).
    # position_index stays None because there is no edge to count along.
    interior.sort(key=lambda h: (round(h.centroid_px[1]), h.centroid_px[0]))
    for idx, h in enumerate(interior, 1):
        h.position_index = None
        h.hole_id = f"interior_{idx:02d}"
        final_holes.append(h)

    return final_holes


def run_step08(
    session_id: str,
    candidates: List[HoleRecord],
    stamp_frame: StampFrame,
    session_manager: Optional[SessionManager] = None
) -> List[HoleRecord]:
    """Execute Step 8: Hole-Type Classification & Edge Assignment."""
    if session_manager is None:
        session_manager = SessionManager()

    session_dir = session_manager.get_session_dir(session_id)
    manifest = session_manager.get_manifest_manager(session_id)
    stages_dir = session_dir / "stages"
    stages_dir.mkdir(parents=True, exist_ok=True)

    # 1. Classify type (edge_tooth vs interior_hole)
    for h in candidates:
        h.type = classify_hole_type(h.raw_outline_px, stamp_frame.boundary_polygon)

    # 2. Assign edge and sequential 1D position_index
    classified_holes = assign_edge_and_position(candidates, stamp_frame.stamp_bbox)

    # 3. Persist finalized holes.json
    out_file = stages_dir / "holes.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump([h.to_dict() for h in classified_holes], f, indent=2)

    # 4. Update manifest holes_total
    manifest.set_holes_total(len(classified_holes))
    manifest.add_stage_entry(
        step_id=8,
        phase=2,
        stage_name="hole_tooth_classification",
        output_path="stages/holes.json",
        status="complete"
    )

    return classified_holes
