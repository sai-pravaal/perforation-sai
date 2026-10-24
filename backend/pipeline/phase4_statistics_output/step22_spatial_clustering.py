"""Step 22: Spatial Clustering & Candidate Region Localization.

File: backend/pipeline/phase4_statistics_output/step22_spatial_clustering.py
Within each edge's ordered hole sequence, finds runs of consecutive flagged
holes (optionally bridging a single unflagged gap) and reports each as one
candidate suspect region, converting the run's position range into physical
coordinates for drawing.
"""

import json
import logging
from typing import Any, Dict, List, Optional

import numpy as np

from backend.config.pipeline_constants import (
    REGION_MERGE_ENABLED,
    REGION_MERGE_MAX_GAP,
    REGION_MIN_RUN_LENGTH,
    REGION_PITCH_GAP_FACTOR,
)
from backend.pipeline.phase2_geometry_classification.step06_stamp_boundary_orientation import StampFrame
from backend.pipeline.phase2_geometry_classification.step07_perforation_candidate_detection import HoleRecord
from backend.session.session_manager import SessionManager

logger = logging.getLogger(__name__)


def _find_runs(flagged_indices: List[int], max_gap: int, merge: bool) -> List[List[int]]:
    """Group ordered-position indices into runs, optionally bridging small gaps.

    Returns a list of [start_idx, end_idx] inclusive index ranges into the ordered list.
    """
    if not flagged_indices:
        return []
    flagged_indices = sorted(flagged_indices)
    runs: List[List[int]] = [[flagged_indices[0], flagged_indices[0]]]
    for idx in flagged_indices[1:]:
        gap = idx - runs[-1][1] - 1  # count of unflagged holes between
        if gap == 0 or (merge and gap <= max_gap):
            runs[-1][1] = idx
        else:
            runs.append([idx, idx])
    return runs


def _region_polygon(members: List[HoleRecord], edge: str, stamp_bbox: Optional[List[float]]) -> List[List[float]]:
    """Physical extent of a run: union of the members' outlines, extended to the stamp edge.

    Uses the Step 6 stamp box (orientation-normalised coordinates) so the
    region covers the stretch of that edge, not just a line through centroids.
    """
    x0 = min(h.bbox_px[0] for h in members)
    y0 = min(h.bbox_px[1] for h in members)
    x1 = max(h.bbox_px[0] + h.bbox_px[2] for h in members)
    y1 = max(h.bbox_px[1] + h.bbox_px[3] for h in members)
    if stamp_bbox and len(stamp_bbox) == 4:
        sx, sy, sw, sh = stamp_bbox
        if edge == "top":
            y0 = min(y0, sy)
        elif edge == "bottom":
            y1 = max(y1, sy + sh)
        elif edge == "left":
            x0 = min(x0, sx)
        elif edge == "right":
            x1 = max(x1, sx + sw)
    return [[x0, y0], [x1, y0], [x1, y1], [x0, y1]]


def _slot_numbers(ordered: List[HoleRecord], edge: str) -> List[int]:
    """Slot index per hole along the edge, counting skipped slots where a hole is missing.

    Consecutive holes normally sit one pitch apart; a gap of about two pitches
    means an undetected hole in between, which must count as an unflagged gap
    rather than make its neighbours look adjacent.
    """
    axis = 0 if edge in ("top", "bottom") else 1
    coords = [h.centroid_px[axis] for h in ordered]
    if len(coords) < 2:
        return list(range(len(coords)))
    diffs = np.diff(coords)
    pitch = float(np.median(diffs[diffs > 0])) if np.any(diffs > 0) else 0.0
    slots = [0]
    for d in diffs:
        step = int(round(d / pitch)) if pitch > 0 and d > REGION_PITCH_GAP_FACTOR * pitch else 1
        slots.append(slots[-1] + max(1, step))
    return slots


def build_candidate_regions(
    holes: List[HoleRecord],
    outlier_flags: Dict[str, Dict[str, Any]],
    stamp_bbox: Optional[List[float]] = None
) -> List[Dict[str, Any]]:
    """Build the CandidateRegion list from per-hole outlier flags.

    A run must contain at least REGION_MIN_RUN_LENGTH flagged holes: one
    isolated flag may be ordinary damage or noise; the spec treats a cluster
    along an edge as the meaningful evidence.
    """
    by_edge: Dict[str, List[HoleRecord]] = {}
    for h in holes:
        if h.edge is None or h.position_index is None:
            continue
        by_edge.setdefault(h.edge, []).append(h)

    regions: List[Dict[str, Any]] = []
    for edge in ("top", "bottom", "left", "right"):
        ordered = sorted(by_edge.get(edge, []), key=lambda h: h.position_index)
        if not ordered:
            continue
        slots = _slot_numbers(ordered, edge)
        slot_to_index = {s: i for i, s in enumerate(slots)}
        flagged_slots = [
            slots[i] for i, h in enumerate(ordered)
            if outlier_flags.get(h.hole_id, {}).get("is_outlier", False)
        ]
        runs = _find_runs(flagged_slots, REGION_MERGE_MAX_GAP, REGION_MERGE_ENABLED)

        n = 0
        for start_s, end_s in runs:
            members = [ordered[slot_to_index[s]] for s in range(start_s, end_s + 1) if s in slot_to_index]
            flagged_count = sum(1 for s in flagged_slots if start_s <= s <= end_s)
            if flagged_count < REGION_MIN_RUN_LENGTH:
                continue
            n += 1
            regions.append({
                "region_id": f"{edge}_region_{n}",
                "edge": edge,
                "hole_ids": [h.hole_id for h in members],
                "start_position": int(members[0].position_index),
                "end_position": int(members[-1].position_index),
                "region_coords_px": _region_polygon(members, edge, stamp_bbox)
            })
    return regions


def run_step22(
    session_id: str,
    holes: List[HoleRecord],
    outlier_flags: Dict[str, Dict[str, Any]],
    stamp_frame: StampFrame,
    session_manager: Optional[SessionManager] = None
) -> List[Dict[str, Any]]:
    """Execute Step 22: Spatial Clustering & Candidate Region Localization."""
    if session_manager is None:
        session_manager = SessionManager()

    session_dir = session_manager.get_session_dir(session_id)
    manifest = session_manager.get_manifest_manager(session_id)
    stages_dir = session_dir / "stages"
    stages_dir.mkdir(parents=True, exist_ok=True)

    regions = build_candidate_regions(holes, outlier_flags, stamp_frame.stamp_bbox if stamp_frame else None)

    with open(stages_dir / "candidate_regions.json", "w", encoding="utf-8") as f:
        json.dump(regions, f, indent=2)

    manifest.add_stage_entry(
        step_id=22,
        phase=4,
        stage_name="spatial_clustering",
        output_path="stages/candidate_regions.json",
        status="complete"
    )

    return regions
