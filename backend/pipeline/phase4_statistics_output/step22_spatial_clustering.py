"""Step 22: Spatial Clustering & Candidate Region Localization.

File: backend/pipeline/phase4_statistics_output/step22_spatial_clustering.py
Within each edge's ordered hole sequence, finds runs of consecutive flagged
holes (optionally bridging a single unflagged gap) and reports each as one
candidate suspect region, converting the run's position range into physical
coordinates for drawing.

It also checks every edge as a whole for perforation consistency (see
edge_consistency): when a whole edge was reperforated, every tooth on it is
altered, so no single tooth stands out against its own edge, but the teeth
are no longer alike (one line, one spacing, one size) as a perforating machine
leaves them. The check uses only the edge's own teeth, as percentages of the
edge's own typical values. Such an edge becomes a candidate region of its own
(basis "edge_consistency").
"""

import json
import logging
from typing import Any, Dict, List, Optional

import numpy as np

from backend.config.pipeline_constants import (
    COMPOSITE_FIELD_ORDER,
    EDGE_CONSISTENCY_MIN_TEETH,
    EDGE_VARIATION_MIN_EXCEEDED,
    EDGE_VARIATION_THRESHOLDS_PCT,
    REGION_MERGE_ENABLED,
    REGION_MERGE_MAX_GAP,
    REGION_MIN_RUN_LENGTH,
    REGION_PITCH_GAP_FACTOR,
)
from backend.pipeline.phase2_geometry_classification.step06_stamp_boundary_orientation import StampFrame
from backend.pipeline.phase2_geometry_classification.step07_perforation_candidate_detection import HoleRecord
from backend.pipeline.phase2_geometry_classification.step09_roi_extraction import compute_crop_size_px
from backend.pipeline.phase1_calibration_registration.step02_lens_and_unit_calibration import CalibrationRecord
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


def _mad(values: np.ndarray) -> float:
    values = values[np.isfinite(values)]
    return float(np.median(np.abs(values - np.median(values)))) if values.size else float("nan")


# Spec descriptor fields whose tooth-to-tooth variation is expressed as a
# percentage of the edge's typical hole radius (they are lengths, and the
# shape terms centre on zero, so a percentage of their own median is undefined).
_RADIUS_SCALED_FIELDS = ["radius_std_um", "fourier_c1", "fourier_c2", "fourier_c3", "fourier_c4", "residual_wobble"]
# Fields expressed as a percentage of their own typical (median) value.
_SELF_SCALED_FIELDS = ["sobel_mean", "sobel_std", "glcm_contrast", "glcm_homogeneity", "shadow_strength"]


def edge_consistency(teeth: List[Dict[str, Any]], edge: str, microns_per_pixel: float) -> Optional[Dict[str, Any]]:
    """How consistent one edge's teeth are with each other, in percent (self-referential).

    A perforating machine punches every hole of a row on one straight line, at
    one spacing, with one size and shape, so on a genuine edge the teeth are
    alike; holes re-punched by hand are not. Each measure is the typical
    tooth-to-tooth variation on this edge (median absolute deviation),
    expressed as a percentage of the edge's own typical value. Nothing is
    compared with any other stamp:
      spacing:     centre-to-centre distance, % of the edge's pitch (gaps over
                   REGION_PITCH_GAP_FACTOR x pitch are missing holes, left out)
      notch_depth: hole centre's distance from the edge's fitted perforation
                   line, % of pitch (deeper/shallower notches)
      hole_size:   mean radius (Step 13), % of the edge's median radius
      spec descriptors: the radius spread, four shape terms and wobble as %
                   of the median radius; Sobel/GLCM roughness and shadow
                   strength as % of their own median; shadow direction as
                   degrees (variation_deg).
    An edge is flagged when at least EDGE_VARIATION_MIN_EXCEEDED of the
    measures listed in EDGE_VARIATION_THRESHOLDS_PCT exceed their threshold.
    """
    if len(teeth) < EDGE_CONSISTENCY_MIN_TEETH:
        return None
    centres = np.array([t["centre_px"] for t in teeth], dtype=np.float64)
    along, across = (centres[:, 0], centres[:, 1]) if edge in ("top", "bottom") else (centres[:, 1], centres[:, 0])
    order = np.argsort(along)
    along, across = along[order], across[order]
    line = np.polyfit(along, across, 1)
    offsets_um = (across - np.polyval(line, along)) * microns_per_pixel
    gaps_um = np.diff(along) * microns_per_pixel
    pitch_um = float(np.median(gaps_um))
    if pitch_um <= 0:
        return None
    gaps_um = gaps_um[gaps_um <= REGION_PITCH_GAP_FACTOR * pitch_um]

    vectors = np.array([[np.nan if v is None else v for v in t["vector"]] for t in teeth], dtype=np.float64)
    column = {name: vectors[:, COMPOSITE_FIELD_ORDER.index(name)] for name in COMPOSITE_FIELD_ORDER}
    radius_um = float(np.nanmedian(column["mean_radius_um"]))

    def pct(spread: float, scale: float) -> float:
        return float(100.0 * spread / scale) if np.isfinite(spread) and np.isfinite(scale) and abs(scale) > 1e-12 \
            else float("nan")

    variation = {
        "spacing": pct(_mad(gaps_um), pitch_um),
        "notch_depth": pct(_mad(offsets_um), pitch_um),
        "hole_size": pct(_mad(column["mean_radius_um"]), radius_um),
    }
    for name in _RADIUS_SCALED_FIELDS:
        variation[name] = pct(_mad(column[name]), radius_um)
    for name in _SELF_SCALED_FIELDS:
        variation[name] = pct(_mad(column[name]), abs(float(np.nanmedian(column[name]))))
    directions = column["shadow_direction_deg"]
    directions = directions[np.isfinite(directions)]
    direction_deg = float("nan")
    if directions.size:
        centre = float(np.degrees(np.arctan2(np.mean(np.sin(np.radians(directions))),
                                             np.mean(np.cos(np.radians(directions))))))
        direction_deg = float(np.median(np.abs((directions - centre + 180.0) % 360.0 - 180.0)))

    exceeded = [m for m, limit in EDGE_VARIATION_THRESHOLDS_PCT.items()
                if np.isfinite(variation.get(m, float("nan"))) and variation[m] > limit]
    return {
        "edge": edge,
        "teeth": len(teeth),
        "pitch_um": pitch_um,
        "radius_um": radius_um,
        "variation_pct": variation,
        "variation_deg": {"shadow_direction_deg": direction_deg},
        "thresholds_pct": dict(EDGE_VARIATION_THRESHOLDS_PCT),
        "exceeded": exceeded,
        "inconsistent": len(exceeded) >= EDGE_VARIATION_MIN_EXCEEDED,
    }


def _edge_tooth_inputs(session_dir, holes: List[HoleRecord], crop_size_px: int) -> Dict[str, List[Dict[str, Any]]]:
    """Per edge, each tooth's circle centre (frame px) and composite vector (Steps 12/18)."""
    out: Dict[str, List[Dict[str, Any]]] = {}
    for h in sorted(holes, key=lambda x: (str(x.edge), x.position_index or 0)):
        if h.type != "edge_tooth" or h.edge is None:
            continue
        hdir = session_dir / "holes" / h.hole_id
        try:
            with open(hdir / "radius_profile.json", "r", encoding="utf-8") as f:
                profile = json.load(f)
            with open(hdir / "descriptors.json", "r", encoding="utf-8") as f:
                vector = json.load(f)["composite"]["vector"]
        except (OSError, KeyError, json.JSONDecodeError):
            continue
        if not any(d is not None for d in profile.get("distances_px", [])):
            continue
        # Crop pixel (u, v) is frame (u + cx - (S-1)/2, v + cy - (S-1)/2) (Step 9).
        half = (crop_size_px - 1) / 2.0
        cx = h.centroid_px[0] + profile["center_px"][0] - half
        cy = h.centroid_px[1] + profile["center_px"][1] - half
        out.setdefault(h.edge, []).append({"centre_px": [cx, cy], "vector": vector, "hole_id": h.hole_id})
    return out


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

    stamp_bbox = stamp_frame.stamp_bbox if stamp_frame else None
    regions = build_candidate_regions(holes, outlier_flags, stamp_bbox)
    for region in regions:
        region["basis"] = "tooth_outliers"

    # Whole-edge consistency (needs the session scale and the Step 9 crop size).
    calib_path = stages_dir / "calibration.json"
    calib = json.load(open(calib_path, "r", encoding="utf-8")) if calib_path.exists() else None
    record = CalibrationRecord(calib.get("distortion_coeffs", [0.0, 0.0, 0.0]), calib["microns_per_pixel"]) if calib else None
    teeth_by_edge = _edge_tooth_inputs(session_dir, holes, compute_crop_size_px(record)) if record else {}
    by_id = {h.hole_id: h for h in holes}
    report = []
    for edge in ("top", "bottom", "left", "right"):
        result = edge_consistency(teeth_by_edge.get(edge, []), edge, record.microns_per_pixel) if record else None
        if result is None:
            continue
        report.append(result)
        if result["inconsistent"]:
            members = [by_id[t["hole_id"]] for t in teeth_by_edge[edge]]
            regions.append({
                "region_id": f"{edge}_edge_consistency",
                "edge": edge,
                "hole_ids": [m.hole_id for m in members],
                "start_position": int(members[0].position_index),
                "end_position": int(members[-1].position_index),
                "region_coords_px": _region_polygon(members, edge, stamp_bbox),
                "basis": "edge_consistency",
                "exceeded": result["exceeded"],
            })
    # Tooth-level regions are reported only on edges that fail the consistency
    # check (decision 2026-10-26): on the genuine reference stamps the
    # per-tooth outliers produced regions on every capture from natural tear
    # variation. Edges the check could not assess (too few teeth) keep theirs.
    # Known cost: a few re-punched teeth on an otherwise consistent edge are
    # no longer reported. Hidden regions stay listed in edge_consistency.json.
    consistent_edges = {r["edge"] for r in report if not r["inconsistent"]}
    for result in report:
        result["hidden_tooth_regions"] = [
            {k: g[k] for k in ("hole_ids", "start_position", "end_position")}
            for g in regions if g["basis"] == "tooth_outliers" and g["edge"] == result["edge"]
            and result["edge"] in consistent_edges
        ]
    regions = [g for g in regions if not (g["basis"] == "tooth_outliers" and g["edge"] in consistent_edges)]
    with open(stages_dir / "edge_consistency.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

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
