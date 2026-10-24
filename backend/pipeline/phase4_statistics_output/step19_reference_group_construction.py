"""Step 19: Reference-Group Construction (Leave-One-Out, Per Edge/Row).

File: backend/pipeline/phase4_statistics_output/step19_reference_group_construction.py
For every hole, builds the list of peer holes it will be compared against:
normally every other hole on the same edge (leave-one-out), falling back to
every other hole on the whole specimen when an edge has too few holes,
tagging such holes low_confidence.
"""

import logging
from typing import Any, Dict, List, Optional

from backend.config.pipeline_constants import MIN_HOLES_PER_EDGE
from backend.pipeline.phase2_geometry_classification.step07_perforation_candidate_detection import HoleRecord
from backend.session.session_manager import SessionManager

logger = logging.getLogger(__name__)


def build_reference_groups(holes: List[HoleRecord]) -> Dict[str, Dict[str, Any]]:
    """Return a ReferenceGroup dict per hole. Pure function of type and edge assignments.

    Teeth and interior holes are never mixed: a tooth's centre comes from a
    circle fit to a partial arc and it has no roundness ratio, so its
    descriptors are measured differently from a closed interior hole's (spec Step 8).
    - Teeth: every other tooth on the same edge; if that edge has fewer than
      MIN_HOLES_PER_EDGE teeth, every other tooth on the specimen ("pooled",
      low_confidence).
    - Interior holes: one pooled group of every other interior hole
      ("interior"), low_confidence when it has fewer than MIN_HOLES_PER_EDGE members.
    """
    teeth = [h for h in holes if h.type != "interior_hole"]
    interior_ids = [h.hole_id for h in holes if h.type == "interior_hole"]
    teeth_ids = [h.hole_id for h in teeth]

    edge_members: Dict[str, List[str]] = {}
    for h in teeth:
        edge_members.setdefault(h.edge, []).append(h.hole_id)

    groups: Dict[str, Dict[str, Any]] = {}
    for h in holes:
        if h.type == "interior_hole":
            comparison = [hid for hid in interior_ids if hid != h.hole_id]
            stratification = "interior"
            low_confidence = len(interior_ids) < MIN_HOLES_PER_EDGE
        else:
            same_edge = edge_members.get(h.edge, []) if h.edge is not None else []
            if h.edge is not None and len(same_edge) >= MIN_HOLES_PER_EDGE:
                comparison = [hid for hid in same_edge if hid != h.hole_id]
                stratification = "edge"
                low_confidence = False
            else:
                comparison = [hid for hid in teeth_ids if hid != h.hole_id]
                stratification = "pooled"
                low_confidence = True

        groups[h.hole_id] = {
            "hole_id": h.hole_id,
            "comparison_group_ids": comparison,
            "stratification": stratification,
            "low_confidence": low_confidence
        }

    return groups


def run_step19(
    session_id: str,
    holes: List[HoleRecord],
    session_manager: Optional[SessionManager] = None
) -> Dict[str, Dict[str, Any]]:
    """Execute Step 19: Reference-Group Construction.

    Not persisted as a separate file (pure function of edge assignments); the
    low_confidence holes are recorded in manifest.json.
    """
    if session_manager is None:
        session_manager = SessionManager()

    manifest = session_manager.get_manifest_manager(session_id)
    groups = build_reference_groups(holes)

    low_conf = [hid for hid, g in groups.items() if g["low_confidence"]]
    manifest.set_low_confidence_holes(low_conf)

    manifest.add_stage_entry(
        step_id=19,
        phase=4,
        stage_name="reference_group_construction",
        output_path="manifest.json",
        status="complete"
    )

    return groups
