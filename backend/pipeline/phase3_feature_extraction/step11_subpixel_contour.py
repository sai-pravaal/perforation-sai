"""Step 11: Subpixel Contour Extraction (Feature 1 - Shape).

File: backend/pipeline/phase3_feature_extraction/step11_subpixel_contour.py
Traces each hole's boundary in its backlit crop at the per-hole midpoint
brightness threshold using marching squares (skimage.measure.find_contours),
producing a smooth ordered polyline whose points sit at fractional pixel
coordinates.
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import numpy as np
from skimage import measure
from skimage.filters import threshold_otsu

from backend.config.pipeline_constants import OUTPUT_DISPLAY_GAMMA
from backend.pipeline.phase2_geometry_classification.step07_perforation_candidate_detection import HoleRecord
from backend.session.session_manager import SessionManager

logger = logging.getLogger(__name__)


def compute_midpoint_threshold(crop: np.ndarray) -> float:
    """Derive the brightness midpoint between the bright hole plateau and dark paper plateau.

    The crop's own histogram is split into two classes with Otsu's method and
    each plateau is that class's median, so the level does not depend on how
    much of the crop the hole covers (a tooth covers well under 5%, which made
    fixed percentiles land both "plateaus" on paper).
    """
    finite = crop[np.isfinite(crop)]
    if finite.size == 0 or float(finite.max()) <= float(finite.min()):
        return float(finite.mean()) if finite.size else 0.5
    split = float(threshold_otsu(finite))
    dark = finite[finite <= split]
    bright = finite[finite > split]
    if dark.size == 0 or bright.size == 0:
        return split
    return (float(np.median(dark)) + float(np.median(bright))) / 2.0


def _is_closed(contour: np.ndarray) -> bool:
    return contour.shape[0] > 2 and np.allclose(contour[0], contour[-1])


def _select_hole_contour(contours: List[np.ndarray], shape: tuple) -> np.ndarray:
    """Pick the contour belonging to the hole the crop is centred on.

    Step 9 centres every crop on the hole, so: prefer the largest closed contour
    that encloses the crop centre; otherwise (teeth and holes cut by the crop
    border give open contours) take the contour passing nearest the centre.
    Neighbouring holes and the stamp edge are never chosen merely for being big.
    """
    cy, cx = (shape[0] - 1) / 2.0, (shape[1] - 1) / 2.0
    enclosing = []
    for c in contours:
        if _is_closed(c):
            poly = c[:, ::-1].astype(np.float32).reshape(-1, 1, 2)  # (x, y)
            if cv2.pointPolygonTest(poly, (cx, cy), False) >= 0:
                enclosing.append((abs(cv2.contourArea(poly)), c))
    if enclosing:
        return max(enclosing, key=lambda t: t[0])[1]
    return min(contours, key=lambda c: float(np.min(np.hypot(c[:, 0] - cy, c[:, 1] - cx))))


def extract_subpixel_outline(backlit_crop: np.ndarray) -> List[List[float]]:
    """Run marching squares at the midpoint threshold and select the hole's contour.

    Returns an ordered list of subpixel [x, y] coordinates (image coordinates).
    """
    threshold = compute_midpoint_threshold(backlit_crop)
    # find_contours returns rows of (row, col) == (y, x) at subpixel precision.
    contours = measure.find_contours(backlit_crop, level=threshold)
    if not contours:
        return []

    best = _select_hole_contour(contours, backlit_crop.shape)
    # Convert (row, col) -> (x, y)
    outline = [[float(pt[1]), float(pt[0])] for pt in best]
    return outline


def run_step11(
    session_id: str,
    holes: List[HoleRecord],
    crops: Dict[str, Dict[str, np.ndarray]],
    session_manager: Optional[SessionManager] = None
) -> Dict[str, List[List[float]]]:
    """Execute Step 11: Subpixel Contour Extraction.

    crops: mapping hole_id -> {frame_tag: ndarray}. Uses the (fused) backlit crop
    per hole. Returns mapping hole_id -> subpixel outline polyline.
    """
    if session_manager is None:
        session_manager = SessionManager()

    session_dir = session_manager.get_session_dir(session_id)
    manifest = session_manager.get_manifest_manager(session_id)
    holes_dir = session_dir / "holes"

    outlines: Dict[str, List[List[float]]] = {}
    image_entries: List[Dict[str, Any]] = []

    for h in holes:
        hole_crops = crops.get(h.hole_id, {})
        backlit = hole_crops.get("backlit")
        if backlit is None:
            logger.warning("Step 11: hole %s has no backlit crop; skipping outline", h.hole_id)
            outlines[h.hole_id] = []
            continue

        outline = extract_subpixel_outline(backlit)
        outlines[h.hole_id] = outline

        h_dir = holes_dir / h.hole_id
        h_dir.mkdir(parents=True, exist_ok=True)
        out_path = h_dir / "subpixel_outline.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump({"hole_id": h.hole_id, "type": h.type, "outline_px": outline}, f, indent=2)

        # Feature 6/8 image: backlit crop with the traced subpixel contour.
        overlay = np.power(np.clip(backlit, 0.0, 1.0), 1.0 / OUTPUT_DISPLAY_GAMMA)
        overlay = cv2.cvtColor((overlay * 255.0).astype(np.uint8), cv2.COLOR_GRAY2BGR)
        if len(outline) >= 2:
            pts = np.round(np.asarray(outline) * 16).astype(np.int32).reshape(-1, 1, 2)
            cv2.polylines(overlay, [pts], isClosed=False, color=(0, 200, 255), thickness=1,
                          lineType=cv2.LINE_AA, shift=4)
        cv2.imwrite(str(h_dir / "contour_overlay.png"), overlay)
        image_entries.append({"path": f"holes/{h.hole_id}/contour_overlay.png",
                              "stage": "subpixel_contour", "hole_id": h.hole_id, "edge": h.edge})

    if image_entries:
        manifest.add_image_entries(image_entries)

    manifest.add_stage_entry(
        step_id=11,
        phase=3,
        stage_name="subpixel_contour",
        output_path="holes/",
        status="complete"
    )

    return outlines
