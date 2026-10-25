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

from backend.config.pipeline_constants import (
    OUTPUT_DISPLAY_GAMMA,
    TOOTH_ARC_END_TRIM_FRACTION,
    TOOTH_ARC_MASK_MARGIN_UM,
)
from backend.pipeline.phase1_calibration_registration.step02_lens_and_unit_calibration import CalibrationRecord
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


def notch_mask_in_crop(notch_outline_px: List[List[float]], centre_px: List[float], shape: tuple,
                      margin_px: float) -> np.ndarray:
    """Rasterise a tooth's Step 7 notch outline (frame coordinates) into its Step 9 crop.

    getRectSubPix maps crop pixel (u, v) to frame (u + cx - (w-1)/2, v + cy - (h-1)/2).
    The mask is dilated by margin_px so the traced edge, which lies on the
    notch boundary, falls inside it.
    """
    h, w = shape
    offset = np.array([centre_px[0] - (w - 1) / 2.0, centre_px[1] - (h - 1) / 2.0])
    pts = np.round(np.asarray(notch_outline_px, dtype=np.float64) - offset).astype(np.int32)
    mask = np.zeros((h, w), np.uint8)
    if pts.shape[0] >= 3:
        cv2.fillPoly(mask, [pts.reshape(-1, 1, 2)], 255)
    k = 2 * max(1, int(round(margin_px))) + 1
    return cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))


def _tooth_arc(contours: List[np.ndarray], mask: np.ndarray) -> Optional[np.ndarray]:
    """The run of traced edge that borders this tooth's own notch.

    In the backlit crop a tooth's notch is open to the bright background, so
    the traced edge runs on along the neighbouring teeth and paper edge. Only
    the longest contiguous run of points inside the notch mask is the
    punched arc; TOOTH_ARC_END_TRIM_FRACTION of its length is dropped at each
    end, where the arc turns into the torn tooth tips.
    """
    best: Optional[np.ndarray] = None
    for c in contours:
        rows = np.clip(np.round(c[:, 0]).astype(int), 0, mask.shape[0] - 1)
        cols = np.clip(np.round(c[:, 1]).astype(int), 0, mask.shape[1] - 1)
        inside = mask[rows, cols] > 0
        start = None
        for i, flag in enumerate(np.append(inside, False)):
            if flag and start is None:
                start = i
            elif not flag and start is not None:
                if best is None or i - start > best.shape[0]:
                    best = c[start:i]
                start = None
    if best is None or best.shape[0] < 5:
        return None
    seg = np.hypot(*np.diff(best, axis=0).T)
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    trim = TOOTH_ARC_END_TRIM_FRACTION * cum[-1]
    keep = (cum >= trim) & (cum <= cum[-1] - trim)
    return best[keep] if np.count_nonzero(keep) >= 5 else best


def extract_subpixel_outline(backlit_crop: np.ndarray, notch_mask: Optional[np.ndarray] = None) -> List[List[float]]:
    """Run marching squares at the midpoint threshold and select the hole's contour.

    notch_mask (edge teeth): restrict the result to the arc of this tooth's
    own notch (see _tooth_arc). Returns an ordered list of subpixel [x, y]
    coordinates (image coordinates).
    """
    threshold = compute_midpoint_threshold(backlit_crop)
    # find_contours returns rows of (row, col) == (y, x) at subpixel precision.
    contours = measure.find_contours(backlit_crop, level=threshold)
    if not contours:
        return []

    best = _tooth_arc(contours, notch_mask) if notch_mask is not None else None
    if best is None:
        best = _select_hole_contour(contours, backlit_crop.shape)
    # Convert (row, col) -> (x, y)
    outline = [[float(pt[1]), float(pt[0])] for pt in best]
    return outline


def run_step11(
    session_id: str,
    holes: List[HoleRecord],
    crops: Dict[str, Dict[str, np.ndarray]],
    session_manager: Optional[SessionManager] = None,
    calibration_record: Optional[CalibrationRecord] = None
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

    upp = calibration_record.microns_per_pixel if calibration_record is not None else None
    if upp is None:
        with open(session_dir / "stages" / "calibration.json", "r", encoding="utf-8") as f:
            upp = float(json.load(f)["microns_per_pixel"])
    margin_px = TOOTH_ARC_MASK_MARGIN_UM / max(1e-6, upp)

    outlines: Dict[str, List[List[float]]] = {}
    image_entries: List[Dict[str, Any]] = []

    for h in holes:
        hole_crops = crops.get(h.hole_id, {})
        backlit = hole_crops.get("backlit")
        if backlit is None:
            logger.warning("Step 11: hole %s has no backlit crop; skipping outline", h.hole_id)
            outlines[h.hole_id] = []
            continue

        notch_mask = None
        if h.type == "edge_tooth" and len(h.raw_outline_px) >= 3:
            notch_mask = notch_mask_in_crop(h.raw_outline_px, h.centroid_px, backlit.shape, margin_px)
        outline = extract_subpixel_outline(backlit, notch_mask)
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
