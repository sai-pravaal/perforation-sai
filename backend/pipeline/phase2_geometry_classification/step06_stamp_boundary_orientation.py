"""Step 6: Stamp Boundary Detection & Orientation Normalization.

File: backend/pipeline/phase2_geometry_classification/step06_stamp_boundary_orientation.py
Separates stamp from dark backdrop in diffuse frame, traces outer contour,
determines rotation angle, and rotates all frames so edges are axis-aligned.
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
    OUTPUT_DISPLAY_GAMMA,
    STAMP_ROTATION_WARN_DEG,
    STAMP_THRESHOLD_BLUR_PX,
)
from backend.pipeline.phase0_ingestion.step01_raw_capture_ingestion import RawFrameRecord
from backend.pipeline.phase2_geometry_classification.step07_perforation_candidate_detection import stamp_silhouette
from backend.session.manifest import ManifestManager
from backend.session.session_manager import SessionManager

logger = logging.getLogger(__name__)


class StampFrame:
    """Represents the coordinate frame of the orientation-normalized stamp."""

    def __init__(
        self,
        boundary_polygon: List[List[float]],
        rotation_deg: float,
        stamp_bbox: List[float]
    ) -> None:
        self.boundary_polygon = [[float(pt[0]), float(pt[1])] for pt in boundary_polygon]
        self.rotation_deg = float(rotation_deg)
        self.stamp_bbox = [float(v) for v in stamp_bbox]  # [x, y, w, h]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "boundary_polygon": self.boundary_polygon,
            "rotation_deg": self.rotation_deg,
            "stamp_bbox": self.stamp_bbox
        }


def detect_stamp_orientation(
    diffuse_image: np.ndarray,
    backlit: bool = False
) -> Tuple[float, List[List[float]], List[float]]:
    """Detect stamp outer boundary and compute rotation angle to square it to axes.

    backlit=True: the image is the (registered) backlit frame, where the paper
    is dark against the bright light box, so the outline is the filled
    silhouette of the largest dark region (the same silhouette Step 7 cuts
    the edge notches from). Otherwise the diffuse frame is split from its
    backdrop with Otsu.

    Returns (rotation_deg, boundary_polygon, stamp_bbox).
    """
    img_u8 = (np.clip(diffuse_image, 0.0, 1.0) * 255.0).astype(np.uint8)
    h, w = img_u8.shape

    blur = cv2.GaussianBlur(img_u8, (STAMP_THRESHOLD_BLUR_PX, STAMP_THRESHOLD_BLUR_PX), 0)
    _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if backlit:
        silhouette = stamp_silhouette(thresh)
        thresh = silhouette if silhouette is not None else cv2.bitwise_not(thresh)

    # Find external contours
    cnts, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        # Fallback to whole frame
        return 0.0, [[0.0, 0.0], [w, 0.0], [w, h], [0.0, h]], [0.0, 0.0, float(w), float(h)]

    largest_cnt = max(cnts, key=cv2.contourArea)

    # Fit minimum area rectangle
    rect = cv2.minAreaRect(largest_cnt)
    (cx, cy), (rw, rh), angle = rect

    # OpenCV minAreaRect angle normalization:
    # We want the angle relative to horizontal / vertical axes
    if rw < rh:
        angle = angle + 90.0
    if angle > 45.0:
        angle = angle - 90.0
    elif angle < -45.0:
        angle = angle + 90.0

    # Approx polygon
    poly = largest_cnt.reshape(-1, 2).tolist()
    # Axis-aligned bbox in image axes. minAreaRect's (rw, rh) follow the
    # rectangle's own axes and can come back swapped (e.g. angle 90 for an
    # axis-aligned stamp), so they must not be used as image width/height.
    bx, by, bw, bh = cv2.boundingRect(largest_cnt)
    bbox = [float(bx), float(by), float(bw), float(bh)]

    return float(angle), poly, bbox


def rotate_image_plane(image: np.ndarray, angle_deg: float) -> np.ndarray:
    """Rotate image about center by angle_deg using affine warp."""
    if abs(angle_deg) < 1e-3:
        return image.copy()

    h, w = image.shape
    center = (w / 2.0, h / 2.0)
    rot_mat = cv2.getRotationMatrix2D(center, angle_deg, 1.0)
    rotated = cv2.warpAffine(
        image, rot_mat, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE
    )
    return rotated.astype(np.float32)


def run_step06(
    session_id: str,
    frames: Dict[str, RawFrameRecord],
    session_manager: Optional[SessionManager] = None
) -> Tuple[Dict[str, RawFrameRecord], StampFrame]:
    """Execute Step 6: Stamp Boundary Detection & Orientation Normalization."""
    if session_manager is None:
        session_manager = SessionManager()

    session_dir = session_manager.get_session_dir(session_id)
    manifest = session_manager.get_manifest_manager(session_id)
    stages_dir = session_dir / "stages"
    stages_dir.mkdir(parents=True, exist_ok=True)

    if "diffuse" not in frames:
        raise KeyError("Missing required 'diffuse' frame for Step 6")

    # The outline comes from the registered backlit frame when present: with
    # a dark-grey backdrop (as in the reference captures) the diffuse frame
    # gives too little stamp/backdrop contrast and Otsu outlined only one lit
    # corner of the stamp. Step 5 registered the backlit frame onto the
    # diffuse one, so both share coordinates.
    outline_tag = "backlit" if "backlit" in frames else "diffuse"
    use_backlit = outline_tag == "backlit"
    angle_deg, raw_poly, raw_bbox = detect_stamp_orientation(frames[outline_tag].brightness_array, use_backlit)

    # minAreaRect reports the tilt clockwise in image coordinates, while
    # getRotationMatrix2D treats positive angles as counter-clockwise, so
    # rotating by +angle_deg cancels the tilt and axis-aligns the stamp edges.
    normalized_frames: Dict[str, RawFrameRecord] = {}
    for tag, frame in frames.items():
        rot_array = rotate_image_plane(frame.brightness_array, angle_deg)
        normalized_frames[tag] = RawFrameRecord(
            tag=tag,
            brightness_array=rot_array,
            source_filename=frame.source_filename,
            torch_meta=frame.torch_meta
        )

    # Re-derive axis-aligned boundary on rotated diffuse frame
    norm_diffuse = normalized_frames["diffuse"].brightness_array
    _, final_poly, final_bbox = detect_stamp_orientation(normalized_frames[outline_tag].brightness_array, use_backlit)

    stamp_frame = StampFrame(
        boundary_polygon=final_poly,
        rotation_deg=angle_deg,
        stamp_bbox=final_bbox
    )

    # Spec side-check: the operator aligns the torch azimuth 0-degree mark with
    # the stamp's row direction, so a large tilt means the two disagree and the
    # torch azimuths (Step 17) are not where the session log says.
    payload = stamp_frame.to_dict()
    if abs(angle_deg) > STAMP_ROTATION_WARN_DEG:
        warning = (f"Stamp is rotated {angle_deg:.1f} deg in the frame; the torch azimuth 0-degree mark "
                   "may not be aligned with the stamp rows. Check the rig before trusting shadow results.")
        logger.warning("Step 6: %s", warning)
        payload["alignment_warning"] = warning

    # Persist StampFrame
    out_path = stages_dir / "stamp_frame.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    # Feature 6 thumbnail: straightened overview with the detected boundary.
    preview = np.power(np.clip(norm_diffuse, 0.0, 1.0), 1.0 / OUTPUT_DISPLAY_GAMMA)
    preview = cv2.cvtColor((preview * 255.0).astype(np.uint8), cv2.COLOR_GRAY2BGR)
    poly = np.array(final_poly, dtype=np.int32).reshape(-1, 1, 2)
    if poly.size:
        cv2.polylines(preview, [poly], isClosed=True, color=(0, 200, 255), thickness=max(1, preview.shape[1] // 600))
    cv2.imwrite(str(stages_dir / "stamp_boundary.png"), preview)
    manifest.add_image_entry(path="stages/stamp_boundary.png", stage="stamp_boundary")

    manifest.add_stage_entry(
        step_id=6,
        phase=2,
        stage_name="stamp_boundary_orientation",
        output_path="stages/stamp_frame.json",
        status="complete"
    )

    return normalized_frames, stamp_frame
