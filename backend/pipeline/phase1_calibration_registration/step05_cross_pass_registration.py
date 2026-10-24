"""Step 5: Cross-Pass Image Registration.

File: backend/pipeline/phase1_calibration_registration/step05_cross_pass_registration.py
Aligns all torch frames and the backlit frame to the diffuse overview frame,
using Enhanced Correlation Coefficient (ECC) with ORB+Homography fallback.
"""

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from backend.config.pipeline_constants import (
    REGISTRATION_ECC_EPS,
    REGISTRATION_ECC_MAX_DIM,
    REGISTRATION_ECC_MAX_ITERS,
    REGISTRATION_ECC_REFINE_ITERS,
    REGISTRATION_MAX_SHIFT_PX,
    REGISTRATION_MIN_ECC_CORRELATION,
    REGISTRATION_ORB_MAX_FEATURES,
    REGISTRATION_RANSAC_REPROJ_PX,
)
from backend.pipeline.phase0_ingestion.step01_raw_capture_ingestion import RawFrameRecord
from backend.pipeline.phase2_geometry_classification.step09_roi_extraction import save_linear_png
from backend.session.manifest import ManifestManager
from backend.session.session_manager import SessionManager

logger = logging.getLogger(__name__)


def _ecc(ref: np.ndarray, tgt: np.ndarray, warp: np.ndarray, motion_type: int,
         max_iters: int, eps: float) -> Tuple[float, np.ndarray]:
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, max_iters, eps)
    return cv2.findTransformECC(ref, tgt, warp, motion_type, criteria, None, 5)


def align_frame_ecc(
    target: np.ndarray,
    reference: np.ndarray,
    motion_type: int = cv2.MOTION_EUCLIDEAN,
    max_iters: int = REGISTRATION_ECC_MAX_ITERS,
    eps: float = REGISTRATION_ECC_EPS
) -> Tuple[Optional[np.ndarray], float]:
    """Align target to reference with ECC, coarse-to-fine. Returns (warp, correlation).

    The warp is solved on copies downscaled to REGISTRATION_ECC_MAX_DIM, then
    refined at full resolution for a few iterations, which keeps full-frame
    macro photos tractable. A result whose correlation is below
    REGISTRATION_MIN_ECC_CORRELATION is rejected (None) so the caller falls back.
    """
    target_f32 = target.astype(np.float32)
    ref_f32 = reference.astype(np.float32)
    h, w = ref_f32.shape
    scale = min(1.0, REGISTRATION_ECC_MAX_DIM / float(max(h, w)))
    warp = np.eye(2, 3, dtype=np.float32)
    try:
        if scale < 1.0:
            size = (max(8, int(round(w * scale))), max(8, int(round(h * scale))))
            ref_small = cv2.resize(ref_f32, size, interpolation=cv2.INTER_AREA)
            tgt_small = cv2.resize(target_f32, size, interpolation=cv2.INTER_AREA)
            _, warp = _ecc(ref_small, tgt_small, warp, motion_type, max_iters, eps)
            warp[:, 2] /= scale
            corr, warp = _ecc(ref_f32, target_f32, warp, motion_type, REGISTRATION_ECC_REFINE_ITERS, eps)
        else:
            corr, warp = _ecc(ref_f32, target_f32, warp, motion_type, max_iters, eps)
    except cv2.error as e:
        logger.debug("ECC alignment failed to converge: %s", e)
        return None, 0.0
    if corr < REGISTRATION_MIN_ECC_CORRELATION:
        logger.debug("ECC correlation %.3f below minimum; rejecting", corr)
        return None, float(corr)
    return warp, float(corr)


def max_corner_displacement(transform: Optional[np.ndarray], shape: Tuple[int, int]) -> float:
    """Largest pixel displacement of the four frame corners under the transform.

    Unlike the translation column alone, this reflects rotation and keystone
    as well, which is what "the platen or tripod moved" looks like.
    """
    if transform is None:
        return 0.0
    h, w = shape
    corners = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], dtype=np.float64)
    if transform.shape == (2, 3):
        moved = corners @ transform[:, :2].T.astype(np.float64) + transform[:, 2].astype(np.float64)
    else:
        homog = np.c_[corners, np.ones(4)] @ transform.T.astype(np.float64)
        moved = homog[:, :2] / homog[:, 2:3]
    return float(np.max(np.linalg.norm(moved - corners, axis=1)))


def align_frame_orb(
    target: np.ndarray,
    reference: np.ndarray,
    max_features: int = REGISTRATION_ORB_MAX_FEATURES
) -> Optional[np.ndarray]:
    """Fallback alignment using ORB keypoint matching and Homography."""
    t_u8 = (np.clip(target, 0.0, 1.0) * 255.0).astype(np.uint8)
    r_u8 = (np.clip(reference, 0.0, 1.0) * 255.0).astype(np.uint8)

    orb = cv2.ORB_create(max_features)
    kp1, des1 = orb.detectAndCompute(t_u8, None)
    kp2, des2 = orb.detectAndCompute(r_u8, None)

    if des1 is None or des2 is None or len(des1) < 10 or len(des2) < 10:
        return None

    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = matcher.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)[:100]

    if len(matches) < 8:
        return None

    pts_target = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    pts_ref = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

    homography, inliers = cv2.findHomography(pts_target, pts_ref, cv2.RANSAC, REGISTRATION_RANSAC_REPROJ_PX)
    return homography


def warp_registered_image(
    image: np.ndarray,
    transform: Optional[np.ndarray],
    ref_shape: Tuple[int, int]
) -> np.ndarray:
    """Apply solved transform (2x3 affine or 3x3 homography) to warp image."""
    if transform is None:
        return image.copy()

    h, w = ref_shape
    if transform.shape == (2, 3):
        return cv2.warpAffine(
            image, transform, (w, h), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP
        )
    elif transform.shape == (3, 3):
        return cv2.warpPerspective(
            image, transform, (w, h), flags=cv2.INTER_LINEAR
        )
    return image.copy()


def run_step05(
    session_id: str,
    frames: Dict[str, RawFrameRecord],
    session_manager: Optional[SessionManager] = None,
    max_shift_px_warning: float = REGISTRATION_MAX_SHIFT_PX
) -> Dict[str, RawFrameRecord]:
    """Execute Step 5: Cross-Pass Image Registration."""
    if session_manager is None:
        session_manager = SessionManager()

    session_dir = session_manager.get_session_dir(session_id)
    manifest = session_manager.get_manifest_manager(session_id)
    stages_dir = session_dir / "stages"
    registered_dir = stages_dir / "registered"
    registered_dir.mkdir(parents=True, exist_ok=True)

    if "diffuse" not in frames:
        raise KeyError("Missing required 'diffuse' reference frame for Step 5 registration")

    ref_frame = frames["diffuse"]
    ref_shape = ref_frame.brightness_array.shape

    registered_frames: Dict[str, RawFrameRecord] = {
        "diffuse": ref_frame
    }

    # Save reference registered image for thumbnail preview
    # 16-bit linear PNGs: thumbnails for Feature 6 and full-precision inputs for resume.
    save_linear_png(registered_dir / "diffuse_registered.png", ref_frame.brightness_array)
    manifest.add_image_entry(
        path="stages/registered/diffuse_registered.png",
        stage="registration"
    )

    targets = ["torch_1", "torch_2", "torch_3", "torch_4", "backlit"]
    movement_warnings: List[str] = []
    per_frame: Dict[str, Dict[str, Any]] = {}

    for tag in targets:
        if tag not in frames:
            continue
        tgt_frame = frames[tag]

        # The backlit frame has inverted contrast relative to the diffuse
        # reference (holes bright, paper darker), and ECC maximises positive
        # correlation, so it is aligned on an inverted copy. The warp is then
        # applied to the original frame.
        align_src = tgt_frame.brightness_array
        if tag == "backlit":
            align_src = float(np.nanmax(align_src)) - align_src

        # 1. Primary: ECC alignment
        transform, correlation = align_frame_ecc(align_src, ref_frame.brightness_array)
        method_used = "ecc"

        # 2. Backup: ORB + Homography
        if transform is None:
            transform = align_frame_orb(align_src, ref_frame.brightness_array)
            method_used = "orb_homography" if transform is not None else "identity"
            if transform is None:
                movement_warnings.append(f"{tag} could not be registered; used unaligned")

        shift_mag = max_corner_displacement(transform, ref_shape)
        per_frame[tag] = {
            "method": method_used,
            "ecc_correlation": round(correlation, 4) if method_used == "ecc" else None,
            "max_corner_shift_px": round(shift_mag, 3),
        }

        if shift_mag > max_shift_px_warning:
            warn_msg = f"{tag} showed possible mid-session movement (shift: {shift_mag:.2f}px)"
            logger.warning(warn_msg)
            movement_warnings.append(warn_msg)

        warped = warp_registered_image(tgt_frame.brightness_array, transform, ref_shape)
        reg_record = RawFrameRecord(
            tag=tag,
            brightness_array=warped,
            source_filename=tgt_frame.source_filename,
            torch_meta=tgt_frame.torch_meta
        )
        registered_frames[tag] = reg_record

        # Save registered preview image and add manifest image entry
        save_linear_png(registered_dir / f"{tag}_registered.png", warped)
        manifest.add_image_entry(
            path=f"stages/registered/{tag}_registered.png",
            stage="registration"
        )

    # Persist registration metadata summary
    reg_summary = {
        "reference_tag": "diffuse",
        "registered_tags": list(registered_frames.keys()),
        "movement_warnings": movement_warnings,
        "per_frame": per_frame,
    }
    with open(stages_dir / "registration_summary.json", "w", encoding="utf-8") as f:
        json.dump(reg_summary, f, indent=2)

    manifest.add_stage_entry(
        step_id=5,
        phase=1,
        stage_name="cross_pass_registration",
        output_path="stages/registered/",
        status="complete"
    )

    return registered_frames
