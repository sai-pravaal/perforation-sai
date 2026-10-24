"""Step 14: Boundary Texture Sampling (Feature 2 - Roughness).

File: backend/pipeline/phase3_feature_extraction/step14_boundary_texture_sampling.py
Walks the subpixel outline at fixed real-world arc-length spacing, and at each
sample cuts a small tangent-aligned patch from whichever of the four torch crops
shows the highest local texture contrast at that location.
"""

import logging
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from backend.config.pipeline_constants import BOUNDARY_SAMPLE_SPACING_UM, BOUNDARY_PATCH_SIZE_PX
from backend.pipeline.phase1_calibration_registration.step02_lens_and_unit_calibration import CalibrationRecord
from backend.pipeline.phase2_geometry_classification.step07_perforation_candidate_detection import HoleRecord
from backend.pipeline.phase2_geometry_classification.step09_roi_extraction import save_linear_png
from backend.session.session_manager import SessionManager

logger = logging.getLogger(__name__)

TORCH_TAGS = ["torch_1", "torch_2", "torch_3", "torch_4"]


def resample_outline(outline: List[List[float]], spacing_px: float) -> List[Tuple[float, float, float]]:
    """Resample a polyline at fixed arc-length spacing.

    Returns a list of (x, y, tangent_deg) sample points.
    """
    pts = np.asarray(outline, dtype=np.float64)
    if pts.shape[0] < 2 or spacing_px <= 0:
        return []

    seg_vecs = np.diff(pts, axis=0)
    seg_len = np.hypot(seg_vecs[:, 0], seg_vecs[:, 1])
    cum = np.concatenate([[0.0], np.cumsum(seg_len)])
    total = float(cum[-1])
    if total <= 0:
        return []

    closed = pts.shape[0] > 2 and np.allclose(pts[0], pts[-1])

    def point_at(arc: float) -> np.ndarray:
        if closed:
            arc = arc % total
        else:
            arc = min(max(arc, 0.0), total)
        idx = int(np.searchsorted(cum, arc, side="right") - 1)
        idx = max(0, min(idx, len(seg_len) - 1))
        frac = (arc - cum[idx]) / max(1e-9, seg_len[idx])
        return pts[idx] + frac * seg_vecs[idx]

    # Tangent from the points half a spacing before and after the sample (spec:
    # "looking at the nearby points just before and after it"), which smooths
    # over single-segment jitter in the subpixel outline.
    half = max(spacing_px / 2.0, 1.0)
    samples: List[Tuple[float, float, float]] = []
    target = 0.0
    while target < total:
        p = point_at(target)
        ahead, behind = point_at(target + half), point_at(target - half)
        tangent_deg = float(np.degrees(np.arctan2(ahead[1] - behind[1], ahead[0] - behind[0])))
        samples.append((float(p[0]), float(p[1]), tangent_deg))
        target += spacing_px
    return samples


def _sobel_magnitude(image: np.ndarray) -> np.ndarray:
    """Sobel gradient magnitude map of a float image."""
    gx = cv2.Sobel(image, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(image, cv2.CV_32F, 0, 1, ksize=3)
    return cv2.magnitude(gx, gy)


def extract_aligned_patch(
    image: np.ndarray,
    center: Tuple[float, float],
    angle_deg: float,
    size: int
) -> np.ndarray:
    """Cut a square patch centered at `center`, rotated so the tangent is horizontal.

    Only the size x size window is resampled: each output pixel (u, v) is read
    from center + R(angle) . ((u, v) - c), so the patch's +x axis runs along
    the tangent. (Previously the whole crop was rotated for every sample.)
    """
    a = np.radians(angle_deg)
    cos_a, sin_a = float(np.cos(a)), float(np.sin(a))
    c = (size - 1) / 2.0
    cx, cy = float(center[0]), float(center[1])
    inverse = np.array([
        [cos_a, -sin_a, cx - cos_a * c + sin_a * c],
        [sin_a, cos_a, cy - sin_a * c - cos_a * c],
    ], dtype=np.float64)
    patch = cv2.warpAffine(
        image.astype(np.float32), inverse, (size, size),
        flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP, borderMode=cv2.BORDER_REPLICATE
    )
    return patch.astype(np.float32)


def run_step14(
    session_id: str,
    holes: List[HoleRecord],
    outlines: Dict[str, List[List[float]]],
    crops: Dict[str, Dict[str, np.ndarray]],
    calibration_record: CalibrationRecord,
    session_manager: Optional[SessionManager] = None
) -> Dict[str, List[np.ndarray]]:
    """Execute Step 14: Boundary Texture Sampling."""
    if session_manager is None:
        session_manager = SessionManager()

    session_dir = session_manager.get_session_dir(session_id)
    manifest = session_manager.get_manifest_manager(session_id)
    holes_dir = session_dir / "holes"

    spacing_px = BOUNDARY_SAMPLE_SPACING_UM / max(1e-6, calibration_record.microns_per_pixel)
    patch_size = BOUNDARY_PATCH_SIZE_PX

    all_patches: Dict[str, List[np.ndarray]] = {}

    for h in holes:
        outline = outlines.get(h.hole_id, [])
        samples = resample_outline(outline, spacing_px)
        hole_crops = crops.get(h.hole_id, {})

        torch_imgs = {t: hole_crops[t] for t in TORCH_TAGS if t in hole_crops}
        if not torch_imgs or not samples:
            all_patches[h.hole_id] = []
            continue

        sobel_maps = {t: _sobel_magnitude(img) for t, img in torch_imgs.items()}

        patches: List[np.ndarray] = []
        patch_dir = holes_dir / h.hole_id / "boundary_patches"
        # Start clean so a re-run with fewer samples leaves no stale patch files.
        if patch_dir.is_dir():
            shutil.rmtree(patch_dir, ignore_errors=True)
        patch_dir.mkdir(parents=True, exist_ok=True)

        for i, (x, y, tangent_deg) in enumerate(samples):
            # Select torch crop with highest local contrast at this point.
            best_tag = None
            best_score = -1.0
            for t, mag in sobel_maps.items():
                local = cv2.getRectSubPix(mag, (patch_size, patch_size), (x, y))
                score = float(np.mean(local))
                if score > best_score:
                    best_score = score
                    best_tag = t

            patch = extract_aligned_patch(torch_imgs[best_tag], (x, y), tangent_deg, patch_size)
            patches.append(patch)

            # 16-bit linear so Step 15 can be resumed from disk without precision loss.
            save_linear_png(patch_dir / f"patch_{i:03d}.png", patch)

        all_patches[h.hole_id] = patches

    manifest.add_stage_entry(
        step_id=14,
        phase=3,
        stage_name="boundary_texture_sampling",
        output_path="holes/",
        status="complete"
    )

    return all_patches
