"""Step 10: Focus-Stack Fusion.

File: backend/pipeline/phase2_geometry_classification/step10_focus_stack_fusion.py
Combines multi-focus depth exposures for a hole into a single all-in-focus
image using Laplacian sharpness mapping and pyramid blending.
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import numpy as np

from backend.config.pipeline_constants import FOCUS_PYRAMID_LEVELS, FOCUS_SHARPNESS_BLUR_PX
from backend.pipeline.phase2_geometry_classification.step07_perforation_candidate_detection import HoleRecord
from backend.pipeline.phase2_geometry_classification.step09_roi_extraction import save_linear_png
from backend.session.manifest import ManifestManager
from backend.session.session_manager import SessionManager

logger = logging.getLogger(__name__)


def compute_laplacian_sharpness(image: np.ndarray) -> np.ndarray:
    """Compute local high-frequency sharpness map using Laplacian filter."""
    lap = cv2.Laplacian(np.asarray(image, dtype=np.float32), cv2.CV_32F, ksize=3)
    sharpness = np.abs(lap)
    # Smooth sharpness map slightly to avoid pixel speckle
    k = FOCUS_SHARPNESS_BLUR_PX if FOCUS_SHARPNESS_BLUR_PX % 2 == 1 else FOCUS_SHARPNESS_BLUR_PX + 1
    return cv2.GaussianBlur(sharpness, (k, k), 0)


def _gaussian_pyramid(img: np.ndarray, levels: int) -> List[np.ndarray]:
    pyr = [img]
    for _ in range(levels):
        pyr.append(cv2.pyrDown(pyr[-1]))
    return pyr


def _laplacian_pyramid(img: np.ndarray, levels: int) -> List[np.ndarray]:
    gauss = _gaussian_pyramid(img, levels)
    lap = []
    for i in range(levels):
        size = (gauss[i].shape[1], gauss[i].shape[0])
        lap.append(gauss[i] - cv2.pyrUp(gauss[i + 1], dstsize=size))
    lap.append(gauss[-1])
    return lap


def fuse_focus_stack(images: List[np.ndarray]) -> np.ndarray:
    """Fuse multiple depth-exposed crops into one sharp composite image.

    Each pixel's source is the sharpest exposure (Laplacian sharpness). The
    per-source selection masks are blended with a Laplacian pyramid (spec
    Step 10) so switching from one depth to another leaves no visible seams.
    """
    if len(images) == 1:
        return images[0].copy()

    stack = [np.asarray(img, dtype=np.float32) for img in images]
    best_idx = np.argmax(np.stack([compute_laplacian_sharpness(img) for img in stack], axis=0), axis=0)

    h, w = stack[0].shape
    levels = max(0, min(FOCUS_PYRAMID_LEVELS, int(np.log2(max(1, min(h, w)))) - 2))
    blended: Optional[List[np.ndarray]] = None
    for i, img in enumerate(stack):
        mask_pyr = _gaussian_pyramid((best_idx == i).astype(np.float32), levels)
        lap_pyr = _laplacian_pyramid(img, levels)
        weighted = [lp * mp for lp, mp in zip(lap_pyr, mask_pyr)]
        blended = weighted if blended is None else [b + c for b, c in zip(blended, weighted)]

    fused = blended[-1]
    for level in range(levels - 1, -1, -1):
        size = (blended[level].shape[1], blended[level].shape[0])
        fused = cv2.pyrUp(fused, dstsize=size) + blended[level]
    return fused.astype(np.float32)


def run_step10(
    session_id: str,
    holes: List[HoleRecord],
    crops: Dict[str, Dict[str, np.ndarray]],
    session_manager: Optional[SessionManager] = None
) -> Dict[str, Dict[str, np.ndarray]]:
    """Execute Step 10: Focus-Stack Fusion."""
    if session_manager is None:
        session_manager = SessionManager()

    session_dir = session_manager.get_session_dir(session_id)
    manifest = session_manager.get_manifest_manager(session_id)
    holes_dir = session_dir / "holes"

    fused_crops: Dict[str, Dict[str, np.ndarray]] = {}

    for h in holes:
        h_id = h.hole_id
        h_crops = crops.get(h_id, {})
        h_dir = holes_dir / h_id / "crops"
        h_dir.mkdir(parents=True, exist_ok=True)

        fused_h: Dict[str, np.ndarray] = {}
        for tag, crop_arr in h_crops.items():
            # In standard single-exposure capture, fuse_focus_stack simply preserves the sharp image
            fused = fuse_focus_stack([crop_arr])
            fused_h[tag] = fused

            # Save fused image to disk at 16-bit linear precision
            save_linear_png(h_dir / f"fused_{tag}.png", fused)

        fused_crops[h_id] = fused_h

    manifest.add_stage_entry(
        step_id=10,
        phase=2,
        stage_name="focus_stack_fusion",
        output_path="holes/",
        status="complete"
    )

    return fused_crops
