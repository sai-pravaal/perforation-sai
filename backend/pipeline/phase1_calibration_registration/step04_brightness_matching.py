"""Step 4: Brightness-Matching Across Four Torch Photos.

File: backend/pipeline/phase1_calibration_registration/step04_brightness_matching.py
Samples a shared undecorated patch of stamp paper across the 4 torch photos,
computes linear rescaling factors to equalize illumination intensity, and flags warnings.
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from skimage.filters import threshold_otsu

from backend.config.pipeline_constants import BRIGHTNESS_MISMATCH_TOLERANCE, BRIGHTNESS_PATCH_SIZE_PX
from backend.pipeline.phase0_ingestion.step01_raw_capture_ingestion import RawFrameRecord
from backend.session.manifest import ManifestManager
from backend.session.session_manager import SessionManager

logger = logging.getLogger(__name__)


def locate_reference_paper_patch(
    torch_frames: Dict[str, np.ndarray],
    patch_size: int = BRIGHTNESS_PATCH_SIZE_PX
) -> Tuple[int, int, int, int]:
    """Identify coordinates (ymin, ymax, xmin, xmax) of a shared undecorated patch of stamp paper.

    The spec needs plain paper with no hole, perforation or print nearby,
    visible in all four torch photos. Candidate patches on a regular grid are
    kept only if they are brighter than the Otsu split of the mean torch image
    (paper, not the dark backdrop or a hole), and the one with the lowest
    texture (coefficient of variation, worst case over the four torches) wins.
    Print, perforations and edges all raise the variation, so they are avoided.
    """
    stack = np.stack([np.asarray(f, dtype=np.float32) for f in torch_frames.values()], axis=0)
    h, w = stack.shape[1:]
    size = max(4, min(patch_size, h, w))
    mean_img = stack.mean(axis=0)
    finite = mean_img[np.isfinite(mean_img)]
    paper_level = float(threshold_otsu(finite)) if finite.size and finite.max() > finite.min() else 0.0

    best: Optional[Tuple[float, int, int]] = None
    for y in range(0, h - size + 1, size):
        for x in range(0, w - size + 1, size):
            patches = stack[:, y:y + size, x:x + size]
            if float(patches.mean()) <= paper_level:
                continue
            means = patches.mean(axis=(1, 2))
            if np.any(means <= 1e-6):
                continue
            cv = float(np.max(patches.std(axis=(1, 2)) / means))
            if best is None or cv < best[0]:
                best = (cv, y, x)

    if best is None:
        # No bright patch found (e.g. tiny frames): fall back to the centre.
        logger.warning("Step 4: no plain-paper patch found; using the frame centre")
        y, x = max(0, h // 2 - size // 2), max(0, w // 2 - size // 2)
    else:
        _, y, x = best
    return y, min(h, y + size), x, min(w, x + size)


def run_step04(
    session_id: str,
    frames: Dict[str, RawFrameRecord],
    session_manager: Optional[SessionManager] = None
) -> Tuple[Dict[str, RawFrameRecord], Dict[str, Any]]:
    """Execute Step 4: Brightness-Matching Across Torch Photos."""
    if session_manager is None:
        session_manager = SessionManager()

    session_dir = session_manager.get_session_dir(session_id)
    manifest = session_manager.get_manifest_manager(session_id)
    stages_dir = session_dir / "stages"
    stages_dir.mkdir(parents=True, exist_ok=True)

    torch_tags = ["torch_1", "torch_2", "torch_3", "torch_4"]
    torch_arrays: Dict[str, np.ndarray] = {}
    for t in torch_tags:
        if t not in frames:
            raise KeyError(f"Missing required torch frame '{t}' for Step 4 brightness matching")
        torch_arrays[t] = frames[t].brightness_array

    ymin, ymax, xmin, xmax = locate_reference_paper_patch(torch_arrays)

    # Compute mean brightness inside patch for each torch
    b_values: Dict[str, float] = {}
    for t in torch_tags:
        patch = torch_arrays[t][ymin:ymax, xmin:xmax]
        b_mean = float(np.mean(patch))
        b_values[t] = b_mean if b_mean > 1e-6 else 1e-6

    b_ref = float(np.mean(list(b_values.values())))

    # Compute correction factors
    correction_factors: Dict[str, float] = {}
    warnings: List[Dict[str, Any]] = []

    for t in torch_tags:
        factor = b_ref / b_values[t]
        correction_factors[t] = round(float(factor), 4)

        # Check tolerance band against pipeline_constants
        if abs(factor - 1.0) > BRIGHTNESS_MISMATCH_TOLERANCE:
            pct = round((factor - 1.0) * 100.0, 1)
            msg = f"{t} brightness deviates by {pct}% from reference average"
            logger.warning(msg)
            warnings.append({
                "torch_id": t,
                "factor": correction_factors[t],
                "deviation_pct": pct,
                "message": msg
            })

    # Apply per-photo whole-frame linear rescaling to torch frames
    matched_frames: Dict[str, RawFrameRecord] = {}
    for tag, frame in frames.items():
        if tag in torch_tags:
            scale = correction_factors[tag]
            scaled_array = (frame.brightness_array * scale).astype(np.float32)
            matched_frames[tag] = RawFrameRecord(
                tag=tag,
                brightness_array=scaled_array,
                source_filename=frame.source_filename,
                torch_meta=frame.torch_meta
            )
        else:
            # Diffuse, backlit, and calibration remain unscaled
            matched_frames[tag] = frame

    # Persist correction data for Feature 4
    correction_payload = {
        "reference_patch_bbox": [xmin, ymin, xmax - xmin, ymax - ymin],
        "b_reference": b_ref,
        "patch_mean_brightness": b_values,
        "correction_factors": correction_factors,
        "tolerance": BRIGHTNESS_MISMATCH_TOLERANCE,
        "warnings": warnings
    }

    out_file = stages_dir / "brightness_correction.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(correction_payload, f, indent=2)

    manifest.add_stage_entry(
        step_id=4,
        phase=1,
        stage_name="brightness_matching",
        output_path="stages/brightness_correction.json",
        status="complete"
    )

    return matched_frames, correction_payload
