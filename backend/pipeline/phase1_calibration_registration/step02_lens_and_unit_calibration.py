"""Step 2: Lens Distortion Correction & Pixel-to-Physical-Unit Calibration.

File: backend/pipeline/phase1_calibration_registration/step02_lens_and_unit_calibration.py
Solves for radial lens distortion coefficients (k1, k2, k3), undistorts all frames,
and derives the session's physical scale factor (microns_per_pixel).
"""

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from backend.config.pipeline_constants import (
    CALIBRATION_DETECTION_GAMMA,
    CALIBRATION_PATTERN_SIZE,
    CALIBRATION_PATTERN_SPACING_UM,
)
from backend.pipeline.phase0_ingestion.step01_raw_capture_ingestion import RawFrameRecord
from backend.session.manifest import ManifestManager
from backend.session.session_manager import SessionManager

logger = logging.getLogger(__name__)

# Camera matrix solved alongside the distortion coefficients by the most recent
# detect_calibration_scale call. Distortion coefficients are only meaningful
# with the intrinsics they were fitted against, so run_step02 undistorts with
# this matrix. It is kept in memory only: the frozen CalibrationRecord contract
# has no camera-matrix field, and undistortion happens only inside Step 2.
_SOLVED_CAMERA_MATRIX: Dict[str, Optional[np.ndarray]] = {"last": None}


class CalibrationRecord:
    """Represents calibration results including lens distortion and physical scale."""

    def __init__(
        self,
        distortion_coeffs: List[float],
        microns_per_pixel: float,
        calibration_source_tag: str = "calibration"
    ) -> None:
        self.distortion_coeffs = [float(k) for k in distortion_coeffs]
        self.microns_per_pixel = float(microns_per_pixel)
        self.calibration_source_tag = calibration_source_tag

    def to_dict(self) -> Dict[str, Any]:
        return {
            "distortion_coeffs": self.distortion_coeffs,
            "microns_per_pixel": self.microns_per_pixel,
            "calibration_source_tag": self.calibration_source_tag
        }


class CalibrationTargetNotFoundError(ValueError):
    """Raised when the calibration target cannot be detected in the calibration frame."""


def _to_detection_u8(linear_image: np.ndarray) -> np.ndarray:
    """Convert a linear luminance plane to display-gamma 8-bit for pattern detection only.

    Linear sensor data is very dark when cast straight to 8 bit, which starves the
    detectors of contrast. The gamma is applied to this detection copy only; the
    measurement frames stay linear.
    """
    lin = np.clip(linear_image, 0.0, 1.0)
    return (np.power(lin, 1.0 / CALIBRATION_DETECTION_GAMMA) * 255.0).astype(np.uint8)


def _undistorted_grid_spacing_px(
    corners: np.ndarray,
    pattern_size: Tuple[int, int],
    camera_matrix: np.ndarray,
    dist: np.ndarray
) -> float:
    """Median spacing (px) between adjacent grid points after lens correction.

    Only true row/column neighbours are used, so the jump from the end of one
    row to the start of the next never enters the median.
    """
    pts = cv2.undistortPoints(
        corners.reshape(-1, 1, 2).astype(np.float32), camera_matrix, dist, P=camera_matrix
    ).reshape(pattern_size[1], pattern_size[0], 2)
    along_rows = np.linalg.norm(np.diff(pts, axis=1), axis=2).ravel()
    along_cols = np.linalg.norm(np.diff(pts, axis=0), axis=2).ravel()
    return float(np.median(np.concatenate([along_rows, along_cols])))


def detect_calibration_scale(
    calib_image: np.ndarray,
    known_pattern_spacing_um: Optional[float] = None,
    pattern_size: Optional[Tuple[int, int]] = None
) -> Tuple[List[float], float]:
    """Detect calibration pattern, solve distortion coefficients, and compute microns_per_pixel.

    Tries circles grid first, then chessboard corners. Spacing and grid size come
    from pipeline_constants unless given explicitly. If no pattern is detected,
    CalibrationTargetNotFoundError is raised: every micrometre value downstream
    depends on this scale, so the pipeline must not continue on a guessed one.
    """
    if known_pattern_spacing_um is None:
        known_pattern_spacing_um = CALIBRATION_PATTERN_SPACING_UM
    if pattern_size is None:
        pattern_size = tuple(CALIBRATION_PATTERN_SIZE)

    img_uint8 = _to_detection_u8(calib_image)
    h, w = img_uint8.shape

    # Try circles grid
    found, corners = cv2.findCirclesGrid(
        img_uint8, pattern_size, flags=cv2.CALIB_CB_SYMMETRIC_GRID
    )
    if not found:
        # Try chessboard
        found, corners = cv2.findChessboardCorners(img_uint8, pattern_size)

    if found and corners is not None:
        # Construct object points in microns
        objp = np.zeros((pattern_size[0] * pattern_size[1], 3), np.float32)
        objp[:, :2] = np.mgrid[0:pattern_size[0], 0:pattern_size[1]].T.reshape(-1, 2) * known_pattern_spacing_um

        # The CalibrationRecord contract stores the radial model [k1, k2, k3]
        # only, so tangential terms are held at zero during the fit.
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
            [objp], [corners], (w, h), None, None,
            flags=cv2.CALIB_ZERO_TANGENT_DIST
        )
        # OpenCV orders distortion as [k1, k2, p1, p2, k3].
        dist_flat = dist.ravel().tolist()
        dist_coeffs = [dist_flat[0], dist_flat[1], dist_flat[4] if len(dist_flat) > 4 else 0.0]
        _SOLVED_CAMERA_MATRIX["last"] = mtx

        # Pixel spacing measured on the lens-corrected grid positions (spec Step 2).
        median_px = _undistorted_grid_spacing_px(corners, pattern_size, mtx, dist)
        if median_px <= 0:
            raise CalibrationTargetNotFoundError("Calibration grid spacing measured as zero pixels.")
        microns_per_pixel = known_pattern_spacing_um / median_px
        return dist_coeffs, microns_per_pixel

    raise CalibrationTargetNotFoundError(
        f"Calibration target ({pattern_size[0]}x{pattern_size[1]} dot grid or chessboard) "
        "was not detected in the calibration photo. Confirm the target is in focus, fully "
        "visible and matches CALIBRATION_PATTERN_SIZE, then re-upload."
    )


def undistort_frame(
    image: np.ndarray,
    dist_coeffs: List[float],
    camera_matrix: Optional[np.ndarray] = None
) -> np.ndarray:
    """Apply radial lens undistortion to a 2D float32 image.

    camera_matrix should be the matrix solved together with dist_coeffs; the
    centred approximation is only a fallback when none is available.
    """
    k = np.array(dist_coeffs, dtype=np.float32)
    if np.all(k == 0.0):
        return image.copy()

    h, w = image.shape
    if camera_matrix is None:
        camera_matrix = np.array([
            [w, 0, w / 2.0],
            [0, w, h / 2.0],
            [0, 0, 1.0]
        ], dtype=np.float32)
    camera_matrix = np.asarray(camera_matrix, dtype=np.float32)

    dist = np.array([k[0], k[1], 0.0, 0.0, k[2] if len(k) > 2 else 0.0], dtype=np.float32)
    undistorted = cv2.undistort(image, camera_matrix, dist)
    return undistorted.astype(np.float32)


def run_step02(
    session_id: str,
    frames: Dict[str, RawFrameRecord],
    session_manager: Optional[SessionManager] = None,
    known_pattern_spacing_um: Optional[float] = None
) -> Tuple[Dict[str, RawFrameRecord], CalibrationRecord]:
    """Execute Step 2: Lens Distortion Correction & Physical Unit Calibration."""
    if session_manager is None:
        session_manager = SessionManager()

    session_dir = session_manager.get_session_dir(session_id)
    manifest = session_manager.get_manifest_manager(session_id)
    stages_dir = session_dir / "stages"
    stages_dir.mkdir(parents=True, exist_ok=True)

    if "calibration" not in frames:
        raise KeyError("Missing required 'calibration' frame from Step 1 output")

    calib_frame = frames["calibration"]
    _SOLVED_CAMERA_MATRIX["last"] = None
    dist_coeffs, microns_per_px = detect_calibration_scale(
        calib_frame.brightness_array,
        known_pattern_spacing_um=known_pattern_spacing_um
    )

    record = CalibrationRecord(
        distortion_coeffs=dist_coeffs,
        microns_per_pixel=microns_per_px,
        calibration_source_tag="calibration"
    )

    # Persist calibration record
    calib_json_path = stages_dir / "calibration.json"
    with open(calib_json_path, "w", encoding="utf-8") as f:
        json.dump(record.to_dict(), f, indent=2)

    # Undistort all frames
    camera_matrix = _SOLVED_CAMERA_MATRIX["last"]
    corrected_frames: Dict[str, RawFrameRecord] = {}
    for tag, frame in frames.items():
        corrected_array = undistort_frame(frame.brightness_array, dist_coeffs, camera_matrix)
        corrected_frames[tag] = RawFrameRecord(
            tag=tag,
            brightness_array=corrected_array,
            source_filename=frame.source_filename,
            torch_meta=frame.torch_meta
        )

    manifest.add_stage_entry(
        step_id=2,
        phase=1,
        stage_name="lens_and_unit_calibration",
        output_path="stages/calibration.json",
        status="complete"
    )

    return corrected_frames, record
