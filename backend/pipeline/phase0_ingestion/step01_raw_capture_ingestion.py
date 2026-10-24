"""Step 1: Raw Capture Ingestion.

File: backend/pipeline/phase0_ingestion/step01_raw_capture_ingestion.py
Decodes camera RAW frames in linear space, collapses them to luminance arrays,
validates session log and camera settings consistency, and attaches torch metadata.
"""

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

try:
    import rawpy
except ImportError:
    rawpy = None

from backend.config.capture_schema import match_filename_to_tag, validate_captured_tags  # noqa: F401 (re-exported)
from backend.config.pipeline_constants import CAMERA_MATCH_KEYS
from backend.session.manifest import ManifestManager
from backend.session.session_manager import SessionManager

logger = logging.getLogger(__name__)


class RawFrameRecord:
    """Represents a decoded linear brightness frame with its metadata."""

    def __init__(
        self,
        tag: str,
        brightness_array: np.ndarray,
        source_filename: str,
        torch_meta: Optional[Dict[str, Any]] = None
    ) -> None:
        self.tag = tag
        self.brightness_array = brightness_array.astype(np.float32)
        self.source_filename = source_filename
        self.torch_meta = torch_meta

    def to_dict(self) -> Dict[str, Any]:
        """Return serializable summary of frame record (without full pixel matrix)."""
        return {
            "tag": self.tag,
            "source_filename": self.source_filename,
            "shape": list(self.brightness_array.shape),
            "mean_brightness": float(np.mean(self.brightness_array)),
            "torch_meta": self.torch_meta
        }


def _srgb_to_linear(encoded: np.ndarray) -> np.ndarray:
    """Inverse sRGB transfer function (IEC 61966-2-1) on values in [0, 1]."""
    return np.where(
        encoded <= 0.04045, encoded / 12.92, np.power((encoded + 0.055) / 1.055, 2.4)
    ).astype(np.float32)


def decode_raw_to_luminance(file_path: Path | str) -> np.ndarray:
    """Decode a RAW image file to linear luminance float32 plane [0.0, 1.0].
    
    Formula: Y = 0.2126 * R + 0.7152 * G + 0.0722 * B
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Source file not found: {path}")

    raw_extensions = {".raf", ".cr2", ".cr3", ".nef", ".arw", ".dng", ".orf", ".rw2"}
    if rawpy is not None and path.suffix.lower() in raw_extensions:
        with rawpy.imread(str(path)) as raw:
            # Linear demosaic: gamma=(1, 1), no auto brightness adjustment
            rgb = raw.postprocess(
                gamma=(1, 1),
                no_auto_bright=True,
                use_camera_wb=True,
                output_bps=16
            ).astype(np.float32) / 65535.0
    else:
        # Fallback for standard image formats or testing arrays
        import cv2
        img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if img is None:
            raise ValueError(f"Unable to read image at {path}")
        max_val = 65535.0 if img.dtype == np.uint16 else 255.0
        encoded = img.astype(np.float32) / max_val
        if img.dtype == np.uint8:
            # 8-bit files (JPEG/PNG) are sRGB gamma-encoded; undo the curve so
            # every later step still sees brightness proportional to light.
            # 16-bit inputs are assumed already linear (e.g. our own exports).
            encoded = _srgb_to_linear(encoded)
        if img.ndim == 2:
            return encoded.astype(np.float32)
        rgb = cv2.cvtColor(encoded, cv2.COLOR_BGR2RGB)

    luminance = (0.2126 * rgb[:, :, 0] + 0.7152 * rgb[:, :, 1] + 0.0722 * rgb[:, :, 2]).astype(np.float32)
    return luminance


def run_step01(
    session_id: str,
    session_manager: Optional[SessionManager] = None
) -> Dict[str, RawFrameRecord]:
    """Execute Step 1: Raw Capture Ingestion."""
    if session_manager is None:
        session_manager = SessionManager()

    session_dir = session_manager.get_session_dir(session_id)
    manifest = session_manager.get_manifest_manager(session_id)
    raw_dir = session_dir / "raw"

    if not raw_dir.exists():
        raise FileNotFoundError(f"Raw capture directory does not exist: {raw_dir}")

    session_log = session_manager.load_session_log(session_id)
    if not session_log:
        raise ValueError(f"Missing session_log.json for session {session_id}")

    torches = session_log.get("torches", [])
    if len(torches) != 4:
        raise ValueError(f"session_log.json must contain exactly 4 torch records, found {len(torches)}")

    camera_settings = session_log.get("camera_settings", {})
    for key in CAMERA_MATCH_KEYS:
        if key not in camera_settings:
            logger.warning(f"Camera setting '{key}' not specified in session_log.json")

    torch_meta_map: Dict[str, Dict[str, Any]] = {
        t.get("torch_id", f"torch_{i+1}"): t
        for i, t in enumerate(torches)
    }

    # Sorted so the result never depends on directory listing order.
    tagged_lists: Dict[str, List[Path]] = {}
    for entry in sorted(raw_dir.iterdir(), key=lambda p: p.name.lower()):
        if entry.is_file():
            tag = match_filename_to_tag(entry.name)
            if tag:
                tagged_lists.setdefault(tag, []).append(entry)
    tagged_counts: Dict[str, int] = {t: len(v) for t, v in tagged_lists.items()}

    validation_res = validate_captured_tags(tagged_counts)
    if not validation_res["ready"]:
        failed_tags = [
            t for t, info in validation_res["tag_status"].items()
            if "Fail" in info["status"]
        ]
        raise ValueError(
            f"Pre-flight schema validation failed for session {session_id}. "
            f"Failed required tags: {failed_tags}"
        )

    # RawFrameRecord holds one frame per tag. Several backlit files (a focus
    # bracket) are accepted by the schema, but Steps 2-9 carry a single backlit
    # frame, so the first (by name) is used and the rest are reported.
    tagged_files: Dict[str, Path] = {}
    for tag, paths in tagged_lists.items():
        tagged_files[tag] = paths[0]
        if len(paths) > 1:
            logger.warning(
                "Step 1: %d '%s' files supplied; using %s (focus-stack inputs are not "
                "carried through Steps 2-9)", len(paths), tag, paths[0].name
            )

    records: Dict[str, RawFrameRecord] = {}
    reference_shape: Optional[tuple] = None
    for tag, file_path in tagged_files.items():
        brightness_plane = decode_raw_to_luminance(file_path)
        # Every later step indexes the same pixel grid across frames.
        if reference_shape is None:
            reference_shape = brightness_plane.shape
        elif brightness_plane.shape != reference_shape:
            raise ValueError(
                f"{file_path.name} is {brightness_plane.shape[1]}x{brightness_plane.shape[0]} px but "
                f"earlier frames are {reference_shape[1]}x{reference_shape[0]} px; all photos in a "
                "session must come from the same camera setup."
            )
        meta = torch_meta_map.get(tag) if tag.startswith("torch_") else None
        record = RawFrameRecord(
            tag=tag,
            brightness_array=brightness_plane,
            source_filename=file_path.name,
            torch_meta=meta
        )
        records[tag] = record

    manifest.set_status("in_progress")
    manifest.add_stage_entry(
        step_id=1,
        phase=0,
        stage_name="raw_capture_ingestion",
        output_path="raw/",
        status="complete"
    )

    return records
