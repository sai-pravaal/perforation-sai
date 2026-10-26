"""Small preview images for the gallery and the live thumbnail strip (Feature 6).

The pipeline writes full-size frames (up to ~24 MP, 16-bit) and per-hole crops.
Decoding those on the GUI thread to draw a 96 px thumbnail froze the window for
seconds, so each manifest image gets a small 8-bit preview made here, in the
pipeline's own thread, right after the step that produced it.
"""

import logging
import re
from pathlib import Path
from typing import Dict, Optional

import cv2
import numpy as np

from backend.config.pipeline_constants import OUTPUT_DISPLAY_GAMMA, THUMBNAIL_MAX_PX
from backend.session.manifest import ManifestManager

logger = logging.getLogger(__name__)

THUMB_DIR = "stages/thumbs"


def make_thumbnail(src: Path, dst: Path, max_px: int = THUMBNAIL_MAX_PX) -> bool:
    """Write an 8-bit preview of src no larger than max_px on its long side.

    16-bit images hold linear light (see save_linear_png), so they get display
    gamma, as the full-size viewer does; 8-bit images are used as they are.
    """
    image = cv2.imread(str(src), cv2.IMREAD_UNCHANGED)
    if image is None:
        return False
    if image.dtype == np.uint16:
        image = (np.power(image.astype(np.float32) / 65535.0, 1.0 / OUTPUT_DISPLAY_GAMMA) * 255.0).round().astype(np.uint8)
    scale = max_px / float(max(image.shape[:2]))
    if scale < 1.0:
        image = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    dst.parent.mkdir(parents=True, exist_ok=True)
    return bool(cv2.imwrite(str(dst), image))


def ensure_thumbnails(session_dir: Path, manifest: ManifestManager) -> int:
    """Create a preview for every manifest image that lacks one; returns how many were made."""
    made: Dict[int, str] = {}
    for entry in manifest.load().get("images", []):
        if entry.get("thumb") or entry.get("token") is None:
            continue
        rel = str(entry.get("path", ""))
        src = session_dir / rel
        if not src.is_file():
            continue
        thumb_rel = f"{THUMB_DIR}/{re.sub(r'[^A-Za-z0-9._-]+', '_', rel)}"
        if not thumb_rel.lower().endswith(".png"):
            thumb_rel += ".png"
        try:
            if make_thumbnail(src, session_dir / thumb_rel):
                made[int(entry["token"])] = thumb_rel
        except Exception:  # a preview is never worth failing the run for
            logger.warning("Could not make a thumbnail for %s", rel, exc_info=True)
    if made:
        manifest.set_image_thumbs(made)
    return len(made)
