"""IPDS Capture Image Schema Specification.

This module defines the required tag specification for captured RAW frames.
It is shared directly between Feature 1 (Pre-Flight Capture Checklist) and
Step 1 (Raw Capture Ingestion) to ensure consistent validation.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional

# Schema definition for capture images:
# Each entry defines:
# - min_count: Minimum required frames with this tag
# - max_count: Maximum allowed frames with this tag (None for unlimited)
# - required: Whether missing frames block pipeline execution
# - description: Human-readable description of the capture requirement
CAPTURE_SCHEMA: Dict[str, Dict[str, Any]] = {
    "diffuse": {
        "min_count": 1,
        "max_count": 1,
        "required": True,
        "description": "Diffuse / overview illumination photograph of stamp"
    },
    "torch_1": {
        "min_count": 1,
        "max_count": 1,
        "required": True,
        "description": "Torch 1 directional raking-light photograph"
    },
    "torch_2": {
        "min_count": 1,
        "max_count": 1,
        "required": True,
        "description": "Torch 2 directional raking-light photograph"
    },
    "torch_3": {
        "min_count": 1,
        "max_count": 1,
        "required": True,
        "description": "Torch 3 directional raking-light photograph"
    },
    "torch_4": {
        "min_count": 1,
        "max_count": 1,
        "required": True,
        "description": "Torch 4 directional raking-light photograph"
    },
    "backlit": {
        "min_count": 1,
        "max_count": None,
        "required": True,
        "description": "Backlit / transillumination photograph (or focus bracket set)"
    },
    "calibration": {
        "min_count": 1,
        "max_count": 1,
        "required": True,
        "description": "Calibration grid/target photograph for optical undistortion and scale"
    },
    "ball_bearing_diffuse": {
        "min_count": 0,
        "max_count": 1,
        "required": False,
        "description": "Optional chrome ball bearing diffuse photograph for Path B calibration"
    },
    "ball_bearing_torch_1": {
        "min_count": 0,
        "max_count": 1,
        "required": False,
        "description": "Optional chrome ball bearing torch 1 photograph for Path B calibration"
    },
    "ball_bearing_torch_2": {
        "min_count": 0,
        "max_count": 1,
        "required": False,
        "description": "Optional chrome ball bearing torch 2 photograph for Path B calibration"
    },
    "ball_bearing_torch_3": {
        "min_count": 0,
        "max_count": 1,
        "required": False,
        "description": "Optional chrome ball bearing torch 3 photograph for Path B calibration"
    },
    "ball_bearing_torch_4": {
        "min_count": 0,
        "max_count": 1,
        "required": False,
        "description": "Optional chrome ball bearing torch 4 photograph for Path B calibration"
    }
}


def match_filename_to_tag(filename: str) -> Optional[str]:
    """Map a capture filename to its CAPTURE_SCHEMA tag by naming convention.

    'torch_1.RAF', 'torch_1_a.CR2' and 'torch_1-2.NEF' all map to 'torch_1'.
    Longer tags are tried first so 'ball_bearing_torch_1' is never read as
    'torch_1'. Shared by upload validation and Step 1 so both always agree.
    """
    stem = Path(filename).stem.lower()
    for tag in sorted(CAPTURE_SCHEMA.keys(), key=len, reverse=True):
        if stem == tag or stem.startswith((tag + "_", tag + "-", tag + ".")):
            return tag
    return None


def validate_captured_tags(tagged_counts: Dict[str, int]) -> Dict[str, Any]:
    """Validate a dictionary of {tag: count} against CAPTURE_SCHEMA.
    
    Returns a dictionary with:
    - 'ready': bool (True if all required tags pass)
    - 'tag_status': dict of tag -> {'status': 'Pass'|'Fail (missing)'|'Fail (duplicate)', 'count': int, 'required': bool}
    """
    ready = True
    tag_status = {}

    for tag, rules in CAPTURE_SCHEMA.items():
        count = tagged_counts.get(tag, 0)
        required = rules["required"]
        min_c = rules["min_count"]
        max_c = rules["max_count"]

        if count < min_c:
            if required:
                status = "Fail (missing)"
                ready = False
            else:
                status = "Optional (not provided)"
        elif max_c is not None and count > max_c:
            status = "Fail (duplicate/over max)"
            if required:
                ready = False
        else:
            status = "Pass"

        tag_status[tag] = {
            "status": status,
            "count": count,
            "required": required,
            "description": rules["description"]
        }

    return {
        "ready": ready,
        "tag_status": tag_status
    }
