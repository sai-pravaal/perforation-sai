"""Step 18: Composite Signature Assembly & Normalization.

File: backend/pipeline/phase3_feature_extraction/step18_composite_signature_assembly.py
Concatenates the shape (Step 13), roughness (Step 15), and shadow (Step 17)
descriptors for one hole into a single ordered 14-element vector using the
fixed COMPOSITE_FIELD_ORDER. No rescaling is performed here (deferred to
Step 20 per comparison group). Completes Phase 3 for the hole.
"""

import json
import logging
import math
from pathlib import Path
from typing import Any, Dict, List, Optional

from backend.config.pipeline_constants import COMPOSITE_FIELD_ORDER
from backend.pipeline.phase2_geometry_classification.step07_perforation_candidate_detection import HoleRecord
from backend.session.session_manager import SessionManager

logger = logging.getLogger(__name__)


def _load_descriptors(descriptors_path: Path) -> Dict[str, Any]:
    if not descriptors_path.exists():
        return {}
    try:
        with open(descriptors_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return {}


def assemble_vector(descriptors: Dict[str, Any]) -> List[float]:
    """Build the 14-element ordered feature vector from the three descriptor blocks."""
    shape = descriptors.get("shape", {})
    roughness = descriptors.get("roughness", {})
    shadow = descriptors.get("shadow", {})
    fourier = shape.get("fourier_coeffs") or []

    def num(value: Any) -> float:
        """Missing or None descriptor values become NaN ("not measured"), never 0.0."""
        return float(value) if value is not None else float("nan")

    def fourier_at(i: int) -> float:
        return num(fourier[i]) if i < len(fourier) else float("nan")

    field_values: Dict[str, float] = {
        "mean_radius_um": num(shape.get("mean_radius_um")),
        "radius_std_um": num(shape.get("radius_std_um")),
        "fourier_c1": fourier_at(0),
        "fourier_c2": fourier_at(1),
        "fourier_c3": fourier_at(2),
        "fourier_c4": fourier_at(3),
        "roundness_ratio": num(shape.get("roundness_ratio")),
        "sobel_mean": num(roughness.get("sobel_mean")),
        "sobel_std": num(roughness.get("sobel_std")),
        "glcm_contrast": num(roughness.get("glcm_contrast")),
        "glcm_homogeneity": num(roughness.get("glcm_homogeneity")),
        "residual_wobble": num(roughness.get("residual_wobble")),
        "shadow_strength": num(shadow.get("strength")),
        "shadow_direction_deg": num(shadow.get("direction_deg")),
    }
    return [field_values[name] for name in COMPOSITE_FIELD_ORDER]


def run_step18(
    session_id: str,
    holes: List[HoleRecord],
    session_manager: Optional[SessionManager] = None
) -> Dict[str, Dict[str, Any]]:
    """Execute Step 18: Composite Signature Assembly."""
    if session_manager is None:
        session_manager = SessionManager()

    session_dir = session_manager.get_session_dir(session_id)
    manifest = session_manager.get_manifest_manager(session_id)
    holes_dir = session_dir / "holes"

    composites: Dict[str, Dict[str, Any]] = {}

    for h in holes:
        desc_path = holes_dir / h.hole_id / "descriptors.json"
        descriptors = _load_descriptors(desc_path)
        vector = assemble_vector(descriptors)

        composite = {
            "vector": vector,
            "field_names": list(COMPOSITE_FIELD_ORDER)
        }
        composites[h.hole_id] = composite  # in memory: NaN marks "not measured"

        # On disk: null instead of NaN, which is not valid JSON for other parsers.
        descriptors["composite"] = {
            "vector": [None if (isinstance(v, float) and math.isnan(v)) else v for v in vector],
            "field_names": list(COMPOSITE_FIELD_ORDER),
        }
        with open(desc_path, "w", encoding="utf-8") as f:
            json.dump(descriptors, f, indent=2)

        manifest.increment_holes_processed()

    manifest.add_stage_entry(
        step_id=18,
        phase=3,
        stage_name="composite_signature_assembly",
        output_path="holes/",
        status="complete"
    )

    return composites
