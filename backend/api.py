"""IPDS backend API bridge.

This module exposes the frozen backend API surface declared in the project
specification. It intentionally stays thin: each function delegates to the
existing backend services and catches exceptions so the UI layer can surface a
friendly message without crashing.
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from skimage.filters import threshold_otsu

from backend.config.capture_schema import validate_captured_tags
from backend.config.pipeline_constants import (
    BRIGHTNESS_MISMATCH_TOLERANCE,
    COMPOSITE_FIELD_ORDER,
    EDGE_VARIATION_MIN_EXCEEDED,
    VERDICT_LIKELY_EDGE_FRACTION,
    MAD_SCALE_CONSTANT,
    MIN_HOLES_PER_EDGE,
    FIELD_DRIVER_Z,
    FIELD_NOTABLE_Z,
    OUTLIER_TAIL_PROBABILITY,
    SUSPICION_BANDS,
    QUALITY_BACKLIT_SEPARATION_PASS,
    QUALITY_BACKLIT_SEPARATION_WARN,
    QUALITY_TORCH_WARN_MULTIPLIER,
    get_pipeline_constants_dict,
)
from backend.pipeline.phase0_ingestion.step01_raw_capture_ingestion import (
    decode_raw_to_luminance,
    match_filename_to_tag,
)
from backend.pipeline.phase1_calibration_registration.step02_lens_and_unit_calibration import (
    CalibrationTargetNotFoundError,
    detect_calibration_scale,
)
from backend.pipeline.phase2_geometry_classification.step07_perforation_candidate_detection import HoleRecord
from backend.pipeline.phase4_statistics_output.step19_reference_group_construction import build_reference_groups
from backend.pipeline.phase4_statistics_output.step20_robust_stats_deviation_score import robust_z_scores
from backend.pipeline.runner import PipelineRunner
from backend.reporting.pdf_report import PdfReportGenerator
from backend.session.archive import ArchiveError, SessionArchiver, SessionExistsError, SessionImporter
from backend.session.session_manager import SessionManager
from backend.validation.validation_engine import ValidationEngine


logger = logging.getLogger(__name__)


def _session_manager() -> SessionManager:
    return SessionManager()


def _manifest(session_id: str) -> Dict[str, Any]:
    return _session_manager().get_manifest_manager(session_id).load()


def _safe_error(message: str) -> Dict[str, Any]:
    """Structured failure returned to the UI. When called while handling an
    exception, the full traceback goes to the log (never to the user)."""
    if sys.exc_info()[0] is not None:
        logger.error(message, exc_info=True)
    else:
        logger.warning(message)
    return {"success": False, "error": message}


def _read_json(path: Path) -> Any:
    if not path.exists():
        return None
    try:
        with open(path, "r", encoding="utf-8") as handle:
            return json.load(handle)
    except (json.JSONDecodeError, OSError):
        return None


def _with_abs_paths(session_id: str, images: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Copy image entries adding abs_path (session-relative path resolved), for display only."""
    session_dir = _session_manager().get_session_dir(session_id)
    return [{**img, "abs_path": str(session_dir / str(img.get("path", ""))),
             "thumb_abs_path": str(session_dir / img["thumb"]) if img.get("thumb") else None} for img in images]


def validate_upload(session_id: str, tags: Dict[str, str]) -> Dict[str, Any]:
    """Feature 1: validate required capture tags for the session."""
    try:
        counts: Dict[str, int] = {}
        for key, value in (tags or {}).items():
            try:
                counts[str(key)] = int(value)
            except (TypeError, ValueError):
                counts[str(key)] = 0
        result = validate_captured_tags(counts)
        result["session_id"] = session_id
        return result
    except Exception as exc:  # pragma: no cover - defensive UI safety
        return _safe_error(f"Unable to validate upload: {exc}")


def _upload_result(session_id: str, counts: Dict[str, int]) -> Dict[str, Any]:
    """Shared payload for upload/import: what is now in raw/ plus the Feature 1 checklist."""
    result = validate_captured_tags(counts)
    result.update({
        "success": True,
        "error": None,
        "session_id": session_id,
        "files": _session_manager().list_raw_files(session_id),
    })
    return result


def upload_files(session_id: str, files: Dict[str, List[str]]) -> Dict[str, Any]:
    """Feature 1 (amended freeze, #30): copy user-chosen files into raw/ under their tags.

    files maps tag -> list of source paths (from the picker / drag-and-drop).
    Returns the raw/ file list and the per-tag checklist for the new state.
    """
    try:
        counts = _session_manager().store_raw_files(session_id, files or {})
        return _upload_result(session_id, counts)
    except Exception as exc:
        return _safe_error(f"Unable to store the capture files: {exc}")


def import_folder(session_id: str, folder: str) -> Dict[str, Any]:
    """Feature 1 (amended freeze, #31): take capture files from a folder by their names."""
    try:
        counts = _session_manager().import_raw_folder(session_id, folder)
        return _upload_result(session_id, counts)
    except Exception as exc:
        return _safe_error(f"Unable to import the folder: {exc}")


def get_raw_files(session_id: str) -> Dict[str, Any]:
    """Feature 1 (amended freeze, #32): files already in the session's raw/ folder and their checklist."""
    try:
        manager = _session_manager()
        if not manager.session_exists(session_id):
            return _upload_result_empty(session_id)
        return _upload_result(session_id, manager.get_tagged_files(session_id))
    except Exception as exc:
        return _safe_error(f"Unable to list the session's capture files: {exc}")


def _upload_result_empty(session_id: str) -> Dict[str, Any]:
    result = validate_captured_tags({})
    result.update({"success": True, "error": None, "session_id": session_id, "files": []})
    return result


def generate_report(session_id: str) -> str:
    """Feature 20 (amended freeze, #34): build results/report.pdf and return its path.

    Same convention as export_session: the path on success, the standard
    {"success": False, "error": ...} dict on failure.
    """
    try:
        return PdfReportGenerator(_session_manager()).generate_report(session_id)
    except Exception as exc:
        return _safe_error(f"Unable to generate the PDF report: {exc}")  # type: ignore[return-value]


def start_pipeline(session_id: str) -> Dict[str, Any]:
    """Process button (amended freeze, #33): run Steps 1-23 on the session's raw/ files.

    Blocking; the frontend calls it through Client.call_async and follows
    progress with get_pipeline_status. Requires the Feature 1 checklist to pass
    and the Feature 3 acknowledgement (enforced by the runner).
    """
    try:
        manager = _session_manager()
        check = validate_captured_tags(manager.get_tagged_files(session_id))
        if not check.get("ready"):
            missing = [t for t, s in check["tag_status"].items() if s["status"].startswith("Fail")]
            return _safe_error(f"Required captures are missing or duplicated: {', '.join(missing)}.")
        return PipelineRunner(manager).run(session_id)
    except Exception as exc:
        return _safe_error(f"Unable to start processing: {exc}")


def compute_elevation(height: float, distance: float, correction: float = 0.0) -> float:
    """Feature 2: compute torch elevation from height and distance."""
    try:
        return SessionManager.compute_elevation(float(height), float(distance), float(correction))
    except Exception as exc:  # pragma: no cover - defensive UI safety
        return float(correction)


def save_session_log(session_id: str, log_data: Dict[str, Any]) -> bool:
    """Feature 2: persist the session log."""
    try:
        manager = _session_manager()
        if not manager.session_exists(session_id):
            manager.create_session(session_id, str((log_data or {}).get("specimen_id", "")))
        return manager.save_session_log(session_id, log_data or {})
    except Exception:
        return False


def get_session_log(session_id: str) -> Dict[str, Any]:
    """Feature 2 (amended freeze, #35): the saved session log, so the form can be reloaded.

    Returns {"success": True, "error": None, "exists": bool, "log": dict | None}.
    "exists" is False for a session that has no log file yet.
    """
    try:
        manager = _session_manager()
        log = manager.load_session_log(session_id) if manager.session_exists(session_id) else None
        return {"success": True, "error": None, "exists": log is not None, "log": log}
    except Exception as exc:
        return _safe_error(f"Unable to read the session log: {exc}")


def acknowledge_shadow_bleed_check(session_id: str) -> bool:
    """Feature 3: record that a shadow-bleed check was acknowledged."""
    try:
        return _session_manager().set_shadow_bleed_acknowledged(session_id, True)
    except Exception:
        return False


def get_brightness_warnings(session_id: str) -> List[Dict[str, Any]]:
    """Feature 4: read brightness mismatches and warnings for the current session."""
    try:
        session_dir = _session_manager().get_session_dir(session_id)
        stage_path = session_dir / "stages" / "brightness_correction.json"
        payload = _read_json(stage_path) or {}
        factors = payload.get("correction_factors") or {}
        flagged = {w.get("torch_id"): w for w in payload.get("warnings") or [] if isinstance(w, dict)}
        # One entry per torch (Feature 4 shows all four chips, amber when out of tolerance).
        rows = []
        for torch_id in ("torch_1", "torch_2", "torch_3", "torch_4"):
            if torch_id not in factors:
                continue
            factor = float(factors[torch_id])
            warn = flagged.get(torch_id)
            rows.append({
                "torch_id": torch_id,
                "factor": factor,
                "deviation_pct": round((factor - 1.0) * 100.0, 1),
                "warning": warn is not None,
                "message": warn.get("message") if warn else "Within tolerance",
                "tolerance": payload.get("tolerance"),
            })
        return rows
    except Exception:
        return []


def get_pipeline_status(session_id: str) -> Dict[str, Any]:
    """Feature 5: current stage and status information for a session."""
    try:
        manager = _session_manager()
        if not manager.session_exists(session_id):
            return {"session_id": session_id, "status": "not_found", "stages": [], "holes_total": 0, "holes_processed": 0}
        manifest = manager.get_manifest_manager(session_id).load()
        return {
            "session_id": session_id,
            "status": manifest.get("status", "not_started"),
            "stages": manifest.get("stages", []),
            "holes_total": manifest.get("holes_total", 0),
            "holes_processed": manifest.get("holes_processed", 0),
            "low_confidence_holes": manifest.get("low_confidence_holes", []),
        }
    except Exception as exc:
        return _safe_error(f"Unable to read pipeline status: {exc}")


def get_manifest_diff(session_id: str, since_token: int = 0) -> Dict[str, Any]:
    """Feature 6: return the manifest changes since a given token."""
    try:
        manager = _session_manager()
        if not manager.session_exists(session_id):
            return {"session_id": session_id, "status": "not_found", "latest_token": 0, "new_images": [], "stages": [], "mod_token": 0}
        diff = manager.get_manifest_manager(session_id).get_manifest_diff(since_token)
        new_images = diff.get("new_images") or []
        if diff.get("status") == "in_progress":
            # A step's images are listed the moment they are written, a little
            # before their small previews exist (the pipeline makes those right
            # after the step). Hand the UI only images that are ready, in order,
            # and leave the token behind the first one that is not.
            ready = []
            for image in new_images:
                if not image.get("thumb"):
                    break
                ready.append(image)
            new_images = ready
            diff["latest_token"] = ready[-1]["token"] if ready else since_token
        diff["new_images"] = _with_abs_paths(session_id, new_images)
        return diff
    except Exception as exc:
        return _safe_error(f"Unable to read manifest diff: {exc}")


def get_hole_progress(session_id: str) -> Dict[str, Any]:
    """Feature 7: current hole processing progress."""
    try:
        manifest = _manifest(session_id)
        total = int(manifest.get("holes_total", 0))
        processed = int(manifest.get("holes_processed", 0))
        return {
            "session_id": session_id,
            "status": manifest.get("status", "not_started"),
            "holes_total": total,
            "holes_processed": processed,
            "remaining": max(total - processed, 0),
        }
    except Exception as exc:
        return _safe_error(f"Unable to read hole progress: {exc}")


def get_comparison_pair(session_id: str, image_id_a: str, image_id_b: str) -> Dict[str, Any]:
    """Feature 8: fetch a pair of images for side-by-side comparison."""
    try:
        images = (_manifest(session_id).get("images") or [])
        lookup = {str(img.get("path", "")): img for img in images}
        missing = [i for i in (image_id_a, image_id_b) if i not in lookup]
        if missing:
            return _safe_error(f"Image not found in this session: {', '.join(missing)}")
        a, b = _with_abs_paths(session_id, [lookup[image_id_a], lookup[image_id_b]])
        return {"success": True, "error": None,
                "pair": [{"image_id": image_id_a, "image": a}, {"image_id": image_id_b, "image": b}]}
    except Exception as exc:
        return _safe_error(f"Unable to fetch comparison pair: {exc}")


def query_gallery(session_id: str, filters: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Feature 10: query the session gallery using simple filters."""
    try:
        images = (_manifest(session_id).get("images") or [])
        criteria = dict(filters or {})
        # Feature 10's "flagged only" switch maps onto the manifest's flagged field.
        if criteria.pop("flagged_only", False):
            criteria["flagged"] = True
        filtered: List[Dict[str, Any]] = []
        for image in images:
            match = True
            for key, value in criteria.items():
                if value is None:
                    continue
                if image.get(key) != value:
                    match = False
                    break
            if match:
                filtered.append(image)
        return _with_abs_paths(session_id, filtered)
    except Exception:
        return []


def _hole_dir(session_id: str, hole_id: str) -> Path:
    """Per-hole folder; hole ids are single path components (e.g. 'top_04')."""
    if not hole_id or any(ch in str(hole_id) for ch in ("/", "\\", ":")) or hole_id in {".", ".."}:
        raise ValueError(f"Invalid hole id: {hole_id!r}")
    return _session_manager().get_session_dir(session_id) / "holes" / hole_id


def _hole_record(session_id: str, hole_id: str) -> Dict[str, Any]:
    return next((r for r in _hole_records(session_id) if r.get("hole_id") == hole_id), {})


def _fmt(value: Any, digits: int = 2, unit: str = "") -> str:
    return "n/a" if value is None else f"{float(value):.{digits}f}{unit}"


def get_hole_tooltip(session_id: str, hole_id: str) -> Dict[str, Any]:
    """Feature 11: key descriptor values for a hover tooltip (read, never recomputed)."""
    try:
        hole_path = _hole_dir(session_id, hole_id)
        score = _read_json(hole_path / "score.json") or {}
        desc = _read_json(hole_path / "descriptors.json") or {}
        rec = _hole_record(session_id, hole_id)
        shape = desc.get("shape") or {}
        rough = desc.get("roughness") or {}
        shadow = desc.get("shadow") or {}
        outlier = score.get("outlier") or {}
        lines = [
            f"Mean radius: {_fmt(shape.get('mean_radius_um'), 1, ' um')}",
            f"Roundness: {_fmt(shape.get('roundness_ratio'), 3)}",
            f"Roughness (contrast): {_fmt(rough.get('glcm_contrast'), 2)}",
            f"Shadow: {_fmt(shadow.get('strength'), 3)} @ {_fmt(shadow.get('direction_deg'), 0, ' deg')}",
            f"Deviation score: {_fmt(score.get('combined_score'), 2)}",
        ]
        return {
            "hole_id": hole_id,
            "type": rec.get("type"),
            "edge": rec.get("edge"),
            "position_index": rec.get("position_index"),
            "score": score.get("combined_score"),
            "outlier": outlier.get("is_outlier"),
            "band": suspicion_band(outlier.get("tail_probability")),
            "tail_probability": outlier.get("tail_probability"),
            "low_confidence": hole_id in set(_manifest(session_id).get("low_confidence_holes") or []),
            "values": {
                "mean_radius_um": shape.get("mean_radius_um"),
                "roundness_ratio": shape.get("roundness_ratio"),
                "glcm_contrast": rough.get("glcm_contrast"),
                "shadow_strength": shadow.get("strength"),
                "shadow_direction_deg": shadow.get("direction_deg"),
            },
            "summary": "\n".join(lines),
        }
    except Exception as exc:
        return _safe_error(f"Unable to load hole tooltip: {exc}")


FIELD_CATEGORY = {
    **{f: "shape" for f in COMPOSITE_FIELD_ORDER[:7]},
    **{f: "roughness" for f in COMPOSITE_FIELD_ORDER[7:12]},
    **{f: "shadow" for f in COMPOSITE_FIELD_ORDER[12:]},
}


def _composite_vector(session_id: str, hole_id: str) -> List[Optional[float]]:
    desc = _read_json(_hole_dir(session_id, hole_id) / "descriptors.json") or {}
    vec = (desc.get("composite") or {}).get("vector") or []
    out: List[Optional[float]] = []
    for v in vec:
        try:
            fv = float(v)
            out.append(fv if np.isfinite(fv) else None)
        except (TypeError, ValueError):
            out.append(None)
    return out


def _reference_group(session_id: str, hole_id: str) -> Dict[str, Any]:
    """Step 19 group for one hole (a pure function of the stored edge assignments)."""
    holes = [
        HoleRecord(r.get("hole_id"), r.get("centroid_px", [0, 0]), r.get("bbox_px", [0, 0, 0, 0]),
                   r.get("raw_outline_px", []), r.get("type"), r.get("edge"), r.get("position_index"))
        for r in _hole_records(session_id)
    ]
    return build_reference_groups(holes).get(hole_id, {})


def _group_description(group: Dict[str, Any], rec: Dict[str, Any]) -> str:
    n = len(group.get("comparison_group_ids", []))
    if group.get("stratification") == "edge":
        return f"Compared against {n} other teeth on the {str(rec.get('edge', '')).title()} edge."
    if group.get("stratification") == "interior":
        note = (f" Fewer than {MIN_HOLES_PER_EDGE} interior holes, so confidence is lower."
                if group.get("low_confidence") else "")
        return f"Compared against {n} other interior (full) holes on the specimen.{note}"
    return (f"Compared against {n} other teeth across the whole specimen (pooled): fewer than "
            f"{MIN_HOLES_PER_EDGE} teeth were available on this edge, so confidence is lower.")


def get_hole_breakdown(session_id: str, hole_id: str) -> Dict[str, Any]:
    """Feature 12: per-descriptor value vs. group median/MAD/z, sorted by |z|, drivers flagged.

    Every number is read from Step 18/20 outputs; nothing is recomputed.
    """
    try:
        hole_path = _hole_dir(session_id, hole_id)
        descriptors = _read_json(hole_path / "descriptors.json") or {}
        score = _read_json(hole_path / "score.json") or {}
        values = _composite_vector(session_id, hole_id)
        medians = score.get("group_median") or []
        mads = score.get("group_mad") or []
        zs = score.get("per_field_z") or []
        outlier = score.get("outlier") or {}
        threshold = outlier.get("threshold_used")
        rows = []
        for i, field in enumerate(COMPOSITE_FIELD_ORDER):
            z = float(zs[i]) if i < len(zs) and zs[i] is not None else None
            rows.append({
                "descriptor": field,
                "category": FIELD_CATEGORY[field],
                "hole_value": values[i] if i < len(values) else None,
                "group_median": medians[i] if i < len(medians) else None,
                "group_mad": mads[i] if i < len(mads) else None,
                "z": z,
                # Per-field driver: |z| beyond the two-sided Bonferroni level for 14 fields.
                "driver": z is not None and abs(z) > FIELD_DRIVER_Z,
            })
        rows.sort(key=lambda r: abs(r["z"]) if r["z"] is not None else -1.0, reverse=True)
        rec = _hole_record(session_id, hole_id)
        group = _reference_group(session_id, hole_id)
        return {
            "hole_id": hole_id,
            "type": rec.get("type"),
            "edge": rec.get("edge"),
            "combined_score": score.get("combined_score"),
            "threshold_used": threshold,
            "tail_probability": outlier.get("tail_probability"),
            "is_outlier": outlier.get("is_outlier"),
            "rows": rows,
            "comparison_group": {
                "ids": group.get("comparison_group_ids", []),
                "stratification": group.get("stratification"),
                "low_confidence": bool(group.get("low_confidence")),
                "description": _group_description(group, rec),
            },
            "descriptors": descriptors,
            "score": score,
        }
    except Exception as exc:
        return _safe_error(f"Unable to load hole breakdown: {exc}")


def _most_typical_hole(session_id: str, group_ids: List[str]) -> Optional[str]:
    """Feature 13 auto-reference: the group member closest to the group median.

    Reuses Step 20's robust z on the group members themselves (each member
    against the rest of the group) and picks the smallest combined deviation.
    """
    vectors = {gid: _composite_vector(session_id, gid) for gid in group_ids}
    vectors = {k: v for k, v in vectors.items() if v}
    if len(vectors) < 2:
        return next(iter(vectors), None)
    best_id, best = None, float("inf")
    for gid, vec in vectors.items():
        others = np.array([[np.nan if x is None else x for x in v] for k, v in vectors.items() if k != gid], dtype=float)
        z, _, _ = robust_z_scores(np.array([np.nan if x is None else x for x in vec], dtype=float), others)
        dev = float(np.sqrt(np.sum(z ** 2)))
        if dev < best:
            best_id, best = gid, dev
    return best_id


def _pair_side(session_id: str, hole_id: str) -> Dict[str, Any]:
    crops_dir = _hole_dir(session_id, hole_id) / "crops"
    return {
        "hole_id": hole_id,
        "values": _composite_vector(session_id, hole_id),
        "crops": {tag: str(crops_dir / f"{tag}.png") for tag in ("backlit", "torch_1")},
        "low_confidence": hole_id in set(_manifest(session_id).get("low_confidence_holes") or []),
    }


def get_pairwise_comparison(session_id: str, hole_id: str, compare_to: Optional[str] = None) -> Dict[str, Any]:
    """Feature 13: selected hole vs. a reference hole (auto: the group's most typical)."""
    try:
        auto = not compare_to
        if auto:
            group = _reference_group(session_id, hole_id)
            compare_to = _most_typical_hole(session_id, group.get("comparison_group_ids", []))
            if not compare_to:
                return _safe_error("No reference hole is available for comparison.")
        left = _pair_side(session_id, hole_id)
        right = _pair_side(session_id, compare_to)
        table = []
        for i, field in enumerate(COMPOSITE_FIELD_ORDER):
            a = left["values"][i] if i < len(left["values"]) else None
            b = right["values"][i] if i < len(right["values"]) else None
            table.append({"descriptor": field, "hole_a": a, "hole_b": b,
                          "delta": (a - b) if a is not None and b is not None else None})
        return {
            "success": True,
            "error": None,
            "hole_id": hole_id,
            "compare_to": compare_to,
            "auto_selected": auto,
            "left": left,
            "right": right,
            "table": table,
        }
    except Exception as exc:
        return _safe_error(f"Unable to compare holes: {exc}")


def _edge_hole_rows(session_id: str, edge: Optional[str]) -> List[Dict[str, Any]]:
    """Hole-level rows (position order) for one edge, or all holes when edge is None/'all'."""
    scope = None if edge in (None, "", "all", "All") else edge
    low_conf = set(_manifest(session_id).get("low_confidence_holes") or [])
    rows = []
    for rec in _hole_records(session_id):
        if scope is not None and rec.get("edge") != scope:
            continue
        hid = rec.get("hole_id")
        hole_path = _hole_dir(session_id, hid)
        desc = _read_json(hole_path / "descriptors.json") or {}
        score = _read_json(hole_path / "score.json") or {}
        rows.append({
            "hole_id": hid,
            "edge": rec.get("edge"),
            "position_index": rec.get("position_index"),
            "type": rec.get("type"),
            "shadow": desc.get("shadow") or {},
            "roughness": desc.get("roughness") or {},
            "combined_score": score.get("combined_score"),
            "is_outlier": bool((score.get("outlier") or {}).get("is_outlier")),
            "low_confidence": hid in low_conf,
        })
    rows.sort(key=lambda r: (str(r["edge"]), r["position_index"] or 0))
    return rows


def get_shadow_vectors(session_id: str, edge: Optional[str] = None) -> Dict[str, Any]:
    """Feature 14: shadow strength/direction per hole for one edge (or all), with outlier flags."""
    try:
        rows = _edge_hole_rows(session_id, edge)
        holes = [{k: r[k] for k in ("hole_id", "edge", "position_index", "shadow", "is_outlier", "low_confidence")}
                 for r in rows]
        return {"edge": edge or "all", "holes": holes}
    except Exception as exc:
        return _safe_error(f"Unable to summarize shadow vectors: {exc}")


FIELD_LABELS = {
    "mean_radius_um": "Mean radius (um)", "radius_std_um": "Radius variation (um)",
    "fourier_c1": "Fourier harmonic 1 (um)", "fourier_c2": "Fourier harmonic 2 (um)",
    "fourier_c3": "Fourier harmonic 3 (um)", "fourier_c4": "Fourier harmonic 4 (um)",
    "roundness_ratio": "Roundness ratio", "sobel_mean": "Edge gradient mean",
    "sobel_std": "Edge gradient spread", "glcm_contrast": "GLCM contrast",
    "glcm_homogeneity": "GLCM homogeneity", "residual_wobble": "Residual wobble (um)",
    "shadow_strength": "Shadow strength", "shadow_direction_deg": "Shadow direction (deg)",
}
CATEGORY_MEANING = {
    "shape": "how round versus flat or angular the hole outline is",
    "roughness": "how torn and fibrous versus clean the edge is",
    "shadow": "which way, and how strongly, the paper was pushed at the hole",
}


def get_math_explanation(session_id: str, hole_id: str) -> Dict[str, Any]:
    """Feature 15: per-category formula text filled with this hole's stored numbers.

    Built from Step 13/15/17 descriptors and Step 20 statistics already on
    disk; returns one {formula_lines, interpretation} block per category.
    """
    try:
        breakdown = get_hole_breakdown(session_id, hole_id)
        if breakdown.get("success") is False:
            return breakdown
        score = breakdown.get("score") or {}
        blocks: Dict[str, Dict[str, Any]] = {}
        for category in ("shape", "roughness", "shadow"):
            rows = [r for r in breakdown["rows"] if r["category"] == category]
            rows.sort(key=lambda r: COMPOSITE_FIELD_ORDER.index(r["descriptor"]))
            lines = []
            for r in rows:
                label = FIELD_LABELS.get(r["descriptor"], r["descriptor"])
                if r["hole_value"] is None or r["group_median"] is None or r["z"] is None:
                    lines.append(f"{label}: not measured for this hole")
                    continue
                lines.append(
                    f"{label}: {r['hole_value']:.3g} vs. group median {r['group_median']:.3g}, "
                    f"MAD {r['group_mad']:.3g} -> robust z = {r['z']:.2f}"
                )
            measured = [abs(r["z"]) for r in rows if r["z"] is not None]
            peak = max(measured) if measured else 0.0
            meaning = CATEGORY_MEANING[category]
            if peak > FIELD_DRIVER_Z:
                interpretation = f"This hole is well outside its group on {meaning}; this category helps drive the flag."
            elif peak > FIELD_NOTABLE_Z:
                interpretation = f"This hole is somewhat unusual on {meaning}, but below the flagging level."
            else:
                interpretation = f"This hole is within the normal spread of its group on {meaning}."
            blocks[category] = {"formula_lines": lines, "interpretation": interpretation}
        return {
            "hole_id": hole_id,
            "method": score.get("method", "summed_z"),
            "formula": f"robust z = {MAD_SCALE_CONSTANT} x (hole value - group median) / group MAD",
            "summary": (
                f"Combined deviation {_fmt(breakdown.get('combined_score'), 2)} "
                f"({score.get('method', 'summed_z')}); a genuine hole in a group like this scores at least "
                f"this high with probability {_fmt(breakdown.get('tail_probability'), 4)} "
                f"(flagged below {OUTLIER_TAIL_PROBABILITY}). "
                + breakdown["comparison_group"]["description"]
            ),
            "blocks": blocks,
            "descriptors": breakdown.get("descriptors"),
        }
    except Exception as exc:
        return _safe_error(f"Unable to load math explanation: {exc}")


def get_edge_summary(session_id: str, edge: str) -> Dict[str, Any]:
    """Feature 16: one edge's holes in position order with shadow and roughness series."""
    try:
        regions = _read_json(_session_manager().get_session_dir(session_id) / "stages" / "candidate_regions.json") or []
        matching = [r for r in regions if r.get("edge") == edge]
        rows = _edge_hole_rows(session_id, edge)
        series = [{
            "hole_id": r["hole_id"],
            "position_index": r["position_index"],
            "shadow_strength": r["shadow"].get("strength"),
            "shadow_direction_deg": r["shadow"].get("direction_deg"),
            "glcm_contrast": r["roughness"].get("glcm_contrast"),
            "is_outlier": r["is_outlier"],
            "low_confidence": r["low_confidence"],
        } for r in rows]
        return {"edge": edge, "holes": series, "regions": matching, "count": len(matching)}
    except Exception as exc:
        return _safe_error(f"Unable to summarize edge: {exc}")


def get_citation(descriptor_type: str) -> Dict[str, Any]:
    """Feature 17: return the static literature citation attached to a descriptor."""
    try:
        source = _read_json(Path(__file__).resolve().parent / "citations" / "descriptor_sources.json") or {}
        entry = source.get(str(descriptor_type), {})
        return {"descriptor_type": descriptor_type, "citation": entry}
    except Exception as exc:
        return _safe_error(f"Unable to load citation: {exc}")


BAND_LABELS = {"genuine": "low", "suspicious": "medium", "highly_suspicious": "high"}
EDGE_MEASURE_LABELS = {"spacing": "spacing between teeth", "notch_depth": "notch depth",
                       "hole_size": "hole size", **FIELD_LABELS}


def suspicion_band(tail_probability: Optional[float]) -> Optional[str]:
    """Feature 18 band ("low" / "medium" / "high") from the hole's tail probability (Step 21)."""
    if tail_probability is None:
        return None
    p = float(tail_probability)
    for key, (lo, hi) in SUSPICION_BANDS.items():
        if lo < p <= hi or (lo == 0.0 and p <= hi):
            return BAND_LABELS.get(key, key)
    return "low"


def _hole_records(session_id: str) -> List[Dict[str, Any]]:
    data = _read_json(_session_manager().get_session_dir(session_id) / "stages" / "holes.json")
    return data if isinstance(data, list) else []


def _top_drivers(score: Dict[str, Any], limit: int = 3) -> List[Dict[str, Any]]:
    """Descriptor fields with the largest |robust z| (already computed in Step 20)."""
    z = score.get("per_field_z") or []
    pairs = [(name, float(v)) for name, v in zip(COMPOSITE_FIELD_ORDER, z) if v is not None]
    pairs.sort(key=lambda p: abs(p[1]), reverse=True)
    return [{"descriptor": n, "z": round(v, 3)} for n, v in pairs[:limit] if v != 0.0]


def _overall_verdict(session_id: str, holes_total: int) -> Dict[str, Any]:
    """One consolidated, plain-language result for the whole stamp (Results tab headline).

    Built only from the edge consistency check (Step 22), the check that
    judges a stamp against its own teeth. A flagged hole is a tooth on an
    edge whose teeth are not alike. "Evidence strength" is the share of
    checked edges that fail: it grades how widespread the inconsistency is and
    is NOT a calibrated probability that the stamp is altered.
    """
    report = _read_json(_session_manager().get_session_dir(session_id) / "stages" / "edge_consistency.json")
    edges = [e for e in (report if isinstance(report, list) else []) if isinstance(e, dict)]
    failed = [e for e in edges if e.get("inconsistent")]
    checked, n_failed = len(edges), len(failed)
    teeth_flagged = sum(int(e.get("teeth", 0)) for e in failed)
    edge_lines = [{
        "edge": e.get("edge"),
        "teeth": e.get("teeth"),
        "inconsistent": bool(e.get("inconsistent")),
        "detail": ("teeth not alike in " + ", ".join(EDGE_MEASURE_LABELS.get(m, m) for m in e.get("exceeded", [])))
                  if e.get("inconsistent") else "teeth are alike",
    } for e in edges]
    if checked == 0:
        level, evidence = "not_assessed", None
        headline = "No overall result: too few teeth per edge to check consistency."
        explanation = ("Each edge needs at least 5 teeth. Review the individual holes below.")
    elif n_failed == 0:
        level, evidence = "none", 0
        headline = f"0 of {holes_total} holes flagged. No sign of reperforation."
        explanation = (f"The teeth on all {checked} checked edges are alike in spacing, notch depth and size, "
                       f"as a perforating machine leaves them.")
    else:
        evidence = int(round(100.0 * n_failed / checked))
        level = "likely" if n_failed / checked >= VERDICT_LIKELY_EDGE_FRACTION else "possible"
        verb = "is likely" if level == "likely" else "may have been"
        headline = (f"{teeth_flagged} of {holes_total} holes flagged. This stamp {verb} reperforated "
                    f"(evidence strength {evidence}%).")
        names = ", ".join(str(e.get("edge")) for e in failed)
        explanation = (f"{n_failed} of {checked} checked edges have teeth that are not alike ({names}). "
                       f"Genuine perforation is uniform along an edge; hand re-punched holes are not.")
    return {
        "level": level,
        "headline": headline,
        "explanation": explanation,
        "evidence_pct": evidence,
        "evidence_note": ("Evidence strength is the share of checked edges that fail the consistency check. "
                          "It is not a calibrated probability: treat it as a guide for expert review."),
        "holes_flagged": teeth_flagged,
        "holes_total": holes_total,
        "edges_checked": checked,
        "edges_flagged": n_failed,
        "edges": edge_lines,
    }


def get_results_summary(session_id: str) -> Dict[str, Any]:
    """Feature 18: per-hole and per-region suspicion bands, always with their evidence."""
    try:
        session_dir = _session_manager().get_session_dir(session_id)
        low_conf = set(_manifest(session_id).get("low_confidence_holes") or [])
        regions = _read_json(session_dir / "stages" / "candidate_regions.json") or []
        region_of = {hid: r.get("region_id") for r in regions for hid in r.get("hole_ids", [])}
        rows: List[Dict[str, Any]] = []
        for rec in _hole_records(session_id):
            hid = rec.get("hole_id")
            score = _read_json(session_dir / "holes" / str(hid) / "score.json") or {}
            outlier = score.get("outlier") or {}
            rows.append({
                "hole_id": hid,
                "edge": rec.get("edge"),
                "position_index": rec.get("position_index"),
                "type": rec.get("type"),
                "combined_score": score.get("combined_score"),
                "is_outlier": outlier.get("is_outlier"),
                "threshold_used": outlier.get("threshold_used"),
                "tail_probability": outlier.get("tail_probability"),
                "band": suspicion_band(outlier.get("tail_probability")),
                "drivers": _top_drivers(score),
                "low_confidence": hid in low_conf,
                "region_id": region_of.get(hid),
            })
        order = {"high": 0, "medium": 1, "low": 2, None: 3}
        by_id = {r["hole_id"]: r for r in rows}
        region_rows = []
        for r in regions:
            members = [by_id[h] for h in r.get("hole_ids", []) if h in by_id]
            band = min((m["band"] for m in members), key=lambda b: order.get(b, 3), default=None)
            evidence = (f"{str(r.get('edge', '')).title()} edge, holes "
                        f"{r.get('start_position')}-{r.get('end_position')}")
            if r.get("basis") == "edge_consistency":
                # Whole-edge inconsistency, graded by how many measures exceed their threshold.
                exceeded = r.get("exceeded") or []
                band = "high" if len(exceeded) > EDGE_VARIATION_MIN_EXCEEDED else "medium"
                evidence += (": teeth not alike in " + ", ".join(EDGE_MEASURE_LABELS.get(m, m) for m in exceeded))
            region_rows.append({
                "region_id": r.get("region_id"),
                "edge": r.get("edge"),
                "start_position": r.get("start_position"),
                "end_position": r.get("end_position"),
                "band": band,
                "basis": r.get("basis", "tooth_outliers"),
                "evidence": evidence,
                "low_confidence": any(m["low_confidence"] for m in members),
            })
        return {"session_id": session_id, "holes": rows, "regions": region_rows,
                "verdict": _overall_verdict(session_id, len(rows))}
    except Exception as exc:
        return _safe_error(f"Unable to load results summary: {exc}")


def get_region_summary(session_id: str, region_id: str) -> Dict[str, Any]:
    """Feature 19: summarize a specific candidate region."""
    try:
        regions = _read_json(_session_manager().get_session_dir(session_id) / "stages" / "candidate_regions.json") or []
        region = next((r for r in regions if r.get("region_id") == region_id), None)
        if region is None:
            return {"success": False, "region": None, "error": f"Region {region_id} not found."}
        low_conf = set(_manifest(session_id).get("low_confidence_holes") or [])
        members = []
        per_field: Dict[str, List[float]] = {f: [] for f in COMPOSITE_FIELD_ORDER}
        for hid in region.get("hole_ids", []):
            score = _read_json(_hole_dir(session_id, hid) / "score.json") or {}
            zs = score.get("per_field_z") or []
            for field, z in zip(COMPOSITE_FIELD_ORDER, zs):
                if z is not None:
                    per_field[field].append(float(z))
            members.append({
                "hole_id": hid,
                "combined_score": score.get("combined_score"),
                "is_outlier": bool((score.get("outlier") or {}).get("is_outlier")),
                "low_confidence": hid in low_conf,
                "crop": str(_hole_dir(session_id, hid) / "crops" / "backlit.png"),
            })
        # Feature 19: aggregate each descriptor's z across the cluster.
        table = [{
            "descriptor": f,
            "category": FIELD_CATEGORY[f],
            "mean_z": float(np.mean(v)) if v else None,
            "mean_abs_z": float(np.mean(np.abs(v))) if v else None,
        } for f, v in per_field.items()]
        table.sort(key=lambda r: r["mean_abs_z"] if r["mean_abs_z"] is not None else -1.0, reverse=True)
        dominant = [r["descriptor"] for r in table[:3] if (r["mean_abs_z"] or 0.0) > 0.0]
        category_strength: Dict[str, float] = {}
        for r in table:
            if r["mean_abs_z"] is not None:
                category_strength[r["category"]] = max(category_strength.get(r["category"], 0.0), r["mean_abs_z"])
        lead = max(category_strength, key=category_strength.get) if category_strength else None
        statement = (f"This region is flagged mainly on {lead} "
                     f"({', '.join(FIELD_LABELS.get(d, d) for d in dominant)})." if lead else
                     "No descriptor stands out across this region.")
        consistency = None
        if region.get("basis") == "edge_consistency":
            report = _read_json(_session_manager().get_session_dir(session_id) / "stages" / "edge_consistency.json") or []
            consistency = next((e for e in report if e.get("edge") == region.get("edge")), None)
            if consistency:
                pct, limits = consistency["variation_pct"], consistency["thresholds_pct"]
                parts = ", ".join(f"{EDGE_MEASURE_LABELS.get(m, m)} varies {pct[m]:.1f}% (limit {limits[m]:.1f}%)"
                                  for m in consistency["exceeded"])
                statement = (f"The teeth on this edge are not alike: {parts}. A perforating machine leaves every "
                             f"hole of a row alike (one line, one spacing, one size); holes re-punched by hand "
                             f"are not. Measured on this stamp's own teeth only.")
        return {
            "success": True,
            "error": None,
            "region": region,
            "edge_consistency": consistency,
            "title": f"{str(region.get('edge', '')).title()} edge, holes "
                     f"{region.get('start_position')}-{region.get('end_position')}",
            "members": members,
            "descriptor_table": table,
            "dominant_descriptors": dominant,
            "dominant_category": lead,
            "statement": statement,
            "low_confidence": any(m["low_confidence"] for m in members),
        }
    except Exception as exc:
        return _safe_error(f"Unable to load region summary: {exc}")


def get_version_metadata(session_id: str) -> Dict[str, Any]:
    """Features 20 and 28: the version block stamped into the manifest when the run started.

    Never substitutes the current environment: versions reported for a result
    must be the ones that produced it. An unstamped (never-run) session gets a
    structured "not recorded" failure instead.
    """
    try:
        manifest = _session_manager().get_manifest_manager(session_id).load()
        metadata = manifest.get("version_metadata") or {}
        if not metadata:
            return _safe_error("No version metadata recorded yet: this session has not been run.")
        return metadata
    except Exception as exc:
        return _safe_error(f"Unable to load version metadata: {exc}")


def check_incomplete_session() -> Optional[Dict[str, Any]]:
    """Feature 22: return a resumable incomplete session if one exists."""
    try:
        return _session_manager().check_incomplete_session()
    except Exception:
        return None


def resume_session(session_id: str) -> Dict[str, Any]:
    """Feature 22: resume a session from the last completed step."""
    try:
        return _session_manager().resume_session(session_id)
    except Exception as exc:
        return _safe_error(f"Unable to resume session: {exc}")


def reset_session(session_id: str) -> bool:
    """Feature 23: delete the session directory."""
    try:
        return _session_manager().reset_session(session_id)
    except Exception:
        return False


def reprocess_from_stage(session_id: str, stage_id: str, corrected_inputs: Dict[str, Any] = None) -> Dict[str, Any]:
    """Feature 24: re-run the pipeline from a specific stage.

    stage_id is a step number 1-23 ("7", "step_7", "stage_07"). An unreadable
    id is rejected rather than silently becoming a full re-run.
    corrected_inputs may carry {"session_log": {...}}: those top-level keys are
    merged into session_log.json before re-running (e.g. corrected torch
    geometry before reprocessing from Step 3). Other keys are rejected.
    """
    try:
        text = str(stage_id).strip().lower().replace("step_", "").replace("stage_", "")
        try:
            step_id = int(text)
        except ValueError:
            return _safe_error(f"Unknown stage '{stage_id}'. Choose a step from 1 to 23.")
        if not 1 <= step_id <= 23:
            return _safe_error(f"Stage {step_id} is out of range. Choose a step from 1 to 23.")

        manager = _session_manager()
        if corrected_inputs:
            unknown = set(corrected_inputs) - {"session_log"}
            if unknown:
                return _safe_error(f"Unsupported corrected inputs: {', '.join(sorted(unknown))}.")
            patch = corrected_inputs.get("session_log") or {}
            if not isinstance(patch, dict):
                return _safe_error("corrected_inputs['session_log'] must be an object.")
            log = manager.load_session_log(session_id) or {}
            log.update(patch)
            manager.save_session_log(session_id, log)

        return PipelineRunner(manager).reprocess_from_stage(session_id, step_id)
    except Exception as exc:
        return _safe_error(f"Unable to reprocess from stage: {exc}")


def get_config_snapshot(session_id: str) -> Dict[str, Any]:
    """Feature 25: return the current config snapshot and version metadata."""
    try:
        return {
            "session_id": session_id,
            "constants": get_pipeline_constants_dict(),
            "version_metadata": get_version_metadata(session_id),
        }
    except Exception as exc:
        return _safe_error(f"Unable to read config snapshot: {exc}")


def run_validation(specimen_set_id: str) -> Dict[str, Any]:
    """Feature 26: run the validation benchmark for the specified specimen set."""
    try:
        return ValidationEngine(_session_manager()).run_validation(specimen_set_id)
    except Exception as exc:
        return _safe_error(f"Validation could not run: {exc}")


def get_validation_report(run_id: str) -> Dict[str, Any]:
    """Feature 26: fetch a saved validation report."""
    try:
        report = ValidationEngine(_session_manager()).get_validation_report(run_id)
        if report is None:
            return {"success": False, "report": None, "error": f"Run {run_id} not found."}
        return {"success": True, "report": report}
    except Exception as exc:
        return _safe_error(f"Unable to load validation report: {exc}")


def export_session(session_id: str, destination: Optional[str] = None) -> str:
    """Feature 27: package the session directory into a zip archive.

    destination (amended freeze): the .zip path the user chose; without it a
    timestamped archive is written next to the sessions folder. Any session
    can be exported, finished or not: the user asks for it explicitly.
    Returns the archive path on success. On failure it returns the standard
    structured failure dict ({"success": False, "error": ...}), like every other
    API function, so an error message can never be mistaken for a path.
    """
    try:
        return SessionArchiver(_session_manager()).export_session(
            session_id, allow_incomplete=True, destination=destination)
    except Exception as exc:
        return _safe_error(f"Unable to export session: {exc}")  # type: ignore[return-value]


def import_session(archive_path: str, new_session_id: Optional[str] = None) -> Dict[str, Any]:
    """Feature 27 (amended freeze, #36): restore a session from an exported archive.

    Returns {"success": True, "error": None, "session_id", "status"}. When the
    archive's session name is already taken: {"success": False, "error": ...,
    "exists": True, "session_id": <taken id>, "suggested_id": <free id>} so the
    caller can ask for another name and call again with new_session_id.
    """
    try:
        manager = _session_manager()
        result = SessionImporter(manager).import_session(archive_path, new_session_id)
        return {"success": True, "error": None, **result}
    except SessionExistsError as exc:
        taken = exc.session_id
        suggestion, n = f"{taken}_imported", 2
        while manager.session_exists(suggestion):
            suggestion, n = f"{taken}_imported{n}", n + 1
        return {"success": False, "error": str(exc), "exists": True, "session_id": taken,
                "suggested_id": suggestion}
    except ArchiveError as exc:
        return _safe_error(str(exc))
    except Exception as exc:
        return _safe_error(f"Unable to import the session: {exc}")


def _grade(value: float, pass_at: float, warn_at: float, higher_is_better: bool) -> str:
    if higher_is_better:
        return "pass" if value >= pass_at else ("warn" if value >= warn_at else "fail")
    return "pass" if value <= pass_at else ("warn" if value <= warn_at else "fail")


def _raw_files_by_tag(session_id: str) -> Dict[str, List[Path]]:
    raw_dir = _session_manager().get_session_dir(session_id) / "raw"
    found: Dict[str, List[Path]] = {}
    if raw_dir.is_dir():
        for entry in sorted(raw_dir.iterdir()):
            tag = match_filename_to_tag(entry.name) if entry.is_file() else None
            if tag:
                found.setdefault(tag, []).append(entry)
    return found


def _check_backlit(files: Dict[str, List[Path]]) -> Dict[str, Any]:
    """Bimodality of the backlit frame: Otsu between-class / total variance (0..1)."""
    name = "Backlit contrast"
    if not files.get("backlit"):
        return {"criterion": name, "status": "fail", "value": None,
                "message": "No backlit photo uploaded."}
    img = decode_raw_to_luminance(files["backlit"][0])
    flat = img[np.isfinite(img)].ravel()
    total_var = float(np.var(flat))
    if total_var <= 0:
        return {"criterion": name, "status": "fail", "value": 0.0,
                "message": "Backlit photo is uniform; holes are not visible."}
    split = float(threshold_otsu(flat))
    lo, hi = flat[flat <= split], flat[flat > split]
    w0, w1 = lo.size / flat.size, hi.size / flat.size
    between = w0 * w1 * (float(hi.mean()) - float(lo.mean())) ** 2 if lo.size and hi.size else 0.0
    separation = between / total_var
    status = _grade(separation, QUALITY_BACKLIT_SEPARATION_PASS, QUALITY_BACKLIT_SEPARATION_WARN, True)
    message = {
        "pass": "Clear bright-hole / dark-paper separation.",
        "warn": "Weak hole/paper separation; check backlight evenness and exposure.",
        "fail": "Backlit photo is not bimodal; re-shoot with the light source directly behind the stamp.",
    }[status]
    return {"criterion": name, "status": status, "value": round(separation, 3), "message": message}


def _check_torch_exposure(files: Dict[str, List[Path]]) -> Dict[str, Any]:
    """Spread of the four torch frames' mean brightness, relative to their average."""
    name = "Torch exposure consistency"
    tags = ["torch_1", "torch_2", "torch_3", "torch_4"]
    missing = [t for t in tags if not files.get(t)]
    if missing:
        return {"criterion": name, "status": "fail", "value": None,
                "message": f"Missing torch photos: {', '.join(missing)}."}
    means = [float(np.mean(decode_raw_to_luminance(files[t][0]))) for t in tags]
    ref = float(np.mean(means))
    if ref <= 0:
        return {"criterion": name, "status": "fail", "value": None, "message": "Torch photos are black."}
    deviation = max(abs(m - ref) / ref for m in means)
    status = _grade(deviation, BRIGHTNESS_MISMATCH_TOLERANCE,
                    BRIGHTNESS_MISMATCH_TOLERANCE * QUALITY_TORCH_WARN_MULTIPLIER, False)
    message = {
        "pass": "The four torch photos have comparable exposure.",
        "warn": "One torch photo is noticeably brighter or dimmer; check battery, distance and ambient light.",
        "fail": "Torch exposures differ too much; re-shoot with matched torches.",
    }[status]
    return {"criterion": name, "status": status, "value": round(deviation, 3), "message": message}


def _check_calibration(files: Dict[str, List[Path]]) -> Dict[str, Any]:
    """Run Step 2's pattern detection early (Feature 29: fail the gate if not found)."""
    name = "Calibration target detected"
    if not files.get("calibration"):
        return {"criterion": name, "status": "fail", "value": None,
                "message": "No calibration photo uploaded."}
    try:
        _, microns_per_px = detect_calibration_scale(decode_raw_to_luminance(files["calibration"][0]))
    except CalibrationTargetNotFoundError:
        return {"criterion": name, "status": "fail", "value": None,
                "message": "Calibration target not detected; confirm the calibration photo is in focus "
                           "and fully visible, then re-upload."}
    return {"criterion": name, "status": "pass", "value": round(microns_per_px, 3),
            "message": f"Target found ({microns_per_px:.2f} um per pixel)."}


def run_quality_gate(session_id: str) -> Dict[str, Any]:
    """Feature 29: pre-run capture quality checks on the uploaded raw images.

    Returns one pass/warn/fail record per criterion. Any fail blocks processing
    (passed=False); warnings pass but must be acknowledged by the user.
    """
    try:
        files = _raw_files_by_tag(session_id)
        criteria = [_check_backlit(files), _check_torch_exposure(files), _check_calibration(files)]
        return {
            "session_id": session_id,
            "passed": all(c["status"] != "fail" for c in criteria),
            "needs_acknowledgement": any(c["status"] == "warn" for c in criteria),
            "criteria": criteria,
            "issues": [f"{c['criterion']}: {c['message']}" for c in criteria if c["status"] != "pass"],
        }
    except Exception as exc:
        return _safe_error(f"Unable to run quality gate: {exc}")


__all__ = [
    "upload_files",
    "import_folder",
    "get_raw_files",
    "start_pipeline",
    "generate_report",
    "validate_upload",
    "compute_elevation",
    "save_session_log",
    "get_session_log",
    "acknowledge_shadow_bleed_check",
    "get_brightness_warnings",
    "get_pipeline_status",
    "get_manifest_diff",
    "get_hole_progress",
    "get_comparison_pair",
    "query_gallery",
    "get_hole_tooltip",
    "get_hole_breakdown",
    "get_pairwise_comparison",
    "get_shadow_vectors",
    "get_math_explanation",
    "get_edge_summary",
    "get_citation",
    "get_results_summary",
    "get_region_summary",
    "get_version_metadata",
    "check_incomplete_session",
    "resume_session",
    "reset_session",
    "reprocess_from_stage",
    "get_config_snapshot",
    "run_validation",
    "get_validation_report",
    "export_session",
    "import_session",
    "run_quality_gate",
]
