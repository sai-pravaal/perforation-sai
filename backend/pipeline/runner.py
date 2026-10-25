"""Pipeline Execution Engine (runner.py).

File: backend/pipeline/runner.py
The single execution engine for the 23-step IPDS pipeline. It knows the fixed
step order and phase grouping, threads each step's in-memory output to the next,
emits progress events consumed by Features 5 and 7, writes manifest status
transitions for crash-resumable state, and supports resuming/reprocessing from
an arbitrary stage by reconstructing the required context from persisted outputs.

Contains zero Qt/PySide6 imports.
"""

import json
import logging
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import cv2
import numpy as np

from backend.pipeline.phase0_ingestion.step01_raw_capture_ingestion import RawFrameRecord, run_step01
from backend.pipeline.phase1_calibration_registration.step02_lens_and_unit_calibration import (
    CalibrationRecord,
    run_step02,
)
from backend.pipeline.phase1_calibration_registration.step03_illumination_geometry import run_step03
from backend.pipeline.phase1_calibration_registration.step04_brightness_matching import run_step04
from backend.pipeline.phase1_calibration_registration.step05_cross_pass_registration import run_step05
from backend.pipeline.phase2_geometry_classification.step06_stamp_boundary_orientation import (
    StampFrame,
    rotate_image_plane,
    run_step06,
)
from backend.pipeline.phase2_geometry_classification.step07_perforation_candidate_detection import (
    HoleRecord,
    run_step07,
)
from backend.pipeline.phase2_geometry_classification.step08_hole_tooth_classification import run_step08
from backend.pipeline.phase2_geometry_classification.step09_roi_extraction import (
    compute_crop_size_px,
    extract_hole_crops,
    load_linear_png,
    run_step09,
)
from backend.pipeline.phase2_geometry_classification.step10_focus_stack_fusion import fuse_hole_crops, run_step10
from backend.pipeline.phase3_feature_extraction.step11_subpixel_contour import run_step11
from backend.pipeline.phase3_feature_extraction.step12_centroid_radius_profile import run_step12
from backend.pipeline.phase3_feature_extraction.step13_shape_deviation_descriptor import run_step13
from backend.pipeline.phase3_feature_extraction.step14_boundary_texture_sampling import PATCH_ARRAY_FILE, run_step14
from backend.pipeline.phase3_feature_extraction.step15_roughness_descriptor import run_step15
from backend.pipeline.phase3_feature_extraction.step16_angular_brightness_sampling import run_step16
from backend.pipeline.phase3_feature_extraction.step17_directional_profile_fitting import run_step17
from backend.pipeline.phase3_feature_extraction.step18_composite_signature_assembly import run_step18
from backend.pipeline.phase4_statistics_output.step19_reference_group_construction import (
    build_reference_groups,
    run_step19,
)
from backend.pipeline.phase4_statistics_output.step20_robust_stats_deviation_score import run_step20
from backend.pipeline.phase4_statistics_output.step21_outlier_thresholding import run_step21
from backend.pipeline.phase4_statistics_output.step22_spatial_clustering import run_step22
from backend.pipeline.phase4_statistics_output.step23_final_output_generation import run_step23
from backend.reporting.version_stamp import stamp_session
from backend.session.session_manager import SessionManager

logger = logging.getLogger(__name__)

# Fixed step -> phase mapping (matches the frozen 5-phase grouping).
STEP_PHASE: Dict[int, int] = {
    1: 0,
    2: 1, 3: 1, 4: 1, 5: 1,
    6: 2, 7: 2, 8: 2, 9: 2, 10: 2,
    11: 3, 12: 3, 13: 3, 14: 3, 15: 3, 16: 3, 17: 3, 18: 3,
    19: 4, 20: 4, 21: 4, 22: 4, 23: 4,
}
PHASE_NAMES: Dict[int, str] = {
    0: "Ingestion",
    1: "Calibration & Registration",
    2: "Geometry & Classification",
    3: "Feature Extraction",
    4: "Statistical Detection & Output",
}
TORCH_TAGS = ["torch_1", "torch_2", "torch_3", "torch_4"]
FIRST_STEP = 1
LAST_STEP = 23
# Last step whose inputs exist only as in-memory frames from the step before it
# (Step 6 consumes Step 5's registered frames); restarts at or before it rebuild
# the chain from raw/.
FRAME_CHAIN_LAST_STEP = 6
# Last step that measures on image data (Step 16 samples the torch crops);
# restarts up to here rebuild frames and crops from raw/ at full precision.
LAST_IMAGE_STEP = 16


class PipelineRunner:
    """Executes the 23-step pipeline in order, with resume-from-stage support."""

    def __init__(self, session_manager: Optional[SessionManager] = None) -> None:
        self.session_manager = session_manager or SessionManager()

    # ------------------------------------------------------------------ public
    def run(
        self,
        session_id: str,
        start_step: int = FIRST_STEP,
        progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
        require_shadow_bleed_ack: bool = True,
    ) -> Dict[str, Any]:
        """Execute steps start_step..23. Returns a summary dict.

        On a fresh run (start_step == 1) the version stamp is written first.
        Progress events {step_id, phase, phase_name, status} are delivered to
        progress_callback and mirrored into manifest status transitions.
        A fresh run requires the Feature 3 shadow-bleed acknowledgement in the
        session log; validation runs over pre-captured specimens opt out.
        """
        if start_step == FIRST_STEP and require_shadow_bleed_ack:
            log = self.session_manager.load_session_log(session_id) or {}
            if not log.get("shadow_bleed_acknowledged"):
                message = ("The shadow-bleed check has not been acknowledged for this session. "
                           "Confirm it on the Session Log tab before processing.")
                self._emit(progress_callback, FIRST_STEP, "failed", detail=message)
                return {"status": "failed", "failed_step": FIRST_STEP, "error": message}

        manifest = self.session_manager.get_manifest_manager(session_id)
        if start_step == FIRST_STEP:
            # A fresh run starts from a clean slate: earlier stage entries,
            # images, hole counters and derived files would otherwise be
            # duplicated or double-counted.
            manifest.invalidate_from_stage(FIRST_STEP)
            self.session_manager.clear_outputs_from_stage(session_id, FIRST_STEP)
        manifest.set_status("in_progress")

        ctx: Dict[str, Any] = {"session_id": session_id}
        if start_step == FIRST_STEP:
            stamp_session(session_id, self.session_manager)
        else:
            # Measurement frames exist at full precision only in memory (the
            # PNGs on disk are rounded, clipped display copies). A restart that
            # needs image data therefore recomputes the frame chain (Steps 1-6)
            # from the RAW files in raw/; those steps are deterministic and
            # their outputs are simply rewritten in place.
            replay_to = (start_step - 1 if start_step <= FRAME_CHAIN_LAST_STEP
                         else FRAME_CHAIN_LAST_STEP if start_step <= LAST_IMAGE_STEP else 0)
            try:
                for step_id in range(FIRST_STEP, replay_to + 1):
                    self._execute_step(step_id, ctx)
                if start_step > FRAME_CHAIN_LAST_STEP:
                    self._rebuild_context(session_id, start_step, ctx)
            except Exception as exc:
                logger.exception("Rebuilding context for step %d failed", start_step)
                manifest.set_status("failed")
                self._emit(progress_callback, start_step, "failed", detail=str(exc))
                return {"status": "failed", "failed_step": start_step, "error": str(exc)}

        for step_id in range(start_step, LAST_STEP + 1):
            self._emit(progress_callback, step_id, "started")
            try:
                self._execute_step(step_id, ctx)
            except Exception as exc:  # surfaced to the caller, not crashed
                logger.exception("Pipeline step %d failed for session %s", step_id, session_id)
                manifest.add_stage_entry(
                    step_id=step_id,
                    phase=STEP_PHASE[step_id],
                    stage_name=f"step_{step_id:02d}",
                    output_path="",
                    status="failed",
                )
                manifest.set_status("failed")
                self._emit(progress_callback, step_id, "failed", detail=str(exc))
                return {"status": "failed", "failed_step": step_id, "error": str(exc)}
            self._emit(progress_callback, step_id, "complete")

        manifest.set_status("complete")
        return {"status": "complete", "last_step": LAST_STEP}

    def reprocess_from_stage(
        self,
        session_id: str,
        start_step: int,
        progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
    ) -> Dict[str, Any]:
        """Invalidate manifest entries at/after start_step, then resume from it."""
        manifest = self.session_manager.get_manifest_manager(session_id)
        manifest.invalidate_from_stage(start_step)
        self.session_manager.clear_outputs_from_stage(session_id, start_step)
        return self.run(session_id, start_step=start_step, progress_callback=progress_callback)

    # --------------------------------------------------------------- internals
    def _emit(
        self,
        callback: Optional[Callable[[Dict[str, Any]], None]],
        step_id: int,
        status: str,
        detail: Optional[str] = None,
    ) -> None:
        if callback is None:
            return
        phase = STEP_PHASE[step_id]
        event = {"step_id": step_id, "phase": phase, "phase_name": PHASE_NAMES[phase], "status": status}
        if detail is not None:
            event["detail"] = detail
        try:
            callback(event)
        except Exception:  # a UI callback must never break the pipeline
            logger.warning("progress_callback raised; ignoring", exc_info=True)

    def _execute_step(self, step_id: int, ctx: Dict[str, Any]) -> None:
        """Run one step, reading and writing its pieces of the in-memory context."""
        sid = ctx["session_id"]
        sm = self.session_manager

        if step_id == 1:
            ctx["frames"] = run_step01(sid, sm)
        elif step_id == 2:
            ctx["frames"], ctx["calibration"] = run_step02(sid, ctx["frames"], sm)
        elif step_id == 3:
            ctx["illumination"] = run_step03(sid, ctx["frames"], sm)
        elif step_id == 4:
            ctx["frames"], ctx["brightness"] = run_step04(sid, ctx["frames"], sm)
        elif step_id == 5:
            ctx["frames"] = run_step05(sid, ctx["frames"], sm)
        elif step_id == 6:
            ctx["frames"], ctx["stamp_frame"] = run_step06(sid, ctx["frames"], sm)
        elif step_id == 7:
            ctx["candidates"] = run_step07(sid, ctx["frames"], ctx["calibration"], sm)
        elif step_id == 8:
            ctx["holes"] = run_step08(sid, ctx["candidates"], ctx["stamp_frame"], sm)
            if not ctx["holes"]:
                # Every later step would loop over nothing and the session would
                # end "complete" with empty results; stop with a usable reason.
                raise ValueError(
                    "No perforation holes or teeth were detected in the backlit photo. "
                    "Check the backlit exposure and the calibration scale, then re-run."
                )
        elif step_id == 9:
            ctx["crops"] = run_step09(sid, ctx["holes"], ctx["frames"], ctx["calibration"], sm)
        elif step_id == 10:
            ctx["fused"] = run_step10(sid, ctx["holes"], ctx["crops"], sm)
        elif step_id == 11:
            ctx["outlines"] = run_step11(sid, ctx["holes"], self._working_crops(ctx), sm)
        elif step_id == 12:
            ctx["profiles"] = run_step12(sid, ctx["holes"], ctx["outlines"], sm)
        elif step_id == 13:
            ctx["shape"] = run_step13(sid, ctx["holes"], ctx["profiles"], ctx["outlines"], ctx["calibration"], sm)
        elif step_id == 14:
            ctx["patches"] = run_step14(sid, ctx["holes"], ctx["outlines"], self._working_crops(ctx), ctx["calibration"], sm)
        elif step_id == 15:
            ctx["roughness"] = run_step15(sid, ctx["holes"], ctx["patches"], ctx["profiles"], sm, ctx.get("calibration"))
        elif step_id == 16:
            ctx["tables"] = run_step16(sid, ctx["holes"], ctx["profiles"], self._working_crops(ctx), ctx["calibration"], sm)
        elif step_id == 17:
            ctx["shadow"] = run_step17(sid, ctx["holes"], ctx["tables"], ctx["illumination"], sm)
        elif step_id == 18:
            ctx["composites"] = run_step18(sid, ctx["holes"], sm)
        elif step_id == 19:
            ctx["reference_groups"] = run_step19(sid, ctx["holes"], sm)
        elif step_id == 20:
            ctx["scores"] = run_step20(sid, ctx["holes"], ctx["composites"], ctx["reference_groups"], sm)
        elif step_id == 21:
            ctx["flags"] = run_step21(sid, ctx["holes"], ctx["scores"], sm)
        elif step_id == 22:
            ctx["regions"] = run_step22(sid, ctx["holes"], ctx["flags"], ctx["stamp_frame"], sm)
        elif step_id == 23:
            overview = ctx["frames"]["diffuse"].brightness_array
            ctx["outputs"] = run_step23(sid, ctx["holes"], overview, ctx["reference_groups"], ctx["regions"], sm)
        else:
            raise ValueError(f"Unknown pipeline step id: {step_id}")

    @staticmethod
    def _working_crops(ctx: Dict[str, Any]) -> Dict[str, Dict[str, np.ndarray]]:
        """Prefer focus-fused crops (Step 10) when present, else raw crops (Step 9)."""
        return ctx.get("fused") or ctx.get("crops") or {}

    # ------------------------------------------------------- context rebuilding
    def _rebuild_context(self, session_id: str, start_step: int, ctx: Dict[str, Any]) -> None:
        """Reconstruct the in-memory context needed to resume at start_step from disk.

        Only used for start steps after FRAME_CHAIN_LAST_STEP. For restarts up
        to LAST_IMAGE_STEP the caller has already replayed Steps 1-6 from raw/,
        so frames and crops are full precision; patches come from Step 14's
        float32 array. Only Step 23's drawing canvas uses a PNG display copy.
        """
        session_dir = self.session_manager.get_session_dir(session_id)
        stages = session_dir / "stages"
        holes_dir = session_dir / "holes"

        # Steps 1-6 were already replayed from raw/ for image-dependent restarts;
        # their in-memory results take precedence over the JSON copies.
        if "calibration" not in ctx:
            calibration = self._load_calibration(stages)
            if calibration is not None:
                ctx["calibration"] = calibration

        if "illumination" not in ctx:
            illum = self._load_json(stages / "illumination_geometry.json")
            if isinstance(illum, list):
                ctx["illumination"] = illum

        if "stamp_frame" not in ctx:
            stamp_frame = self._load_stamp_frame(stages)
            if stamp_frame is not None:
                ctx["stamp_frame"] = stamp_frame
        stamp_frame = ctx.get("stamp_frame")

        holes = self._load_holes(stages)
        if holes:
            ctx["holes"] = holes
            ctx["reference_groups"] = build_reference_groups(holes)

        # Step 23 only draws on the overview, so the display copy is enough there.
        if start_step == 23 and "frames" not in ctx:
            frames = self._load_registered_frames(session_id, session_dir)
            if frames and stamp_frame is not None:
                for rec in frames.values():
                    rec.brightness_array = rotate_image_plane(rec.brightness_array, stamp_frame.rotation_deg)
            if frames:
                ctx["frames"] = frames

        # Per-hole crops, cut again from the full-precision replayed frames
        # (identical to what Steps 9-10 produced in the original run).
        if 10 <= start_step <= LAST_IMAGE_STEP and holes and "frames" in ctx:
            ctx["crops"] = extract_hole_crops(holes, ctx["frames"], compute_crop_size_px(ctx["calibration"]))
            if start_step >= 11:
                ctx["fused"] = {hid: fuse_hole_crops(c) for hid, c in ctx["crops"].items()}

        # Outlines / profiles for steps 12..15.
        if 12 <= start_step <= 15 and holes:
            ctx["outlines"] = self._load_outlines(holes_dir, holes)
        if 13 <= start_step <= 16 and holes:
            ctx["profiles"] = self._load_profiles(holes_dir, holes)

        # Boundary patches for step 15.
        if start_step == 15 and holes:
            ctx["patches"] = self._load_patches(holes_dir, holes)

        # Angular tables for step 17.
        if start_step == 17 and holes:
            ctx["tables"] = self._load_tables(holes_dir, holes)

        # Composites for step 20; scores/flags/regions for steps 21..23.
        if start_step >= 19 and holes:
            ctx["composites"] = self._load_composites(holes_dir, holes)
        if start_step >= 21 and holes:
            ctx["scores"] = self._load_scores(holes_dir, holes)
        if start_step >= 22 and holes:
            ctx["flags"] = self._load_flags(holes_dir, holes)
        if start_step == 23:
            ctx["regions"] = self._load_json(stages / "candidate_regions.json") or []

    # ---------------------------------------------------------------- loaders
    @staticmethod
    def _load_json(path: Path) -> Any:
        if not path.exists():
            return None
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            return None

    def _load_calibration(self, stages: Path) -> Optional[CalibrationRecord]:
        data = self._load_json(stages / "calibration.json")
        if not data:
            return None
        return CalibrationRecord(
            distortion_coeffs=data.get("distortion_coeffs", [0.0, 0.0, 0.0]),
            microns_per_pixel=data.get("microns_per_pixel", 10.0),
            calibration_source_tag=data.get("calibration_source_tag", "calibration"),
        )

    def _load_stamp_frame(self, stages: Path) -> Optional[StampFrame]:
        data = self._load_json(stages / "stamp_frame.json")
        if not data:
            return None
        return StampFrame(
            boundary_polygon=data.get("boundary_polygon", []),
            rotation_deg=data.get("rotation_deg", 0.0),
            stamp_bbox=data.get("stamp_bbox", [0.0, 0.0, 0.0, 0.0]),
        )

    def _load_holes(self, stages: Path) -> List[HoleRecord]:
        data = self._load_json(stages / "holes.json")
        if not isinstance(data, list):
            return []
        holes: List[HoleRecord] = []
        for d in data:
            holes.append(HoleRecord(
                hole_id=d.get("hole_id"),
                centroid_px=d.get("centroid_px", [0.0, 0.0]),
                bbox_px=d.get("bbox_px", [0.0, 0.0, 0.0, 0.0]),
                raw_outline_px=d.get("raw_outline_px", []),
                hole_type=d.get("type"),
                edge=d.get("edge"),
                position_index=d.get("position_index"),
            ))
        return holes

    def _load_registered_frames(self, session_id: str, session_dir: Path) -> Dict[str, RawFrameRecord]:
        reg_dir = session_dir / "stages" / "registered"
        frames: Dict[str, RawFrameRecord] = {}
        if not reg_dir.exists():
            return frames
        session_log = self.session_manager.load_session_log(session_id) or {}
        torch_meta_by_tag = {t.get("torch_id"): t for t in session_log.get("torches", [])}
        for png in reg_dir.glob("*_registered.png"):
            tag = png.name[: -len("_registered.png")]
            arr = load_linear_png(png)
            if arr is None:
                continue
            frames[tag] = RawFrameRecord(
                tag=tag,
                brightness_array=arr,
                source_filename=png.name,
                torch_meta=torch_meta_by_tag.get(tag),
            )
        return frames

    def _load_patches(self, holes_dir: Path, holes: List[HoleRecord]) -> Dict[str, List[np.ndarray]]:
        """Step 14 patches: the exact float32 array, else (older sessions) the PNG copies."""
        out: Dict[str, List[np.ndarray]] = {}
        for h in holes:
            patch_dir = holes_dir / h.hole_id / "boundary_patches"
            patches: List[np.ndarray] = []
            exact = patch_dir / PATCH_ARRAY_FILE
            if exact.is_file():
                patches = list(np.load(exact))
            elif patch_dir.is_dir():
                for png in sorted(patch_dir.glob("patch_*.png")):
                    img = load_linear_png(png)
                    if img is not None:
                        patches.append(img)
            out[h.hole_id] = patches
        return out

    def _load_outlines(self, holes_dir: Path, holes: List[HoleRecord]) -> Dict[str, List[List[float]]]:
        out: Dict[str, List[List[float]]] = {}
        for h in holes:
            data = self._load_json(holes_dir / h.hole_id / "subpixel_outline.json")
            out[h.hole_id] = data.get("outline_px", []) if isinstance(data, dict) else []
        return out

    def _load_profiles(self, holes_dir: Path, holes: List[HoleRecord]) -> Dict[str, Dict[str, Any]]:
        out: Dict[str, Dict[str, Any]] = {}
        for h in holes:
            data = self._load_json(holes_dir / h.hole_id / "radius_profile.json")
            if isinstance(data, dict):
                out[h.hole_id] = data
        return out

    def _load_tables(self, holes_dir: Path, holes: List[HoleRecord]) -> Dict[str, Dict[str, Any]]:
        out: Dict[str, Dict[str, Any]] = {}
        for h in holes:
            data = self._load_json(holes_dir / h.hole_id / "angular_brightness_table.json")
            if isinstance(data, dict):
                out[h.hole_id] = data
        return out

    def _load_composites(self, holes_dir: Path, holes: List[HoleRecord]) -> Dict[str, Dict[str, Any]]:
        out: Dict[str, Dict[str, Any]] = {}
        for h in holes:
            data = self._load_json(holes_dir / h.hole_id / "descriptors.json")
            if isinstance(data, dict) and "composite" in data:
                comp = dict(data["composite"])
                # JSON stores "not measured" as null; Step 20 expects NaN.
                comp["vector"] = [float("nan") if v is None else v for v in comp.get("vector") or []]
                out[h.hole_id] = comp
        return out

    def _load_scores(self, holes_dir: Path, holes: List[HoleRecord]) -> Dict[str, Dict[str, Any]]:
        out: Dict[str, Dict[str, Any]] = {}
        for h in holes:
            data = self._load_json(holes_dir / h.hole_id / "score.json")
            if isinstance(data, dict):
                out[h.hole_id] = data
        return out

    def _load_flags(self, holes_dir: Path, holes: List[HoleRecord]) -> Dict[str, Dict[str, Any]]:
        out: Dict[str, Dict[str, Any]] = {}
        for h in holes:
            data = self._load_json(holes_dir / h.hole_id / "score.json")
            if isinstance(data, dict) and "outlier" in data:
                out[h.hole_id] = data["outlier"]
        return out
