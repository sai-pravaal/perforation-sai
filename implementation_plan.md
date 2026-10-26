# IPDS Implementation Plan & Frozen Architectural Conventions

## Executive Overview
The Intra-Specimen Perforation Deviation Signature (IPDS) application is a standalone desktop application for reperforation detection on postage stamps. The architecture separates the PySide6 user interface (`frontend/`) and the computational/data processing engine (`backend/`) through a strict API bridge (`backend/api.py` and `frontend/client.py`).

Per the project specifications and hierarchy of authority, all architectural conventions, data contracts, constants, and execution phases are frozen herein.

---

## User Review Required
> [!IMPORTANT]
> **Validation Data Rule**: Under no circumstances will synthetic or simulated stamp images / specimen datasets be generated. All image processing steps are built against the exact specifications and verified with synthetic numeric matrices / mocks during unit testing, with real capture images tested only when provided by the user.
> 
> **Development Order**: We will strictly proceed phase by phase (Phases 1 through 18). Each phase must be explicitly approved before moving to the next. At the end of each phase, a comprehensive summary document (adequate and detailed enough for an agent to fully understand all context, no strict 50-line limit, plain text only, no emoji or diagrams) will be created in `phases/phase_N_summary.md`.

---

## 1. Frozen Data Contracts

All data contracts represent schemas serialized to JSON or kept in-memory across the pipeline.

### 1.1 `SessionLog` (`data/sessions/<session_id>/session_log.json`)
- `session_id`: `str` — Unique session identifier (directory name under `data/sessions/`).
- `specimen_id`: `str` — User-assigned specimen identifier.
- `torches`: `list[dict]` — List of 4 entries for torches 1 to 4:
  - `torch_id`: `str` (`"torch_1"`, `"torch_2"`, `"torch_3"`, `"torch_4"`)
  - `azimuth_deg`: `float`
  - `height_cm`: `float`
  - `distance_cm`: `float`
  - `manual_correction_deg`: `float` (default `0.0`)
  - `computed_elevation_deg`: `float` ($\arctan(\text{height}/\text{distance}) + \text{correction}$)
- `camera_settings`: `dict` — `{aperture: float, shutter_speed: float, iso: int, white_balance: str}` (must match across all frames).
- `ball_bearing_used`: `bool` — Whether Path B calibration photos were supplied.
- `capture_timestamp`: `str` — ISO-8601 formatted timestamp.
- `operator_notes`: `str` — Optional freeform notes.

### 1.2 `RawFrameRecord` (in-memory / working cache)
- `tag`: `str` (`"diffuse"`, `"torch_1"`, `"torch_2"`, `"torch_3"`, `"torch_4"`, `"backlit"`, `"calibration"`, `"ball_bearing_diffuse"`, `"ball_bearing_torch_1"`, `"ball_bearing_torch_2"`, `"ball_bearing_torch_3"`, `"ball_bearing_torch_4"`)
- `brightness_array`: `numpy.ndarray (float32, [H, W])` — Linear luminance ($0.2126R + 0.7152G + 0.0722B$).
- `source_filename`: `str` — Original filename.
- `torch_meta`: `dict | None` — Copy of matching `SessionLog.torches[i]` entry (torch frames only).

### 1.3 `CalibrationRecord` (`data/sessions/<session_id>/stages/calibration.json`)
- `distortion_coeffs`: `list[float]` — $[k_1, k_2, k_3]$ radial distortion coefficients.
- `microns_per_pixel`: `float` — Physical scale conversion factor.
- `calibration_source_tag`: `str` — Tag of calibration frame (`"calibration"`).

### 1.4 `IlluminationGeometry` (`data/sessions/<session_id>/stages/illumination_geometry.json`)
- Array of 4 torch records:
  - `torch_index`: `int` ($1 \le i \le 4$)
  - `azimuth_deg`: `float` ($0.0 \le \theta < 360.0$)
  - `elevation_deg`: `float`
  - `source_path`: `str` (`"calculated"` or `"ball_bearing"`)

### 1.5 `StampFrame` (`data/sessions/<session_id>/stages/stamp_frame.json`)
- `boundary_polygon`: `list[list[float]]` — Stamp outer polygon in rotation-normalized coordinates (`[[x, y], ...]`).
- `rotation_deg`: `float` — Angle applied to rotate stamp axis-aligned.
- `stamp_bbox`: `list[float]` — `[x, y, w, h]` bounding box in normalized space.

### 1.6 `HoleRecord` (`data/sessions/<session_id>/stages/holes.json`)
- `hole_id`: `str` — Stable unique identifier (e.g., `"top_04"`).
- `type`: `str` — `"interior_hole"` or `"edge_tooth"`.
- `edge`: `str | None` — `"top"`, `"bottom"`, `"left"`, `"right"`, or `None` for unassigned interior holes.
- `position_index`: `int` — 1-based sequential slot index along edge.
- `centroid_px`: `list[float]` — `[x, y]` coordinates in normalized stamp coordinates.
- `bbox_px`: `list[float]` — `[x, y, w, h]` bounding box in normalized stamp coordinates.
- `raw_outline_px`: `list[list[float]]` — Coarse outline polygon.

### 1.7 `RadiusProfile` (`data/sessions/<session_id>/holes/<hole_id>/radius_profile.json`)
- `center_px`: `list[float]` — Subpixel centroid or circle fit `[x, y]`.
- `angles_deg`: `list[float]` — 72 sample angles ($0^\circ, 5^\circ, \dots, 355^\circ$).
- `distances_px`: `list[float | None]` — Measured radial distance per angle (`None` for tooth gap).
- `valid_mask`: `list[bool]` — 72 booleans indicating valid perimeter sample.

### 1.8 `ShapeDescriptor` (`data/sessions/<session_id>/holes/<hole_id>/descriptors.json` -> key: `shape`)
- `mean_radius_um`: `float`
- `radius_std_um`: `float`
- `fourier_coeffs`: `list[float]` — Magnitudes of harmonics 1 to 4.
- `roundness_ratio`: `float | None` — $4\pi \cdot \text{area} / \text{perimeter}^2$ (`None` / NaN for teeth).

### 1.9 `RoughnessDescriptor` (`data/sessions/<session_id>/holes/<hole_id>/descriptors.json` -> key: `roughness`)
- `sobel_mean`: `float`
- `sobel_std`: `float`
- `glcm_contrast`: `float`
- `glcm_homogeneity`: `float`
- `residual_wobble`: `float`

### 1.10 `ShadowDescriptor` (`data/sessions/<session_id>/holes/<hole_id>/descriptors.json` -> key: `shadow`)
- `strength`: `float`
- `direction_deg`: `float`
- `per_point_confidence_summary`: `dict` — `{mean: float, min: float, max: float}`

### 1.11 `CompositeSignature` (`data/sessions/<session_id>/holes/<hole_id>/descriptors.json` -> key: `composite`)
- `vector`: `list[float]` — 14-element ordered feature vector.
- `field_names`: `list[str]` — Parallel array of 14 feature field names.

### 1.12 `ReferenceGroup` (in-memory / exposed via API)
- `hole_id`: `str`
- `comparison_group_ids`: `list[str]` — IDs of peer holes in leave-one-out reference set.
- `stratification`: `str` — `"edge"` (normal) or `"pooled"` (fallback).
- `low_confidence`: `bool` — `True` when stratification is pooled.

### 1.13 `DeviationScore` (`data/sessions/<session_id>/holes/<hole_id>/score.json`)
- `per_field_z`: `list[float]` — Robust z-score per feature in composite vector.
- `combined_score`: `float` — Robust aggregate score.
- `method`: `str` — `"summed_z"` or `"mahalanobis"`.
- `group_median`: `list[float]` — Reference group median per field.
- `group_mad`: `list[float]` — Reference group MAD per field.

### 1.14 `OutlierFlag` (`data/sessions/<session_id>/holes/<hole_id>/score.json` -> key: `outlier`)
- `is_outlier`: `bool` — Whether combined_score exceeded cutoff.
- `threshold_used`: `float` — Cutoff threshold applied.

### 1.15 `CandidateRegion` (`data/sessions/<session_id>/stages/candidate_regions.json`)
- `region_id`: `str` — E.g., `"top_region_1"`.
- `edge`: `str` — `"top"`, `"bottom"`, `"left"`, or `"right"`.
- `hole_ids`: `list[str]` — Consecutive member hole IDs.
- `start_position`: `int` — Starting slot index.
- `end_position`: `int` — Ending slot index.
- `region_coords_px`: `list[list[float]]` — Physical polygon coordinates for drawing.

### 1.16 `Manifest` (`data/sessions/<session_id>/manifest.json`)
- `session_id`: `str`
- `status`: `str` — `"not_started"`, `"in_progress"`, `"complete"`, or `"failed"`.
- `stages`: `list[dict]` — Incremental list of `{step_id: int, phase: int, stage_name: str, output_path: str, timestamp: str, status: str}`.
- `images`: `list[dict]` — Displayable outputs `{path: str, stage: str, hole_id: str | None, edge: str | None, flagged: bool | None}`.
- `holes_total`: `int` — Populated at Step 8.
- `holes_processed`: `int` — Incremented as holes finish Step 18.
- `low_confidence_holes`: `list[str]` — Holes flagged low_confidence.
- `version_metadata`: `dict` — App version, config snapshot, library versions.

### 1.17 `ValidationResult` (`data/sessions/<validation_run_id>/validation_report.json`)
- `specimen_id`: `str`
- `ground_truth_region`: `dict | None`
- `predicted_region`: `dict | None`
- `confusion_counts`: `dict` — `{tp: int, fp: int, fn: int, tn: int}`
- `roc_points`: `list[dict]` — `[{threshold: float, tpr: float, fpr: float}, ...]`
- `repeatability_deltas`: `list[dict] | None` — `[{hole_id: str, score_delta: float}, ...]`

---

## 2. Frozen `pipeline_constants.py`

Every tunable threshold and constant lives in `backend/config/pipeline_constants.py`:

```python
# Fixed composite signature vector field order (14 fields)
COMPOSITE_FIELD_ORDER = [
    "mean_radius_um",
    "radius_std_um",
    "fourier_c1",
    "fourier_c2",
    "fourier_c3",
    "fourier_c4",
    "roundness_ratio",
    "sobel_mean",
    "sobel_std",
    "glcm_contrast",
    "glcm_homogeneity",
    "residual_wobble",
    "shadow_strength",
    "shadow_direction_deg"
]

# Robust Statistics Constants
MAD_SCALE_CONSTANT = 0.6745            # Step 20 MAD normal consistency factor
COMBINATION_METHOD = "summed_z"         # Step 20: "summed_z" or "mahalanobis"
OUTLIER_CUTOFF = 3.0                   # Step 21 default threshold

# Geometric and Feature Extraction Settings
FOURIER_FIT_ORDER = 4                  # Step 13 1st to 4th harmonic
MIN_HOLES_PER_EDGE = 5                 # Step 19 fallback threshold for pooling
RADIUS_PROFILE_NUM_ANGLES = 72         # Step 12 5-degree increments (360 / 5)

# Verification and Region Merging
BRIGHTNESS_MISMATCH_TOLERANCE = 0.15   # Step 4 & Feature 4 warning tolerance
REGION_MERGE_MAX_GAP = 1               # Step 22 maximum gap to bridge

# Suspicion Bands for Evidence Display (Feature 18)
SUSPICION_BANDS = {
    "genuine": (0.0, 2.5),
    "suspicious": (2.5, 3.5),
    "highly_suspicious": (3.5, float("inf"))
}
```

---

## 3. Python Naming Conventions & Code Rules
- **Files**: `snake_case.py` (matching `IPDS_Project_File_Structure.docx`).
- **Classes**: `PascalCase` (e.g., `ManifestManager`, `SessionManager`, `ValidationEngine`).
- **Functions & Methods**: `snake_case` (e.g., `validate_upload`, `reprocess_from_stage`).
- **Constants**: `UPPER_SNAKE_CASE` (e.g., `MAD_SCALE_CONSTANT`).
- **Variables & Parameters**: `snake_case` (e.g., `session_id`, `hole_id`).
- **No Qt in Backend**: `backend/` must have 0 imports from `PySide6` or `Qt`.
- **No Scientific/Image Libraries in Frontend**: `frontend/` must have 0 imports from `cv2`, `numpy`, `scipy`, `rawpy`, `skimage`, `shapely`.
- **No Direct I/O in UI**: Tabs and widgets communicate strictly via `frontend/client.py` -> `backend/api.py`.
- **Safe Calls in Client**: Every call in `frontend/client.py` wraps exceptions via a `safe_call` wrapper, returning structured success/failure dicts with friendly messages.
  - *Decision (2026-10-25, SESS-06):* this structured-result convention is the single error-reporting method. The frontend specification's alternative (re-raising a `ClientError`) is not used: a returned result can never escape a Qt slot half-handled, every view handles it through one function (`Client.unwrap`), and the message shown is always readable. Backend API functions return `{"success": False, "error": ...}` on failure and log the full traceback; the UI shows only the message.

---

## 4. Frozen `backend/api.py` Function Signatures

All 29 functions defined across Usability Features 1–29:

1. `validate_upload(session_id: str, tags: dict[str, str]) -> dict` (Feature 1)
2. `compute_elevation(height: float, distance: float, correction: float = 0.0) -> float` (Feature 2)
3. `save_session_log(session_id: str, log_data: dict) -> bool` (Feature 2)
4. `acknowledge_shadow_bleed_check(session_id: str) -> bool` (Feature 3)
5. `get_brightness_warnings(session_id: str) -> list[dict]` (Feature 4)
6. `get_pipeline_status(session_id: str) -> dict` (Feature 5)
7. `get_manifest_diff(session_id: str, since_token: int = 0) -> dict` (Feature 6)
8. `get_hole_progress(session_id: str) -> dict` (Feature 7)
9. `get_comparison_pair(session_id: str, image_id_a: str, image_id_b: str) -> dict` (Feature 8)
10. `query_gallery(session_id: str, filters: dict) -> list[dict]` (Feature 10)
11. `get_hole_tooltip(session_id: str, hole_id: str) -> dict` (Feature 11)
12. `get_hole_breakdown(session_id: str, hole_id: str) -> dict` (Feature 12)
13. `get_pairwise_comparison(session_id: str, hole_id: str, compare_to: str) -> dict` (Feature 13)
14. `get_shadow_vectors(session_id: str, edge: str) -> dict` (Feature 14)
15. `get_math_explanation(session_id: str, hole_id: str) -> dict` (Feature 15)
16. `get_edge_summary(session_id: str, edge: str) -> dict` (Feature 16)
17. `get_citation(descriptor_type: str) -> dict` (Feature 17)
18. `get_results_summary(session_id: str) -> dict` (Feature 18)
19. `get_region_summary(session_id: str, region_id: str) -> dict` (Feature 19)
20. `get_version_metadata(session_id: str) -> dict` (Features 20, 28)
21. `check_incomplete_session() -> dict | None` (Feature 22)
22. `resume_session(session_id: str) -> dict` (Feature 22)
23. `reset_session(session_id: str) -> bool` (Feature 23)
24. `reprocess_from_stage(session_id: str, stage_id: str, corrected_inputs: dict = None) -> dict` (Feature 24)
25. `get_config_snapshot(session_id: str) -> dict` (Feature 25)
26. `run_validation(specimen_set_id: str) -> dict` (Feature 26)
27. `get_validation_report(run_id: str) -> dict` (Feature 26)
28. `export_session(session_id: str) -> str` (Feature 27)
29. `run_quality_gate(session_id: str) -> dict` (Feature 29)

**Amendment (2026-10-25, approved by the project owner):** the freeze is extended so the
application can receive files and start a run, which the original 29 functions could not do.
30. `upload_files(session_id: str, files: dict[str, list[str]]) -> dict` (Feature 1 — copy tagged files into `raw/` under canonical names)
31. `import_folder(session_id: str, folder: str) -> dict` (Feature 1 — take convention-named files from a folder, the original workflow)
32. `get_raw_files(session_id: str) -> dict` (Feature 1 — list files already in `raw/` with their tags and the checklist)
33. `start_pipeline(session_id: str) -> dict` (Process button — run Steps 1–23; called off the GUI thread)
34. `generate_report(session_id: str) -> str` (Feature 20 — write `results/report.pdf`; returns its path, or the structured failure dict)
36. `import_session(archive_path: str, new_session_id: str | None = None) -> dict` (Feature 27 — restore a session from an exported zip; `export_session` also gained an optional `destination` path and exports unfinished sessions)
35. `get_session_log(session_id: str) -> dict` (Feature 2 — read back the saved session log so the form reloads without re-entering data; `{success, error, exists, log}`)

---

## 5. Folder Import Paths & Module Resolution

- **Root Directory**: `e:/coding/perforation project`.
- The root directory contains `backend/`, `frontend/`, `data/`, `tests/`, `docs/`, `phases/`, and `requirements.txt`.
- Module resolution is rooted at the workspace root, making `backend` and `frontend` top-level packages.
- Backend modules import from `backend...` (e.g., `from backend.config.pipeline_constants import MAD_SCALE_CONSTANT`).
- Frontend modules import from `frontend...` (e.g., `from frontend.client import client`).
- `client.py` imports `from backend.api import ...`.

---

## 6. Phase-by-Phase Roadmap

### Phase 1: Foundation & Configuration
- Files: `backend/config/pipeline_constants.py`, `backend/config/capture_schema.py`, `backend/citations/descriptor_sources.json`, `requirements.txt`, directory skeleton with `__init__.py` files.
- Summary: `phases/phase_1_summary.md`.

### Phase 2: Session State Layer
- Files: `backend/session/manifest.py`, `backend/session/session_manager.py`.
- Summary: `phases/phase_2_summary.md`.

### Phase 3: Ingestion, Calibration & Registration (Steps 1–5)
- Files: `backend/pipeline/phase0_ingestion/step01_raw_capture_ingestion.py`, `backend/pipeline/phase1_calibration_registration/step02_lens_and_unit_calibration.py`, `step03_illumination_geometry.py`, `step04_brightness_matching.py`, `step05_cross_pass_registration.py`.
- Summary: `phases/phase_3_summary.md`.

### Phase 4: Geometry, Localization & Classification (Steps 6–10)
- Files: `backend/pipeline/phase2_geometry_classification/step06_stamp_boundary_orientation.py`, `step07_perforation_candidate_detection.py`, `step08_hole_tooth_classification.py`, `step09_roi_extraction.py`, `step10_focus_stack_fusion.py`.
- Summary: `phases/phase_4_summary.md`.

### Phase 5: Per-Hole Feature Extraction (Steps 11–18)
- Files: `backend/pipeline/phase3_feature_extraction/step11_subpixel_contour.py` through `step18_composite_signature_assembly.py`.
- Summary: `phases/phase_5_summary.md`.

### Phase 6: Statistical Detection & Output (Steps 19–23)
- Files: `backend/pipeline/phase4_statistics_output/step19_reference_group_construction.py` through `step23_final_output_generation.py`.
- Summary: `phases/phase_6_summary.md`.

### Phase 7: Pipeline Orchestration & Version Stamping
- Files: `backend/pipeline/runner.py`, `backend/reporting/version_stamp.py`.
- Summary: `phases/phase_7_summary.md`.

### Phase 8: Reporting, Archiving & Validation Engine
- Files: `backend/reporting/pdf_report.py`, `backend/session/archive.py`, `backend/validation/validation_engine.py`, `backend/validation/labeled_specimens/` (empty directory).
- Summary: `phases/phase_8_summary.md`.

### Phase 9: Backend API Bridge
- Files: `backend/api.py` (all 29 functions implemented).
- Summary: `phases/phase_9_summary.md`.

### Phase 10: Frontend Application Shell
- Files: `frontend/main.py`, `frontend/main_window.py`, `frontend/client.py`, `frontend/theme/style.qss`.
- Summary: `phases/phase_10_summary.md`.

### Phase 11: Frontend Category A: Capture & Session Setup (Features 1–4)
- Files: `frontend/tabs/upload_tab.py`, `frontend/widgets/checklist_widget.py`, `frontend/tabs/session_log_tab.py`.
- Summary: `phases/phase_11_summary.md`.

### Phase 12: Frontend Category B: Processing Transparency (Features 5–7)
- Files: `frontend/tabs/processing_tab.py`, `frontend/widgets/progress_stepper.py`, `frontend/widgets/thumbnail_strip.py`.
- Summary: `phases/phase_12_summary.md`.

### Phase 13: Frontend Category C: View Tab (Features 8–11)
- Files: `frontend/tabs/view_tab.py`, `frontend/widgets/graphics_view_sync.py`, `frontend/widgets/breakdown_table.py` (tooltip mode).
- Summary: `phases/phase_13_summary.md`.

### Phase 14: Frontend Category D: Math Explanation & Traceability (Features 12–17)
- Files: `frontend/widgets/breakdown_table.py` (extended), `frontend/widgets/compass_plot.py`, `frontend/widgets/math_block.py`, `frontend/tabs/results_tab.py`.
- Summary: `phases/phase_14_summary.md`.

### Phase 15: Frontend Category E: Results & Reporting (Features 18–21)
- Files: `frontend/tabs/results_tab.py` (extended for summaries, low-confidence badges, PDF export).
- Summary: `phases/phase_15_summary.md`.

### Phase 16: Frontend Category F: Session Continuity (Features 22–24)
- Files: `frontend/dialogs/resume_session_dialog.py`, `frontend/dialogs/reset_confirm_dialog.py`, `frontend/tabs/processing_tab.py` (extended for reprocess).
- Summary: `phases/phase_16_summary.md`.

### Phase 17: Frontend Category G: Validation & Patent-Support (Features 25–29)
- Files: `frontend/tabs/validation_tab.py`, extensions to `results_tab.py`, `main_window.py`, and `upload_tab.py`.
- Summary: `phases/phase_17_summary.md`.

### Phase 18: Testing
- Files: `tests/phase0/` through `tests/phase4/`, `tests/test_validation_engine.py`.
- Summary: `phases/phase_18_summary.md`.

---

## 7. Verification Plan
- **Phase Gate Verifications**: After each phase, verify that imports resolve, functions conform to frozen contracts, syntax is clean, and the phase completion summary adheres to the rules.
- **Rule Verification**: Verify zero Qt imports in `backend/`, zero scientific imports in `frontend/`, zero synthetic images or mock specimen photographs created.
