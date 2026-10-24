Phase 4 Completion Summary: Geometry, Localization and Classification (Steps 6-10)

Note: This summary was authored during a post-hoc review. The Phase 4 code files already existed on disk but no summary had been produced and the phase was never confirmed. Each file was read, cross-referenced against IPDS_Backend_Algorithm_Pipeline_Steps.docx (steps 6-10), byte-compiled successfully, and verified against the frozen file structure and manifest contract.

Files Created and Purpose:
backend/pipeline/phase2_geometry_classification/step06_stamp_boundary_orientation.py implements Step 6. It separates the stamp from the dark backdrop in the diffuse frame via Otsu threshold, traces the outer contour with findContours, reads the tilt with minAreaRect, rotates every frame by the negative angle to axis-align the stamp, re-derives the axis-aligned boundary, and persists the StampFrame class (boundary_polygon, rotation_deg, stamp_bbox) to stages/stamp_frame.json.
step07_perforation_candidate_detection.py implements Step 7. It Otsu-thresholds the backlit frame, runs connectedComponentsWithStats, converts an expected physical hole-diameter range to a pixel-area window using microns_per_pixel, and emits preliminary HoleRecord rows (centroid_px, bbox_px, raw_outline_px; type/edge/position unset) to stages/holes.json. HoleRecord is defined here.
step08_hole_tooth_classification.py implements Step 8. It classifies each candidate as edge_tooth or interior_hole by proximity of its outline to the boundary polygon (pointPolygonTest), assigns each to the nearest of the four edges, sorts along the edge, assigns 1-based position_index, sets hole_id as edge_NN, rewrites holes.json, and sets manifest holes_total.
step09_roi_extraction.py implements Step 9. It computes a physical crop size (crop_scale_factor x expected hole diameter) converted to pixels via microns_per_pixel, extracts subpixel-centered square crops with getRectSubPix for diffuse/torch_1-4/backlit, and writes them to holes/<hole_id>/crops/<tag>.png.
step10_focus_stack_fusion.py implements Step 10. It computes per-pixel Laplacian sharpness maps and selects the sharpest source per pixel. With single-exposure capture it passes the one crop through unchanged and writes holes/<hole_id>/crops/fused_<tag>.png.

Backend API Function Signatures Added:
None. Per the development-order rules backend/api.py is built in Phase 9. The callable stage entry points added are run_step06, run_step07, run_step08, run_step09, run_step10, to be sequenced later by runner.py.

Key Data Structures and JSON Formats Used:
StampFrame (stages/stamp_frame.json): boundary_polygon list[[x,y]], rotation_deg float, stamp_bbox [x,y,w,h]. HoleRecord (stages/holes.json): hole_id, type, edge, position_index, centroid_px, bbox_px, raw_outline_px. Per-hole crop images written under holes/<hole_id>/crops/. All steps append stage entries to manifest.json via add_stage_entry(step_id, phase=2, stage_name, output_path, status); Step 8 additionally calls set_holes_total. All method signatures match backend/session/manifest.py and session_manager.py.

Ambiguities Encountered and Resolution:
Step 8 nearest-edge assignment is done using distance of each centroid to the four bbox lines rather than shapely line-segment projection; for the axis-aligned frame produced by Step 6 this is equivalent and was accepted.

Deviations From Spec Identified During Review (not yet corrected):
1. Step 6 omits the required cross-check of computed rotation against the torch azimuth 0-degree capture alignment and its quality warning.
2. Step 7 silently discards over-sized blobs; the spec requires over-sized blobs be flagged for manual review rather than dropped.
3. Steps 7, 8, 9 hardcode tunable thresholds inline (min/max diameter mm, edge-proximity margin, edge_tooth fraction 0.15, expected hole diameter, crop scale factor, crop-size clamp). This violates the frozen rule that every tunable constant lives only in backend/config/pipeline_constants.py.
4. Step 8 forces every hole (including interior_hole) onto an edge; the HoleRecord contract and spec allow edge and position_index to be null for interior holes not adjacent to an edge.
5. Step 10 does not implement the specified Laplacian-pyramid seam blend on the multi-depth branch; this is inert under single-exposure capture but is a latent gap if a real focus stack is supplied.

Decisions Made Not Explicitly in Documents:
Crop dimension is forced odd and clamped to the range 33-257 px for a symmetric center. Step 10 treats each lighting tag as an independent single-image stack when no depth series exists, acting as a pass-through. Verified zero Qt/PySide6 imports across all five files.

Update (2026-10-25 audit): Deviation 1 fixed: Step 6 compares the measured tilt with STAMP_ROTATION_WARN_DEG and records alignment_warning in stamp_frame.json. Deviation 2 fixed: Step 7 writes over-sized blobs (up to HOLE_OVERSIZE_REVIEW_FACTOR times the maximum plausible area) to stages/oversize_candidates.json for manual review. Deviation 3 fixed: all Step 7, 8 and 9 tunables now live in pipeline_constants.py. Deviation 5 fixed: Step 10 blends a real focus stack with a Laplacian pyramid (FOCUS_PYRAMID_LEVELS). Deviation 4 (interior holes forced onto an edge) remains open as PIPE-05 pending a decision on how interior holes are grouped. Also fixed in this phase's files: Step 6 rotation sign (frames are now rotated by +rotation_deg) and stamp_bbox taken from cv2.boundingRect so width and height cannot swap; Step 9 crop no longer clamped to 257 px and crops are saved as 16-bit linear PNGs; Step 6 writes a stamp_boundary.png thumbnail and Step 9 registers every crop in the manifest with hole_id and edge.
