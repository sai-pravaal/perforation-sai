"""Behaviour tests for audited defects (synthetic numeric matrices only; no specimen images)."""

import json

import cv2
import numpy as np
import pytest

from backend.session.manifest import ManifestManager, ManifestReadError
from backend.session.session_manager import SessionManager


def _tilted_rectangle(tilt_deg: float, size=(300.0, 200.0)) -> np.ndarray:
    img = np.zeros((600, 800), np.float32)
    box = cv2.boxPoints(((400.0, 300.0), size, tilt_deg)).astype(np.int32)
    cv2.fillPoly(img, [box], 1.0)
    return img


@pytest.mark.parametrize("tilt", [0.0, 3.0, -4.0])
def test_step06_rotation_cancels_tilt(tilt):
    """PIPE-01: rotating by +detected angle leaves the stamp axis-aligned."""
    from backend.pipeline.phase2_geometry_classification.step06_stamp_boundary_orientation import (
        detect_stamp_orientation, rotate_image_plane)
    angle, _, _ = detect_stamp_orientation(_tilted_rectangle(tilt))
    residual, _, _ = detect_stamp_orientation(rotate_image_plane(_tilted_rectangle(tilt), angle))
    assert abs(residual) < 0.5


def test_step06_bbox_width_height_not_swapped():
    """PIPE-02: a 300x200 axis-aligned stamp reports w=300, h=200."""
    from backend.pipeline.phase2_geometry_classification.step06_stamp_boundary_orientation import detect_stamp_orientation
    _, _, bbox = detect_stamp_orientation(_tilted_rectangle(0.0))
    assert abs(bbox[2] - 300) <= 3 and abs(bbox[3] - 200) <= 3


def test_step02_missing_target_raises():
    """PIPE-04: no silent 10 um/px fallback."""
    from backend.pipeline.phase1_calibration_registration.step02_lens_and_unit_calibration import (
        CalibrationTargetNotFoundError, detect_calibration_scale)
    with pytest.raises(CalibrationTargetNotFoundError):
        detect_calibration_scale(np.zeros((200, 200), np.float32))


@pytest.mark.parametrize("bad", ["", ".", "..", "../x", "a/b", "a\\b"])
def test_session_ids_cannot_escape_root(tmp_path, bad):
    """SESS-01: reset_session can never address the sessions root or outside it."""
    sm = SessionManager(tmp_path / "sessions")
    sm.create_session("keep_me")
    with pytest.raises(ValueError):
        sm.reset_session(bad)
    assert sm.session_exists("keep_me")


def test_manifest_unreadable_is_not_replaced(tmp_path):
    """SESS-02: a corrupt manifest raises instead of being overwritten with a blank one."""
    mm = ManifestManager(tmp_path / "s")
    mm.set_status("in_progress")
    mm.manifest_path.write_text("{ not json", encoding="utf-8")
    with pytest.raises(ManifestReadError):
        mm.load()


def test_manifest_invalidate_prunes_images_and_keeps_tokens_monotonic(tmp_path):
    """SESS-03: reprocess drops later-stage images; new tokens never repeat."""
    mm = ManifestManager(tmp_path / "s")
    t1 = mm.add_image_entry("stages/registered/a.png", "registration")
    t2 = mm.add_image_entry("results/annotated_overview.png", "final_output")
    mm.invalidate_from_stage(23)
    stages = [i["stage"] for i in mm.load()["images"]]
    assert stages == ["registration"]
    t3 = mm.add_image_entry("results/annotated_overview.png", "final_output")
    assert t3 > t2 > t1


def test_zero_mad_does_not_hide_outlier():
    """STAT-03: a hole far from an otherwise identical group gets a large z."""
    from backend.pipeline.phase4_statistics_output.step20_robust_stats_deviation_score import robust_z_scores
    z, _, _ = robust_z_scores(np.array([50.0]), np.array([[1.0], [1.0], [1.0], [2.0]]))
    assert z[0] > 10


def test_missing_descriptor_is_nan_not_zero():
    """STAT-04: unmeasured values enter the composite vector as NaN."""
    from backend.pipeline.phase3_feature_extraction.step18_composite_signature_assembly import assemble_vector
    vec = assemble_vector({"shape": {}, "roughness": {}, "shadow": {"strength": None}})
    assert all(np.isnan(v) for v in vec)


def test_radius_profile_is_interpolated_on_rays():
    """PIPE-16: an ellipse's radius is exact at 0 and 90 degrees."""
    from backend.pipeline.phase3_feature_extraction.step12_centroid_radius_profile import build_radius_profile
    th = np.linspace(0, 2 * np.pi, 61)
    outline = [[50 + 20 * np.cos(t), 50 + 10 * np.sin(t)] for t in th]
    prof = build_radius_profile(outline, "interior_hole")
    assert prof["distances_px"][0] == pytest.approx(20.0, abs=0.05)
    assert prof["distances_px"][18] == pytest.approx(10.0, abs=0.05)


def test_single_flagged_hole_is_not_a_region():
    """STAT-07: a lone flag is noise; two adjacent flags form a region."""
    from backend.pipeline.phase2_geometry_classification.step07_perforation_candidate_detection import HoleRecord
    from backend.pipeline.phase4_statistics_output.step22_spatial_clustering import build_candidate_regions
    holes = [HoleRecord(f"top_{i + 1:02d}", [10 + 20 * i, 5], [7 + 20 * i, 2, 6, 6], [], "edge_tooth", "top", i + 1)
             for i in range(6)]
    flags = lambda ids: {h.hole_id: {"is_outlier": h.hole_id in ids} for h in holes}  # noqa: E731
    assert build_candidate_regions(holes, flags({"top_03"}), [0, 0, 200, 100]) == []
    assert len(build_candidate_regions(holes, flags({"top_03", "top_04"}), [0, 0, 200, 100])) == 1


def test_mahalanobis_fallback_records_summed_z(tmp_path, monkeypatch):
    """STAT-05: score.json records the method actually used."""
    import backend.pipeline.phase4_statistics_output.step20_robust_stats_deviation_score as s20
    from backend.pipeline.phase2_geometry_classification.step07_perforation_candidate_detection import HoleRecord
    monkeypatch.setattr(s20, "COMBINATION_METHOD", "mahalanobis")
    sm = SessionManager(tmp_path / "sessions")
    sm.create_session("s")
    holes = [HoleRecord(f"top_{i}", [0, 0], [0, 0, 1, 1], [], "edge_tooth", "top", i) for i in range(1, 4)]
    comps = {h.hole_id: {"vector": [float(i)] * 14} for i, h in enumerate(holes)}
    groups = {h.hole_id: {"comparison_group_ids": [g.hole_id for g in holes if g is not h]} for h in holes}
    scores = s20.run_step20("s", holes, comps, groups, sm)
    assert all(s["method"] == "summed_z" for s in scores.values())


def test_pooled_scale_used_on_large_edges_and_calibrated(tmp_path):
    """STAT-01 follow-up: edges with >= 8 teeth borrow the spread across edges; null is calibrated."""
    import backend.pipeline.phase4_statistics_output.step20_robust_stats_deviation_score as s20
    from backend.pipeline.phase2_geometry_classification.step07_perforation_candidate_detection import HoleRecord
    from backend.pipeline.phase4_statistics_output.step19_reference_group_construction import build_reference_groups
    from backend.pipeline.phase4_statistics_output.step21_outlier_thresholding import run_step21
    rng = np.random.default_rng(11)
    sm = SessionManager(tmp_path / "sessions")
    sm.create_session("s")
    holes, comps = [], {}
    for e, edge in enumerate(["top", "bottom", "left", "right"]):
        n = 10 if e < 2 else 6
        for i in range(n):
            hid = f"{edge}_{i + 1:02d}"
            v = rng.standard_normal(14)
            v[6] = np.nan
            holes.append(HoleRecord(hid, [i, e], [i, e, 1, 1], [], "edge_tooth", edge, i + 1))
            comps[hid] = {"vector": list(v)}
    scores = s20.run_step20("s", holes, comps, build_reference_groups(holes), sm)
    assert scores["top_01"]["scale_method"] == "pooled_edge_factor"
    assert scores["top_01"]["pool_sizes"][0] == 9
    assert scores["left_01"]["scale_method"] == "group_mad"  # only 6 teeth on that edge
    flags = run_step21("s", holes, scores, sm)
    assert all(f["tail_probability"] is None or 0 < f["tail_probability"] <= 1 for f in flags.values())


def test_session_log_save_merges_acknowledgement(tmp_path):
    """SESS-04: saving the form keeps the shadow-bleed acknowledgement."""
    sm = SessionManager(tmp_path / "sessions")
    sm.create_session("s")
    sm.set_shadow_bleed_acknowledged("s", True)
    sm.save_session_log("s", {"session_id": "s", "torches": []})
    assert sm.load_session_log("s")["shadow_bleed_acknowledged"] is True


def test_run_refuses_without_shadow_bleed_ack(tmp_path):
    """SESS-04: a fresh run needs the Feature 3 acknowledgement."""
    from backend.pipeline.runner import PipelineRunner
    sm = SessionManager(tmp_path / "sessions")
    sm.create_session("s")
    result = PipelineRunner(sm).run("s")
    assert result["status"] == "failed" and "shadow-bleed" in result["error"]


def _dot_grid_frame(pitch_px: float = 60.0, dot_r: float = 8.0, shape=(2400, 3600)) -> np.ndarray:
    """Grey frame with a 7x7 grid of dark dots near the centre (target spans ~10% of the frame)."""
    img = np.full(shape, 0.06, np.float32)
    cy, cx = shape[0] / 2.0, shape[1] / 2.0
    for r in range(7):
        for c in range(7):
            cv2.circle(img, (int(round(cx + (c - 3) * pitch_px)), int(round(cy + (r - 3) * pitch_px))),
                       int(dot_r), 0.01, -1, lineType=cv2.LINE_AA)
    return img


def test_step02_small_grid_gives_scale_without_fitted_distortion():
    """CAL-01/02: a small central dot grid is found quickly and does not yield fitted distortion."""
    import time
    from backend.pipeline.phase1_calibration_registration.step02_lens_and_unit_calibration import (
        detect_calibration_scale)
    start = time.time()
    dist, upp = detect_calibration_scale(_dot_grid_frame(), known_pattern_spacing_um=1000.0)
    assert time.time() - start < 20.0
    assert dist == [0.0, 0.0, 0.0]
    assert upp == pytest.approx(1000.0 / 60.0, rel=0.01)


def test_manifest_image_entry_replaced_not_duplicated(tmp_path):
    """Replaying a step rewrites the same file; its manifest entry must not be duplicated."""
    m = ManifestManager(tmp_path / "s")
    first = m.add_image_entry("stages/stamp_boundary.png", "stamp_boundary")
    m.add_image_entries([{"path": "holes/a/crops/backlit.png", "stage": "roi_crop"}])
    second = m.add_image_entry("stages/stamp_boundary.png", "stamp_boundary")
    m.add_image_entries([{"path": "holes/a/crops/backlit.png", "stage": "roi_crop"}])
    paths = [img["path"] for img in m.load()["images"]]
    assert sorted(paths) == ["holes/a/crops/backlit.png", "stages/stamp_boundary.png"]
    assert second > first


def test_step07_finds_each_edge_notch_of_a_separated_stamp():
    """PIPE-07: notches open to the bright background are found one per notch."""
    from backend.pipeline.phase2_geometry_classification.step07_perforation_candidate_detection import (
        detect_perforation_candidates)
    upp = 10.0                      # 1.2 mm holes -> radius 60 px
    img = np.ones((1400, 1800), np.float32)                      # bright light box
    cv2.rectangle(img, (300, 300), (1500, 1100), 0.1, -1)        # paper
    xs = range(500, 1400, 200)                                   # 5 notches top and bottom
    ys = range(500, 1000, 200)                                   # 3 notches left and right
    for x in xs:
        cv2.circle(img, (x, 300), 60, 1.0, -1)
        cv2.circle(img, (x, 1100), 60, 1.0, -1)
    for y in ys:
        cv2.circle(img, (300, y), 60, 1.0, -1)
        cv2.circle(img, (1500, y), 60, 1.0, -1)
    cv2.circle(img, (900, 700), 60, 1.0, -1)                     # one interior (closed) hole
    found = detect_perforation_candidates(img, upp)
    assert len(found) == 2 * len(xs) + 2 * len(ys) + 1


def test_step13_tooth_arc_terms_are_stable_under_noise():
    """A noisy circular arc (~150 deg, as a tooth shows) gives small shape terms;
    the old 4-harmonic Fourier fit amplified the same noise by up to ~500x."""
    from backend.pipeline.phase3_feature_extraction.step13_shape_deviation_descriptor import fit_shape_model
    rng = np.random.default_rng(0)
    angles = [5.0 * i for i in range(72)]
    valid = [15 <= a <= 165 for a in angles]
    dist = [50.0 + rng.normal(0, 0.3) if v else None for a, v in zip(angles, valid)]
    terms, _, _ = fit_shape_model(angles, dist, valid, 4, partial_arc=True)
    assert max(abs(t) for t in terms) < 1.0          # radius noise is 0.3 px


def test_step17_torch_brightness_level_does_not_create_a_shadow():
    """A torch that is uniformly brighter at a hole must not produce a shadow vector."""
    from backend.pipeline.phase3_feature_extraction.step17_directional_profile_fitting import run_step17
    from backend.pipeline.phase2_geometry_classification.step07_perforation_candidate_detection import HoleRecord
    import tempfile
    rows = [{"angle_deg": a, "brightness": [1.0 * 1.3, 1.0, 1.0, 1.0]} for a in range(0, 360, 5)]
    table = {"torch_order": ["torch_1", "torch_2", "torch_3", "torch_4"], "rows": rows}
    geometry = [{"torch_index": i + 1, "azimuth_deg": 90.0 * i} for i in range(4)]
    with tempfile.TemporaryDirectory() as tmp:
        sm = SessionManager(tmp)
        sm.create_session("s")
        hole = HoleRecord("top_01", [0, 0], [0, 0, 1, 1], [], "edge_tooth", "top", 1)
        out = run_step17("s", [hole], {"top_01": table}, geometry, sm)
    assert out["top_01"]["strength"] == pytest.approx(0.0, abs=1e-9)


def test_step22_edge_consistency_separates_machine_and_irregular_edges():
    """Self-referential: teeth alike (one line, one spacing, one size) pass; unlike teeth are flagged."""
    from backend.config.pipeline_constants import COMPOSITE_FIELD_ORDER
    from backend.pipeline.phase4_statistics_output.step22_spatial_clustering import edge_consistency
    rng = np.random.default_rng(1)
    upp, pitch_px, radius_um = 10.0, 150.0, 500.0
    def teeth(spread):
        out = []
        for i in range(14):
            vector = [0.0] * len(COMPOSITE_FIELD_ORDER)
            vector[COMPOSITE_FIELD_ORDER.index("mean_radius_um")] = radius_um * (1 + rng.normal(0, spread))
            out.append({"centre_px": [100 + i * pitch_px * (1 + rng.normal(0, spread)),
                                      200 + rng.normal(0, spread * pitch_px)],
                        "vector": vector, "hole_id": f"top_{i + 1:02d}"})
        return out
    alike, unlike = edge_consistency(teeth(0.01), "top", upp), edge_consistency(teeth(0.15), "top", upp)
    assert not alike["inconsistent"]
    assert unlike["inconsistent"] and {"spacing", "notch_depth"} <= set(unlike["exceeded"])
    assert edge_consistency(teeth(0.01)[:4], "top", upp) is None   # too few teeth to judge


def test_thumbnails_are_small_and_the_diff_waits_for_them(tmp_path, monkeypatch):
    """The UI gets 8-bit previews, never full frames, and only for images whose preview exists."""
    import backend.api as api
    from backend.pipeline.thumbnails import ensure_thumbnails
    root = tmp_path / "sessions"
    monkeypatch.setattr(api, "_session_manager", lambda: SessionManager(root))
    sm = SessionManager(root)
    sm.create_session("t")
    sdir = sm.get_session_dir("t")
    (sdir / "stages").mkdir(exist_ok=True)
    big = np.full((1200, 1800), 20000, np.uint16)            # a "full frame" linear 16-bit PNG
    cv2.imwrite(str(sdir / "stages" / "frame.png"), big)
    cv2.imwrite(str(sdir / "stages" / "other.png"), big)
    manifest = sm.get_manifest_manager("t")
    manifest.set_status("in_progress")
    manifest.add_image_entry("stages/frame.png", "registration")
    manifest.add_image_entry("stages/other.png", "registration")

    # Written but no preview yet: nothing is offered to the UI, the token does not advance.
    early = api.get_manifest_diff("t", 0)
    assert early["new_images"] == [] and early["latest_token"] == 0

    assert ensure_thumbnails(sdir, manifest) == 2
    ready = api.get_manifest_diff("t", 0)
    assert len(ready["new_images"]) == 2
    thumb = cv2.imread(ready["new_images"][0]["thumb_abs_path"], cv2.IMREAD_UNCHANGED)
    assert thumb.dtype == np.uint8 and max(thumb.shape[:2]) == 256
    assert thumb[0, 0] == round(255 * (20000 / 65535) ** (1 / 2.2))   # display gamma applied to linear data
    assert ensure_thumbnails(sdir, manifest) == 0                       # already done: not repeated


def _verdict_session(tmp_path, monkeypatch, edges, n_holes=48):
    import backend.api as api
    root = tmp_path / "sessions"
    monkeypatch.setattr(api, "_session_manager", lambda: SessionManager(root))
    sm = SessionManager(root)
    sm.create_session("v")
    stages = sm.get_session_dir("v") / "stages"
    stages.mkdir(exist_ok=True)
    holes = [{"hole_id": f"top_{i:02d}", "type": "edge_tooth", "edge": "top", "position_index": i,
              "centroid_px": [0, 0], "bbox_px": [0, 0, 1, 1], "raw_outline_px": []} for i in range(1, n_holes + 1)]
    (stages / "holes.json").write_text(json.dumps(holes))
    (stages / "edge_consistency.json").write_text(json.dumps(edges))
    return api


def _edge(name, teeth, bad):
    return {"edge": name, "teeth": teeth, "inconsistent": bad, "exceeded": ["spacing", "notch_depth"] if bad else []}


def test_overall_verdict_wording_and_evidence_strength(tmp_path, monkeypatch):
    four = lambda *bad: [_edge(n, 12, n in bad) for n in ("top", "bottom", "left", "right")]
    api = _verdict_session(tmp_path, monkeypatch, four())
    v = api.get_results_summary("v")["verdict"]
    assert v["level"] == "none" and v["evidence_pct"] == 0 and v["holes_flagged"] == 0
    assert v["headline"].startswith("0 of 48 holes flagged")

    api = _verdict_session(tmp_path, monkeypatch, four("left"))
    v = api.get_results_summary("v")["verdict"]
    assert v["level"] == "possible" and v["evidence_pct"] == 25 and v["holes_flagged"] == 12
    assert "12 of 48 holes flagged" in v["headline"] and "may have been" in v["headline"] and "25%" in v["headline"]

    api = _verdict_session(tmp_path, monkeypatch, four("top", "bottom", "right"))
    v = api.get_results_summary("v")["verdict"]
    assert v["level"] == "likely" and v["evidence_pct"] == 75 and "is likely" in v["headline"]
    assert "not a calibrated probability" in v["evidence_note"]

    api = _verdict_session(tmp_path, monkeypatch, [])           # no edge had 5 teeth
    v = api.get_results_summary("v")["verdict"]
    assert v["level"] == "not_assessed" and v["evidence_pct"] is None
