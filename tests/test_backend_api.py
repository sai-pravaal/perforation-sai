import json
from pathlib import Path

import pytest

from backend.session.session_manager import SessionManager


@pytest.fixture
def temp_session(tmp_path, monkeypatch):
    """A session in a temporary root; backend.api is pointed at it so no tracked data is touched."""
    import backend.api as api

    root = tmp_path / "sessions"
    manager = SessionManager(root)
    monkeypatch.setattr(api, "_session_manager", lambda: SessionManager(root))
    session_id = "api_test_session"
    manager.create_session(session_id, "specimen_001")
    return manager, session_id


def test_backend_api_exports_required_functions(temp_session):
    manager, session_id = temp_session

    import backend.api as api

    expected = {
        "validate_upload",
        "compute_elevation",
        "save_session_log",
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
        "run_quality_gate",
    }

    assert expected.issubset(set(dir(api)))

    log = {
        "session_id": session_id,
        "specimen_id": "specimen_001",
        "torches": [{
            "torch_id": "torch_1",
            "azimuth_deg": 0.0,
            "height_cm": 10.0,
            "distance_cm": 50.0,
            "manual_correction_deg": 0.0,
        }],
        "camera_settings": {"aperture": 2.8, "shutter_speed": 1 / 200, "iso": 100, "white_balance": "daylight"},
        "ball_bearing_used": False,
        "capture_timestamp": "2026-01-01T00:00:00Z",
        "operator_notes": "test",
    }

    assert api.save_session_log(session_id, log) is True
    assert api.compute_elevation(10.0, 50.0, 0.0) == pytest.approx(11.31, abs=1e-2)
    assert api.get_pipeline_status(session_id)["session_id"] == session_id
    assert api.get_manifest_diff(session_id)["session_id"] == session_id
    assert isinstance(api.get_config_snapshot(session_id), dict)
    # Never-run session: metadata is "not recorded", not the current environment.
    assert api.get_version_metadata(session_id)["success"] is False
    from backend.reporting.version_stamp import stamp_session
    stamp_session(session_id, manager)
    assert api.get_version_metadata(session_id)["app_version"] == "1.0.0"

    manager.reset_session(session_id)
    assert api.check_incomplete_session() is None
