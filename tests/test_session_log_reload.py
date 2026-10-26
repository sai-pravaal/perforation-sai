"""Saving a session log and reloading it into a fresh form (no re-entering data)."""

import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication  # noqa: E402

from backend.session.session_manager import SessionManager  # noqa: E402


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def api_in_tmp(tmp_path, monkeypatch):
    import backend.api as api
    root = tmp_path / "sessions"
    monkeypatch.setattr(api, "_session_manager", lambda: SessionManager(root))
    return api


def _form(api):
    from frontend.client import Client
    from frontend.tabs.session_log_tab import SessionLogTab
    return SessionLogTab(Client(api))


def _fill(tab):
    tab.specimen_id.setText("stamp_2_nomperf")
    tab.operator_notes.setText("macro lens")
    for card in tab.torch_cards.values():
        card.height_cm.setValue(9.2)
        card.distance_cm.setValue(30.0)
    tab.torch_cards["torch_4"].distance_cm.setValue(27.5)
    tab.torch_cards["torch_2"].correction_deg.setValue(1.5)
    tab.aperture.setValue(8.0)
    tab.shutter_speed.setText("1/125")
    tab.iso.setValue(100)
    tab.white_balance.setText("5200K")
    tab.ball_bearing_used.setChecked(True)


def test_saved_log_reloads_into_a_fresh_form(app, api_in_tmp):
    first = _form(api_in_tmp)
    first.set_session_id("s1")
    _fill(first)
    first.shadow_bleed.setChecked(True)
    first.save_log()
    assert first.message.text() == "Session log saved."

    # A new form, as after restarting the application.
    second = _form(api_in_tmp)
    saved, acked = [], []
    second.log_saved.connect(saved.append)
    second.shadow_bleed_acknowledged.connect(acked.append)
    second.set_session_id("s1")

    assert second.specimen_id.text() == "stamp_2_nomperf"
    assert second.operator_notes.text() == "macro lens"
    assert second.torch_cards["torch_1"].height_cm.value() == pytest.approx(9.2)
    assert second.torch_cards["torch_4"].distance_cm.value() == pytest.approx(27.5)
    assert second.torch_cards["torch_2"].correction_deg.value() == pytest.approx(1.5)
    assert "18.5" in second.torch_cards["torch_4"].elevation.text()
    assert second.aperture.value() == pytest.approx(8.0)
    assert second.shutter_speed.text() == "1/125"
    assert second.iso.value() == 100
    assert second.white_balance.text() == "5200K"
    assert second.ball_bearing_used.isChecked()
    assert second.shadow_bleed.isChecked()
    # The Processing gate sees a complete, acknowledged log (last emitted values).
    assert saved[-1] is True and acked[-1] is True
    reloaded, original = second.build_log(), first.build_log()
    reloaded.pop("capture_timestamp"), original.pop("capture_timestamp")   # taken at the moment of each call
    assert reloaded == original


def test_session_without_a_log_keeps_typed_values_and_locks_the_gate(app, api_in_tmp):
    tab = _form(api_in_tmp)
    tab.torch_cards["torch_1"].height_cm.setValue(9.2)          # typed before a session exists
    saved = []
    tab.log_saved.connect(saved.append)
    tab.set_session_id("brand_new")
    assert tab.torch_cards["torch_1"].height_cm.value() == pytest.approx(9.2)
    assert saved == [False]


def test_same_session_announced_again_does_not_overwrite_edits(app, api_in_tmp):
    first = _form(api_in_tmp)
    first.set_session_id("s1")
    _fill(first)
    first.save_log()
    tab = _form(api_in_tmp)
    tab.set_session_id("s1")
    tab.torch_cards["torch_1"].height_cm.setValue(11.0)          # unsaved edit
    tab.set_session_id("s1")                                     # e.g. the Upload tab re-announces the id
    assert tab.torch_cards["torch_1"].height_cm.value() == pytest.approx(11.0)


def test_get_session_log_api(api_in_tmp):
    assert api_in_tmp.get_session_log("nope") == {"success": True, "error": None, "exists": False, "log": None}
    api_in_tmp.save_session_log("s9", {"specimen_id": "x", "torches": []})
    result = api_in_tmp.get_session_log("s9")
    assert result["exists"] and result["log"]["specimen_id"] == "x"


def test_only_session_and_torch_geometry_are_required_to_save(app, api_in_tmp):
    tab = _form(api_in_tmp)
    tab.set_session_id("s2")
    for card in tab.torch_cards.values():
        card.height_cm.setValue(9.2)
        card.distance_cm.setValue(30.0)
    assert tab._missing_fields() == []                      # no specimen, camera settings or notes
    tab.save_log()
    assert tab.message.text() == "Session log saved."
    saved = api_in_tmp.get_session_log("s2")["log"]
    assert saved["camera_settings"] == {"aperture": None, "shutter_speed": None, "iso": None, "white_balance": None}

    reloaded = _form(api_in_tmp)
    reloaded.set_session_id("s2")
    assert reloaded.aperture.value() == 0 and reloaded.iso.value() == 0 and reloaded.shutter_speed.text() == ""
    assert reloaded.torch_cards["torch_2"].distance_cm.value() == pytest.approx(30.0)

    # Torch geometry is still required (Step 3 computes the elevation from it).
    empty = _form(api_in_tmp)
    empty.set_session_id("s3")
    assert set(empty._missing_fields()) == {"torch_1", "torch_2", "torch_3", "torch_4"}


def test_archive_name_falls_back_to_session_id(tmp_path):
    from backend.session.archive import SessionArchiver
    sm = SessionManager(tmp_path / "sessions")
    sm.create_session("run7")                              # empty specimen id
    sm.get_manifest_manager("run7").set_status("complete")
    assert "run7_" in SessionArchiver(sm).export_session("run7")
