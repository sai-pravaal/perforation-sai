"""Reset Session returns the whole application to a clean, just-launched state."""

import os
import time

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication, QMessageBox  # noqa: E402

from backend.session.session_manager import SessionManager  # noqa: E402


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def window(app, tmp_path, monkeypatch):
    import backend.api as api
    from frontend.client import Client
    from frontend.dialogs.reset_confirm_dialog import ResetConfirmDialog
    from frontend.main_window import MainWindow

    root = tmp_path / "sessions"
    monkeypatch.setattr(api, "_session_manager", lambda: SessionManager(root))
    monkeypatch.setattr(ResetConfirmDialog, "confirm", classmethod(lambda cls, parent, sid: True))
    notices = []
    monkeypatch.setattr(QMessageBox, "information", lambda *a, **k: notices.append(a[2]))
    monkeypatch.setattr(QMessageBox, "warning", lambda *a, **k: notices.append(a[2]))
    w = MainWindow(Client(api))
    w.notices, w.root = notices, root
    return w


def _use_session(w, sid="s1"):
    """Create a session on disk and put data into every kind of widget state."""
    SessionManager(w.root).create_session(sid, "stamp")
    w.upload_tab.session_id.setText(sid)
    w.set_session_id(sid)
    log = w.session_log_tab
    log.specimen_id.setText("stamp")
    log.operator_notes.setText("notes")
    for card in log.torch_cards.values():
        card.height_cm.setValue(9.2)
        card.distance_cm.setValue(30.0)
    log.aperture.setValue(8)
    log.shutter_speed.setText("1/125")
    log.iso.setValue(100)
    log.white_balance.setText("5200K")
    log.ball_bearing_used.setChecked(True)
    log.shadow_bleed.setChecked(True)
    log.save_log()
    w.processing_tab.message.setText("Pipeline status: complete")
    w.processing_tab.banner.setText("Step 3 failed")
    w.view_tab.stage_filter.setCurrentIndex(2)
    w.view_tab.flagged_only.setChecked(True)
    w.results_tab.message.setText("Loaded results")


def test_reset_gives_a_clean_state(window):
    w = window
    _use_session(w)
    old_tabs = [w.tabs.widget(i) for i in range(w.tabs.count())]
    w.tabs.setCurrentWidget(w.results_tab)

    w.reset_session()

    # Disk and client
    assert not SessionManager(w.root).session_exists("s1")
    assert w.client.get_current_session_id() is None
    # Every tab is a new object, and the old ones are gone
    new_tabs = [w.tabs.widget(i) for i in range(w.tabs.count())]
    assert len(new_tabs) == len(old_tabs) == 6
    assert all(n is not o for n, o in zip(new_tabs, old_tabs))
    assert w.tabs.currentIndex() == 0
    # Forms are blank
    log = w.session_log_tab
    assert log.session_id.text() == "" and log.specimen_id.text() == "" and log.operator_notes.text() == ""
    assert all(c.height_cm.value() == 0 and c.distance_cm.value() == 0 for c in log.torch_cards.values())
    assert log.aperture.value() == 0 and log.iso.value() == 0
    assert log.shutter_speed.text() == "" and log.white_balance.text() == ""
    assert not log.ball_bearing_used.isChecked() and not log.shadow_bleed.isChecked()
    assert w.upload_tab.session_id.text() == "" and w.upload_tab.files.rowCount() == 0
    assert w.upload_tab.tagged_files() == {} and not w.upload_tab.process_button.isEnabled()
    assert w.processing_tab.banner.isHidden() and w.processing_tab.thumbnails is not None
    assert w.view_tab.stage_filter.currentIndex() == 0 and not w.view_tab.flagged_only.isChecked()
    assert w.view_tab.gallery.count() == 0
    # Window chrome and gating
    assert w.session_label.text() == "No active session" and w.phase_label.text() == ""
    assert w.tabs.isTabEnabled(w.tabs.indexOf(w.processing_tab)) is False
    assert (w._checklist_ready, w._log_saved, w._bleed_acknowledged) == (False, False, False)


def test_a_new_session_after_reset_starts_with_nothing_from_the_old_one(window):
    w = window
    _use_session(w, "s1")
    w.reset_session()
    SessionManager(w.root).create_session("s2")
    w.upload_tab.session_id.setText("s2")
    w.set_session_id("s2")
    log = w.session_log_tab
    assert log.session_id.text() == "s2"
    assert log.specimen_id.text() == "" and log.torch_cards["torch_1"].height_cm.value() == 0
    assert not log.shadow_bleed.isChecked()


def test_reset_with_nothing_on_disk_still_cleans_the_ui(window):
    w = window
    w.upload_tab.session_id.setText("ghost")
    w.set_session_id("ghost")                       # never created on disk
    w.session_log_tab.specimen_id.setText("typed")
    w.reset_session()
    assert w.session_log_tab.specimen_id.text() == ""
    assert w.client.get_current_session_id() is None


def test_reset_is_refused_while_an_operation_is_running(window):
    w = window
    _use_session(w)
    w.client._pending.add(object())                 # a background call is in flight
    before = w.tabs.widget(0)
    w.reset_session()
    assert SessionManager(w.root).session_exists("s1")
    assert w.tabs.widget(0) is before and w.session_log_tab.specimen_id.text() == "stamp"
    assert any("Wait" in str(n) for n in w.notices)


def test_results_of_calls_started_before_a_reset_are_ignored(window, app):
    w = window
    got = []
    w.client.call_async("compute_elevation", 9.2, 30.0, 0.0, on_done=got.append)
    w.client.reset_state()
    deadline = time.time() + 5
    while w.client.has_pending_calls() and time.time() < deadline:
        app.processEvents()
        time.sleep(0.01)
    app.processEvents()
    assert got == []
    w.client.call_async("compute_elevation", 9.2, 30.0, 0.0, on_done=got.append)   # new generation delivers
    deadline = time.time() + 5
    while not got and time.time() < deadline:
        app.processEvents()
        time.sleep(0.01)
    assert len(got) == 1


def test_reset_session_is_idempotent(tmp_path):
    sm = SessionManager(tmp_path / "sessions")
    assert sm.reset_session("never_existed") is True
    sm.create_session("x")
    assert sm.reset_session("x") is True and not sm.session_exists("x")


def test_results_tab_shows_the_overall_result_and_an_empty_regions_message(window):
    w = window
    tab = w.results_tab
    tab._show_verdict({"level": "possible", "headline": "12 of 48 holes flagged. This stamp may have been reperforated "
                       "(evidence strength 25%).", "explanation": "1 of 4 checked edges ...", "evidence_note": "not a probability",
                       "edges": [{"edge": "left", "detail": "teeth not alike in spacing"}, {"edge": "top", "detail": "teeth are alike"}]})
    assert "12 of 48 holes flagged" in tab.verdict_headline.text() and "25%" in tab.verdict_headline.text()
    assert tab.verdict_headline.property("severity") == "medium"
    assert "Left: teeth not alike in spacing" in tab.verdict_edges.text() and "Top: teeth are alike" in tab.verdict_edges.text()
    assert tab.verdict_box.toolTip() == "not a probability"

    tab._session_id = "x"
    tab._call = lambda fn, *a: {"holes": [], "regions": [], "verdict": None} if fn == "get_results_summary" else None
    tab.load_results_summary()
    assert tab.region_title.text() == "No candidate regions"
    assert tab.region_list.count() == 1 and "No candidate regions" in tab.region_list.item(0).text()
    assert "No edge failed" in tab.region_statement.text()


def test_results_tab_width_does_not_grow_with_region_size(window):
    """A region with many member holes must not widen the tab (it once reached 2,900 px)."""
    tab = window.results_tab
    tab._session_id = "x"
    few = tab.minimumSizeHint().width()
    members = [{"hole_id": f"left_{i:02d}", "crop": None, "low_confidence": False} for i in range(1, 18)]
    tab._call = lambda fn, *a: {"title": "Left edge, holes 1-17", "statement": "x " * 300, "members": members,
                                "descriptor_table": [], "dominant_descriptors": []}
    tab.load_region_summary("left_edge_consistency")
    assert tab.region_members.count() == 17
    assert tab.minimumSizeHint().width() <= few + 20


def test_region_card_does_not_overlap_in_a_short_window_and_strip_scrolls_with_the_wheel(window, app):
    from PySide6.QtCore import QPoint, QPointF, Qt
    from PySide6.QtGui import QWheelEvent
    w = window
    w.resize(1400, 520)
    w.show()
    tab = w.results_tab
    w.tabs.setCurrentWidget(tab)
    tab.views.setCurrentIndex(3)
    tab._session_id = "x"
    members = [{"hole_id": f"bottom_{i:02d}", "crop": None, "low_confidence": False} for i in range(1, 17)]
    tab._call = lambda fn, *a: {"title": "Bottom edge", "statement": "The teeth are not alike. " * 12, "members": members,
                                "descriptor_table": [], "dominant_descriptors": []}
    tab.load_region_summary("r")
    for _ in range(3):          # widgets added to a layout are sized on later event-loop turns
        app.processEvents()
    strip, statement = tab.region_members_scroll, tab.region_statement
    strip_bottom = strip.mapTo(tab, strip.rect().bottomLeft()).y()
    assert statement.mapTo(tab, statement.rect().topLeft()).y() >= strip_bottom      # text no longer sits on the scrollbar
    bar = strip.horizontalScrollBar()
    assert bar.maximum() > 0
    wheel = QWheelEvent(QPointF(50, 20), QPointF(50, 20), QPoint(0, 0), QPoint(0, -120), Qt.MouseButton.NoButton,
                        Qt.KeyboardModifier.NoModifier, Qt.ScrollPhase.NoScrollPhase, False)
    strip.wheelEvent(wheel)
    assert bar.value() > 0
