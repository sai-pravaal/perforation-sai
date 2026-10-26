"""Export a session to a chosen file, import it back (same or new name), and reject bad archives."""

import json
import os
import zipfile

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication, QFileDialog, QInputDialog, QMessageBox  # noqa: E402

from backend.session.session_manager import SessionManager  # noqa: E402


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def api_tmp(tmp_path, monkeypatch):
    import backend.api as api
    root = tmp_path / "sessions"
    monkeypatch.setattr(api, "_session_manager", lambda: SessionManager(root))
    return api, SessionManager(root), tmp_path


def _make_session(sm, sid="stampA", status="in_progress"):
    sm.create_session(sid, "my stamp")
    d = sm.get_session_dir(sid)
    (d / "raw" / "diffuse.arw").write_bytes(b"RAW" * 1000)
    (d / "holes" / "top_01").mkdir()
    (d / "holes" / "top_01" / "score.json").write_text(json.dumps({"combined_score": 1.5}))
    sm.save_session_log(sid, {"specimen_id": "my stamp", "torches": [], "shadow_bleed_acknowledged": True})
    sm.get_manifest_manager(sid).set_status(status)
    return d


def test_export_to_chosen_file_then_import_restores_everything(api_tmp):
    api, sm, tmp = api_tmp
    _make_session(sm)
    target = tmp / "out" / "my_export.zip"
    path = api.export_session("stampA", str(target))
    assert path == str(target) and target.is_file()        # an unfinished session exports too
    assert zipfile.ZipFile(target).namelist()[0].startswith("stampA/")

    sm.reset_session("stampA")
    result = api.import_session(str(target))
    assert result["success"] and result["session_id"] == "stampA" and result["status"] == "in_progress"
    d = sm.get_session_dir("stampA")
    assert (d / "raw" / "diffuse.arw").read_bytes() == b"RAW" * 1000
    assert json.loads((d / "holes" / "top_01" / "score.json").read_text())["combined_score"] == 1.5
    assert sm.load_session_log("stampA")["shadow_bleed_acknowledged"] is True
    assert not [p for p in sm.sessions_root.iterdir() if p.name.startswith(".import_")]   # no staging left behind


def test_import_under_a_new_name_updates_the_ids(api_tmp):
    api, sm, tmp = api_tmp
    _make_session(sm)
    target = tmp / "a.zip"
    api.export_session("stampA", str(target))
    clash = api.import_session(str(target))                 # original still present
    assert clash["success"] is False and clash["exists"] and clash["session_id"] == "stampA"
    assert clash["suggested_id"] == "stampA_imported"
    ok = api.import_session(str(target), clash["suggested_id"])
    assert ok["success"] and ok["session_id"] == "stampA_imported"
    assert sm.get_manifest_manager("stampA_imported").load()["session_id"] == "stampA_imported"
    assert sm.load_session_log("stampA_imported")["session_id"] == "stampA_imported"
    assert sm.session_exists("stampA")                      # the original is untouched


def test_bad_archives_are_rejected_and_nothing_is_created(api_tmp):
    api, sm, tmp = api_tmp
    notzip = tmp / "x.zip"
    notzip.write_text("hello")
    assert api.import_session(str(notzip))["success"] is False

    two = tmp / "two.zip"
    with zipfile.ZipFile(two, "w") as zf:
        zf.writestr("a/manifest.json", "{}")
        zf.writestr("b/manifest.json", "{}")
    assert "exactly one" in api.import_session(str(two))["error"]

    nomanifest = tmp / "nomanifest.zip"
    with zipfile.ZipFile(nomanifest, "w") as zf:
        zf.writestr("s1/readme.txt", "x")
    assert "manifest.json" in api.import_session(str(nomanifest))["error"]

    slip = tmp / "slip.zip"
    with zipfile.ZipFile(slip, "w") as zf:
        zf.writestr("s1/manifest.json", "{}")
        zf.writestr("s1/../../evil.txt", "x")
    assert api.import_session(str(slip))["success"] is False
    assert not (tmp / "evil.txt").exists() and not (sm.sessions_root.parent / "evil.txt").exists()

    badname = tmp / "badname.zip"
    with zipfile.ZipFile(badname, "w") as zf:
        zf.writestr("s1/manifest.json", "{}")
    assert api.import_session(str(badname), "../escape")["success"] is False
    assert [p.name for p in sm.sessions_root.iterdir()] == []     # nothing was created by any of them


@pytest.fixture
def window(app, api_tmp, monkeypatch):
    import backend.api as api
    from frontend.client import Client
    from frontend.main_window import MainWindow
    _, sm, tmp = api_tmp
    shown = []
    monkeypatch.setattr(QMessageBox, "information", lambda *a, **k: shown.append(("info", a[2])))
    monkeypatch.setattr(QMessageBox, "warning", lambda *a, **k: shown.append(("warn", a[2])))
    w = MainWindow(Client(api))
    w.shown, w.sm, w.tmp = shown, sm, tmp
    return w


def _wait(app, w, seconds=10):
    import time
    end = time.time() + seconds
    while w.client.has_pending_calls() and time.time() < end:
        app.processEvents()
        time.sleep(0.01)
    app.processEvents()


def test_export_and_import_through_the_window(window, app, monkeypatch):
    w = window
    _make_session(w.sm)
    w.upload_tab.session_id.setText("stampA")
    w.set_session_id("stampA")
    target = str(w.tmp / "exported.zip")
    monkeypatch.setattr(QFileDialog, "getSaveFileName", lambda *a, **k: (target, ""))
    w.export_session()
    _wait(app, w)
    assert os.path.isfile(target)
    assert w.shown[-1][0] == "info" and target in w.shown[-1][1]          # the user is told where it went

    # Results-tab button goes through the same path
    w.shown.clear()
    w.results_tab.export_button.click()
    _wait(app, w)
    assert w.shown and w.shown[-1][0] == "info"

    # Import: the name is taken, so the user is asked for another and the session opens
    monkeypatch.setattr(QFileDialog, "getOpenFileName", lambda *a, **k: (target, ""))
    monkeypatch.setattr(QInputDialog, "getText", lambda *a, **k: ("stampA_copy", True))
    w.import_session()
    _wait(app, w)
    _wait(app, w)
    assert w.sm.session_exists("stampA_copy")
    assert w.client.get_current_session_id() == "stampA_copy"
    assert w.upload_tab.session_id.text() == "stampA_copy"
    assert w.session_log_tab.specimen_id.text() == "my stamp"             # form reloaded from the imported log


def test_cancelled_dialogs_do_nothing(window, app, monkeypatch):
    w = window
    _make_session(w.sm)
    w.set_session_id("stampA")
    monkeypatch.setattr(QFileDialog, "getSaveFileName", lambda *a, **k: ("", ""))
    monkeypatch.setattr(QFileDialog, "getOpenFileName", lambda *a, **k: ("", ""))
    w.export_session()
    w.import_session()
    assert not w.client.has_pending_calls() and w.shown == []


def test_import_menu_entry_exists(window):
    texts = [a.text() for m in window.menuBar().actions() for a in (m.menu().actions() if m.menu() else [])]
    assert "Import Session…" in texts and "Export Session…" in texts
