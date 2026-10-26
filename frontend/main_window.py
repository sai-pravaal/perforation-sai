"""Main application window for the IPDS desktop shell."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from PySide6.QtGui import QAction
from PySide6.QtWidgets import QFileDialog, QInputDialog, QLabel, QMainWindow, QMessageBox, QTabWidget

from frontend.client import Client
from frontend.dialogs.reset_confirm_dialog import ResetConfirmDialog
from frontend.tabs.processing_tab import ProcessingTab
from frontend.tabs.results_tab import ResultsTab
from frontend.tabs.session_log_tab import SessionLogTab
from frontend.tabs.upload_tab import UploadTab
from frontend.tabs.validation_tab import ValidationTab
from frontend.tabs.view_tab import ViewTab


class MainWindow(QMainWindow):
    """Tab container, session-wide actions and the global tab-locking rule."""

    def __init__(self, client: Optional[Client] = None, session_id: Optional[str] = None) -> None:
        super().__init__()
        self.setWindowTitle("IPDS Reperforation Detection")
        self.client = client or Client()

        self.tabs = QTabWidget()
        self.setCentralWidget(self.tabs)

        self._build_menus()
        self.session_label = QLabel("No active session")
        self.phase_label = QLabel("")
        self.statusBar().addWidget(self.session_label)
        self.statusBar().addPermanentWidget(self.phase_label)

        # Gating state (Features 1-3).
        self._checklist_ready = False
        self._log_saved = False
        self._bleed_acknowledged = False
        self._build_tabs()

        if session_id:
            # A resumed session already passed capture setup in an earlier run.
            self._checklist_ready = self._log_saved = self._bleed_acknowledged = True
            self.set_session_id(session_id)
        self._apply_gating()
        self.resize(1280, 860)

    def _build_tabs(self) -> None:
        """Create every tab fresh and wire it (also used by Reset, so nothing is carried over)."""
        # Pipeline order, left to right.
        self.upload_tab = UploadTab(self.client)
        self.session_log_tab = SessionLogTab(self.client)
        self.processing_tab = ProcessingTab(self.client)
        self.view_tab = ViewTab(self.client)
        self.results_tab = ResultsTab(self.client)
        self.validation_tab = ValidationTab(self.client)
        for widget, title in (
            (self.upload_tab, "Upload"),
            (self.session_log_tab, "Session Log"),
            (self.processing_tab, "Processing"),
            (self.view_tab, "View"),
            (self.results_tab, "Results"),
            (self.validation_tab, "Validation"),
        ):
            self.tabs.addTab(widget, title)

        self.upload_tab.checklist_ready.connect(self._on_checklist_ready)
        self.upload_tab.session_changed.connect(self.set_session_id)
        self.session_log_tab.log_saved.connect(self._on_log_saved)
        self.session_log_tab.shadow_bleed_acknowledged.connect(self._on_bleed_acknowledged)
        self.processing_tab.phase_changed.connect(self.phase_label.setText)
        self.processing_tab.pipeline_finished.connect(self._on_pipeline_finished)
        self.upload_tab.process_requested.connect(self._on_process_requested)
        self.processing_tab.edit_session_log_requested.connect(
            lambda _step: self.tabs.setCurrentWidget(self.session_log_tab))
        self.processing_tab.image_selected.connect(self._on_thumbnail_selected)
        self.results_tab.export_requested.connect(self.export_session)

    def _discard_tabs(self) -> None:
        """Remove and delete every tab (and with them their timers, images and widget state)."""
        old = [self.tabs.widget(i) for i in range(self.tabs.count())]
        self.tabs.clear()  # removes the pages without deleting them
        for widget in old:
            widget.hide()
            widget.deleteLater()

    # ------------------------------------------------------------- session
    def set_session_id(self, session_id: Optional[str]) -> None:
        """Share the active session id with the client and every tab."""
        self.client.set_current_session_id(session_id)
        for tab in (self.upload_tab, self.session_log_tab, self.processing_tab,
                    self.view_tab, self.results_tab):
            setter = getattr(tab, "set_session_id", None)
            if setter is not None:
                setter(session_id)
        self.session_label.setText(f"Session: {session_id}" if session_id else "No active session")

    # -------------------------------------------------------------- gating
    def _on_checklist_ready(self, ready: bool) -> None:
        self._checklist_ready = bool(ready)
        self._apply_gating()

    def _on_log_saved(self, ok: bool) -> None:
        self._log_saved = bool(ok)
        self._apply_gating()

    def _on_bleed_acknowledged(self, ok: bool) -> None:
        self._bleed_acknowledged = bool(ok)
        self._apply_gating()

    def _apply_gating(self) -> None:
        """Only Processing is locked: it needs the checklist, the saved log and the acknowledgement.

        The Session Log form can be filled in at any time (it was locked until
        the upload checklist passed, so torch geometry could not be entered
        first), and View/Results/Validation just show whatever the session
        already holds.
        """
        for index in range(self.tabs.count()):
            self.tabs.setTabEnabled(index, True)
        processing_index = self.tabs.indexOf(self.processing_tab)
        self.tabs.setTabEnabled(
            processing_index, self._checklist_ready and self._log_saved and self._bleed_acknowledged
        )

    def _on_process_requested(self, session_id: str) -> None:
        """Feature 29 gate passed on the Upload tab: start the real run."""
        if not self.tabs.isTabEnabled(self.tabs.indexOf(self.processing_tab)):
            QMessageBox.information(self, "Session log needed",
                                    "Save the session log and acknowledge the shadow-bleed check first.")
            self.tabs.setCurrentWidget(self.session_log_tab)
            return
        self.tabs.setCurrentWidget(self.processing_tab)
        self.processing_tab.start_pipeline(session_id)

    def _on_thumbnail_selected(self, image: dict) -> None:
        selector = getattr(self.view_tab, "select_image", None)
        if selector is not None and self.tabs.isTabEnabled(self.tabs.indexOf(self.view_tab)):
            self.tabs.setCurrentWidget(self.view_tab)
            selector(image)

    def _on_pipeline_finished(self, status: str) -> None:
        if status == "complete":
            self.tabs.setCurrentWidget(self.results_tab)

    # --------------------------------------------------------------- menus
    def _build_menus(self) -> None:
        file_menu = self.menuBar().addMenu("&File")
        export_action = QAction("Export Session…", self)
        export_action.triggered.connect(self.export_session)
        import_action = QAction("Import Session…", self)
        import_action.triggered.connect(self.import_session)
        reset_action = QAction("Reset Session…", self)
        reset_action.triggered.connect(self.reset_session)
        quit_action = QAction("Quit", self)
        quit_action.triggered.connect(self.close)
        file_menu.addAction(export_action)
        file_menu.addAction(import_action)
        file_menu.addAction(reset_action)
        file_menu.addSeparator()
        file_menu.addAction(quit_action)

        help_menu = self.menuBar().addMenu("&Help")
        about_action = QAction("About", self)
        about_action.triggered.connect(self.show_about)
        help_menu.addAction(about_action)

    def _require_session(self) -> Optional[str]:
        session_id = self.client.get_current_session_id()
        if not session_id:
            QMessageBox.information(self, "No session", "Open or create a session first.")
        return session_id

    def _busy(self) -> bool:
        if self.client.has_pending_calls():
            QMessageBox.information(self, "Busy", "Wait for the running operation (processing, upload, export "
                                    "or validation) to finish first.")
            return True
        return False

    def export_session(self) -> None:
        """Feature 27: choose where to save, then zip the session there in the background."""
        session_id = self._require_session()
        if not session_id or self._busy():
            return
        target, _ = QFileDialog.getSaveFileName(self, "Export session", f"{session_id}.zip", "Session archive (*.zip)")
        if not target:
            return
        self.statusBar().showMessage(f"Exporting session {session_id}…")
        self.client.call_async("export_session", session_id, target, on_done=self._on_export_done)

    def _on_export_done(self, response) -> None:
        self.statusBar().clearMessage()
        archive, error = Client.unwrap(response, "The session could not be exported.")
        if error or not isinstance(archive, str) or not Path(archive).exists():
            QMessageBox.warning(self, "Export failed", str(error or "The session could not be exported."))
            return
        size_mb = Path(archive).stat().st_size / 1e6
        QMessageBox.information(self, "Session exported", f"Saved {size_mb:.0f} MB to:\n{archive}")

    def import_session(self) -> None:
        """Feature 27: restore a session from an exported archive and open it."""
        if self._busy():
            return
        path, _ = QFileDialog.getOpenFileName(self, "Import session", "", "Session archive (*.zip)")
        if not path:
            return
        self._start_import(path, None)

    def _start_import(self, path: str, new_id) -> None:
        self.statusBar().showMessage("Importing session…")
        self.client.call_async("import_session", path, new_id, on_done=lambda r: self._on_import_done(path, r))

    def _on_import_done(self, path: str, response) -> None:
        self.statusBar().clearMessage()
        result = response.get("result", response) if isinstance(response, dict) else {}
        if isinstance(result, dict) and result.get("exists"):
            # The archive's session name is taken: let the user pick another.
            name, ok = QInputDialog.getText(
                self, "Session already exists",
                f"A session named '{result.get('session_id')}' already exists.\nImport it under this name:",
                text=str(result.get("suggested_id", "")))
            if ok and name.strip():
                self._start_import(path, name.strip())
            return
        payload, error = Client.unwrap(response, "The session could not be imported.")
        if error or not isinstance(payload, dict):
            QMessageBox.warning(self, "Import failed", str(error or "The session could not be imported."))
            return
        session_id = payload["session_id"]
        self.set_session_id(session_id)
        self.tabs.setCurrentWidget(self.results_tab if payload.get("status") == "complete" else self.upload_tab)
        self.statusBar().showMessage(f"Imported session {session_id} ({payload.get('status')}).", 8000)

    def reset_session(self) -> None:
        """Feature 23: delete the session only after explicit confirmation."""
        session_id = self._require_session()
        if not session_id:
            return
        if self.client.has_pending_calls():
            # A run, upload or export is still reading or writing the session folder.
            QMessageBox.information(self, "Busy", "Wait for the running operation (processing, upload, export "
                                    "or validation) to finish before resetting the session.")
            return
        if not ResetConfirmDialog.confirm(self, session_id):
            return
        response = self.client.call("reset_session", session_id)
        if response.get("success") is False or not response.get("result"):
            QMessageBox.warning(self, "Reset failed", response.get("error", "The session could not be deleted."))
            return
        self._start_clean()

    def _start_clean(self) -> None:
        """Return the whole application to its just-launched state: no session, every tab new."""
        self.client.reset_state()
        self._discard_tabs()
        self._checklist_ready = self._log_saved = self._bleed_acknowledged = False
        self._build_tabs()
        self.session_label.setText("No active session")
        self.phase_label.setText("")
        self.statusBar().clearMessage()
        self._apply_gating()
        self.tabs.setCurrentIndex(0)

    def show_about(self) -> None:
        """Feature 28: show version and constants metadata read-only."""
        session_id = self.client.get_current_session_id()
        lines = ["IPDS — Intra-Specimen Perforation Deviation Signature"]
        if session_id:
            response = self.client.call("get_version_metadata", session_id)
            meta = response.get("result") if response.get("success") is not False else None
            if isinstance(meta, dict) and "app_version" in meta:
                lines.append(f"App version: {meta.get('app_version')}")
                lines.append(f"Python: {meta.get('python_version')}")
                libs = meta.get("library_versions") or {}
                lines.append("Libraries: " + ", ".join(f"{k} {v}" for k, v in libs.items()))
                constants = meta.get("config_snapshot") or {}
                lines.append(f"Constants recorded: {len(constants)}")
        else:
            lines.append("Open a session to see the versions and constants recorded for its run.")
        QMessageBox.about(self, "About IPDS", "\n".join(lines))
