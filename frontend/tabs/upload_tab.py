"""Capture upload, pre-flight checklist and quality gate (Features 1 and 29)."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QTableWidget,
    QVBoxLayout,
    QWidget,
)

from frontend.client import Client
from frontend.widgets.checklist_widget import CHECKLIST_ITEMS, ChecklistWidget
from frontend.widgets.scroll import install_scroll_area

UNTAGGED = "(choose tag)"
TAG_CHOICES = [UNTAGGED] + [tag for tag, _ in CHECKLIST_ITEMS]
STATUS_ICON = {"pass": "✓", "warn": "⚠", "fail": "✗"}


def _guess_tag(filename: str) -> str:
    """Pre-select a tag when the file follows the capture naming convention (user can change it)."""
    stem = Path(filename).stem.lower()
    for tag in sorted(TAG_CHOICES[1:], key=len, reverse=True):
        if stem == tag or stem.startswith((tag + "_", tag + "-", tag + ".")):
            return tag
    return UNTAGGED


class QualityGateCard(QFrame):
    """Feature 29 results: one pass/warn/fail row per criterion with acknowledgement."""

    continue_requested = Signal()

    def __init__(self) -> None:
        super().__init__()
        self.setObjectName("card")
        self._grid = QGridLayout()
        self._acks: List[QCheckBox] = []
        self.summary = QLabel()
        self.summary.setWordWrap(True)
        self.continue_button = QPushButton("Continue to processing")
        self.continue_button.setEnabled(False)
        self.continue_button.clicked.connect(self.continue_requested.emit)
        title = QLabel("Quality check")
        title.setObjectName("cardTitle")
        layout = QVBoxLayout(self)
        layout.addWidget(title)
        layout.addLayout(self._grid)
        layout.addWidget(self.summary)
        layout.addWidget(self.continue_button)
        self.hide()

    def show_result(self, payload: Dict) -> None:
        while self._grid.count():
            item = self._grid.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        self._acks = []
        criteria = payload.get("criteria", []) if isinstance(payload, dict) else []
        any_fail = False
        for row, crit in enumerate(criteria):
            status = crit.get("status", "fail")
            any_fail = any_fail or status == "fail"
            icon = QLabel(STATUS_ICON.get(status, "?"))
            icon.setProperty("status", status)
            name = QLabel(crit.get("criterion", ""))
            message = QLabel(crit.get("message", ""))
            message.setWordWrap(True)
            message.setProperty("status", status)
            self._grid.addWidget(icon, row, 0)
            self._grid.addWidget(name, row, 1)
            self._grid.addWidget(message, row, 2)
            if status == "warn":
                ack = QCheckBox("I acknowledge this warning")
                ack.toggled.connect(self._refresh_continue)
                self._acks.append(ack)
                self._grid.addWidget(ack, row, 3)
        self._any_fail = any_fail
        if any_fail:
            self.summary.setText("Processing is blocked until every failed check passes. Fix the capture and re-upload.")
        elif self._acks:
            self.summary.setText("Acknowledge each warning to continue.")
        else:
            self.summary.setText("All checks passed. Starting processing…")
        self.show()
        self._refresh_continue()
        if not any_fail and not self._acks:
            # Bound to this card: deleting the tab (Reset) cancels the pending start.
            QTimer.singleShot(1000, self, self.continue_requested.emit)

    def _refresh_continue(self) -> None:
        self.continue_button.setEnabled(not self._any_fail and all(a.isChecked() for a in self._acks))


class UploadTab(QWidget):
    """Upload area with per-file tags, live checklist, Process button and quality gate."""

    checklist_ready = Signal(bool)
    session_changed = Signal(str)
    process_requested = Signal(str)

    def __init__(self, client: Optional[Client] = None) -> None:
        super().__init__()
        self.client = client or Client()
        self.setAcceptDrops(True)
        self.session_id = QLineEdit()
        self.session_id.setPlaceholderText("Session identifier")
        self.session_id.editingFinished.connect(self.load_existing_files)

        drop = QLabel("Drop capture files here, or")
        drop.setAlignment(Qt.AlignmentFlag.AlignCenter)
        pick = QPushButton("Choose files…")
        pick.clicked.connect(self.choose_files)
        import_button = QPushButton("Import folder…")
        import_button.setToolTip("Copy every file named by the capture convention "
                                 "(diffuse, torch_1..torch_4, backlit, calibration, ball_bearing_*) from a folder.")
        import_button.clicked.connect(self.import_folder)
        self.import_button = import_button
        drop_row = QHBoxLayout()
        drop_row.addStretch(1)
        drop_row.addWidget(drop)
        drop_row.addWidget(pick)
        drop_row.addWidget(import_button)
        drop_row.addStretch(1)
        drop_frame = QFrame()
        drop_frame.setObjectName("card")
        drop_frame.setLayout(drop_row)

        self.files = QTableWidget(0, 2)
        self.files.setHorizontalHeaderLabels(("File", "Tag"))
        self.files.horizontalHeader().setStretchLastSection(True)
        self._paths: List[str] = []
        self._ready = False

        self.checklist = ChecklistWidget()
        self.message = QLabel("Add the capture files and assign a tag to each.")
        self.message.setWordWrap(True)
        self.process_button = QPushButton("Process")
        self.process_button.setEnabled(False)
        self.process_button.clicked.connect(self.run_quality_gate)
        self.quality_card = QualityGateCard()
        self.quality_card.continue_requested.connect(self._continue)

        form = QFormLayout()
        form.addRow("Session", self.session_id)
        layout = QVBoxLayout(install_scroll_area(self))
        layout.addLayout(form)
        layout.addWidget(drop_frame)
        layout.addWidget(self.files)
        layout.addWidget(self.checklist)
        layout.addWidget(self.process_button)
        layout.addWidget(self.quality_card)
        layout.addWidget(self.message)

    # ------------------------------------------------------------ session
    def set_session_id(self, session_id: Optional[str]) -> None:
        if (session_id or "") != self.session_id.text().strip():
            self.session_id.setText(session_id or "")
            self.load_existing_files()

    # -------------------------------------------------------------- files
    def _clear_files(self) -> None:
        self._paths = []
        self.files.setRowCount(0)

    def load_existing_files(self) -> None:
        """Show files already in the session's raw/ folder (named by the capture convention)."""
        session_id = self.session_id.text().strip()
        if not session_id:
            return
        payload, error = Client.unwrap(self.client.call("get_raw_files", session_id), "Unable to read the session folder.")
        if error:
            self.message.setText(error)
            return
        self._show_raw(payload)

    def _show_raw(self, payload: Dict) -> None:
        self._clear_files()
        for record in payload.get("files", []):
            self._add_row(record["path"], record.get("tag") or UNTAGGED)
        self.validate_capture()

    def import_folder(self) -> None:
        """Copy convention-named files from a folder into the session (the original workflow)."""
        session_id = self.session_id.text().strip()
        if not session_id:
            self.message.setText("Enter the session identifier first.")
            return
        folder = QFileDialog.getExistingDirectory(self, "Choose a folder of capture files")
        if not folder:
            return
        # Copying a set of RAW files takes seconds, so it runs off the GUI thread.
        self._set_busy(True, f"Importing from {folder}…")
        self.client.call_async("import_folder", session_id, folder,
                               on_done=lambda r, f=folder: self._on_imported(f, r))

    def _on_imported(self, folder: str, response: Dict) -> None:
        payload, error = Client.unwrap(response, "Import failed.")
        if error:
            self._set_busy(False, error)
            return
        self._set_busy(False, "")
        self._show_raw(payload)
        self.message.setText(f"Imported {len(payload.get('files', []))} file(s) from {folder}.")
    def choose_files(self) -> None:
        paths, _ = QFileDialog.getOpenFileNames(self, "Choose capture files")
        self.add_files(paths)

    def dragEnterEvent(self, event) -> None:  # noqa: N802 (Qt override)
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event) -> None:  # noqa: N802 (Qt override)
        self.add_files([u.toLocalFile() for u in event.mimeData().urls() if u.isLocalFile()])

    def add_files(self, paths: List[str]) -> None:
        for path in paths:
            if path and path not in self._paths:
                self._add_row(path, _guess_tag(path))
        self.validate_capture()

    def _add_row(self, path: str, tag: str) -> None:
        self._paths.append(path)
        row = self.files.rowCount()
        self.files.insertRow(row)
        self.files.setCellWidget(row, 0, QLabel(Path(path).name))
        combo = QComboBox()
        combo.addItems(TAG_CHOICES)
        combo.setCurrentText(tag)
        combo.currentTextChanged.connect(self.validate_capture)
        self.files.setCellWidget(row, 1, combo)

    def tagged_files(self) -> Dict[str, List[str]]:
        """Tag -> list of chosen file paths (untagged files are ignored)."""
        out: Dict[str, List[str]] = {}
        for row, path in enumerate(self._paths):
            combo = self.files.cellWidget(row, 1)
            tag = combo.currentText() if combo else UNTAGGED
            if tag != UNTAGGED:
                out.setdefault(tag, []).append(path)
        return out

    # ---------------------------------------------------------- checklist
    def validate_capture(self) -> None:
        """Feature 1: re-validate on every file add or tag change."""
        counts = {tag: str(len(paths)) for tag, paths in self.tagged_files().items()}
        response = self.client.call("validate_upload", self.session_id.text().strip(), counts)
        if response.get("success") is False:
            self.message.setText(response.get("error", "Capture validation failed."))
            ready = False
        else:
            ready = self.checklist.update_result(response)
            untagged = len(self._paths) - sum(len(v) for v in self.tagged_files().values())
            note = f" {untagged} file(s) still need a tag." if untagged else ""
            self.message.setText(("All required captures are present." if ready
                                  else "Required captures are missing or duplicated.") + note)
        session_id = self.session_id.text().strip()
        ready = ready and bool(session_id)
        self._ready = ready
        self.process_button.setEnabled(ready)
        if session_id:
            self.session_changed.emit(session_id)
        self.checklist_ready.emit(ready)

    # ------------------------------------------------------- quality gate
    def run_quality_gate(self) -> None:
        """Feature 29: run the pre-check on Process; only a pass (or acknowledged warn) continues.

        Storing the files and the check itself decode full-size RAW frames, so
        both run off the GUI thread (the window froze for minutes before).
        """
        session_id = self.session_id.text().strip()
        self._set_busy(True, "Storing capture files…")
        # Write the tagged files into the session's raw/ folder first: the
        # quality gate and the pipeline both read from there.
        self.client.call_async("upload_files", session_id, self.tagged_files(),
                               on_done=lambda r, sid=session_id: self._on_stored(sid, r))

    def _on_stored(self, session_id: str, response: Dict) -> None:
        stored, error = Client.unwrap(response, "Unable to store the capture files.")
        if error:
            self._set_busy(False, error)
            return
        if not stored.get("ready"):
            self.checklist.update_result(stored)
            self._set_busy(False, "The stored capture set is incomplete; check the checklist.")
            return
        self.message.setText("Running quality check (decoding the RAW frames)…")
        self.client.call_async("run_quality_gate", session_id, on_done=self._on_quality_checked)

    def _on_quality_checked(self, response: Dict) -> None:
        result, error = Client.unwrap(response, "Quality check could not run.")
        if error:
            self._set_busy(False, error)
            return
        self._set_busy(False, "")
        self.quality_card.show_result(result)

    def _set_busy(self, busy: bool, text: str) -> None:
        """Lock the controls that would start a second upload or check while one is running."""
        self.process_button.setEnabled(not busy and self._ready)
        self.import_button.setEnabled(not busy)
        self.process_button.setText("Working…" if busy else "Process")
        self.message.setText(text)

    def _continue(self) -> None:
        self.process_requested.emit(self.session_id.text().strip())
