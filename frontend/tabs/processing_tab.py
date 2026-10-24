"""Processing transparency and session continuity (Features 4-7, 22, 24)."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from PySide6.QtCore import QTimer, Signal
from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QMenu,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from frontend.client import Client
from frontend.widgets.progress_stepper import PHASE_NAMES, STEP_PHASE, ProgressStepper
from frontend.widgets.thumbnail_strip import ThumbnailStrip

POLL_MS = 500
STEP_NAMES = {
    1: "Raw capture ingestion", 2: "Lens & unit calibration", 3: "Illumination geometry",
    4: "Brightness matching", 5: "Cross-pass registration", 6: "Stamp boundary & orientation",
    7: "Perforation candidates", 8: "Hole / tooth classification", 9: "ROI extraction",
    10: "Focus-stack fusion", 11: "Subpixel contour", 12: "Centroid & radius profile",
    13: "Shape descriptor", 14: "Boundary texture sampling", 15: "Roughness descriptor",
    16: "Angular brightness sampling", 17: "Directional profile fit", 18: "Composite signature",
    19: "Reference groups", 20: "Robust statistics", 21: "Outlier thresholding",
    22: "Spatial clustering", 23: "Final output",
}
# Steps whose inputs the user edits on the Session Log tab (torch geometry).
SESSION_LOG_STEPS = {3}


class ProcessingTab(QWidget):
    """Stepper, brightness chips, per-hole counter, live thumbnails, start/resume/reprocess."""

    phase_changed = Signal(str)
    pipeline_finished = Signal(str)
    edit_session_log_requested = Signal(int)
    image_selected = Signal(dict)

    def __init__(self, client: Optional[Client] = None) -> None:
        super().__init__()
        self.client = client or Client()
        self._session_id: Optional[str] = None
        self._since_token = 0
        self._running = False
        self._chips_loaded = False

        self.banner = QLabel()
        self.banner.setProperty("status", "fail")
        self.banner.setWordWrap(True)
        self.banner.hide()
        dismiss = QPushButton("Dismiss")
        dismiss.clicked.connect(self.banner.hide)
        dismiss.clicked.connect(dismiss.hide)
        dismiss.hide()
        self._dismiss = dismiss

        self.start_button = QPushButton("Start processing")
        self.start_button.clicked.connect(lambda: self.start_pipeline())
        self.resume_button = QPushButton("Resume")
        self.resume_button.clicked.connect(self.resume_session)
        self.reprocess_button = QToolButton()
        self.reprocess_button.setText("Reprocess from…")
        self.reprocess_button.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
        self.reprocess_button.setMenu(self._build_reprocess_menu())
        self.reprocess_button.setEnabled(False)

        toolbar = QHBoxLayout()
        toolbar.addWidget(self.start_button)
        toolbar.addWidget(self.resume_button)
        toolbar.addWidget(self.reprocess_button)
        toolbar.addStretch(1)

        self.stepper = ProgressStepper()
        self.chips_row = QHBoxLayout()
        self.chips: Dict[str, QLabel] = {}
        for torch in ("torch_1", "torch_2", "torch_3", "torch_4"):
            chip = QLabel(f"T{torch[-1]}: —")
            chip.setObjectName("chip")
            self.chips[torch] = chip
            self.chips_row.addWidget(chip)
        self.chips_row.addStretch(1)
        self.chips_box = QWidget()
        self.chips_box.setLayout(self.chips_row)
        self.chips_box.hide()

        self.hole_label = QLabel("Hole 0 / 0 processed")
        self.hole_bar = QProgressBar()
        self.hole_bar.setRange(0, 0)
        self.hole_label.hide()
        self.hole_bar.hide()

        self.thumbnails = ThumbnailStrip()
        self.thumbnails.image_clicked.connect(self.image_selected.emit)
        self.message = QLabel("Processing has not started.")
        self.message.setWordWrap(True)

        banner_row = QHBoxLayout()
        banner_row.addWidget(self.banner, 1)
        banner_row.addWidget(dismiss)

        layout = QVBoxLayout(self)
        layout.addLayout(banner_row)
        layout.addLayout(toolbar)
        layout.addWidget(self.stepper)
        layout.addWidget(self.chips_box)
        layout.addWidget(self.hole_label)
        layout.addWidget(self.hole_bar)
        layout.addWidget(self.thumbnails)
        layout.addWidget(self.message)
        layout.addStretch(1)

        self._timer = QTimer(self)
        self._timer.setInterval(POLL_MS)
        self._timer.timeout.connect(self.refresh_status)

    # ------------------------------------------------------------ session
    def set_session_id(self, session_id: Optional[str]) -> None:
        self._session_id = session_id or None
        self._since_token = 0
        self._chips_loaded = False
        self.thumbnails.clear()
        self.chips_box.hide()
        if self._session_id:
            self.refresh_status()

    # -------------------------------------------------------------- start
    def start_pipeline(self, session_id: Optional[str] = None) -> None:
        """Begin a fresh run (Step 1) in the background and start live updates."""
        sid = session_id or self._session_id
        if not sid:
            self._show_error("No active session.")
            return
        self._session_id = sid
        self._launch(1, "Processing started.", fresh=True)

    def _launch(self, step: int, note: str, fresh: bool = False) -> None:
        self._since_token = 0
        self._chips_loaded = False
        self.thumbnails.clear()
        self.chips_box.hide()
        self.banner.hide()
        self._dismiss.hide()
        self._running = True
        self.start_button.setEnabled(False)
        self.reprocess_button.setEnabled(False)
        self.message.setText(note)
        if fresh:
            self.client.call_async("start_pipeline", self._session_id, on_done=self._on_run_done)
        else:
            self.client.call_async("reprocess_from_stage", self._session_id, str(step), None,
                                   on_done=self._on_run_done)
        self._timer.start()

    def _on_run_done(self, response: Dict[str, Any]) -> None:
        self._running = False
        self.start_button.setEnabled(True)
        self.refresh_status()
        self._timer.stop()
        result = response.get("result", response) if response.get("success") is not False else response
        status = result.get("status") if isinstance(result, dict) else None
        if response.get("success") is False or status == "failed":
            step = result.get("failed_step") if isinstance(result, dict) else None
            detail = (result.get("error") if isinstance(result, dict) else None) or response.get("error", "")
            self._show_error(f"Step {step} ({STEP_NAMES.get(step, '')}) failed: {detail}" if step else str(detail))
            self.pipeline_finished.emit("failed")
        else:
            self.message.setText("Processing complete.")
            self.pipeline_finished.emit("complete")

    # ------------------------------------------------------------- polling
    def refresh_status(self) -> None:
        """One poll: stepper, chips, hole counter, new thumbnails (Features 4-7)."""
        sid = self._session_id
        if not sid:
            return
        status = self._result(self.client.call("get_pipeline_status", sid))
        progress = self._result(self.client.call("get_hole_progress", sid))
        diff = self._result(self.client.call("get_manifest_diff", sid, self._since_token))
        if not isinstance(status, dict):
            return
        active = self.stepper.update_status(status, progress if isinstance(progress, dict) else None)
        if active is not None:
            self.phase_changed.emit(PHASE_NAMES[active])
        done_steps = {s.get("step_id") for s in status.get("stages", []) if s.get("status") == "complete"}

        # Feature 4: brightness chips once Step 4 has completed.
        if 4 in done_steps and not self._chips_loaded:
            self._load_chips(sid)

        # Feature 7: per-hole counter only while Phase 3 runs.
        in_phase3 = active == 3 and status.get("status") == "in_progress"
        self.hole_label.setVisible(in_phase3)
        self.hole_bar.setVisible(in_phase3)
        if in_phase3 and isinstance(progress, dict):
            total = int(progress.get("holes_total", 0))
            done = int(progress.get("holes_processed", 0))
            self.hole_bar.setRange(0, total)
            self.hole_bar.setValue(done)
            self.hole_label.setText(f"Hole {done} / {total} processed")

        # Feature 6: append only new manifest images.
        if isinstance(diff, dict):
            self.thumbnails.low_confidence_holes = set(status.get("low_confidence_holes") or [])
            new = diff.get("new_images") or []
            if new:
                self.thumbnails.append_images(new)
            self._since_token = int(diff.get("latest_token", self._since_token) or self._since_token)

        self.reprocess_button.setEnabled(not self._running and status.get("status") in {"complete", "failed"})
        if not self._running:
            self.message.setText(f"Pipeline status: {status.get('status', 'unknown')}")

    def _load_chips(self, sid: str) -> None:
        rows = self._result(self.client.call("get_brightness_warnings", sid))
        if not isinstance(rows, list):
            return
        for row in rows:
            chip = self.chips.get(row.get("torch_id"))
            if chip is None:
                continue
            chip.setText(f"T{row['torch_id'][-1]}: {float(row.get('factor', 0.0)):.2f}")
            chip.setToolTip(str(row.get("message", "")))
            chip.setProperty("severity", "warning" if row.get("warning") else "")
            chip.style().unpolish(chip)
            chip.style().polish(chip)
        self.chips_box.show()
        self._chips_loaded = True

    # ------------------------------------------------------ resume (F22)
    def resume_session(self) -> None:
        """Reload the session's state; if it stopped early, offer to continue from the next step."""
        sid = self._session_id
        if not sid:
            self._show_error("No active session.")
            return
        info, error = Client.unwrap(self.client.call("resume_session", sid), "Unable to resume the session.")
        if error or not isinstance(info, dict):
            self._show_error(error or "Unable to resume the session.")
            return
        self.refresh_status()
        last = int(info.get("last_completed_step", 0) or 0)
        if info.get("status") == "complete" or last >= 23:
            self.message.setText("This session already completed all 23 steps.")
            return
        nxt = last + 1
        answer = QMessageBox.question(
            self, "Continue processing?",
            f"The session stopped after step {last}. Continue from step {nxt} ({STEP_NAMES[nxt]})?",
        )
        if answer == QMessageBox.StandardButton.Yes:
            self._launch(nxt, f"Continuing from step {nxt}.")

    # ---------------------------------------------------- reprocess (F24)
    def _build_reprocess_menu(self) -> QMenu:
        menu = QMenu(self)
        for phase_index, phase_name in enumerate(PHASE_NAMES):
            sub = menu.addMenu(phase_name)
            for step, ph in STEP_PHASE.items():
                if ph == phase_index:
                    action = sub.addAction(f"{step}. {STEP_NAMES[step]}")
                    action.triggered.connect(lambda _=False, s=step: self.reprocess_from_stage(s))
        return menu

    def reprocess_from_stage(self, step_id: Optional[int] = None) -> None:
        """Feature 24: confirm, then rerun from the chosen step in the background."""
        if not self._session_id or step_id is None:
            return
        if step_id in SESSION_LOG_STEPS:
            answer = QMessageBox.question(
                self, "Edit inputs first?",
                f"Step {step_id} uses the torch geometry from the Session Log. Open the Session Log "
                "tab to correct it first? Choose No to reprocess with the saved values.",
            )
            if answer == QMessageBox.StandardButton.Yes:
                self.edit_session_log_requested.emit(step_id)
                return
        answer = QMessageBox.question(
            self, "Reprocess",
            f"This will discard all results from step {step_id} ({STEP_NAMES[step_id]}) onward. Continue?",
        )
        if answer == QMessageBox.StandardButton.Yes:
            self._launch(step_id, f"Reprocessing from step {step_id}.")

    # ------------------------------------------------------------- helpers
    @staticmethod
    def _result(response: Dict[str, Any]) -> Any:
        payload, _error = Client.unwrap(response)
        return payload

    def _show_error(self, text: str) -> None:
        self.banner.setText(text)
        self.banner.show()
        self._dismiss.show()
