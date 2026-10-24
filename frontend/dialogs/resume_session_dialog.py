"""Resume-on-launch dialog (Feature 22)."""

from __future__ import annotations

from typing import Any, Dict, Optional

from PySide6.QtWidgets import QDialog, QHBoxLayout, QLabel, QPushButton, QVBoxLayout

from frontend.client import Client


def _last_stage_name(info: Dict[str, Any]) -> str:
    stages = (info.get("manifest") or {}).get("stages") or []
    done = [s for s in stages if s.get("status") == "complete"]
    if not done:
        return "before the first step"
    last = max(done, key=lambda s: s.get("step_id", 0))
    return f"step {last.get('step_id')} ({str(last.get('stage_name', '')).replace('_', ' ')})"


class ResumeSessionDialog(QDialog):
    """Modal 'Resume Previous Session?' prompt shown by main.py before MainWindow.

    Resume calls resume_session and accepts (session_id is set); Discard or
    Escape rejects, leaving session_id as None.
    """

    def __init__(self, client: Optional[Client] = None, info: Optional[Dict[str, Any]] = None) -> None:
        super().__init__()
        self.client = client or Client()
        self.info = info or {}
        self.session_id: Optional[str] = None
        self.setWindowTitle("Resume Previous Session?")
        self.setModal(True)

        candidate = self.info.get("session_id", "")
        specimen = self.info.get("specimen_id") or candidate
        body = QLabel(
            f"A previous session (specimen: {specimen}) was interrupted at: "
            f"{_last_stage_name(self.info)}. Would you like to resume it?"
        )
        body.setWordWrap(True)
        self.message = QLabel()
        self.message.setWordWrap(True)

        resume_button = QPushButton("Resume")
        resume_button.setObjectName("primary")
        resume_button.setDefault(True)
        resume_button.clicked.connect(self.resume_found)
        discard_button = QPushButton("Discard")
        discard_button.clicked.connect(self.reject)

        buttons = QHBoxLayout()
        buttons.addStretch(1)
        buttons.addWidget(discard_button)
        buttons.addWidget(resume_button)

        layout = QVBoxLayout(self)
        layout.addWidget(body)
        layout.addWidget(self.message)
        layout.addLayout(buttons)

    def check_incomplete(self) -> None:
        """Refresh the candidate from the backend (kept for callers that build the dialog empty)."""
        response = self.client.call("check_incomplete_session")
        payload = response.get("result") if response.get("success") is not False else None
        self.info = payload if isinstance(payload, dict) else {}

    def resume_found(self) -> None:
        """Resume the interrupted session, then close returning its id."""
        candidate = self.info.get("session_id")
        if not candidate:
            self.message.setText("No interrupted session was found.")
            return
        response = self.client.call("resume_session", candidate)
        if response.get("success") is False:
            self.message.setText(response.get("error", "Unable to resume the session."))
            return
        self.session_id = candidate
        self.accept()
