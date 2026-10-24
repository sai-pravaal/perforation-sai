"""Application entry point for the IPDS desktop shell."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

# Allow running this file directly (python frontend/main.py) as well as with -m.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from PySide6.QtWidgets import QApplication, QDialog

from frontend.client import Client
from frontend.dialogs.resume_session_dialog import ResumeSessionDialog
from frontend.main_window import MainWindow

STYLE_SHEET = Path(__file__).resolve().parent / "theme" / "style.qss"


def main() -> int:
    """Boot sequence: theme first, then the resume check, then the main window."""
    # Technical error detail (tracebacks) goes to the log; the UI shows short messages.
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    app = QApplication(sys.argv)
    app.setApplicationName("IPDS")
    app.setApplicationDisplayName("IPDS Reperforation Detection")
    # Applied before any widget exists so even the resume dialog is themed.
    if STYLE_SHEET.exists():
        app.setStyleSheet(STYLE_SHEET.read_text(encoding="utf-8"))

    client = Client()
    session_id = None
    found = client.call("check_incomplete_session")
    incomplete = found.get("result") if found.get("success") is not False else None
    if isinstance(incomplete, dict) and incomplete.get("session_id"):
        dialog = ResumeSessionDialog(client, incomplete)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            session_id = dialog.session_id

    window = MainWindow(client=client, session_id=session_id)
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
