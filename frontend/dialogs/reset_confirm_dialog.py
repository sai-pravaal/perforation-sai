"""Session-reset confirmation (Feature 23)."""

from __future__ import annotations

from typing import Optional

from PySide6.QtWidgets import QMessageBox, QWidget

RESET_WARNING = (
    "This will permanently delete all images and data for this session. This cannot be undone."
)


class ResetConfirmDialog(QMessageBox):
    """Warning message box with Cancel as the default (Enter-safe) button.

    Only an explicit click on the destructive "Yes, Reset" button confirms.
    The caller performs client.reset_session after confirmation.
    """

    def __init__(self, parent: Optional[QWidget] = None, session_id: Optional[str] = None) -> None:
        super().__init__(parent)
        self.setIcon(QMessageBox.Icon.Warning)
        self.setWindowTitle("Reset Session")
        self.setText(RESET_WARNING)
        if session_id:
            self.setInformativeText(f"Session: {session_id}")
        self.cancel_button = self.addButton("Cancel", QMessageBox.ButtonRole.RejectRole)
        self.reset_button = self.addButton("Yes, Reset", QMessageBox.ButtonRole.DestructiveRole)
        self.reset_button.setObjectName("destructive")
        self.setDefaultButton(self.cancel_button)
        self.setEscapeButton(self.cancel_button)

    def confirmed(self) -> bool:
        """Show modally; True only when "Yes, Reset" was clicked."""
        self.exec()
        return self.clickedButton() is self.reset_button

    @classmethod
    def confirm(cls, parent: Optional[QWidget], session_id: Optional[str]) -> bool:
        return cls(parent, session_id).confirmed()
