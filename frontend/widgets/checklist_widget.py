"""Pre-flight capture checklist (Feature 1)."""

from __future__ import annotations

from typing import Any, Dict

from PySide6.QtWidgets import QFrame, QGridLayout, QLabel

# Display order and plain-language names; the backend schema decides pass/fail.
CHECKLIST_ITEMS = [
    ("diffuse", "Diffuse / Overview"),
    ("torch_1", "Torch 1 (0°)"),
    ("torch_2", "Torch 2 (90°)"),
    ("torch_3", "Torch 3 (180°)"),
    ("torch_4", "Torch 4 (270°)"),
    ("backlit", "Backlit / Transillumination"),
    ("calibration", "Calibration target"),
    ("ball_bearing_diffuse", "Ball bearing — diffuse"),
    ("ball_bearing_torch_1", "Ball bearing — torch 1"),
    ("ball_bearing_torch_2", "Ball bearing — torch 2"),
    ("ball_bearing_torch_3", "Ball bearing — torch 3"),
    ("ball_bearing_torch_4", "Ball bearing — torch 4"),
]


def _reason(info: Dict[str, Any]) -> str:
    """Specific reason text for a row instead of a bare red cross."""
    status = str(info.get("status", ""))
    count = int(info.get("count", 0) or 0)
    if status.startswith("Fail (missing"):
        return f"{count} found, at least 1 needed"
    if status.startswith("Fail (duplicate"):
        return f"{count} found, expected 1"
    if status.startswith("Optional"):
        return "optional — not supplied"
    if status == "Pass":
        return f"{count} found"
    return status or "not checked"


class ChecklistWidget(QFrame):
    """One row per capture item with a status icon and the specific reason."""

    def __init__(self) -> None:
        super().__init__()
        self.setObjectName("card")
        self._icons: Dict[str, QLabel] = {}
        self._reasons: Dict[str, QLabel] = {}
        grid = QGridLayout(self)
        grid.setColumnStretch(2, 1)
        for row, (tag, name) in enumerate(CHECKLIST_ITEMS):
            icon = QLabel("○")
            icon.setFixedWidth(22)
            reason = QLabel("not checked")
            reason.setProperty("status", "")
            optional = tag.startswith("ball_bearing")
            label = QLabel(name + ("  (optional)" if optional else ""))
            grid.addWidget(icon, row, 0)
            grid.addWidget(label, row, 1)
            grid.addWidget(reason, row, 2)
            self._icons[tag] = icon
            self._reasons[tag] = reason
        self.ready = False

    def update_result(self, result: Dict[str, Any]) -> bool:
        """Render a validate_upload payload; returns overall readiness."""
        payload = result.get("result", result) if isinstance(result, dict) else {}
        statuses = payload.get("tag_status", {}) if isinstance(payload, dict) else {}
        for tag, icon in self._icons.items():
            info = statuses.get(tag, {})
            status = str(info.get("status", ""))
            if status == "Pass":
                icon.setText("✓")
                state = "pass"
            elif status.startswith("Fail"):
                icon.setText("✗")
                state = "fail"
            else:
                icon.setText("○")
                state = ""
            for widget in (icon, self._reasons[tag]):
                widget.setProperty("status", state)
                widget.style().unpolish(widget)
                widget.style().polish(widget)
            self._reasons[tag].setText(_reason(info) if info else "not checked")
        self.ready = bool(payload.get("ready", False)) if isinstance(payload, dict) else False
        return self.ready
