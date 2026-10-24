"""Structured session log form and shadow-bleed check (Features 2 and 3)."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, Optional

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QDoubleSpinBox,
    QFormLayout,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from frontend.client import Client

# Fixed torch azimuths from the capture protocol (Feature 2), laid out to mirror
# the physical rig: Torch 1 top-left ... Torch 4 bottom-right.
TORCH_AZIMUTHS = {"torch_1": 0.0, "torch_2": 90.0, "torch_3": 180.0, "torch_4": 270.0}
TORCH_GRID_POS = {"torch_1": (0, 0), "torch_2": (0, 1), "torch_3": (1, 0), "torch_4": (1, 1)}
# Display-only guidance band from the hardware protocol (15-20 deg is a starting
# point, not a hard rule); used to tint the label, never to block saving.
ELEVATION_GUIDANCE_DEG = (15.0, 20.0)
SHADOW_BLEED_TEXT = (
    "Zoom into two adjacent holes at the chosen elevation angle and confirm no shadow overlap."
)


class TorchCard(QFrame):
    """One torch's height/distance/correction inputs with a live elevation label."""

    def __init__(self, torch_id: str, client: Client) -> None:
        super().__init__()
        self.setObjectName("card")
        self.torch_id = torch_id
        self.client = client
        self.height_cm = self._spin(0.0, 500.0)
        self.distance_cm = self._spin(0.0, 500.0)
        self.correction_deg = self._spin(-45.0, 45.0)
        self.elevation = QLabel("Elevation: —")
        self.elevation.setProperty("severity", "")

        form = QFormLayout(self)
        title = QLabel(f"{torch_id.replace('_', ' ').title()} — azimuth {TORCH_AZIMUTHS[torch_id]:.0f}°")
        title.setObjectName("cardTitle")
        form.addRow(title)
        form.addRow("Height (cm)", self.height_cm)
        form.addRow("Distance (cm)", self.distance_cm)
        form.addRow("Manual correction (°)", self.correction_deg)
        form.addRow(self.elevation)

        for spin in (self.height_cm, self.distance_cm, self.correction_deg):
            spin.valueChanged.connect(self.update_elevation)

    @staticmethod
    def _spin(lo: float, hi: float) -> QDoubleSpinBox:
        spin = QDoubleSpinBox()
        spin.setRange(lo, hi)
        spin.setDecimals(2)
        spin.setSingleStep(0.5)
        return spin

    def update_elevation(self) -> None:
        """Feature 2: recompute this torch's elevation through the backend on every change."""
        if self.distance_cm.value() <= 0:
            self.elevation.setText("Elevation: enter a distance")
            self._set_severity("")
            return
        response = self.client.call(
            "compute_elevation",
            self.height_cm.value(),
            self.distance_cm.value(),
            self.correction_deg.value(),
        )
        if response.get("success") is False:
            self.elevation.setText(response.get("error", "Elevation unavailable"))
            return
        value = float(response.get("result", 0.0))
        lo, hi = ELEVATION_GUIDANCE_DEG
        outside = not (lo <= value <= hi)
        self.elevation.setText(f"Elevation: {value:.1f}°" + ("  (outside 15–20° guide)" if outside else ""))
        self.elevation.setToolTip("15–20° is a starting point, not a hard rule." if outside else "")
        self._set_severity("warning" if outside else "")

    def _set_severity(self, value: str) -> None:
        self.elevation.setProperty("severity", value)
        self.elevation.style().unpolish(self.elevation)
        self.elevation.style().polish(self.elevation)

    def is_complete(self) -> bool:
        return self.height_cm.value() > 0 and self.distance_cm.value() > 0

    def to_record(self) -> Dict[str, Any]:
        return {
            "torch_id": self.torch_id,
            "azimuth_deg": TORCH_AZIMUTHS[self.torch_id],
            "height_cm": self.height_cm.value(),
            "distance_cm": self.distance_cm.value(),
            "manual_correction_deg": self.correction_deg.value(),
        }


class SessionLogTab(QWidget):
    """Feature 2 structured session log form with Feature 3's shadow-bleed row."""

    log_saved = Signal(bool)
    shadow_bleed_acknowledged = Signal(bool)

    def __init__(self, client: Optional[Client] = None) -> None:
        super().__init__()
        self.client = client or Client()
        self.session_id = QLineEdit()
        self.session_id.setPlaceholderText("Session identifier")
        self.specimen_id = QLineEdit()
        self.specimen_id.setPlaceholderText("Specimen identifier")
        self.operator_notes = QLineEdit()

        self.torch_cards = {tid: TorchCard(tid, self.client) for tid in TORCH_AZIMUTHS}
        grid = QGridLayout()
        for tid, card in self.torch_cards.items():
            grid.addWidget(card, *TORCH_GRID_POS[tid])

        camera = QFrame()
        camera.setObjectName("card")
        self.aperture = QDoubleSpinBox()
        self.aperture.setRange(0.0, 64.0)
        self.aperture.setDecimals(1)
        self.shutter_speed = QLineEdit()
        self.shutter_speed.setPlaceholderText("e.g. 1/125")
        self.iso = QSpinBox()
        self.iso.setRange(0, 409600)
        self.white_balance = QLineEdit()
        self.white_balance.setPlaceholderText("e.g. daylight or 5200K")
        self.ball_bearing_used = QCheckBox("Ball-bearing calibration photos supplied (Path B)")
        camera_form = QFormLayout(camera)
        camera_title = QLabel("Camera settings (identical for every shot)")
        camera_title.setObjectName("cardTitle")
        camera_form.addRow(camera_title)
        camera_form.addRow("Aperture (f/)", self.aperture)
        camera_form.addRow("Shutter speed", self.shutter_speed)
        camera_form.addRow("ISO", self.iso)
        camera_form.addRow("White balance", self.white_balance)
        camera_form.addRow(self.ball_bearing_used)

        self.shadow_bleed = QCheckBox(SHADOW_BLEED_TEXT)
        self.shadow_bleed.toggled.connect(self.acknowledge_shadow_bleed)
        self.message = QLabel()
        self.message.setWordWrap(True)
        save_button = QPushButton("Save session log")
        save_button.clicked.connect(self.save_log)

        ids = QFormLayout()
        ids.addRow("Session", self.session_id)
        ids.addRow("Specimen", self.specimen_id)
        ids.addRow("Operator notes", self.operator_notes)

        bleed_row = QHBoxLayout()
        bleed_row.addWidget(self.shadow_bleed)

        layout = QVBoxLayout(self)
        layout.addLayout(ids)
        layout.addLayout(grid)
        layout.addWidget(camera)
        layout.addLayout(bleed_row)
        layout.addWidget(save_button)
        layout.addWidget(self.message)

    def set_session_id(self, session_id: Optional[str]) -> None:
        self.session_id.setText(session_id or "")

    def _missing_fields(self) -> list[str]:
        missing = [tid for tid, card in self.torch_cards.items() if not card.is_complete()]
        if not self.session_id.text().strip():
            missing.append("session")
        if not self.specimen_id.text().strip():
            missing.append("specimen")
        if self.aperture.value() <= 0:
            missing.append("aperture")
        if not self.shutter_speed.text().strip():
            missing.append("shutter speed")
        if self.iso.value() <= 0:
            missing.append("ISO")
        if not self.white_balance.text().strip():
            missing.append("white balance")
        return missing

    def build_log(self) -> Dict[str, Any]:
        return {
            "session_id": self.session_id.text().strip(),
            "specimen_id": self.specimen_id.text().strip(),
            "torches": [card.to_record() for card in self.torch_cards.values()],
            "camera_settings": {
                "aperture": self.aperture.value(),
                "shutter_speed": self.shutter_speed.text().strip(),
                "iso": self.iso.value(),
                "white_balance": self.white_balance.text().strip(),
            },
            "ball_bearing_used": self.ball_bearing_used.isChecked(),
            "capture_timestamp": datetime.now(timezone.utc).isoformat(),
            "operator_notes": self.operator_notes.text(),
        }

    def save_log(self) -> None:
        """Validate client-side, then persist the full four-torch session log."""
        missing = self._missing_fields()
        if missing:
            self.message.setText("Fill in before saving: " + ", ".join(missing))
            self.log_saved.emit(False)
            return
        session_id = self.session_id.text().strip()
        response = self.client.call("save_session_log", session_id, self.build_log())
        ok = response.get("success") is not False and bool(response.get("result"))
        self.message.setText("Session log saved." if ok else response.get("error", "Unable to save session log."))
        self.log_saved.emit(ok)
        if ok and self.shadow_bleed.isChecked():
            # Re-record the acknowledgement so it travels with the saved log.
            self.acknowledge_shadow_bleed(True)

    def acknowledge_shadow_bleed(self, checked: bool) -> None:
        """Feature 3: record the operator acknowledgement through the backend API."""
        if not checked:
            self.shadow_bleed_acknowledged.emit(False)
            return
        session_id = self.session_id.text().strip()
        if not session_id:
            self.message.setText("Enter the session id before acknowledging the shadow-bleed check.")
            self.shadow_bleed.blockSignals(True)
            self.shadow_bleed.setChecked(False)
            self.shadow_bleed.blockSignals(False)
            return
        response = self.client.call("acknowledge_shadow_bleed_check", session_id)
        ok = response.get("success") is not False and bool(response.get("result"))
        self.message.setText("Shadow-bleed check acknowledged." if ok else "Unable to record acknowledgement.")
        self.shadow_bleed_acknowledged.emit(ok)
