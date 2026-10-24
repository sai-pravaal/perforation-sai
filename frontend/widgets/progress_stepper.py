"""Five-phase pipeline stepper (Feature 5)."""

from __future__ import annotations

from typing import Any, Dict, Iterable, Optional, Set

from PySide6.QtCore import QPointF, QRectF, Qt, QTimer
from PySide6.QtGui import QColor, QFont, QPainter, QPen
from PySide6.QtWidgets import QSizePolicy, QWidget

PHASE_NAMES = [
    "Ingestion",
    "Calibration & Registration",
    "Geometry & Classification",
    "Feature Extraction",
    "Statistical Detection & Output",
]
# Same fixed step -> phase grouping as backend/pipeline/runner.py.
STEP_PHASE: Dict[int, int] = {1: 0, **{s: 1 for s in range(2, 6)}, **{s: 2 for s in range(6, 11)},
                              **{s: 3 for s in range(11, 19)}, **{s: 4 for s in range(19, 24)}}

ACCENT = QColor("#2E7D6B")
MUTED = QColor("#9E9E9E")
FAIL = QColor("#E53935")
TEXT = QColor("#E0E0E0")


class ProgressStepper(QWidget):
    """Five connected nodes: completed (filled, ticked), active (pulsing), upcoming (hollow)."""

    def __init__(self) -> None:
        super().__init__()
        self.setMinimumHeight(86)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.completed: Set[int] = set()
        self.active: Optional[int] = None
        self.failed: Optional[int] = None
        self._pulse = 1.0
        self._pulse_dir = -0.08
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._tick)

    # ---------------------------------------------------------------- state
    def set_state(self, completed: Iterable[int], active: Optional[int], failed: Optional[int] = None) -> None:
        self.completed = set(completed)
        self.active = active
        self.failed = failed
        if active is not None and failed is None:
            self._timer.start(60)
        else:
            self._timer.stop()
            self._pulse = 1.0
        self.update()

    def update_status(self, status: Dict[str, Any], progress: Optional[Dict[str, Any]] = None) -> Optional[int]:
        """Derive phase state from get_pipeline_status; returns the active phase index."""
        stages = status.get("stages", []) if isinstance(status, dict) else []
        done_steps = {s.get("step_id") for s in stages if s.get("status") == "complete"}
        failed_steps = [s.get("step_id") for s in stages if s.get("status") == "failed"]
        completed = {p for p in range(5) if all(st in done_steps for st, ph in STEP_PHASE.items() if ph == p)}
        failed = STEP_PHASE.get(failed_steps[-1]) if failed_steps else None
        run_status = status.get("status") if isinstance(status, dict) else None
        active = None
        if run_status == "in_progress" and failed is None:
            active = next((p for p in range(5) if p not in completed), None)
        self.set_state(completed, active, failed)
        return active if active is not None else failed

    # --------------------------------------------------------------- paint
    def _tick(self) -> None:
        self._pulse += self._pulse_dir
        if self._pulse <= 0.35 or self._pulse >= 1.0:
            self._pulse_dir = -self._pulse_dir
        self.update()

    def paintEvent(self, _event) -> None:  # noqa: N802 (Qt override)
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        w = self.width()
        n = len(PHASE_NAMES)
        margin = 60
        span = max(1, w - 2 * margin)
        xs = [margin + span * i / (n - 1) for i in range(n)]
        y = 24.0
        r = 10.0
        painter.setPen(QPen(MUTED, 2))
        painter.drawLine(QPointF(xs[0], y), QPointF(xs[-1], y))
        font = QFont(painter.font())
        font.setPointSizeF(max(7.5, font.pointSizeF() - 1))
        painter.setFont(font)
        for i, x in enumerate(xs):
            center = QPointF(x, y)
            if i == self.failed:
                painter.setPen(QPen(FAIL, 2))
                painter.setBrush(FAIL)
                painter.drawEllipse(center, r, r)
            elif i in self.completed:
                painter.setPen(QPen(ACCENT, 2))
                painter.setBrush(ACCENT)
                painter.drawEllipse(center, r, r)
                painter.setPen(QPen(Qt.GlobalColor.white, 2))
                painter.drawLine(QPointF(x - 4, y), QPointF(x - 1, y + 4))
                painter.drawLine(QPointF(x - 1, y + 4), QPointF(x + 5, y - 4))
            elif i == self.active:
                colour = QColor(ACCENT)
                colour.setAlphaF(self._pulse)
                painter.setPen(QPen(colour, 3))
                painter.setBrush(Qt.BrushStyle.NoBrush)
                painter.drawEllipse(center, r, r)
            else:
                painter.setPen(QPen(MUTED, 2))
                painter.setBrush(QColor("#1E1E1E"))
                painter.drawEllipse(center, r, r)
            painter.setPen(TEXT if (i in self.completed or i == self.active) else MUTED)
            painter.drawText(QRectF(x - 70, y + 16, 140, 40),
                             int(Qt.AlignmentFlag.AlignHCenter | Qt.TextFlag.TextWordWrap), PHASE_NAMES[i])
        painter.end()
