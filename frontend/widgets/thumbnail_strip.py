"""Live thumbnail strip for stage outputs (Feature 6)."""

from __future__ import annotations

from collections import deque
from typing import Any, Deque, Dict, Iterable, Optional, Set

from PySide6.QtCore import QTimer, Qt, Signal
from PySide6.QtWidgets import QHBoxLayout, QLabel, QPushButton, QScrollArea, QVBoxLayout, QWidget

from frontend.widgets.graphics_view_sync import load_thumbnail_pixmap
from frontend.widgets.scroll import HorizontalStrip

THUMB_PX = 96
BATCH_SIZE = 8          # thumbnails built per timer tick, so a step adding hundreds of images never freezes the window
BATCH_INTERVAL_MS = 15


def low_confidence_badge(hole_record: Dict[str, Any]) -> Optional[QLabel]:
    """Feature 21 shared badge: 'LC' label for a hole on a pooled comparison group."""
    if not hole_record or not hole_record.get("low_confidence"):
        return None
    badge = QLabel("LC")
    badge.setProperty("lowConfidence", "true")
    badge.setToolTip("Pooled comparison group used — fewer than 5 holes were available on this edge.")
    return badge


class ThumbnailStrip(HorizontalStrip):
    """Horizontally scrolling 96x96 thumbnails appended as manifest entries appear."""

    image_clicked = Signal(dict)

    def __init__(self) -> None:
        super().__init__()
        self.setWidgetResizable(True)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setFixedHeight(THUMB_PX + 64)
        self._content = QWidget()
        self._row = QHBoxLayout(self._content)
        self._row.setAlignment(Qt.AlignmentFlag.AlignLeft)
        self.setWidget(self._content)
        self.low_confidence_holes: Set[str] = set()
        self._queue: Deque[Dict[str, Any]] = deque()
        self._timer = QTimer(self)
        self._timer.setInterval(BATCH_INTERVAL_MS)
        self._timer.timeout.connect(self._add_batch)

    def clear(self) -> None:
        self._queue.clear()
        self._timer.stop()
        while self._row.count():
            item = self._row.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

    def update_images(self, images: Iterable[Dict[str, Any]]) -> None:
        """Replace the strip contents."""
        self.clear()
        self.append_images(images)

    def append_images(self, images: Iterable[Dict[str, Any]]) -> None:
        """Queue new manifest entries; they are added a few at a time, newest kept in view."""
        self._queue.extend(images)
        if self._queue and not self._timer.isActive():
            self._timer.start()

    def _add_batch(self) -> None:
        for _ in range(min(BATCH_SIZE, len(self._queue))):
            self._row.addWidget(self._thumb(self._queue.popleft()))
        bar = self.horizontalScrollBar()
        bar.setValue(bar.maximum())
        if not self._queue:
            self._timer.stop()

    def pending_count(self) -> int:
        """Thumbnails queued but not yet built."""
        return len(self._queue)

    def _thumb(self, image: Dict[str, Any]) -> QWidget:
        cell = QWidget()
        col = QVBoxLayout(cell)
        col.setContentsMargins(2, 2, 2, 2)
        button = QPushButton()
        button.setFixedSize(THUMB_PX, THUMB_PX)
        pixmap = load_thumbnail_pixmap(image)
        if not pixmap.isNull():
            button.setIcon(pixmap.scaled(THUMB_PX, THUMB_PX, Qt.AspectRatioMode.KeepAspectRatio,
                                         Qt.TransformationMode.SmoothTransformation))
            button.setIconSize(button.size())
        else:
            button.setText("n/a")
        button.setToolTip(str(image.get("path", "")))
        button.clicked.connect(lambda _=False, img=dict(image): self.image_clicked.emit(img))
        caption_row = QHBoxLayout()
        hole = image.get("hole_id") or ""
        caption = QLabel(f"{image.get('stage', '')}\n{hole}".strip())
        caption.setAlignment(Qt.AlignmentFlag.AlignCenter)
        caption_row.addWidget(caption)
        badge = low_confidence_badge({"low_confidence": hole in self.low_confidence_holes})
        if badge is not None:
            caption_row.addWidget(badge)
        col.addWidget(button)
        col.addLayout(caption_row)
        return cell
