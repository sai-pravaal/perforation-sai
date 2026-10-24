"""Live thumbnail strip for stage outputs (Feature 6)."""

from __future__ import annotations

from typing import Any, Dict, Iterable, Optional, Set

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import QHBoxLayout, QLabel, QPushButton, QScrollArea, QVBoxLayout, QWidget

from frontend.widgets.graphics_view_sync import load_display_pixmap

THUMB_PX = 96


def low_confidence_badge(hole_record: Dict[str, Any]) -> Optional[QLabel]:
    """Feature 21 shared badge: 'LC' label for a hole on a pooled comparison group."""
    if not hole_record or not hole_record.get("low_confidence"):
        return None
    badge = QLabel("LC")
    badge.setProperty("lowConfidence", "true")
    badge.setToolTip("Pooled comparison group used — fewer than 5 holes were available on this edge.")
    return badge


class ThumbnailStrip(QScrollArea):
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

    def clear(self) -> None:
        while self._row.count():
            item = self._row.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

    def update_images(self, images: Iterable[Dict[str, Any]]) -> None:
        """Replace the strip contents."""
        self.clear()
        self.append_images(images)

    def append_images(self, images: Iterable[Dict[str, Any]]) -> None:
        """Append new manifest entries and keep the newest visible."""
        for image in images:
            self._row.addWidget(self._thumb(image))
        bar = self.horizontalScrollBar()
        bar.setValue(bar.maximum())

    def _thumb(self, image: Dict[str, Any]) -> QWidget:
        cell = QWidget()
        col = QVBoxLayout(cell)
        col.setContentsMargins(2, 2, 2, 2)
        button = QPushButton()
        button.setFixedSize(THUMB_PX, THUMB_PX)
        pixmap = load_display_pixmap(image.get("abs_path"))
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
