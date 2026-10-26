"""Vertical scrolling wrapper for tabs whose content can be taller than the window."""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QFrame, QScrollArea, QVBoxLayout, QWidget


class HorizontalStrip(QScrollArea):
    """Scroll area for a single row of items: the mouse wheel scrolls it sideways.

    A plain QScrollArea only moves a horizontal bar with Shift+wheel or by
    dragging the (thin) bar, which reads as "will not scroll".
    """

    def wheelEvent(self, event) -> None:  # noqa: N802 (Qt override)
        bar = self.horizontalScrollBar()
        if bar.maximum() > 0:
            delta = event.angleDelta().y() or event.angleDelta().x()
            bar.setValue(bar.value() - delta)
            event.accept()
        else:
            super().wheelEvent(event)


def install_scroll_area(tab: QWidget) -> QWidget:
    """Give `tab` an outer scroll area and return the inner widget to build the layout on.

    Without this a tab's minimum height (e.g. the session log form, about
    780 px) becomes the window's minimum height, and anything below the screen
    edge is unreachable. Content keeps the viewport's width and scrolls
    vertically only.
    """
    content = QWidget()
    area = QScrollArea()
    area.setWidgetResizable(True)
    area.setFrameShape(QFrame.Shape.NoFrame)
    area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
    area.setWidget(content)
    outer = QVBoxLayout(tab)
    outer.setContentsMargins(0, 0, 0, 0)
    outer.addWidget(area)
    return content
