"""'Show the Math' blocks with citation info icons (Features 15 and 17)."""

from __future__ import annotations

from typing import Any, Callable, Dict, Mapping, Optional

from PySide6.QtCore import QPoint, Qt
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QToolBox,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

CATEGORIES = [("shape", "Shape"), ("roughness", "Roughness"), ("shadow", "Shadow")]

CitationProvider = Callable[[str], Optional[Dict[str, Any]]]


class CitationPopover(QFrame):
    """Small popup anchored to an info icon; closes on any outside click or Escape."""

    def __init__(self, citation: Mapping[str, Any], parent: Optional[QWidget] = None) -> None:
        super().__init__(parent, Qt.WindowType.Popup)
        self.setObjectName("card")
        self.setMaximumWidth(420)
        layout = QVBoxLayout(self)
        title = QLabel(str(citation.get("descriptor_name", "Source")))
        title.setObjectName("cardTitle")
        text = QLabel(str(citation.get("citation_text", "No citation recorded.")))
        text.setWordWrap(True)
        source = QLabel("Source: " + str(citation.get("source_reference", "—")))
        source.setWordWrap(True)
        source.setProperty("status", "")
        layout.addWidget(title)
        layout.addWidget(text)
        layout.addWidget(source)


def make_info_button(descriptor_type: str, provider: Optional[CitationProvider], parent: QWidget) -> QToolButton:
    """Feature 17 info icon: a 14px 'i' that opens the descriptor's citation popover."""
    button = QToolButton(parent)
    button.setText("i")
    button.setFixedSize(18, 18)
    button.setToolTip(f"Literature source for {descriptor_type}")

    def show() -> None:
        citation = provider(descriptor_type) if provider else None
        popover = CitationPopover(citation or {}, button)
        popover.move(button.mapToGlobal(QPoint(0, button.height())))
        popover.show()

    button.clicked.connect(show)
    return button


class MathBlock(QWidget):
    """Three collapsible sections (Shape / Roughness / Shadow) filled with the hole's real numbers."""

    def __init__(self, citation_provider: Optional[CitationProvider] = None) -> None:
        super().__init__()
        self.citation_provider = citation_provider
        self.method = QLabel("Method: not loaded")
        self.summary = QLabel("Select a hole to see how its score was calculated.")
        self.summary.setWordWrap(True)
        self.formula = QLabel("")
        self.formula.setWordWrap(True)
        self.formula.setStyleSheet("font-family: Consolas, Menlo, monospace;")
        self.toolbox = QToolBox()
        self._bodies: Dict[str, QLabel] = {}
        self._interpretations: Dict[str, QLabel] = {}
        for key, title in CATEGORIES:
            page = QWidget()
            page_layout = QVBoxLayout(page)
            header = QHBoxLayout()
            header.addWidget(QLabel(f"{title} — source"))
            header.addWidget(make_info_button(key, self._provide, page))
            header.addStretch(1)
            body = QLabel("—")
            body.setWordWrap(True)
            body.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
            body.setStyleSheet("font-family: Consolas, Menlo, monospace;")
            interpretation = QLabel("")
            interpretation.setWordWrap(True)
            page_layout.addLayout(header)
            page_layout.addWidget(body)
            page_layout.addWidget(interpretation)
            self._bodies[key] = body
            self._interpretations[key] = interpretation
            self.toolbox.addItem(page, title)
        stats_row = QHBoxLayout()
        stats_row.addWidget(self.method)
        stats_row.addWidget(make_info_button("statistics", self._provide, self))
        stats_row.addStretch(1)
        layout = QVBoxLayout(self)
        layout.addLayout(stats_row)
        layout.addWidget(self.formula)
        layout.addWidget(self.summary)
        layout.addWidget(self.toolbox)

    def _provide(self, descriptor_type: str) -> Optional[Dict[str, Any]]:
        return self.citation_provider(descriptor_type) if self.citation_provider else None

    def update_explanation(self, payload: Mapping[str, Any]) -> None:
        """Display get_math_explanation output (no recalculation in the UI)."""
        self.method.setText(f"Method: {payload.get('method', 'unknown')}")
        self.formula.setText(str(payload.get("formula", "")))
        self.summary.setText(str(payload.get("summary", "")))
        blocks = payload.get("blocks") or {}
        for key, _ in CATEGORIES:
            block = blocks.get(key) or {}
            self._bodies[key].setText("\n".join(block.get("formula_lines") or ["—"]))
            self._interpretations[key].setText(str(block.get("interpretation", "")))
        # Start each newly selected hole collapsed (no page expanded beyond the first header).
        self.toolbox.setCurrentIndex(0)
