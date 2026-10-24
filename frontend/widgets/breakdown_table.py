"""Per-hole 'why flagged' breakdown and pairwise tables (Features 11-13, 21)."""

from __future__ import annotations

from typing import Any, Mapping, Optional

from PySide6.QtCore import Signal
from PySide6.QtGui import QBrush, QColor
from PySide6.QtWidgets import QHBoxLayout, QLabel, QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget

from frontend.widgets.math_block import CitationProvider, make_info_button
from frontend.widgets.thumbnail_strip import low_confidence_badge

DRIVER_TINT = QColor(229, 57, 53, 38)  # #E53935 at ~15%


def _fmt(value: Any) -> str:
    if value is None:
        return "—"
    try:
        return f"{float(value):.4g}"
    except (TypeError, ValueError):
        return str(value)


class BreakdownTable(QWidget):
    """Header (comparison group + LC badge) over a descriptor table with driver rows tinted."""

    descriptor_clicked = Signal(str)

    def __init__(self, citation_provider: Optional[CitationProvider] = None) -> None:
        super().__init__()
        self.header = QLabel("Select a hole.")
        self.header.setWordWrap(True)
        self._badge_slot = QHBoxLayout()
        self.table = QTableWidget(0, 5)
        self.table.setHorizontalHeaderLabels(("Descriptor", "Hole value", "Group median", "Group MAD", "z-score"))
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.table.cellClicked.connect(self._on_cell_clicked)
        head = QHBoxLayout()
        head.addWidget(self.header, 1)
        head.addLayout(self._badge_slot)
        head.addWidget(make_info_button("statistics", citation_provider, self))
        layout = QVBoxLayout(self)
        layout.addLayout(head)
        layout.addWidget(self.table)

    # ------------------------------------------------------------ helpers
    def _set_badge(self, record: Mapping[str, Any]) -> None:
        while self._badge_slot.count():
            item = self._badge_slot.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        badge = low_confidence_badge(dict(record))
        if badge is not None:
            self._badge_slot.addWidget(badge)

    def _set_rows(self, headers, rows, tints=None) -> None:
        self.table.clear()
        self.table.setColumnCount(len(headers))
        self.table.setHorizontalHeaderLabels(headers)
        self.table.setRowCount(len(rows))
        for r, values in enumerate(rows):
            for c, value in enumerate(values):
                item = QTableWidgetItem(value if isinstance(value, str) else _fmt(value))
                if tints and tints[r]:
                    item.setBackground(QBrush(DRIVER_TINT))
                self.table.setItem(r, c, item)

    def _on_cell_clicked(self, row: int, _col: int) -> None:
        item = self.table.item(row, 0)
        if item is not None and item.data(256):
            self.descriptor_clicked.emit(str(item.data(256)))

    # ----------------------------------------------------- Feature 12
    def update_breakdown(self, payload: Mapping[str, Any]) -> None:
        """Rows already sorted by |z| (backend); full-row tint on driver descriptors."""
        rows = payload.get("rows") or []
        group = payload.get("comparison_group") or {}
        self.header.setText(group.get("description") or f"Hole {payload.get('hole_id', '')}")
        self._set_badge({"low_confidence": group.get("low_confidence")})
        self._set_rows(
            ("Descriptor", "Hole value", "Group median", "Group MAD", "z-score"),
            [(r.get("descriptor", ""), r.get("hole_value"), r.get("group_median"), r.get("group_mad"), r.get("z"))
             for r in rows],
            [bool(r.get("driver")) for r in rows],
        )
        for i, r in enumerate(rows):
            self.table.item(i, 0).setData(256, r.get("category"))

    # ----------------------------------------------------- Feature 13
    def update_pairwise(self, payload: Mapping[str, Any]) -> None:
        table = payload.get("table") or []
        a, b = payload.get("hole_id", "A"), payload.get("compare_to", "B")
        self.header.setText(f"{a} vs {b}" + ("  (auto: most typical)" if payload.get("auto_selected") else ""))
        self._set_badge(payload.get("left") or {})
        self._set_rows(("Descriptor", str(a), str(b), "Delta"),
                       [(r.get("descriptor", ""), r.get("hole_a"), r.get("hole_b"), r.get("delta")) for r in table])

    # ----------------------------------------------------- Feature 11
    def update_tooltip(self, payload: Mapping[str, Any]) -> None:
        values = payload.get("values") or {}
        self.header.setText(f"{payload.get('hole_id', '')} — {payload.get('type') or ''}")
        self._set_badge(payload)
        self._set_rows(("Field", "Value"), [(k, v) for k, v in values.items()])

    def update_traceability(self, payload: Mapping[str, Any]) -> None:
        self.header.setText(str(payload.get("summary", "")))
        self._set_rows(("Field", "Value"), [("method", str(payload.get("method", "")))])
