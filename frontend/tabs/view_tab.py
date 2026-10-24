"""View tab: filterable gallery, image comparison and hover tooltips (Features 8-11)."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from PySide6.QtCore import QSize, Qt
from PySide6.QtGui import QIcon
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from frontend.client import Client
from frontend.widgets.graphics_view_sync import GraphicsViewSync, load_display_pixmap

ALL = "All"
# Manifest image stage labels written by the pipeline, with readable names.
STAGE_LABELS = {
    "registration": "Registration (step 5)",
    "stamp_boundary": "Stamp boundary (step 6)",
    "roi_crop": "Hole crops (step 9)",
    "subpixel_contour": "Subpixel contour (step 11)",
    "final_output": "Annotated overview (step 23)",
}
EDGES = ["top", "bottom", "left", "right"]
GALLERY_ICON = 96


class ViewTab(QWidget):
    """Gallery on the left, comparison panel on the right (resizable splitter)."""

    def __init__(self, client: Optional[Client] = None) -> None:
        super().__init__()
        self.client = client or Client()
        self._session_id: Optional[str] = None
        self._slots: List[Optional[Dict[str, Any]]] = [None, None]
        self._tooltip_cache: Dict[str, str] = {}

        # Filter bar (Feature 10)
        self.stage_filter = QComboBox()
        self.stage_filter.addItem(ALL, None)
        for key, label in STAGE_LABELS.items():
            self.stage_filter.addItem(label, key)
        self.hole_filter = QComboBox()
        self.hole_filter.addItem(ALL, None)
        self.edge_filter = QComboBox()
        self.edge_filter.addItem(ALL, None)
        for e in EDGES:
            self.edge_filter.addItem(e.title(), e)
        self.flagged_only = QCheckBox("Flagged only")
        for combo in (self.stage_filter, self.hole_filter, self.edge_filter):
            combo.currentIndexChanged.connect(self.refresh_gallery)
        self.flagged_only.toggled.connect(self.refresh_gallery)
        filters = QHBoxLayout()
        for label, widget in (("Stage", self.stage_filter), ("Hole", self.hole_filter), ("Edge", self.edge_filter)):
            filters.addWidget(QLabel(label))
            filters.addWidget(widget)
        filters.addWidget(self.flagged_only)
        filters.addStretch(1)

        self.gallery = QListWidget()
        self.gallery.setViewMode(QListWidget.ViewMode.IconMode)
        self.gallery.setIconSize(QSize(GALLERY_ICON, GALLERY_ICON))
        self.gallery.setResizeMode(QListWidget.ResizeMode.Adjust)
        self.gallery.setMouseTracking(True)
        self.gallery.itemClicked.connect(self._on_item_clicked)
        self.gallery.itemEntered.connect(self._on_item_hovered)
        hint = QLabel("Click: image A · Ctrl+click: image B")
        hint.setProperty("status", "")

        left = QWidget()
        left_layout = QVBoxLayout(left)
        left_layout.addLayout(filters)
        left_layout.addWidget(self.gallery)
        left_layout.addWidget(hint)

        self.comparison = GraphicsViewSync()
        self.message = QLabel("Choose a session to see its images.")
        right = QWidget()
        right_layout = QVBoxLayout(right)
        right_layout.addWidget(self.comparison)
        right_layout.addWidget(self.message)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(left)
        splitter.addWidget(right)
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 7)
        layout = QVBoxLayout(self)
        layout.addWidget(splitter)

    # ------------------------------------------------------------ session
    def set_session_id(self, session_id: Optional[str]) -> None:
        self._session_id = session_id or None
        self._tooltip_cache.clear()
        self._slots = [None, None]
        self.refresh_gallery()

    def showEvent(self, event) -> None:  # noqa: N802 (Qt override)
        super().showEvent(event)
        if self._session_id:
            self.refresh_gallery()

    # ------------------------------------------------------------ gallery
    def refresh_gallery(self) -> None:
        """Feature 10: re-query the gallery with the current filter state."""
        self.gallery.clear()
        if not self._session_id:
            return
        filters = {
            "stage": self.stage_filter.currentData(),
            "hole_id": self.hole_filter.currentData(),
            "edge": self.edge_filter.currentData(),
            "flagged_only": self.flagged_only.isChecked(),
        }
        records, error = Client.unwrap(self.client.call("query_gallery", self._session_id, filters),
                                       "Unable to load gallery.")
        if error:
            self.message.setText(error)
            return
        records = records or []
        self._refresh_hole_choices(records)
        for record in records:
            item = QListWidgetItem(f"{STAGE_LABELS.get(record.get('stage'), record.get('stage', ''))}\n"
                                   f"{record.get('hole_id') or ''}".strip())
            pixmap = load_display_pixmap(record.get("abs_path"))
            if not pixmap.isNull():
                item.setIcon(QIcon(pixmap.scaled(GALLERY_ICON, GALLERY_ICON, Qt.AspectRatioMode.KeepAspectRatio,
                                                 Qt.TransformationMode.SmoothTransformation)))
            item.setData(Qt.ItemDataRole.UserRole, record)
            self.gallery.addItem(item)
        self.message.setText(f"{self.gallery.count()} image(s).")

    def _refresh_hole_choices(self, records: List[Dict[str, Any]]) -> None:
        if self.hole_filter.currentData() is not None:
            return  # keep the user's current choice and list
        holes = sorted({r.get("hole_id") for r in records if r.get("hole_id")})
        self.hole_filter.blockSignals(True)
        self.hole_filter.clear()
        self.hole_filter.addItem(ALL, None)
        for hid in holes:
            self.hole_filter.addItem(hid, hid)
        self.hole_filter.blockSignals(False)

    # --------------------------------------------------------- comparison
    def _on_item_clicked(self, item: QListWidgetItem) -> None:
        record = item.data(Qt.ItemDataRole.UserRole) or {}
        ctrl = bool(QApplication.keyboardModifiers() & Qt.KeyboardModifier.ControlModifier)
        self._assign(1 if ctrl else 0, record)

    def select_image(self, record: Dict[str, Any]) -> None:
        """Pre-select an image clicked in the Processing tab's thumbnail strip."""
        self._assign(0, record)

    def _assign(self, slot: int, record: Dict[str, Any]) -> None:
        self._slots[slot] = record
        self.comparison.set_slot(slot, record)
        if all(self._slots):
            self.compare_images()

    def compare_images(self) -> None:
        """Feature 8: resolve both selections through the backend and render the pair."""
        a, b = self._slots
        if not (a and b and self._session_id):
            return
        payload, error = Client.unwrap(
            self.client.call("get_comparison_pair", self._session_id, a.get("path"), b.get("path")),
            "Unable to load comparison.")
        if error:
            self.message.setText(error)
            return
        self.comparison.set_pair(payload.get("pair", []))
        self.message.setText("Comparison loaded.")

    # ------------------------------------------------------------ tooltips
    def _on_item_hovered(self, item: QListWidgetItem) -> None:
        """Feature 11: attach the hole's descriptor values as the item tooltip (cached per hole)."""
        record = item.data(Qt.ItemDataRole.UserRole) or {}
        hole_id = record.get("hole_id")
        if not hole_id or not self._session_id or item.toolTip():
            return
        text = self._tooltip_cache.get(hole_id)
        if text is None:
            payload, _error = Client.unwrap(self.client.call("get_hole_tooltip", self._session_id, hole_id))
            payload = payload or {}
            text = f"{hole_id}\n{payload.get('summary', 'No values available yet.')}"
            if payload.get("low_confidence"):
                text += "\nLC: pooled comparison group (fewer than 5 holes on this edge)"
            self._tooltip_cache[hole_id] = text
        item.setToolTip(text)

    # Kept for callers of the earlier interface.
    def load_tooltip(self, hole_id: str) -> str:
        return self._tooltip_cache.get(hole_id, "")
