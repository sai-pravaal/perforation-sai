"""Results, traceability and reporting (Features 12-21, 25)."""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QComboBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QScrollArea,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from frontend.client import Client
from frontend.widgets.breakdown_table import BreakdownTable
from frontend.widgets.compass_plot import CompassPlot
from frontend.widgets.graphics_view_sync import load_display_pixmap
from frontend.widgets.math_block import MathBlock
from frontend.widgets.thumbnail_strip import low_confidence_badge

AUTO = "(auto: most typical)"
EDGES = ["top", "bottom", "left", "right"]


def _clear_layout(layout) -> None:
    while layout.count():
        item = layout.takeAt(0)
        if item.widget():
            item.widget().deleteLater()
        elif item.layout():
            _clear_layout(item.layout())


class _PillButton(QPushButton):
    """Feature 18 pill: band colour + identifying text, never the label alone."""

    def __init__(self, text: str, band: Optional[str]) -> None:
        super().__init__(text)
        colours = {"high": "#E53935", "medium": "#FFB300", "low": "#42A5F5"}
        fg = "#1E1E1E" if band == "medium" else "#FFFFFF"
        self.setStyleSheet(f"background-color: {colours.get(band, '#3A3A3A')}; color: {fg};"
                           "border-radius: 10px; padding: 3px 12px;")


class ResultsTab(QWidget):
    """Summary header, Hole Detail / Shadow Vectors / Edge Summary / Regions sub-views, settings."""

    def __init__(self, client: Optional[Client] = None) -> None:
        super().__init__()
        self.client = client or Client()
        self._session_id: Optional[str] = None
        self._citations: Dict[str, Dict[str, Any]] = {}
        self._loaded = False

        # ---- summary header (Features 18, 20)
        self.pills = QHBoxLayout()
        self.pills.setAlignment(Qt.AlignmentFlag.AlignLeft)
        self.summary_list = QListWidget()  # evidence table travelling with the pills
        self.summary_list.setMaximumHeight(140)
        self.summary_list.itemClicked.connect(self._on_summary_item)
        self.low_confidence = QLabel("Low-confidence holes: none loaded")
        refresh = QPushButton("Refresh results")
        refresh.clicked.connect(self.load_results_summary)
        self.export_button = QPushButton("Export session")
        self.export_button.clicked.connect(self.export_session)
        self.pdf_button = QPushButton("Export PDF Report")
        self.pdf_button.clicked.connect(self.export_pdf)
        header_row = QHBoxLayout()
        header_row.addLayout(self.pills, 1)
        header_row.addWidget(refresh)
        header_row.addWidget(self.export_button)
        header_row.addWidget(self.pdf_button)

        # ---- Hole Detail (Features 12, 13, 15, 17)
        self.hole_select = QComboBox()
        self.hole_select.currentIndexChanged.connect(self._on_hole_selected)
        self.breakdown = BreakdownTable(self._citation)
        self.breakdown.descriptor_clicked.connect(self._open_math_category)
        self.compare_select = QComboBox()
        self.compare_select.currentIndexChanged.connect(self.load_pairwise)
        self.auto_tag = QLabel("(auto)")
        self.pair_cards = QHBoxLayout()
        self.pair_table = BreakdownTable(self._citation)
        self.math = MathBlock(self._citation)

        detail = QWidget()
        detail_layout = QVBoxLayout(detail)
        sel = QFormLayout()
        sel.addRow("Hole", self.hole_select)
        detail_layout.addLayout(sel)
        detail_layout.addWidget(self.breakdown)
        cmp_row = QHBoxLayout()
        cmp_row.addWidget(QLabel("Compare against:"))
        cmp_row.addWidget(self.compare_select)
        cmp_row.addWidget(self.auto_tag)
        cmp_row.addStretch(1)
        detail_layout.addLayout(cmp_row)
        detail_layout.addLayout(self.pair_cards)
        detail_layout.addWidget(self.pair_table)
        detail_layout.addWidget(self.math)
        detail_scroll = QScrollArea()
        detail_scroll.setWidgetResizable(True)
        detail_scroll.setWidget(detail)

        # ---- Shadow vectors (Feature 14)
        self.vector_scope = QComboBox()
        self.vector_scope.addItem("All", None)
        for e in EDGES:
            self.vector_scope.addItem(e.title(), e)
        self.vector_scope.currentIndexChanged.connect(self.load_shadow_vectors)
        self.compass = CompassPlot()
        vectors = QWidget()
        vectors_layout = QVBoxLayout(vectors)
        scope_row = QHBoxLayout()
        scope_row.addWidget(QLabel("Edge"))
        scope_row.addWidget(self.vector_scope)
        scope_row.addStretch(1)
        vectors_layout.addLayout(scope_row)
        vectors_layout.addWidget(self.compass)

        # ---- Edge summary (Feature 16)
        self.edge_select = QComboBox()
        for e in EDGES:
            self.edge_select.addItem(e.title(), e)
        self.edge_select.currentIndexChanged.connect(self.load_edge_summary)
        self.edge_plot = CompassPlot()
        edge = QWidget()
        edge_layout = QVBoxLayout(edge)
        edge_row = QHBoxLayout()
        edge_row.addWidget(QLabel("Edge"))
        edge_row.addWidget(self.edge_select)
        edge_row.addStretch(1)
        edge_layout.addLayout(edge_row)
        edge_layout.addWidget(self.edge_plot)

        # ---- Regions (Feature 19)
        self.region_list = QListWidget()
        self.region_list.setMaximumWidth(220)
        self.region_list.currentItemChanged.connect(lambda cur, _prev: self.load_region_summary(
            cur.data(Qt.ItemDataRole.UserRole) if cur else None))
        self.region_title = QLabel("Select a region.")
        self.region_title.setObjectName("cardTitle")
        self.region_statement = QLabel("")
        self.region_statement.setWordWrap(True)
        self.region_members = QHBoxLayout()
        self.region_members.setAlignment(Qt.AlignmentFlag.AlignLeft)
        self.region_table = QTableWidget(0, 3)
        self.region_table.setHorizontalHeaderLabels(("Descriptor", "Mean z", "Mean |z|"))
        self.region_table.horizontalHeader().setStretchLastSection(True)
        region_card = QFrame()
        region_card.setObjectName("card")
        card_layout = QVBoxLayout(region_card)
        card_layout.addWidget(self.region_title)
        card_layout.addLayout(self.region_members)
        card_layout.addWidget(self.region_statement)
        card_layout.addWidget(self.region_table)
        regions = QWidget()
        regions_layout = QHBoxLayout(regions)
        regions_layout.addWidget(self.region_list)
        regions_layout.addWidget(region_card, 1)

        self.views = QTabWidget()
        self.views.addTab(detail_scroll, "Hole Detail")
        self.views.addTab(vectors, "Shadow Vectors")
        self.views.addTab(edge, "Edge Summary")
        self.views.addTab(regions, "Regions")

        # ---- Settings & thresholds (Feature 25), collapsed and lazily loaded
        self.settings_box = QGroupBox("Settings & Thresholds Used in This Run")
        self.settings_box.setCheckable(True)
        self.settings_box.setChecked(False)
        self.settings_table = QTableWidget(0, 2)
        self.settings_table.setHorizontalHeaderLabels(("Constant", "Value"))
        self.settings_table.horizontalHeader().setStretchLastSection(True)
        self.settings_table.setVisible(False)
        settings_layout = QVBoxLayout(self.settings_box)
        settings_layout.addWidget(self.settings_table)
        self.settings_box.toggled.connect(self._on_settings_toggled)

        self.message = QLabel("")
        self.message.setWordWrap(True)
        layout = QVBoxLayout(self)
        layout.addLayout(header_row)
        layout.addWidget(self.summary_list)
        layout.addWidget(self.low_confidence)
        layout.addWidget(self.views, 1)
        layout.addWidget(self.settings_box)
        layout.addWidget(self.message)

    # ------------------------------------------------------------- session
    def set_session_id(self, session_id: Optional[str]) -> None:
        self._session_id = session_id or None
        self._loaded = False
        self.settings_table.setRowCount(0)
        if self.isVisible() and self._session_id:
            self.load_results_summary()

    def showEvent(self, event) -> None:  # noqa: N802 (Qt override)
        super().showEvent(event)
        if self._session_id and not self._loaded:
            self.load_results_summary()

    def _call(self, fn: str, *args) -> Optional[Any]:
        """Shared response handling: inline error, None on failure."""
        payload, error = Client.unwrap(self.client.call(fn, *args), f"{fn} failed.")
        if error:
            self.message.setText(error)
        return payload

    def _citation(self, descriptor_type: str) -> Optional[Dict[str, Any]]:
        if descriptor_type not in self._citations:
            payload = self._call("get_citation", descriptor_type)
            self._citations[descriptor_type] = (payload or {}).get("citation") or {}
        return self._citations[descriptor_type]

    # -------------------------------------------------------- Feature 18
    def load_results_summary(self) -> None:
        """Pills per flagged region (or standalone flagged hole), evidence list, LC holes."""
        if not self._session_id:
            return
        payload = self._call("get_results_summary", self._session_id)
        if payload is None:
            return
        self._loaded = True
        holes: List[Dict[str, Any]] = payload.get("holes", [])
        regions: List[Dict[str, Any]] = payload.get("regions", [])
        _clear_layout(self.pills)
        in_region = {h["hole_id"] for h in holes if h.get("region_id")}
        for r in regions:
            pill = _PillButton(f"{r.get('evidence')}: {str(r.get('band') or '').upper()}", r.get("band"))
            pill.clicked.connect(lambda _=False, rid=r.get("region_id"): self._goto_region(rid))
            self.pills.addWidget(pill)
        for h in holes:
            if h.get("is_outlier") and h["hole_id"] not in in_region:
                pill = _PillButton(f"{h['hole_id']}: {str(h.get('band') or '').upper()}", h.get("band"))
                pill.clicked.connect(lambda _=False, hid=h["hole_id"]: self._goto_hole(hid))
                self.pills.addWidget(pill)
        if not self.pills.count():
            self.pills.addWidget(QLabel("No candidate regions or flagged holes."))

        self.summary_list.clear()
        for h in holes:
            drivers = ", ".join(f"{d['descriptor']} (z {d['z']:+.1f})" for d in h.get("drivers", []))
            score = h.get("combined_score")
            text = (f"{h['hole_id']}  [{h.get('band') or '—'}]  score "
                    f"{score:.2f}" if isinstance(score, (int, float)) else f"{h['hole_id']}  score n/a")
            if drivers:
                text += f"  · drivers: {drivers}"
            if h.get("low_confidence"):
                text += "  · LC"
            item = QListWidgetItem(text)
            item.setData(Qt.ItemDataRole.UserRole, h["hole_id"])
            self.summary_list.addItem(item)
        low = [h["hole_id"] for h in holes if h.get("low_confidence")]
        self.low_confidence.setText("Low-confidence holes (pooled comparison): " + (", ".join(low) if low else "none"))

        self._populate_holes([h["hole_id"] for h in holes])
        self.region_list.clear()
        for r in regions:
            item = QListWidgetItem(f"{r.get('evidence')} [{r.get('band')}]")
            item.setData(Qt.ItemDataRole.UserRole, r.get("region_id"))
            self.region_list.addItem(item)
        self.load_shadow_vectors()
        self.load_edge_summary()
        self.message.setText("")

    def _populate_holes(self, hole_ids: List[str]) -> None:
        for combo, extra in ((self.hole_select, None), (self.compare_select, AUTO)):
            combo.blockSignals(True)
            combo.clear()
            if extra:
                combo.addItem(extra, None)
            for hid in hole_ids:
                combo.addItem(hid, hid)
            combo.blockSignals(False)
        if hole_ids:
            self._on_hole_selected()

    def _on_summary_item(self, item: QListWidgetItem) -> None:
        self._goto_hole(item.data(Qt.ItemDataRole.UserRole))

    def _goto_hole(self, hole_id: str) -> None:
        index = self.hole_select.findData(hole_id)
        if index >= 0:
            self.hole_select.setCurrentIndex(index)
        self.views.setCurrentIndex(0)

    def _goto_region(self, region_id: str) -> None:
        self.views.setCurrentIndex(3)
        for i in range(self.region_list.count()):
            if self.region_list.item(i).data(Qt.ItemDataRole.UserRole) == region_id:
                self.region_list.setCurrentRow(i)

    # ------------------------------------------------- Features 12, 13, 15
    def _on_hole_selected(self) -> None:
        hole_id = self.hole_select.currentData()
        if not hole_id or not self._session_id:
            return
        breakdown = self._call("get_hole_breakdown", self._session_id, hole_id)
        if breakdown:
            self.breakdown.update_breakdown(breakdown)
        self.compare_select.blockSignals(True)
        self.compare_select.setCurrentIndex(0)
        self.compare_select.blockSignals(False)
        self.load_pairwise()
        self.load_math()

    def load_pairwise(self) -> None:
        hole_id = self.hole_select.currentData()
        if not hole_id or not self._session_id:
            return
        compare_to = self.compare_select.currentData()
        payload = self._call("get_pairwise_comparison", self._session_id, hole_id, compare_to)
        self.auto_tag.setVisible(compare_to is None)
        if not payload:
            return
        _clear_layout(self.pair_cards)
        for side in ("left", "right"):
            rec = payload.get(side) or {}
            card = QFrame()
            card.setObjectName("card")
            col = QVBoxLayout(card)
            title_row = QHBoxLayout()
            title_row.addWidget(QLabel(str(rec.get("hole_id", ""))))
            badge = low_confidence_badge(rec)
            if badge is not None:
                title_row.addWidget(badge)
            title_row.addStretch(1)
            col.addLayout(title_row)
            crops = QHBoxLayout()
            for tag, path in (rec.get("crops") or {}).items():
                label = QLabel()
                pixmap = load_display_pixmap(path)
                if not pixmap.isNull():
                    label.setPixmap(pixmap.scaled(140, 140, Qt.AspectRatioMode.KeepAspectRatio,
                                                  Qt.TransformationMode.SmoothTransformation))
                else:
                    label.setText(f"{tag}: n/a")
                label.setToolTip(tag)
                crops.addWidget(label)
            col.addLayout(crops)
            self.pair_cards.addWidget(card)
        self.pair_table.update_pairwise(payload)

    def load_math(self) -> None:
        hole_id = self.hole_select.currentData()
        if not hole_id or not self._session_id:
            return
        payload = self._call("get_math_explanation", self._session_id, hole_id)
        if payload:
            self.math.update_explanation(payload)

    def _open_math_category(self, category: str) -> None:
        index = {"shape": 0, "roughness": 1, "shadow": 2}.get(category)
        if index is not None:
            self.math.toolbox.setCurrentIndex(index)

    # -------------------------------------------------- Features 14, 16
    def load_shadow_vectors(self) -> None:
        if not self._session_id:
            return
        payload = self._call("get_shadow_vectors", self._session_id, self.vector_scope.currentData())
        if payload is not None:
            self.compass.update_vectors(payload.get("holes", []))

    def load_edge_summary(self) -> None:
        if not self._session_id:
            return
        payload = self._call("get_edge_summary", self._session_id, self.edge_select.currentData())
        if payload is not None:
            self.edge_plot.update_edge_summary(payload)

    # -------------------------------------------------------- Feature 19
    def load_region_summary(self, region_id: Optional[str] = None) -> None:
        if not region_id or not self._session_id:
            return
        payload = self._call("get_region_summary", self._session_id, region_id)
        if payload is None:
            return
        self.region_title.setText(str(payload.get("title", region_id)))
        self.region_statement.setText(str(payload.get("statement", "")))
        _clear_layout(self.region_members)
        for m in payload.get("members", []):
            button = QPushButton(m.get("hole_id", ""))
            pixmap = load_display_pixmap(m.get("crop"))
            if not pixmap.isNull():
                button.setIcon(pixmap.scaled(64, 64, Qt.AspectRatioMode.KeepAspectRatio))
            button.clicked.connect(lambda _=False, hid=m.get("hole_id"): self._goto_hole(hid))
            self.region_members.addWidget(button)
            badge = low_confidence_badge(m)
            if badge is not None:
                self.region_members.addWidget(badge)
        table = payload.get("descriptor_table", [])
        self.region_table.setRowCount(len(table))
        dominant = set(payload.get("dominant_descriptors") or [])
        for r, row in enumerate(table):
            for c, value in enumerate((row.get("descriptor"), row.get("mean_z"), row.get("mean_abs_z"))):
                text = value if isinstance(value, str) else ("—" if value is None else f"{value:.2f}")
                item = QTableWidgetItem(text)
                if row.get("descriptor") in dominant:
                    item.setBackground(Qt.GlobalColor.darkRed)
                self.region_table.setItem(r, c, item)

    # -------------------------------------------------------- Feature 25
    def _on_settings_toggled(self, expanded: bool) -> None:
        self.settings_table.setVisible(expanded)
        if expanded and self.settings_table.rowCount() == 0:
            self.load_config_snapshot()

    def load_config_snapshot(self) -> None:
        if not self._session_id:
            return
        payload = self._call("get_config_snapshot", self._session_id)
        if payload is None:
            return
        meta = payload.get("version_metadata") or {}
        constants = meta.get("config_snapshot") if isinstance(meta, dict) and meta.get("config_snapshot") else payload.get("constants", {})
        self.settings_table.setRowCount(len(constants))
        for r, (key, value) in enumerate(constants.items()):
            self.settings_table.setItem(r, 0, QTableWidgetItem(str(key)))
            self.settings_table.setItem(r, 1, QTableWidgetItem(str(value)))

    # ---------------------------------------------------- Features 27-29
    def load_version_metadata(self) -> None:
        if not self._session_id:
            return
        meta = self._call("get_version_metadata", self._session_id)
        if meta:
            self.message.setText(f"App {meta.get('app_version')} · Python {meta.get('python_version')} · "
                                 f"captured {meta.get('captured_at')}")

    def export_session(self) -> None:
        """Request a packaged session archive in the background."""
        if not self._session_id:
            return
        self.message.setText("Exporting session…")
        self.client.call_async("export_session", self._session_id, on_done=self._on_export_done)

    def export_pdf(self) -> None:
        """Feature 20: build the report in the background, then let the user choose where to save it."""
        if not self._session_id:
            return
        self.pdf_button.setEnabled(False)
        self.message.setText("Generating PDF report…")
        self.client.call_async("generate_report", self._session_id, on_done=self._on_pdf_done)

    def _on_pdf_done(self, response: Dict[str, Any]) -> None:
        self.pdf_button.setEnabled(True)
        path, error = Client.unwrap(response, "The PDF report could not be generated.")
        if error or not isinstance(path, str):
            self.message.setText(error or "The PDF report could not be generated.")
            return
        target, _ = QFileDialog.getSaveFileName(self, "Save PDF report", Path(path).name, "PDF (*.pdf)")
        if target:
            shutil.copy2(path, target)
            self.message.setText(f"Report saved to {target}")
        else:
            self.message.setText(f"Report written to {path}")

    def _on_export_done(self, response: Dict[str, Any]) -> None:
        path, error = Client.unwrap(response, "Export failed.")
        self.message.setText(error or f"Session archive written to {path}")

    def run_quality_gate(self) -> None:
        if not self._session_id:
            return
        payload = self._call("run_quality_gate", self._session_id)
        if payload:
            self.message.setText("Quality gate: " + ("; ".join(payload.get("issues", [])) or "all checks passed"))
