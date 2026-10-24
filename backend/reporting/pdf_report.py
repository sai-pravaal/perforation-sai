"""Exportable PDF Report (Feature 20).

File: backend/reporting/pdf_report.py
Lays out, in the order the specification gives: title/session metadata, Step 23's
annotated overview image, a legend, the full per-hole results table, and the
region/summary findings, followed by Feature 28's version-metadata block. It
pulls directly from existing outputs (results_table.csv, candidate_regions.json,
score.json, manifest.json) and performs no new computation. ReportLab is
imported lazily so this module loads even when the package is not installed.

Contains zero Qt/PySide6 imports.
"""

import csv
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from backend.config.pipeline_constants import OUTLIER_TAIL_PROBABILITY, SUSPICION_BANDS
from backend.session.session_manager import SessionManager

logger = logging.getLogger(__name__)

BAND_LABELS = {"genuine": "Low", "suspicious": "Medium", "highly_suspicious": "High"}
# Results table split so every column fits a landscape page at a readable size.
IDENTITY_COLUMNS = ["hole_id", "type", "edge", "position_index", "combined_score",
                    "tail_probability", "is_outlier", "low_confidence"]


def _load_json(path: Path) -> Any:
    if not path.exists():
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return None


def _read_results_table(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    with open(path, "r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _cell(value: Any) -> str:
    """Compact display of a CSV value: 4 significant digits, empty for missing."""
    if value in (None, ""):
        return "—"
    try:
        f = float(value)
    except (TypeError, ValueError):
        return str(value)
    if f != 0.0 and (abs(f) < 1e-3 or abs(f) >= 1e5):
        return f"{f:.2e}"
    return f"{f:.4g}"


def _band(tail_probability: Any) -> str:
    try:
        p = float(tail_probability)
    except (TypeError, ValueError):
        return "—"
    for key, (lo, hi) in SUSPICION_BANDS.items():
        if lo < p <= hi or (lo == 0.0 and p <= hi):
            return BAND_LABELS[key]
    return "Low"


class PdfReportGenerator:
    """Lays out and writes the session report PDF from existing session outputs."""

    def __init__(self, session_manager: Optional[SessionManager] = None) -> None:
        self.session_manager = session_manager or SessionManager()

    def generate_report(self, session_id: str) -> str:
        """Assemble results/report.pdf and return its absolute path."""
        # Lazy import keeps the module importable without reportlab installed.
        from reportlab.lib import colors
        from reportlab.lib.pagesizes import landscape, letter
        from reportlab.lib.styles import getSampleStyleSheet
        from reportlab.lib.units import inch
        from reportlab.platypus import (
            Image,
            KeepTogether,
            PageBreak,
            Paragraph,
            SimpleDocTemplate,
            Spacer,
            Table,
            TableStyle,
        )

        session_dir = self.session_manager.get_session_dir(session_id)
        results_dir = session_dir / "results"
        annotated = results_dir / "annotated_overview.png"
        table_path = results_dir / "results_table.csv"
        if not annotated.exists() or not table_path.exists():
            raise FileNotFoundError("The session has no final outputs yet; run the pipeline to Step 23 first.")

        manifest_data = self.session_manager.get_manifest_manager(session_id).load()
        # Only the block stamped at run start describes the run; never substitute
        # the environment the report happens to be generated in.
        version_meta = manifest_data.get("version_metadata") or {}
        session_log = self.session_manager.load_session_log(session_id) or {}
        rows = _read_results_table(table_path)
        regions = _load_json(session_dir / "stages" / "candidate_regions.json") or []

        out_path = results_dir / "report.pdf"
        styles = getSampleStyleSheet()
        small = styles["Normal"].clone("small", fontSize=7, leading=8.5)
        doc = SimpleDocTemplate(str(out_path), pagesize=landscape(letter),
                                leftMargin=0.5 * inch, rightMargin=0.5 * inch,
                                topMargin=0.5 * inch, bottomMargin=0.5 * inch)
        frame_w = doc.width
        story: List[Any] = []

        def grid(data: List[List[Any]], widths: List[float], font: float = 7) -> Table:
            t = Table(data, colWidths=widths, repeatRows=1)
            t.setStyle(TableStyle([
                ("FONTSIZE", (0, 0), (-1, -1), font),
                ("GRID", (0, 0), (-1, -1), 0.25, colors.grey),
                ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ]))
            return t

        # 1. Title / session metadata.
        story.append(Paragraph("IPDS Reperforation Analysis Report", styles["Title"]))
        flagged = [r for r in rows if str(r.get("is_outlier")).lower() == "true"]
        for line in [
            f"Session ID: {session_id}",
            f"Specimen ID: {session_log.get('specimen_id', 'N/A')}",
            f"Capture timestamp: {session_log.get('capture_timestamp', 'N/A')}",
            f"Status: {manifest_data.get('status', 'unknown')}",
            f"Holes analysed: {len(rows)} (flagged for review: {len(flagged)}; candidate regions: {len(regions)})",
            "This report flags candidates for expert review. It is not a verdict on authenticity.",
        ]:
            story.append(Paragraph(line, styles["Normal"]))
        story.append(PageBreak())

        # 2. Annotated overview image, with 3. the legend directly beneath it.
        story.append(Paragraph("Annotated Overview", styles["Heading2"]))
        story.append(Image(str(annotated), width=frame_w, height=5.2 * inch, kind="proportional"))
        legend = [
            ["Marker", "Meaning"],
            ["Blue → red dot", "Hole deviation score relative to its own group's cutoff (red at or above the cutoff)"],
            ["Red ring", f"Outlier: a genuine hole would score this high with probability below {OUTLIER_TAIL_PROBABILITY:g}"],
            ["Yellow box", "Candidate region: a run of flagged holes along one edge"],
            ["Band Low / Medium / High", "Tail probability above 5% / between 5% and 0.1% / below 0.1%"],
            ["LC", "Low confidence: pooled or small comparison group"],
        ]
        story.append(KeepTogether([Paragraph("Legend", styles["Heading3"]),
                                   grid(legend, [1.8 * inch, frame_w - 1.8 * inch], 8)]))
        story.append(PageBreak())

        # 4. Full per-hole results table (two parts so every column fits the page).
        story.append(Paragraph("Per-Hole Results", styles["Heading2"]))
        id_header = IDENTITY_COLUMNS + ["band"]
        id_rows = [id_header] + [[_cell(r.get(c)) for c in IDENTITY_COLUMNS] + [_band(r.get("tail_probability"))]
                                 for r in rows]
        story.append(grid(id_rows, [frame_w / len(id_header)] * len(id_header)))
        story.append(Spacer(1, 0.2 * inch))
        story.append(Paragraph("Descriptor values (µm where applicable)", styles["Heading3"]))
        desc_cols = [c for c in (rows[0].keys() if rows else [])
                     if c not in IDENTITY_COLUMNS + ["threshold_used"]]
        desc_header = ["hole_id"] + desc_cols
        desc_rows = [[Paragraph(h, small) for h in desc_header]] + [
            [r.get("hole_id", "")] + [_cell(r.get(c)) for c in desc_cols] for r in rows]
        story.append(grid(desc_rows, [frame_w / len(desc_header)] * len(desc_header), 6))
        story.append(PageBreak())

        # 5. Region / summary findings.
        story.append(Paragraph("Region and Summary Findings", styles["Heading2"]))
        by_id = {r.get("hole_id"): r for r in rows}
        if regions:
            region_rows = [["Region", "Edge", "Positions", "Holes", "Highest band"]]
            order = {"High": 0, "Medium": 1, "Low": 2, "—": 3}
            for reg in regions:
                bands = [_band(by_id.get(h, {}).get("tail_probability")) for h in reg.get("hole_ids", [])]
                region_rows.append([
                    reg.get("region_id", ""), str(reg.get("edge", "")).title(),
                    f"{reg.get('start_position')}–{reg.get('end_position')}",
                    Paragraph(", ".join(reg.get("hole_ids", [])), small),
                    min(bands, key=lambda b: order.get(b, 3)) if bands else "—",
                ])
            story.append(grid(region_rows, [1.4 * inch, 0.9 * inch, 1.0 * inch, frame_w - 4.6 * inch, 1.3 * inch], 8))
        else:
            story.append(Paragraph("No candidate regions were found.", styles["Normal"]))
        if flagged:
            standalone = [r["hole_id"] for r in flagged
                          if not any(r["hole_id"] in reg.get("hole_ids", []) for reg in regions)]
            if standalone:
                story.append(Paragraph("Flagged holes outside any region: " + ", ".join(standalone), styles["Normal"]))
        story.append(Spacer(1, 0.25 * inch))

        # 6. Feature 28 version / method metadata block.
        story.append(Paragraph("Method &amp; Version Metadata", styles["Heading2"]))
        if not version_meta:
            story.append(Paragraph("Version metadata was not recorded for this session.", styles["Normal"]))
        story.append(Paragraph(f"App version: {version_meta.get('app_version', 'N/A')}", styles["Normal"]))
        story.append(Paragraph(f"Python: {version_meta.get('python_version', 'N/A')}", styles["Normal"]))
        story.append(Paragraph(f"Captured at: {version_meta.get('captured_at', 'N/A')}", styles["Normal"]))
        libs = version_meta.get("library_versions", {})
        if libs:
            story.append(Paragraph("Libraries: " + ", ".join(f"{k} {v}" for k, v in libs.items()), styles["Normal"]))
        constants = version_meta.get("config_snapshot") or {}
        if constants:
            story.append(Spacer(1, 0.1 * inch))
            story.append(Paragraph("Constants used in this run", styles["Heading3"]))
            const_rows = [["Constant", "Value"]] + [[k, Paragraph(str(v), small)] for k, v in constants.items()]
            story.append(grid(const_rows, [3.0 * inch, frame_w - 3.0 * inch]))

        doc.build(story)
        logger.info("Generated PDF report for session %s at %s", session_id, out_path)
        return str(out_path)
