"""Validation mode: run a labelled specimen set and read its accuracy report (Feature 26)."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from PySide6.QtWidgets import (
    QFormLayout,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QProgressBar,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from frontend.client import Client

BG, FG, GRID = "#1E1E1E", "#E0E0E0", "#3A3A3A"


class ValidationTab(QWidget):
    """Left: specimen set and run controls. Right: counts card, ROC plot, per-specimen and repeatability tables."""

    def __init__(self, client: Optional[Client] = None) -> None:
        super().__init__()
        self.client = client or Client()
        self.specimen_set_id = QLineEdit()
        self.specimen_set_id.setPlaceholderText("Folder under backend/validation/labeled_specimens/")
        self.run_id = QLineEdit()
        self.run_id.setPlaceholderText("Validation run identifier")
        self.run_button = QPushButton("Run validation")
        self.run_button.clicked.connect(self.run_validation)
        report_button = QPushButton("Load validation report")
        report_button.clicked.connect(self.load_report)
        self.busy = QProgressBar()
        self.busy.setRange(0, 0)
        self.busy.hide()
        self.message = QLabel("No validation run loaded.")
        self.message.setWordWrap(True)

        left = QWidget()
        form = QFormLayout(left)
        form.addRow("Specimen set", self.specimen_set_id)
        form.addRow(self.run_button)
        form.addRow("Run", self.run_id)
        form.addRow(report_button)
        form.addRow(self.busy)
        form.addRow(self.message)

        # Confusion-count card
        card = QFrame()
        card.setObjectName("card")
        grid = QGridLayout(card)
        self.counts: Dict[str, QLabel] = {}
        for col, key in enumerate(("tp", "fp", "fn", "tn")):
            value = QLabel("—")
            value.setStyleSheet("font-size: 26px; font-weight: 600;")
            self.counts[key] = value
            grid.addWidget(value, 0, col)
            grid.addWidget(QLabel(key.upper()), 1, col)

        self.figure = Figure(figsize=(5, 3), facecolor=BG)
        self.canvas = FigureCanvasQTAgg(self.figure)

        self.specimens = QTableWidget(0, 4)
        self.specimens.setHorizontalHeaderLabels(("Specimen", "Predicted region", "Overlap with truth", "Error"))
        self.specimens.horizontalHeader().setStretchLastSection(True)
        self.repeatability = QTableWidget(0, 3)
        self.repeatability.setHorizontalHeaderLabels(("Specimen", "Hole", "Score delta"))
        self.repeatability.horizontalHeader().setStretchLastSection(True)

        right = QWidget()
        right_layout = QVBoxLayout(right)
        right_layout.addWidget(card)
        right_layout.addWidget(self.canvas)
        right_layout.addWidget(QLabel("Per-specimen results"))
        right_layout.addWidget(self.specimens)
        right_layout.addWidget(QLabel("Repeatability (two independent captures)"))
        right_layout.addWidget(self.repeatability)

        layout = QHBoxLayout(self)
        layout.addWidget(left, 1)
        layout.addWidget(right, 2)

    # --------------------------------------------------------------- run
    def run_validation(self) -> None:
        """Start validation in the background; render the report when it finishes."""
        set_id = self.specimen_set_id.text().strip()
        if not set_id:
            self.message.setText("Enter the specimen set to validate.")
            return
        self.run_button.setEnabled(False)
        self.busy.show()
        self.message.setText("Validation running in the background…")
        self.client.call_async("run_validation", set_id, on_done=self._on_validation_done)

    def _on_validation_done(self, response: Dict[str, Any]) -> None:
        self.run_button.setEnabled(True)
        self.busy.hide()
        report, error = Client.unwrap(response, "Validation failed.")
        if error:
            self.message.setText(error)
            return
        if isinstance(report, dict) and report.get("run_id"):
            self.run_id.setText(str(report["run_id"]))
        self.render_report(report)

    def load_report(self) -> None:
        payload, error = Client.unwrap(self.client.call("get_validation_report", self.run_id.text().strip()),
                                       "Report not found.")
        if error:
            self.message.setText(error)
            return
        self.render_report(payload.get("report", payload))

    # ------------------------------------------------------------ render
    def render_report(self, report: Dict[str, Any]) -> None:
        results: List[Dict[str, Any]] = report.get("results", []) if isinstance(report, dict) else []
        totals = {k: 0 for k in ("tp", "fp", "fn", "tn")}
        for r in results:
            for k in totals:
                totals[k] += int((r.get("confusion_counts") or {}).get(k, 0))
        for k, label in self.counts.items():
            label.setText(str(totals[k]))

        self.specimens.setRowCount(len(results))
        reps: List[tuple] = []
        for row, r in enumerate(results):
            pred = r.get("predicted_region") or {}
            region = (f"{pred.get('edge')} {pred.get('start_position')}-{pred.get('end_position')}" if pred else "none")
            overlap = pred.get("overlap_with_ground_truth") if pred else None
            for col, text in enumerate((r.get("specimen_id", ""), region,
                                        "—" if overlap is None else f"{overlap:.2f}", r.get("error", ""))):
                self.specimens.setItem(row, col, QTableWidgetItem(str(text)))
            for d in r.get("repeatability_deltas") or []:
                reps.append((r.get("specimen_id", ""), d.get("hole_id"), d.get("score_delta")))
        self.repeatability.setRowCount(len(reps))
        for row, (spec, hole, delta) in enumerate(reps):
            for col, text in enumerate((spec, hole, f"{float(delta):.3f}")):
                self.repeatability.setItem(row, col, QTableWidgetItem(str(text)))

        self._plot_roc(results)
        count = report.get("specimen_count", len(results)) if isinstance(report, dict) else 0
        self.message.setText(f"Run {report.get('run_id', '')}: {count} specimen(s)." if count
                             else "No labelled specimens were found for this set.")

    def _plot_roc(self, results: List[Dict[str, Any]]) -> None:
        """ROC-style plot: threshold on x, mean TPR and FPR across specimens on y."""
        by_threshold: Dict[float, List[tuple]] = {}
        for r in results:
            for p in r.get("roc_points") or []:
                by_threshold.setdefault(float(p["threshold"]), []).append((p["tpr"], p["fpr"]))
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        ax.set_facecolor(BG)
        ax.tick_params(colors=FG, labelsize=8)
        for spine in ax.spines.values():
            spine.set_color(GRID)
        if by_threshold:
            xs = sorted(by_threshold)
            tpr = [sum(t for t, _ in by_threshold[x]) / len(by_threshold[x]) for x in xs]
            fpr = [sum(f for _, f in by_threshold[x]) / len(by_threshold[x]) for x in xs]
            ax.plot(xs, tpr, color="#4CAF50", label="TPR")
            ax.plot(xs, fpr, color="#E53935", label="FPR")
            ax.legend(facecolor=BG, edgecolor=GRID, labelcolor=FG, fontsize=8)
        ax.set_xscale("log")
        ax.set_xlabel("Tail-probability cutoff (flag when below)", color=FG, fontsize=8)
        ax.set_ylabel("Rate", color=FG, fontsize=8)
        ax.set_ylim(-0.02, 1.02)
        self.figure.tight_layout()
        self.canvas.draw_idle()
