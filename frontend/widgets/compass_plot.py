"""Shadow-direction compass and edge consistency plots (Features 14 and 16)."""

from __future__ import annotations

import math
from typing import Any, Dict, Iterable, List

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from PySide6.QtWidgets import QVBoxLayout, QWidget

BG = "#1E1E1E"
FG = "#E0E0E0"
GRID = "#3A3A3A"
MUTED = "#9E9E9E"
RED = "#E53935"
BAND = "#FFB300"


class CompassPlot(QWidget):
    """Matplotlib canvas embedded with FigureCanvasQTAgg, dark-themed to match style.qss.

    update_vectors draws Feature 14's polar arrows; update_edge_summary draws
    Feature 16's polar + strip pair for one edge. The canvas is reused and
    redrawn with draw_idle rather than recreated.
    """

    def __init__(self) -> None:
        super().__init__()
        self.figure = Figure(figsize=(6, 3.6), facecolor=BG)
        self.canvas = FigureCanvasQTAgg(self.figure)
        self._annotation = None
        self._artists: List[Any] = []
        self.canvas.mpl_connect("pick_event", self._on_pick)
        layout = QVBoxLayout(self)
        layout.addWidget(self.canvas)

    # ------------------------------------------------------------ styling
    @staticmethod
    def _style_polar(ax) -> None:
        ax.set_facecolor(BG)
        ax.set_theta_zero_location("E")
        ax.set_theta_direction(1)  # counter-clockwise, matching the azimuth convention
        ax.tick_params(colors=FG, labelsize=8)
        ax.grid(color=GRID)
        for spine in ax.spines.values():
            spine.set_color(GRID)

    def _polar_arrows(self, ax, records: Iterable[Dict[str, Any]]) -> None:
        self._artists = []
        peak = 0.0
        for rec in records:
            shadow = rec.get("shadow") or {}
            strength = shadow.get("strength", rec.get("shadow_strength"))
            direction = shadow.get("direction_deg", rec.get("shadow_direction_deg"))
            if strength is None or direction is None:
                continue
            theta = math.radians(float(direction))
            colour = RED if rec.get("is_outlier") else MUTED
            line = ax.annotate("", xy=(theta, float(strength)), xytext=(0, 0),
                               arrowprops=dict(arrowstyle="->", color=colour, lw=1.6))
            marker, = ax.plot([theta], [float(strength)], "o", color=colour, ms=4, picker=6)
            marker._ipds = rec  # type: ignore[attr-defined]
            self._artists.append(line)
            peak = max(peak, float(strength))
        ax.set_rmax(peak * 1.15 if peak > 0 else 1.0)

    # --------------------------------------------------------- Feature 14
    def update_vectors(self, vectors: Iterable[Dict[str, Any]]) -> None:
        self.figure.clear()
        ax = self.figure.add_subplot(111, projection="polar")
        self._style_polar(ax)
        self._polar_arrows(ax, list(vectors))
        ax.set_title("Shadow direction & strength (red = outlier)", color=FG, fontsize=9)
        self._annotation = None
        self.canvas.draw_idle()

    # --------------------------------------------------------- Feature 16
    def update_edge_summary(self, payload: Dict[str, Any]) -> None:
        holes = sorted(payload.get("holes") or [], key=lambda h: h.get("position_index") or 0)
        self.figure.clear()
        polar = self.figure.add_subplot(121, projection="polar")
        self._style_polar(polar)
        self._polar_arrows(polar, holes)
        polar.set_title("Shadow vectors", color=FG, fontsize=9)

        strip = self.figure.add_subplot(122)
        strip.set_facecolor(BG)
        strip.tick_params(colors=FG, labelsize=8)
        for spine in strip.spines.values():
            spine.set_color(GRID)
        xs = [h.get("position_index") for h in holes if h.get("glcm_contrast") is not None]
        ys = [h.get("glcm_contrast") for h in holes if h.get("glcm_contrast") is not None]
        if xs:
            strip.plot(xs, ys, color=MUTED, lw=0.8)
            colours = [RED if h.get("is_outlier") else MUTED for h in holes if h.get("glcm_contrast") is not None]
            strip.scatter(xs, ys, c=colours, s=18, zorder=3)
        for region in payload.get("regions") or []:
            strip.axvspan(float(region.get("start_position", 0)) - 0.5, float(region.get("end_position", 0)) + 0.5,
                          color=BAND, alpha=0.18, lw=0)
        strip.set_xlabel("Position along edge", color=FG, fontsize=8)
        strip.set_ylabel("Roughness (GLCM contrast)", color=FG, fontsize=8)
        strip.set_title(f"{str(payload.get('edge', '')).title()} edge", color=FG, fontsize=9)
        self.figure.tight_layout()
        self._annotation = None
        self.canvas.draw_idle()

    # ------------------------------------------------------------- picking
    def _on_pick(self, event) -> None:
        rec = getattr(event.artist, "_ipds", None)
        if rec is None:
            return
        shadow = rec.get("shadow") or {}
        strength = shadow.get("strength", rec.get("shadow_strength"))
        direction = shadow.get("direction_deg", rec.get("shadow_direction_deg"))
        ax = event.artist.axes
        if self._annotation is not None:
            self._annotation.remove()
        x, y = event.artist.get_data()
        self._annotation = ax.annotate(
            f"{rec.get('hole_id')}\n{float(strength):.3g} @ {float(direction):.0f}°",
            xy=(x[0], y[0]), color=FG, fontsize=8,
            bbox=dict(boxstyle="round", fc="#262626", ec="#2E7D6B"),
        )
        self.canvas.draw_idle()
