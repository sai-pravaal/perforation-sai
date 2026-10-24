"""Image comparison views: side-by-side with synchronized zoom/pan, or alpha overlay (Features 8-9)."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QImage, QPainter, QPixmap, QTransform, qRgb
from PySide6.QtWidgets import (
    QButtonGroup,
    QGraphicsPixmapItem,
    QGraphicsScene,
    QGraphicsView,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSlider,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

# The pipeline stores linear-light 16-bit PNGs; screens expect gamma-encoded
# values, so 16-bit images get this display gamma (presentation only).
DISPLAY_GAMMA = 2.2
_GAMMA_TABLE = [qRgb(v, v, v) for v in (round(255 * ((i / 255.0) ** (1 / DISPLAY_GAMMA))) for i in range(256))]


def load_display_pixmap(path: Optional[str]) -> QPixmap:
    """Load an image the backend pointed to, applying display gamma to linear 16-bit files."""
    if not path:
        return QPixmap()
    image = QImage(path)
    if image.isNull():
        return QPixmap()
    if image.format() in (QImage.Format.Format_Grayscale16, QImage.Format.Format_RGBA64,
                          QImage.Format.Format_RGBX64):
        gray = image.convertToFormat(QImage.Format.Format_Grayscale8)
        data = bytes(gray.constBits())
        indexed = QImage(data, gray.width(), gray.height(), gray.bytesPerLine(), QImage.Format.Format_Indexed8)
        indexed.setColorTable(_GAMMA_TABLE)
        image = indexed.copy()  # detach from `data` before it goes out of scope
    return QPixmap.fromImage(image)


class SyncedGraphicsView(QGraphicsView):
    """QGraphicsView that reports wheel zoom and drag pan so a partner view can mirror them."""

    view_changed = Signal()

    def __init__(self) -> None:
        super().__init__()
        self.setScene(QGraphicsScene(self))
        self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        self.horizontalScrollBar().valueChanged.connect(lambda _v: self.view_changed.emit())
        self.verticalScrollBar().valueChanged.connect(lambda _v: self.view_changed.emit())

    def wheelEvent(self, event) -> None:  # noqa: N802 (Qt override)
        factor = 1.25 if event.angleDelta().y() > 0 else 0.8
        self.scale(factor, factor)
        self.view_changed.emit()

    def set_pixmap(self, pixmap: QPixmap) -> None:
        self.scene().clear()
        if not pixmap.isNull():
            self.scene().addPixmap(pixmap)
            self.setSceneRect(pixmap.rect().toRectF())
            self.fitInView(self.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)


class GraphicsViewSync(QWidget):
    """Feature 8 comparison panel with Feature 9 linked zoom/pan.

    Side-by-Side shows two views whose transform and scroll position mirror
    each other (re-entrancy guarded, toggleable). Overlay stacks image B on
    image A in one view with an alpha slider (pure Qt opacity, no backend call).
    """

    def __init__(self) -> None:
        super().__init__()
        self.left_view = SyncedGraphicsView()
        self.right_view = SyncedGraphicsView()
        self._overlay_scene = QGraphicsScene(self)  # parented so Python does not collect it
        self.overlay_view = QGraphicsView(self._overlay_scene)
        self.overlay_view.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
        self._overlay_item: Optional[QGraphicsPixmapItem] = None
        self._syncing = False
        self._sync_on = True
        self._pixmaps: List[QPixmap] = [QPixmap(), QPixmap()]
        self.left_label = QLabel("Image A: none")
        self.right_label = QLabel("Image B: none")

        side = QWidget()
        side_layout = QHBoxLayout(side)
        for label, view in ((self.left_label, self.left_view), (self.right_label, self.right_view)):
            col = QVBoxLayout()
            col.addWidget(label)
            col.addWidget(view)
            side_layout.addLayout(col)

        overlay = QWidget()
        overlay_layout = QVBoxLayout(overlay)
        overlay_layout.addWidget(self.overlay_view)
        self.alpha = QSlider(Qt.Orientation.Horizontal)
        self.alpha.setRange(0, 100)
        self.alpha.setValue(50)
        self.alpha.valueChanged.connect(self._apply_alpha)
        overlay_layout.addWidget(QLabel("Image B opacity"))
        overlay_layout.addWidget(self.alpha)

        self.stack = QStackedWidget()
        self.stack.addWidget(side)
        self.stack.addWidget(overlay)

        self.side_button = QPushButton("Side-by-Side")
        self.overlay_button = QPushButton("Overlay")
        self.sync_button = QPushButton("Sync: ON")
        for b in (self.side_button, self.overlay_button, self.sync_button):
            b.setCheckable(True)
        self.side_button.setChecked(True)
        self.sync_button.setChecked(True)
        modes = QButtonGroup(self)
        modes.setExclusive(True)
        modes.addButton(self.side_button)
        modes.addButton(self.overlay_button)
        self.side_button.clicked.connect(lambda: self.set_mode("side"))
        self.overlay_button.clicked.connect(lambda: self.set_mode("overlay"))
        self.sync_button.toggled.connect(self.set_sync)

        toolbar = QHBoxLayout()
        toolbar.addWidget(self.side_button)
        toolbar.addWidget(self.overlay_button)
        toolbar.addStretch(1)
        toolbar.addWidget(self.sync_button)

        layout = QVBoxLayout(self)
        layout.addLayout(toolbar)
        layout.addWidget(self.stack)

        self.left_view.view_changed.connect(lambda: self._mirror(self.left_view, self.right_view))
        self.right_view.view_changed.connect(lambda: self._mirror(self.right_view, self.left_view))

    # --------------------------------------------------------------- data
    def set_pair(self, pair: List[Dict[str, Any]]) -> None:
        """Render the two images resolved by get_comparison_pair."""
        records = list(pair)[:2] + [{}] * (2 - len(list(pair)[:2]))
        for i, (record, view, label) in enumerate(zip(records, (self.left_view, self.right_view),
                                                     (self.left_label, self.right_label))):
            image = record.get("image", record) if isinstance(record, dict) else {}
            pixmap = load_display_pixmap(image.get("abs_path"))
            self._pixmaps[i] = pixmap
            view.set_pixmap(pixmap)
            name = "A" if i == 0 else "B"
            label.setText(f"Image {name}: {image.get('stage', '—')} {image.get('hole_id') or ''} "
                          f"{'' if not pixmap.isNull() else '(not available)'}".strip())
        self._build_overlay()
        self._mirror(self.left_view, self.right_view)

    def set_slot(self, slot: int, image: Dict[str, Any]) -> None:
        """Load one image into slot 0 (A) or 1 (B) without waiting for the other."""
        view = self.left_view if slot == 0 else self.right_view
        pixmap = load_display_pixmap(image.get("abs_path"))
        self._pixmaps[slot] = pixmap
        view.set_pixmap(pixmap)
        (self.left_label if slot == 0 else self.right_label).setText(
            f"Image {'A' if slot == 0 else 'B'}: {image.get('stage', '—')} {image.get('hole_id') or ''}".strip()
        )
        self._build_overlay()

    # -------------------------------------------------------------- modes
    def set_mode(self, mode: str) -> None:
        self.stack.setCurrentIndex(1 if mode == "overlay" else 0)
        self.overlay_button.setChecked(mode == "overlay")
        self.side_button.setChecked(mode != "overlay")

    def set_sync(self, on: bool) -> None:
        self._sync_on = bool(on)
        self.sync_button.setText("Sync: ON" if on else "Sync: OFF")
        if on:
            self._mirror(self.left_view, self.right_view)

    def set_zoom(self, factor: float) -> None:
        """Apply the same absolute zoom factor to both side-by-side views."""
        if factor <= 0:
            return
        self.left_view.setTransform(QTransform.fromScale(factor, factor))
        self._mirror(self.left_view, self.right_view)

    def _mirror(self, source: QGraphicsView, target: QGraphicsView) -> None:
        if not self._sync_on or self._syncing:
            return
        self._syncing = True
        try:
            target.setTransform(source.transform())
            target.horizontalScrollBar().setValue(source.horizontalScrollBar().value())
            target.verticalScrollBar().setValue(source.verticalScrollBar().value())
        finally:
            self._syncing = False

    def _build_overlay(self) -> None:
        scene = self.overlay_view.scene()
        scene.clear()
        self._overlay_item = None
        base, top = self._pixmaps
        if not base.isNull():
            scene.addPixmap(base)
        if not top.isNull():
            self._overlay_item = scene.addPixmap(top)
            self._apply_alpha(self.alpha.value())
        rect = scene.itemsBoundingRect()
        if not rect.isEmpty():
            self.overlay_view.fitInView(rect, Qt.AspectRatioMode.KeepAspectRatio)

    def _apply_alpha(self, value: int) -> None:
        if self._overlay_item is not None:
            self._overlay_item.setOpacity(value / 100.0)
