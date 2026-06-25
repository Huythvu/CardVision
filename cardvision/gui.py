"""Phase 1 desktop UI (PySide6): live preview, Start/Stop, drag-to-select region.

Kept fully isolated from the core pipeline — it only *calls* the existing
capture / detect / classify / viz modules, so the CLI keeps working unchanged.

Launch with:  python main.py ui    (or: python -m cardvision.gui)
"""

from __future__ import annotations

import sys
import time

import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets
from PySide6.QtCore import Qt, Signal, Slot

from . import capture, config, formatter, viz

# Backends the worker can run. "detect" needs nothing; "template" needs
# calibrated templates; "ml" needs a YOLO model at config.ML_MODEL_PATH.
BACKENDS = ["detect", "template", "ml"]


class RegionSelector(QtWidgets.QWidget):
    """Translucent full-screen overlay: drag a rectangle to pick the region."""

    regionSelected = Signal(dict)

    def __init__(self):
        super().__init__()
        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.WindowStaysOnTopHint
            | Qt.WindowType.Tool
        )
        self.setCursor(Qt.CursorShape.CrossCursor)
        screen = QtGui.QGuiApplication.primaryScreen().geometry()
        self._screen_origin = screen.topLeft()
        self.setGeometry(screen)
        self._origin = None
        self._rect = QtCore.QRect()

    def paintEvent(self, _event):
        p = QtGui.QPainter(self)
        p.fillRect(self.rect(), QtGui.QColor(0, 0, 0, 90))  # dim everything
        if not self._rect.isNull():
            p.fillRect(self._rect, QtGui.QColor(0, 0, 0, 0))  # punch-through-ish
            pen = QtGui.QPen(QtGui.QColor(0, 220, 0), 2)
            p.setPen(pen)
            p.drawRect(self._rect)
            t = f"{self._rect.width()} x {self._rect.height()}"
            p.drawText(self._rect.topLeft() + QtCore.QPoint(4, -6), t)

    def mousePressEvent(self, e):
        self._origin = e.position().toPoint()
        self._rect = QtCore.QRect(self._origin, self._origin)
        self.update()

    def mouseMoveEvent(self, e):
        if self._origin is not None:
            self._rect = QtCore.QRect(self._origin, e.position().toPoint()).normalized()
            self.update()

    def mouseReleaseEvent(self, _e):
        if self._rect.width() > 5 and self._rect.height() > 5:
            self.regionSelected.emit({
                "left": self._rect.left() + self._screen_origin.x(),
                "top": self._rect.top() + self._screen_origin.y(),
                "width": self._rect.width(),
                "height": self._rect.height(),
            })
        self.close()

    def keyPressEvent(self, e):
        if e.key() == Qt.Key.Key_Escape:
            self.close()


class Worker(QtCore.QObject):
    """Background capture+inference loop; emits annotated frames and labels."""

    frameReady = Signal(object)   # annotated BGR ndarray
    cardsReady = Signal(str)
    statusReady = Signal(str)

    def __init__(self):
        super().__init__()
        self._running = False
        self.region = dict(config.SCREEN_REGION)
        self.backend = "detect"
        self.interval = 0.3
        self._tpl = None
        self._ml = None

    @Slot(dict)
    def setRegion(self, region):
        self.region = dict(region)

    @Slot(str)
    def setBackend(self, backend):
        self.backend = backend

    @Slot(float)
    def setInterval(self, seconds):
        self.interval = max(0.05, seconds)

    def stop(self):
        self._running = False

    @Slot()
    def run(self):
        from . import detect

        self._running = True
        while self._running:
            t0 = time.time()
            try:
                frame = capture.grab_screen(self.region)
            except Exception as exc:  # noqa: BLE001 - surface any capture failure
                self.statusReady.emit(f"capture error: {exc}")
                break
            try:
                annotated, labels = self._process(frame, detect)
            except Exception as exc:  # noqa: BLE001 - keep the loop alive on errors
                annotated, labels = frame, f"error: {exc}"
            self.frameReady.emit(annotated)
            self.cardsReady.emit(labels)
            dt = time.time() - t0
            fps = 1.0 / dt if dt > 0 else 0.0
            self.statusReady.emit(
                f"{self.region['width']}x{self.region['height']} @ {fps:4.1f} fps  [{self.backend}]"
            )
            rest = self.interval - (time.time() - t0)
            if rest > 0:
                time.sleep(rest)
        self.statusReady.emit("stopped")

    def _process(self, frame, detect):
        if self.backend == "ml":
            if self._ml is None:
                from . import mldetect
                self._ml = mldetect.MLCardDetector()
            dets = self._ml.detect(frame)
            labels = "  ".join(formatter.format_rank_suit(d.rank, d.suit, True) for d in dets)
            return viz.annotate_ml(frame, dets), labels or "(no cards)"

        cards = detect.find_cards(frame)
        if self.backend == "template":
            from . import templates
            from .classify import classify_card
            if self._tpl is None:
                self._tpl = (templates.load_rank_templates(), templates.load_suit_templates())
            rr, sr = self._tpl
            results = [classify_card(c.warped, rr, sr) for c in cards]
            labels = "  ".join(formatter.format_card(r, True) for r in results)
            return viz.annotate_detections(frame, cards, results), labels or "(no cards)"

        # detect-only: outline cards, no labels (needs no templates/model)
        return viz.annotate_detections(frame, cards, None), f"{len(cards)} card(s) detected"


class MainWindow(QtWidgets.QMainWindow):
    _setRegion = Signal(dict)
    _setBackend = Signal(str)
    _setInterval = Signal(float)

    def __init__(self):
        super().__init__()
        self.setWindowTitle("CardVision")
        self.resize(900, 640)

        # --- worker thread ---
        self.thread = QtCore.QThread(self)
        self.worker = Worker()
        self.worker.moveToThread(self.thread)
        self.thread.started.connect(self.worker.run)
        self.worker.frameReady.connect(self.onFrame)
        self.worker.cardsReady.connect(self.onCards)
        self.worker.statusReady.connect(self.onStatus)
        self._setRegion.connect(self.worker.setRegion)
        self._setBackend.connect(self.worker.setBackend)
        self._setInterval.connect(self.worker.setInterval)
        self._running = False

        # --- controls ---
        self.startBtn = QtWidgets.QPushButton("Start")
        self.startBtn.setCheckable(True)
        self.startBtn.clicked.connect(self.toggleRun)

        self.regionBtn = QtWidgets.QPushButton("Select Region")
        self.regionBtn.clicked.connect(self.pickRegion)

        self.backendBox = QtWidgets.QComboBox()
        self.backendBox.addItems(BACKENDS)
        self.backendBox.currentTextChanged.connect(self._setBackend.emit)

        self.intervalSlider = QtWidgets.QSlider(Qt.Orientation.Horizontal)
        self.intervalSlider.setRange(100, 1000)  # ms
        self.intervalSlider.setValue(300)
        self.intervalSlider.valueChanged.connect(
            lambda ms: self._setInterval.emit(ms / 1000.0)
        )

        controls = QtWidgets.QHBoxLayout()
        controls.addWidget(self.startBtn)
        controls.addWidget(self.regionBtn)
        controls.addWidget(QtWidgets.QLabel("Backend:"))
        controls.addWidget(self.backendBox)
        controls.addWidget(QtWidgets.QLabel("Interval:"))
        controls.addWidget(self.intervalSlider)
        controls.addStretch(1)

        # --- views ---
        self.view = QtWidgets.QLabel("Press Start (set a region first).")
        self.view.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.view.setMinimumSize(640, 400)
        self.view.setStyleSheet("background:#111; color:#888;")

        self.cardsLabel = QtWidgets.QLabel("—")
        self.cardsLabel.setStyleSheet("font-size:20px; padding:6px;")

        central = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(central)
        layout.addLayout(controls)
        layout.addWidget(self.view, 1)
        layout.addWidget(self.cardsLabel)
        self.setCentralWidget(central)

        self.statusBar().showMessage(
            f"region {config.SCREEN_REGION['width']}x{config.SCREEN_REGION['height']}"
        )

    # --- slots ---
    @Slot()
    def toggleRun(self):
        if self.startBtn.isChecked():
            self._running = True
            self.startBtn.setText("Stop")
            if not self.thread.isRunning():
                self.thread.start()
        else:
            self.startBtn.setText("Start")
            self._running = False
            self.worker.stop()
            self.thread.quit()
            self.thread.wait(2000)

    @Slot()
    def pickRegion(self):
        self._selector = RegionSelector()
        self._selector.regionSelected.connect(self.onRegion)
        self._selector.showFullScreen()

    @Slot(dict)
    def onRegion(self, region):
        self._setRegion.emit(region)
        self.statusBar().showMessage(
            f"region {region['width']}x{region['height']} @ ({region['left']},{region['top']})"
        )

    @Slot(object)
    def onFrame(self, img: np.ndarray):
        img = np.ascontiguousarray(img)
        h, w = img.shape[:2]
        qimg = QtGui.QImage(img.data, w, h, 3 * w, QtGui.QImage.Format.Format_BGR888).copy()
        pix = QtGui.QPixmap.fromImage(qimg).scaled(
            self.view.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self.view.setPixmap(pix)

    @Slot(str)
    def onCards(self, text):
        self.cardsLabel.setText(text)

    @Slot(str)
    def onStatus(self, text):
        self.statusBar().showMessage(text)

    def closeEvent(self, event):
        self.worker.stop()
        self.thread.quit()
        self.thread.wait(2000)
        super().closeEvent(event)


def launch() -> int:
    app = QtWidgets.QApplication(sys.argv)
    win = MainWindow()
    win.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(launch())
