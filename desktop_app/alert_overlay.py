"""
Плавающая подсказка у края экрана: точка в покое, разворот в карточку при алерте.

Работает поверх других окон через Qt — не зависит от Toast Windows и трея.
"""

from __future__ import annotations

from PySide6.QtCore import QEasingCurve, QPropertyAnimation, QRect, Qt, QTimer
from PySide6.QtGui import QColor, QMouseEvent, QPainter, QPainterPath, QPen
from PySide6.QtWidgets import QApplication, QLabel, QVBoxLayout, QWidget


class _DotHandle(QWidget):
    """Круглая ручка в свёрнутом состоянии."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setFixedSize(22, 22)
        self._hover = False
        self.setToolTip("Emotion Checker — здесь появляются уведомления")

    def enterEvent(self, event) -> None:  # noqa: ANN001
        self._hover = True
        self.update()
        super().enterEvent(event)

    def leaveEvent(self, event) -> None:  # noqa: ANN001
        self._hover = False
        self.update()
        super().leaveEvent(event)

    def paintEvent(self, event) -> None:  # noqa: ANN001
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        r = self.rect().adjusted(2, 2, -2, -2)
        path = QPainterPath()
        path.addEllipse(r)
        fill = QColor(40, 120, 72, 220 if self._hover else 160)
        p.fillPath(path, fill)
        pen = QPen(QColor(255, 255, 255, 90))
        pen.setWidth(1)
        p.setPen(pen)
        p.drawPath(path)


class AlertOverlay(QWidget):
    """
    Окно без рамки у нижнего края экрана: точка → разворот с текстом алерта.
    """

    _EXPAND_W = 380
    _MARGIN = 12
    _COLLAPSE_MS_WARNING = 14_000
    _COLLAPSE_MS_CRITICAL = 22_000

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowFlags(
            Qt.WindowType.Tool
            | Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.WindowStaysOnTopHint
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
        self.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating, True)

        self._expanded = False
        self._collapse_timer = QTimer(self)
        self._collapse_timer.setSingleShot(True)
        self._collapse_timer.timeout.connect(self.collapse)

        root = QWidget(self)
        root.setObjectName("card")
        root.setStyleSheet(
            """
            #card {
                background-color: rgba(28, 32, 36, 235);
                border-radius: 12px;
                border: 1px solid rgba(255, 255, 255, 28);
            }
            QLabel#title {
                color: #f0f4f8;
                font-size: 14px;
                font-weight: 600;
            }
            QLabel#body {
                color: #c8d0da;
                font-size: 13px;
            }
            """
        )

        self._title_lbl = QLabel()
        self._title_lbl.setObjectName("title")
        self._title_lbl.setWordWrap(True)
        self._body_lbl = QLabel()
        self._body_lbl.setObjectName("body")
        self._body_lbl.setWordWrap(True)

        lay = QVBoxLayout(root)
        lay.setContentsMargins(16, 14, 16, 14)
        lay.setSpacing(8)
        lay.addWidget(self._title_lbl)
        lay.addWidget(self._body_lbl)

        self._card = root
        self._dot = _DotHandle(self)

        self._card.hide()
        self._dot.show()

        self._geom_anim: QPropertyAnimation | None = None
        self._place_collapsed()

    def showEvent(self, event) -> None:  # noqa: ANN001
        super().showEvent(event)
        if not self._expanded:
            self._place_collapsed()

    def resizeEvent(self, event) -> None:  # noqa: ANN001
        super().resizeEvent(event)
        if self._expanded and self._card.isVisible():
            self._card.setGeometry(self.rect())
        elif not self._expanded:
            self._dot.setGeometry(0, 0, self.width(), self.height())

    def mousePressEvent(self, event: QMouseEvent) -> None:
        if event.button() == Qt.MouseButton.LeftButton and self._expanded:
            self.collapse()
        super().mousePressEvent(event)

    def _stop_geometry_anim(self) -> None:
        if self._geom_anim is None:
            return
        self._geom_anim.stop()
        self._geom_anim.deleteLater()
        self._geom_anim = None

    def _place_collapsed(self) -> None:
        screen = QApplication.primaryScreen()
        if screen is None:
            return
        ag = screen.availableGeometry()
        w, h = 22, 22
        x = ag.right() - w - self._MARGIN + 1
        y = ag.bottom() - h - self._MARGIN + 1
        self.setFixedSize(w, h)
        self.move(x, y)
        self._dot.setGeometry(0, 0, w, h)
        self._dot.show()
        self._card.hide()

    def _severity_style(self, severity: str) -> str:
        if severity == "critical":
            return "border: 1px solid rgba(255, 90, 90, 0.85);"
        if severity == "info":
            return "border: 1px solid rgba(120, 180, 255, 0.55);"
        return "border: 1px solid rgba(255, 200, 80, 0.75);"

    def show_alert(self, title: str, body: str, *, severity: str = "warning") -> None:
        self._stop_geometry_anim()

        self._title_lbl.setText(title or "Emotion Checker")
        self._body_lbl.setText(body or "")
        self._card.setStyleSheet(
            """
            #card {
                background-color: rgba(28, 32, 36, 242);
                border-radius: 12px;
            }
            """
            + self._severity_style(severity)
            + """
            QLabel#title { color: #f0f4f8; font-size: 14px; font-weight: 600; }
            QLabel#body { color: #c8d0da; font-size: 13px; }
            """
        )

        self._collapse_timer.stop()
        ms = (
            self._COLLAPSE_MS_CRITICAL
            if severity == "critical"
            else self._COLLAPSE_MS_WARNING
        )
        self._collapse_timer.start(ms)
        self.expand()

    def expand(self) -> None:
        screen = QApplication.primaryScreen()
        if screen is None:
            return
        ag = screen.availableGeometry()

        self._card.setFixedWidth(self._EXPAND_W)
        self._card.adjustSize()
        card_h = max(96, self._card.sizeHint().height())

        target_w = self._EXPAND_W
        target_h = card_h
        x = ag.right() - target_w - self._MARGIN + 1
        y = ag.bottom() - target_h - self._MARGIN + 1

        start_geo = self.geometry()
        end_geo = QRect(x, y, target_w, target_h)

        self._expanded = True
        self._dot.hide()
        # Пока окно «растёт», карточку не показываем — иначе она сжимается в 22×22.
        self._card.hide()

        self.setMinimumSize(1, 1)
        self.setMaximumSize(16777215, 16777215)

        anim = QPropertyAnimation(self, b"geometry")
        self._geom_anim = anim
        anim.setDuration(220)
        anim.setEasingCurve(QEasingCurve.Type.OutCubic)
        anim.setStartValue(start_geo)
        anim.setEndValue(end_geo)
        anim.finished.connect(self._on_expand_anim_finished)
        anim.start()
        self.raise_()
        self.show()

    def _on_expand_anim_finished(self) -> None:
        if self._geom_anim is not None:
            self._geom_anim.deleteLater()
            self._geom_anim = None
        self._card.setGeometry(self.rect())
        self._card.show()

    def collapse(self) -> None:
        self._collapse_timer.stop()
        self._stop_geometry_anim()

        if not self._expanded:
            self._place_collapsed()
            return

        self._card.hide()

        screen = QApplication.primaryScreen()
        if screen is None:
            return
        ag = screen.availableGeometry()
        end_w, end_h = 22, 22
        end_x = ag.right() - end_w - self._MARGIN + 1
        end_y = ag.bottom() - end_h - self._MARGIN + 1
        end_geo = QRect(end_x, end_y, end_w, end_h)

        anim = QPropertyAnimation(self, b"geometry")
        self._geom_anim = anim
        anim.setDuration(180)
        anim.setEasingCurve(QEasingCurve.Type.InCubic)
        anim.setStartValue(self.geometry())
        anim.setEndValue(end_geo)
        anim.finished.connect(self._after_collapse_anim)
        anim.start()

    def _after_collapse_anim(self) -> None:
        self._expanded = False
        self._card.hide()
        if self._geom_anim is not None:
            self._geom_anim.deleteLater()
            self._geom_anim = None
        self.setFixedSize(22, 22)
        self._place_collapsed()
        self._dot.show()
        self.raise_()
