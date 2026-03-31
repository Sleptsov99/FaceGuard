"""
Emotion Checker — фоновый режим с треем (основной UX).

Запуск из корня репозитория:

    python -m desktop_app.main

Подавление шумных логов TF/MediaPipe — до импорта Qt и движка.
"""

from __future__ import annotations

import sys
from pathlib import Path

from desktop_app.log_silence import apply_log_silence

apply_log_silence()

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from PySide6.QtWidgets import QApplication

from desktop_app.coordinator import DesktopCoordinator


def main() -> None:
    app = QApplication(sys.argv)
    app.setApplicationName("Emotion Checker")
    # Strong Python ref + QObject parent=app: avoids GC destroying tray/timer mid-run.
    _coordinator = DesktopCoordinator(app, camera_source=0)
    setattr(app, "_emotion_checker_coordinator", _coordinator)
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
