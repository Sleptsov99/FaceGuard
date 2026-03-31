# CV Engine

Computer vision engine for facial analysis and emotion detection.

## Run modes

- **Preview (default):** opens an OpenCV window, overlays metrics, quit with `q`.

  `python app/main.py --source camera`

- **Headless:** same pipeline, no `imshow` / `waitKey` (for automation or a future Qt worker).

  `python app/main.py --source camera --headless`

Programmatic use: `EngineSession` in `app/engine/session.py` — `read_and_process()`, `run_loop(preview=False, on_frame=...)`.

**MediaPipe:** landmark detection uses **Tasks** `FaceLandmarker` (`mediapipe>=0.10.30`). The file `face_landmarker.task` is downloaded automatically on first run into `app/landmarks/data/` (needs network once). Older `mp.solutions.face_mesh` is not available in current PyPI builds for recent Python versions.

## Desktop prototype (PySide6)

**Tip:** use a **virtual environment** (`python -m venv .venv`) so this project’s `numpy` / `ml-dtypes` versions are not mixed with a global **TensorFlow** install. Newer TensorFlow needs `ml-dtypes>=0.5.1`; an older pin breaks `import mediapipe` when TF is on `PYTHONPATH` (see `requirements.txt`).

From the **repository root** (with dependencies installed):

```bash
python -m desktop_app.main
```

On **Windows**, alerts use **`windows-toasts` (WinRT)** plus a **tray balloon** so something is visible even if Toast is suppressed. Also install **`winotify`** as fallback. Run `pip install -r requirements.txt`. If Toasts не всплывают: Параметры → Уведомления → разрешить для Python; отключите «Фокусировка внимания». Чтобы не видеть шум MediaPipe в консоли, запускайте **`pythonw -m desktop_app.main`** (без чёрного окна; логи DLL всё равно могут обходить `sys.stderr`).

Primary UX is **tray + background monitoring**: on launch the app stays in the **system tray**, starts **`EngineSession` in a worker thread** automatically, and shows **native notifications** for fatigue-related alerts (with cooldowns in `desktop_app/alert_policy.py`). The **dashboard** (preview, metrics, log) opens from the tray menu or by clicking the icon — it is hidden by default. Closing the dashboard only hides it; use **Выход** in the tray menu to quit. Log noise from TF/MediaPipe is reduced via `desktop_app/log_silence.py` before heavy imports.

## Project Structure

```
cv-engine/
  app/
    main.py           # Entry point
    engine/           # EngineSession + frame results (headless / future UI)
    camera/           # Camera input handling
    landmarks/        # Facial landmark detection
    metrics/          # Metric calculations
    temporal/         # Temporal analysis
    state/            # State management
    api/              # API endpoints
    utils/            # Utility functions
  desktop_app/      # Tray-first PySide6 app (coordinator, worker, alerts)
  tests/
  samples/
  requirements.txt
  README.md
```
