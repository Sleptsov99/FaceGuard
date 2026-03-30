# CV Engine

Computer vision engine for facial analysis and emotion detection.

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Run with camera
```bash
python app/main.py --source camera
```

### Run with video file
```bash
python app/main.py --source video --path samples/input.mp4
```

## Project Structure

```
cv-engine/
  app/
    main.py           # Entry point
    camera/           # Camera input handling
    landmarks/        # Facial landmark detection
    metrics/          # Metric calculations
    temporal/         # Temporal analysis
    state/            # State management
    api/              # API endpoints
    utils/            # Utility functions
  tests/
  samples/
  requirements.txt
  README.md
```
