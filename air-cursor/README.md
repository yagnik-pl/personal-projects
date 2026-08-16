# Air Cursor — Real-Time Gesture-Based Computer Control

A modular, real-time computer vision system that translates webcam-based hand gestures into desktop cursor navigation, click triggers, and system volume controls using MediaPipe, OpenCV, and PyAutoGUI.

---

## Architecture Overview

Air Cursor is designed as a clean 5-module pipeline adhering to single-responsibility principles:

```
                      ┌──────────────────────┐
                      │    Webcam Input      │
                      └──────────┬───────────┘
                                 │ Frame (BGR)
                                 ▼
                      ┌──────────────────────┐
                      │     HandTracker      │
                      │  (MediaPipe Nodes)   │
                      └──────────┬───────────┘
                                 │ Normalized Coordinates (0, 4, 8, 12)
                                 ▼
                      ┌──────────────────────┐
                      │  GeometryProcessor   │
                      │(Active Box + EMA)    │
                      └──────────┬───────────┘
                                 │ Smoothed Screen Coordinates
                                 ▼
                      ┌──────────────────────┐
                      │  GestureController   │
                      │   (State Machine)    │
                      └──────────┬───────────┘
                                 │ ActionType (MOVE / CLICK / ACTION_ENGAGED)
                                 ▼
                      ┌──────────────────────┐
                      │    SystemActuator    │
                      │(PyAutoGUI + pycaw)   │
                      └──────────────────────┘
```

### Module Breakdown

| Module | File | Responsibility |
|---|---|---|
| **Module 1: Vision Engine** | [`src/hand_tracker.py`](./project_code/src/hand_tracker.py) | Ingests frames, executes MediaPipe hand landmark extraction (Wrist: 0, Thumb: 4, Index: 8, Middle: 12), overlays FPS and debug indicators. |
| **Module 2: Math & Mapping** | [`src/geometry_processor.py`](./project_code/src/geometry_processor.py) | Calibrates active bounding margin (100px boundary), maps camera frame to screen dimensions ($1920 \times 1080$), and applies Exponential Moving Average (EMA, $\alpha=0.4$) smoothing. |
| **Module 3: State Machine** | [`src/gesture_controller.py`](./project_code/src/gesture_controller.py) | Evaluates relative landmark distances to distinguish hover cursor tracking, pinch-clicks, and volume engagement without any OS-level side effects. |
| **Module 4: OS Actuation** | [`src/system_actuator.py`](./project_code/src/system_actuator.py) | Dispatches OS-level mouse movements (`pyautogui`), debounced clicks ($0.4\text{s}$ threshold), and endpoint master volume adjustments (`pycaw` with scalar fallbacks). |
| **Module 5: Integration** | [`src/main.py`](./project_code/src/main.py) | High-performance synchronous capture and processing loop with active box visual overlays and responsive exit handling. |

---

## Gestures & Controls

| Gesture | Landmark Condition | Triggered Action | Visual Overlay |
|---|---|---|:---:|
| **Cursor Move** | Index finger extended, default state | Cursor follows index fingertip (EMA smoothed) | Green circle |
| **Left Click** | Thumb tip (4) & Middle tip (12) distance $< 0.04$ | Left mouse click ($0.4\text{s}$ debounce) | Red circle |
| **Volume Control** | Thumb tip (4) & Index tip (8) distance $< 0.03$ | Master volume mapped to vertical hand position | Blue circle |
| **Exit** | Keyboard press `q` | Clean window destruction & camera release | — |

---

## Directory Structure

```
air-cursor/
├── README.md
└── project_code/
    ├── requirements.txt
    └── src/
        ├── geometry_processor.py
        ├── gesture_controller.py
        ├── hand_tracker.py
        ├── main.py
        └── system_actuator.py
```

---

## Quickstart & Installation

### 1. Prerequisites
- Python 3.9+ (Python 3.10 – 3.12 recommended)
- Working webcam
- Windows OS (for `pycaw` audio endpoint integration; mouse navigation works cross-platform)

### 2. Install Dependencies
```bash
cd project_code
pip install -r requirements.txt
```

### 3. Run Application
```bash
python src/main.py
```

### 4. Running Individual Unit Tests
```bash
# Test mathematical geometry processing & coordinate mapping
python src/geometry_processor.py

# Test gesture state evaluation logic
python src/gesture_controller.py
```
