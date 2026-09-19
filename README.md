# Parking Space Detector and Counter

A real-time parking space availability detector and occupancy counter using **YOLO11** and polygon footprint geometric overlap.

---

## Overview

The system uses a 2-step workflow designed for fixed-angle surveillance cameras:
1. **Annotate Spots once**: Interactively click the 4 corners of each parking spot on a video frame. Stored as arbitrary 4-point convex polygons in `spots.json`.
2. **Detect & Count**: Detect vehicles (cars, buses, trucks) frame-by-frame using pretrained YOLO11, determine occupancy per spot via polygon intersection and vehicle ground footprints, apply temporal smoothing to avoid flickers, and output an annotated video with live availability counters.

---

## Architecture & Logic

- **Ground Footprint Approximation**: Uses the bottom 40% of the vehicle bounding box as its ground contact area. This eliminates false positives caused by tall vehicles spilling over into adjacent spots behind them at angled camera views.
- **Dual Occupancy Condition**: A spot is marked occupied if:
  - The bottom-centre of a detected vehicle lies inside the spot polygon, **or**
  - The vehicle's ground footprint covers $\ge 30\%$ (configurable via `--overlap`) of the spot's area.
- **Temporal Smoothing**: Spots must hold a new state for multiple consecutive frames (`--smooth`, default 8) before flipping between Free and Occupied, preventing detection flicker.
- **GPU Acceleration**: Automatically uses CUDA / FP16 if an NVIDIA GPU is detected.

---

## Setup

```bash
conda activate parking-yolo
# or
pip install -r requirements.txt
```

> **Note**: For GPU acceleration, ensure PyTorch with CUDA support is installed (`pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121`).

---

## Usage

### Step 1: Annotate Parking Spots

Launch the interactive spot annotation tool:
```bash
python annotate_spots.py --video video.mp4 --out spots.json
```

**Controls:**
- **Left click**: Add a corner (every 4 corners = 1 parking spot).
- **`u`**: Undo last corner / last spot.
- **`s`**: Save spots to JSON.
- **`q`** or **`ESC`**: Save and exit.

### Step 2: Run Detection and Occupancy Counter

```bash
python detect_parking.py --video video.mp4 --spots spots.json --out output.mp4
```

To run with the local nano model or customize options:
```bash
python detect_parking.py --video video.mp4 --spots spots.json --out output.mp4 --model yolo11n.pt
```

---

## Tuning Options

| Flag | Default | Description |
| :--- | :--- | :--- |
| `--model` | `yolo11m.pt` | Model size (`yolo11n.pt`, `yolo11s.pt`, `yolo11m.pt`, `yolo11l.pt`, etc.). |
| `--imgsz` | `1280` | Inference image size. Larger sizes help detect distant/small cars. |
| `--conf` | `0.3` | Minimum YOLO confidence threshold. Lower to ~0.2 if cars are missed. |
| `--overlap` | `0.30` | Minimum overlap ratio of car footprint over spot area. Raise to 0.4–0.5 if false positives occur. |
| `--smooth` | `8` | Consecutive frames a new state must persist before flipping. Raise if status flickers. |
| `--counts-csv` | `None` | Path to export per-frame counts CSV (`frame,occupied,empty`). |

---

## Project Structure

```
.
├── annotate_spots.py   # Interactive 4-corner polygon annotation tool
├── detect_parking.py   # YOLO vehicle detection & occupancy tracking
├── requirements.txt    # Core dependencies (ultralytics, opencv-python, numpy)
├── yolo11n.pt          # Local pretrained YOLO11 nano weights
└── README.md           # Documentation
```
