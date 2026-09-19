"""
Step 2: Detect vehicles with YOLO, decide occupancy per spot, write annotated mp4.

Usage:
    python detect_parking.py --video parking.mp4 --spots spots.json --out output.mp4

Useful options:
    --model yolo11l.pt     bigger model = more accurate
    --imgsz 1280           larger inference size helps small/distant cars
    --overlap 0.30         min fraction of spot covered by a car footprint
    --smooth 8             frames a new state must persist before it flips
"""
import argparse
import csv
import json

import cv2
import numpy as np
import torch
from ultralytics import YOLO

VEHICLE_CLASSES = [2, 5, 7]  # COCO: car, bus, truck (add 3 for motorcycle if needed)
GREEN = (0, 200, 0)
RED = (0, 0, 220)


def load_spots(path):
    with open(path) as f:
        data = json.load(f)
    return [np.array(p, dtype=np.float32) for p in data["spots"]]


def footprint(box, frac=0.4):
    """Approximate the car's ground footprint as the bottom part of its box."""
    x1, y1, x2, y2 = box
    fy1 = y2 - (y2 - y1) * frac
    return np.array([[x1, fy1], [x2, fy1], [x2, y2], [x1, y2]], dtype=np.float32)


def spot_is_occupied(spot, boxes, overlap_thr):
    """Occupied if a car footprint covers enough of the spot, or its bottom-centre lies in the spot."""
    spot_area = cv2.contourArea(spot)
    if spot_area <= 0:
        return False
    for box in boxes:
        fp = footprint(box)
        # bottom-centre point inside the spot polygon
        bc = (float((box[0] + box[2]) / 2), float(box[3] - (box[3] - box[1]) * 0.1))
        if cv2.pointPolygonTest(spot, bc, False) >= 0:
            return True
        # intersection area (both shapes are convex quads)
        try:
            inter, _ = cv2.intersectConvexConvex(spot, fp)
        except cv2.error:
            inter = 0.0
        if inter / spot_area >= overlap_thr:
            return True
    return False


def draw(frame, spots, states, n_occ, n_free):
    overlay = frame.copy()
    for i, (spot, occ) in enumerate(zip(spots, states)):
        pts = spot.astype(np.int32)
        cv2.fillPoly(overlay, [pts], RED if occ else GREEN)
    frame = cv2.addWeighted(overlay, 0.3, frame, 0.7, 0)
    for i, (spot, occ) in enumerate(zip(spots, states)):
        pts = spot.astype(np.int32)
        color = RED if occ else GREEN
        cv2.polylines(frame, [pts], True, color, 3)
        x, y, bw, bh = cv2.boundingRect(pts)
        label = f"{i + 1}: {'Occupied' if occ else 'Free'}"
        cv2.putText(frame, label, (x, max(15, y - 5)), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, color, 2, cv2.LINE_AA)

    text = f"Occupied: {n_occ} | Empty: {n_free} | Total: {n_occ + n_free}"
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)
    cv2.rectangle(frame, (10, 10), (30 + tw, 30 + th + 10), (0, 0, 0), -1)
    cv2.putText(frame, text, (20, 20 + th + 5), cv2.FONT_HERSHEY_SIMPLEX,
                0.9, (255, 255, 255), 2, cv2.LINE_AA)
    return frame


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--spots", default="spots.json")
    ap.add_argument("--out", default="output.mp4")
    ap.add_argument("--model", default="yolo11m.pt")
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--conf", type=float, default=0.3)
    ap.add_argument("--overlap", type=float, default=0.30)
    ap.add_argument("--smooth", type=int, default=8)
    ap.add_argument("--counts-csv", default=None, help="optional per-frame counts CSV")
    args = ap.parse_args()

    device = 0 if torch.cuda.is_available() else "cpu"
    half = device != "cpu"
    print(f"Device: {'cuda' if half else 'cpu'}")
    model = YOLO(args.model)

    spots = load_spots(args.spots)
    n_spots = len(spots)
    if n_spots == 0:
        raise SystemExit("No spots in the JSON file. Run annotate_spots.py first.")

    cap = cv2.VideoCapture(args.video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    writer = cv2.VideoWriter(args.out, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    states = [None] * n_spots   # stable (smoothed) state per spot
    pending = [0] * n_spots     # consecutive frames disagreeing with stable state
    rows = []
    frame_idx = 0
    n_occ = n_free = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        res = model.predict(frame, imgsz=args.imgsz, conf=args.conf,
                            classes=VEHICLE_CLASSES, device=device,
                            half=half, verbose=False)[0]
        boxes = res.boxes.xyxy.cpu().numpy() if res.boxes is not None else np.empty((0, 4))

        for i, spot in enumerate(spots):
            raw = spot_is_occupied(spot, boxes, args.overlap)
            if states[i] is None:            # first frame: trust detection directly
                states[i] = raw
            elif raw != states[i]:
                pending[i] += 1
                if pending[i] >= args.smooth:  # state held long enough -> flip
                    states[i] = raw
                    pending[i] = 0
            else:
                pending[i] = 0

        n_occ = int(sum(states))
        n_free = n_spots - n_occ
        writer.write(draw(frame, spots, states, n_occ, n_free))
        rows.append((frame_idx, n_occ, n_free))

        frame_idx += 1
        if frame_idx % 50 == 0:
            print(f"Processed {frame_idx}/{total or '?'} frames | occupied={n_occ} empty={n_free}")

    cap.release()
    writer.release()

    if args.counts_csv:
        with open(args.counts_csv, "w", newline="") as f:
            wr = csv.writer(f)
            wr.writerow(["frame", "occupied", "empty"])
            wr.writerows(rows)

    print("\n=== Done ===")
    print(f"Output video : {args.out}")
    print(f"Total spots  : {n_spots}")
    print(f"Final count  : Occupied = {n_occ}, Empty = {n_free}")


if __name__ == "__main__":
    main()
