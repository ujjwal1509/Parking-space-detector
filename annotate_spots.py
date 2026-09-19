"""
Step 1: Define parking spots once (camera is fixed).

Usage:
    python annotate_spots.py --video parking.mp4 --out spots.json

Controls:
    Left click                 : add a corner (every 4 corners = 1 spot)
    Right click / d / Backspace: undo corner / delete previous box
    u                          : undo last corner / last spot
    s                          : save to JSON
    q / ESC                    : save and quit
"""
import argparse
import json
import cv2
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--out", default="spots.json")
    ap.add_argument("--frame", type=int, default=0, help="frame index to annotate on")
    ap.add_argument("--max-display", type=int, default=1280, help="max display width")
    args = ap.parse_args()

    cap = cv2.VideoCapture(args.video)
    cap.set(cv2.CAP_PROP_POS_FRAMES, args.frame)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise SystemExit("Could not read a frame from the video.")

    h, w = frame.shape[:2]
    scale = min(1.0, args.max_display / w)  # display scale only; saved coords are original-res
    spots = []      # list of 4-point polygons (original coordinates)
    current = []    # corners of the spot being drawn (original coordinates)

    def render():
        img = frame.copy()
        for i, poly in enumerate(spots):
            pts = np.array(poly, np.int32)
            cv2.polylines(img, [pts], True, (0, 255, 255), 2)
            cx, cy = pts.mean(axis=0).astype(int)
            cv2.putText(img, str(i + 1), (cx - 8, cy + 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        for p in current:
            cv2.circle(img, tuple(p), 5, (0, 0, 255), -1)
        if len(current) > 1:
            cv2.polylines(img, [np.array(current, np.int32)], False, (0, 0, 255), 2)
        cv2.putText(img, f"Spots: {len(spots)}  (u/d=undo, s=save, q=quit)", (15, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        if scale != 1.0:
            img = cv2.resize(img, None, fx=scale, fy=scale)
        return img

    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            current.append([int(x / scale), int(y / scale)])
            if len(current) == 4:
                spots.append(current.copy())
                current.clear()
        elif event == cv2.EVENT_RBUTTONDOWN:
            if current:
                current.pop()
            elif spots:
                spots.pop()

    def save():
        with open(args.out, "w") as f:
            json.dump({"frame_size": [w, h], "spots": spots}, f, indent=2)
        print(f"Saved {len(spots)} spots to {args.out}")

    cv2.namedWindow("Annotate spots")
    cv2.setMouseCallback("Annotate spots", on_mouse)

    while True:
        cv2.imshow("Annotate spots", render())
        key = cv2.waitKey(20) & 0xFF
        if key in (ord("u"), ord("d"), 8):  # u, d, or Backspace
            if current:
                current.pop()
            elif spots:
                spots.pop()
        elif key == ord("s"):
            save()
        elif key in (ord("q"), 27):
            save()
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
