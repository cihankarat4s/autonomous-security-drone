"""Run a person model on a video and show the boxes live in a window.

Nothing in the video is labelled: every box is the model's own output for that
frame. q or Esc stops, space pauses.

    python training/watch_video.py video.mp4 --model yolo26n_visdrone_sdd.pt --conf 0.25
"""
import argparse
import time

import cv2
from ultralytics import YOLO

from security_drone.detection.detector import Detection, merge_overlapping

WINDOW = "Person detection (q: quit, space: pause)"


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("video")
    parser.add_argument("--model", default="yolo26n_visdrone_sdd.pt")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument("--width", type=int, default=1280, help="window width in px")
    args = parser.parse_args()

    model = YOLO(args.model)
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise SystemExit(f"Cannot open {args.video}")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)

    index, with_person, started = 0, 0, time.monotonic()
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        index += 1
        r = model.predict(frame, conf=args.conf, imgsz=args.imgsz, classes=[0], verbose=False)[0]
        dets = merge_overlapping([Detection(*b, c) for b, c in zip(r.boxes.xyxy.tolist(), r.boxes.conf.tolist())])
        with_person += bool(dets)

        scale = args.width / frame.shape[1]
        view = cv2.resize(frame, (args.width, round(frame.shape[0] * scale)))
        for d in dets:
            x1, y1, x2, y2 = (round(v * scale) for v in (d.x1, d.y1, d.x2, d.y2))
            cv2.rectangle(view, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(view, f"{d.confidence:.2f}", (x1, max(y1 - 5, 14)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)
        fps = index / (time.monotonic() - started)
        cv2.putText(view, f"{index}/{total}  {len(dets)} person  {fps:.1f} fps  {args.model}", (10, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.imshow(WINDOW, view)

        key = cv2.waitKey(1) & 0xFF
        if key == ord(" "):
            key = cv2.waitKey(0) & 0xFF
        if key in (ord("q"), 27):
            break

    cap.release()
    cv2.destroyAllWindows()
    print(f"{with_person}/{index} frames with a person")


if __name__ == "__main__":
    main()
