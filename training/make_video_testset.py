"""Pick evenly spaced video frames and draw numbered candidate boxes for review by eye.

Candidates are the low-confidence boxes of the given models, so a reviewer only
accepts, rejects or adds. A box that no model proposed has to be added by hand.

    python training/make_video_testset.py video1.mp4 video2.mp4 --frames 500 \
        --models a.pt b.pt --target ~/datasets/video_test
"""
import argparse
import json
from pathlib import Path

import cv2
from ultralytics import YOLO

CONF = 0.10
SAME_BOX = 0.5                   # IoU above which two models' boxes count as one


def iou(a, b):
    w = min(a[2], b[2]) - max(a[0], b[0])
    h = min(a[3], b[3]) - max(a[1], b[1])
    if w <= 0 or h <= 0:
        return 0.0
    inter = w * h
    return inter / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter)


def union_boxes(boxes):
    """Most confident first; a box overlapping a kept one is dropped."""
    kept = []
    for box in sorted(boxes, key=lambda b: -b["conf"]):
        if all(iou(box["xyxy"], k["xyxy"]) < SAME_BOX for k in kept):
            kept.append(box)
    return kept


def spaced_indices(total, count):
    """count evenly spaced 0-based frame numbers."""
    step = total / count
    return [int(i * step + step / 2) for i in range(count)]


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("videos", nargs="+")
    parser.add_argument("--frames", type=int, default=500, help="across all videos")
    parser.add_argument("--models", nargs="+", required=True)
    parser.add_argument("--target", type=Path, required=True)
    args = parser.parse_args()

    models = [YOLO(m) for m in args.models]
    (args.target / "images").mkdir(parents=True, exist_ok=True)
    (args.target / "review").mkdir(exist_ok=True)
    caps = [cv2.VideoCapture(v) for v in args.videos]
    totals = [int(c.get(cv2.CAP_PROP_FRAME_COUNT)) for c in caps]
    candidates = {}
    for video, cap, total in zip(args.videos, caps, totals):
        count = round(args.frames * total / sum(totals))
        for index in spaced_indices(total, count):
            cap.set(cv2.CAP_PROP_POS_FRAMES, index)
            ok, frame = cap.read()
            if not ok:
                continue
            name = f"{Path(video).stem}_{index:05d}"
            cv2.imwrite(str(args.target / "images" / f"{name}.jpg"), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
            boxes = []
            for model in models:
                r = model.predict(frame, conf=CONF, imgsz=1280, verbose=False)[0]
                boxes += [{"xyxy": [round(v) for v in b], "conf": round(c, 2), "cls": r.names[int(k)]}
                          for b, c, k in zip(r.boxes.xyxy.tolist(), r.boxes.conf.tolist(), r.boxes.cls.tolist())]
            boxes = union_boxes(boxes)
            candidates[name] = boxes
            for i, box in enumerate(boxes):
                x1, y1, x2, y2 = box["xyxy"]
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 1)
                cv2.putText(frame, str(i), (x1, max(y1 - 2, 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
            cv2.imwrite(str(args.target / "review" / f"{name}.jpg"), frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
    (args.target / "candidates.json").write_text(json.dumps(candidates))
    print(len(candidates), "frames,", sum(map(len, candidates.values())), "candidate boxes")


if __name__ == "__main__":
    main()
