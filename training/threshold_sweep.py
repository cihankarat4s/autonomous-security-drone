"""Trade false alarms against missed people over confidence thresholds.

For each model and threshold, reports
  - the share of person-free frames with any box (false-positive frames), and
  - on frames that contain people, the share of frames with any person (what
    the alarm needs), the mean number of people and the share of frames with
    at least --min-people (how many of the people are still found). Nested
    boxes on one person count once; raw box counts are printed alongside.

Both folders are scored with the same model run at the lowest threshold, so
this is one inference pass per image.

    python training/threshold_sweep.py --models a.pt b.pt \
        --negatives ~/datasets/gazebo_negatives/sun_d \
        --positives ~/datasets/gazebo_urban/patrol_1 --positive-range 49 102

    # held-out check of the chosen threshold
    python training/threshold_sweep.py --models a.pt b.pt \
        --negatives ~/datasets/gazebo_negatives/val_e \
        --positives ~/datasets/gazebo_urban/patrol_2 --positive-glob '*_seq3.jpg'
"""
import argparse
from pathlib import Path

from ultralytics import YOLO


def detections(model, images, imgsz, low):
    """Per image, (confidence, xyxy) of all person boxes above the lowest threshold."""
    out = []
    for image in images:
        result = model.predict(image, conf=low, imgsz=imgsz, classes=[0], verbose=False)[0]
        out.append(list(zip(result.boxes.conf.tolist(), result.boxes.xyxy.tolist())))
    return out


def count_people(boxes, threshold, overlap=0.5):
    """Boxes at or above threshold, with nested boxes counted once.

    The model sometimes puts a small box on a person inside a larger box on
    the same person plus shadow. NMS keeps both (low IoU), so a box that
    covers more than `overlap` of a smaller kept box is the same person.
    """
    kept = []
    for conf, (x1, y1, x2, y2) in sorted(boxes, reverse=True):
        if conf < threshold:
            break
        area = (x2 - x1) * (y2 - y1)
        for kx1, ky1, kx2, ky2 in kept:
            iw = min(x2, kx2) - max(x1, kx1)
            ih = min(y2, ky2) - max(y1, ky1)
            small = min(area, (kx2 - kx1) * (ky2 - ky1))
            if iw > 0 and ih > 0 and iw * ih > overlap * small:
                break
        else:
            kept.append((x1, y1, x2, y2))
    return len(kept)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--models", nargs="+", required=True)
    parser.add_argument("--negatives", type=Path, required=True)
    parser.add_argument("--positives", type=Path, required=True)
    parser.add_argument("--positive-range", type=int, nargs=2, metavar=("START", "END"),
                        help="slice of the sorted positive frames to use, e.g. the hover")
    parser.add_argument("--positive-glob", default="*.jpg",
                        help="e.g. '*_seq3.jpg' for the hover of a patrol capture")
    parser.add_argument("--thresholds", type=float, nargs="+", default=[0.2, 0.25, 0.3, 0.35, 0.4])
    parser.add_argument("--min-people", type=int, default=2)
    parser.add_argument("--imgsz", type=int, default=1280)
    args = parser.parse_args()

    negatives = sorted(args.negatives.expanduser().glob("*.jpg"))
    positives = sorted(args.positives.expanduser().glob(args.positive_glob))
    if args.positive_range:
        positives = positives[slice(*args.positive_range)]
    low = min(args.thresholds)

    print(f"negatives: {args.negatives.name} ({len(negatives)}), "
          f"positives: {args.positives.name} ({len(positives)})")
    print(f"model | conf | FP frames | positives: frames with a person | mean people"
          f" | frames with >= {args.min_people} people | (raw boxes: mean | frames with >= {args.min_people})")
    for model_path in args.models:
        model = YOLO(model_path)
        neg = detections(model, negatives, args.imgsz, low)
        pos = detections(model, positives, args.imgsz, low)
        for t in args.thresholds:
            fp = sum(any(c >= t for c, _ in boxes) for boxes in neg) / len(neg)
            people = [count_people(boxes, t) for boxes in pos]
            raw = [sum(c >= t for c, _ in boxes) for boxes in pos]
            print(f"{Path(model_path).name} | {t:.2f} | {fp:.1%} | "
                  f"{sum(n > 0 for n in people) / len(people):.1%} | "
                  f"{sum(people) / len(people):.2f} | "
                  f"{sum(n >= args.min_people for n in people) / len(people):.1%} | "
                  f"({sum(raw) / len(raw):.2f} | "
                  f"{sum(n >= args.min_people for n in raw) / len(raw):.1%})")


if __name__ == "__main__":
    main()
