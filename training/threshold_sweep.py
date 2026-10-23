"""Trade false alarms against missed people over confidence thresholds.

For each model and threshold, reports
  - the share of person-free frames with any box (false-positive frames), and
  - on frames that contain people, the mean number of boxes and the share of
    frames with at least --min-people boxes (how many people are still found).

Both folders are scored with the same model run at the lowest threshold, so
this is one inference pass per image.

    python training/threshold_sweep.py --models a.pt b.pt \
        --negatives ~/datasets/gazebo_negatives/sun_d \
        --positives ~/datasets/gazebo_urban/patrol_1 --positive-range 49 102
"""
import argparse
from pathlib import Path

from ultralytics import YOLO


def confidences(model, images, imgsz, low):
    """Per image, the confidences of all person boxes above the lowest threshold."""
    out = []
    for image in images:
        result = model.predict(image, conf=low, imgsz=imgsz, classes=[0], verbose=False)[0]
        out.append([float(c) for c in result.boxes.conf])
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--models", nargs="+", required=True)
    parser.add_argument("--negatives", type=Path, required=True)
    parser.add_argument("--positives", type=Path, required=True)
    parser.add_argument("--positive-range", type=int, nargs=2, metavar=("START", "END"),
                        help="slice of the sorted positive frames to use, e.g. the hover")
    parser.add_argument("--thresholds", type=float, nargs="+", default=[0.2, 0.25, 0.3, 0.35, 0.4])
    parser.add_argument("--min-people", type=int, default=2)
    parser.add_argument("--imgsz", type=int, default=1280)
    args = parser.parse_args()

    negatives = sorted(args.negatives.expanduser().glob("*.jpg"))
    positives = sorted(args.positives.expanduser().glob("*.jpg"))
    if args.positive_range:
        positives = positives[slice(*args.positive_range)]
    low = min(args.thresholds)

    print(f"negatives: {args.negatives.name} ({len(negatives)}), "
          f"positives: {args.positives.name} ({len(positives)})")
    print(f"model | conf | FP frames | mean boxes on positives | frames with >= {args.min_people}")
    for model_path in args.models:
        model = YOLO(model_path)
        neg = confidences(model, negatives, args.imgsz, low)
        pos = confidences(model, positives, args.imgsz, low)
        for t in args.thresholds:
            fp = sum(any(c >= t for c in cs) for cs in neg) / len(neg)
            counts = [sum(c >= t for c in cs) for cs in pos]
            mean = sum(counts) / len(counts)
            multi = sum(n >= args.min_people for n in counts) / len(counts)
            print(f"{Path(model_path).name} | {t:.2f} | {fp:.1%} | {mean:.2f} | {multi:.1%}")


if __name__ == "__main__":
    main()
