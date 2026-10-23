"""Count false positives of detection models on person-free images.

Every image in the given folders is a negative, so any person box is a false
positive. Uses the companion's settings (confidence 0.4, 1280 px) by default.

    python training/count_false_positives.py --models yolov8n.pt yolov8n_visdrone_person.pt \
        --folders ~/datasets/gazebo_negatives/sun_b
"""
import argparse
from pathlib import Path

from ultralytics import YOLO


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--models", nargs="+", required=True)
    parser.add_argument("--folders", nargs="+", type=Path, required=True)
    parser.add_argument("--conf", type=float, default=0.4)
    parser.add_argument("--imgsz", type=int, default=1280)
    args = parser.parse_args()

    print("model | folder | images | images with a false positive | boxes")
    for model_path in args.models:
        model = YOLO(model_path)
        for folder in args.folders:
            images = sorted(folder.expanduser().glob("*.jpg"))
            hit_images = boxes = 0
            # One image per call: a list is run as a single batch and fills a 4 GB GPU.
            for image in images:
                result = model.predict(image, conf=args.conf, imgsz=args.imgsz, classes=[0],
                                       verbose=False)[0]
                n = len(result.boxes)
                boxes += n
                hit_images += n > 0
            rate = hit_images / len(images) if images else 0.0
            print(f"{Path(model_path).name} | {folder.name} | {len(images)} | "
                  f"{hit_images} ({rate:.1%}) | {boxes}")


if __name__ == "__main__":
    main()
