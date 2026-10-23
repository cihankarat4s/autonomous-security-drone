"""Pick hard negatives from person-free capture folders into one folder.

Frames are 0.5 s apart, so most are near-duplicates of plain ground. Keeping
all of them would make background images ~28% of the training set, which can
make the model too cautious. This keeps
  - every frame where any of the given models boxes a "person" (at a lower
    confidence than the companion's 0.4, to include near misses), and
  - every n-th of the remaining frames, for variety.

    python training/select_negatives.py --models yolov8n.pt yolov8n_visdrone_person.pt \
        --folders ~/datasets/gazebo_negatives/{sun_a,sun_b,sun_c,urban_a,urban_c} \
        --out ~/datasets/hard_negatives
"""
import argparse
import shutil
from pathlib import Path

from ultralytics import YOLO


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--models", nargs="+", required=True)
    parser.add_argument("--folders", nargs="+", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument("--every", type=int, default=4, help="keep every n-th frame without a hit")
    args = parser.parse_args()

    models = [YOLO(m) for m in args.models]
    args.out.mkdir(parents=True, exist_ok=True)
    total_hard = total_kept = 0
    for folder in args.folders:
        images = sorted(folder.expanduser().glob("*.jpg"))
        hard = kept = 0
        for i, image in enumerate(images):
            # One image per call; a list would be one batch and fill a 4 GB GPU.
            hit = any(len(m.predict(image, conf=args.conf, imgsz=args.imgsz, classes=[0],
                                    verbose=False)[0].boxes) for m in models)
            if hit or i % args.every == 0:
                shutil.copy2(image, args.out / image.name)
                kept += 1
                hard += hit
        print(f"{folder.name}: {len(images)} frames, kept {kept} ({hard} with a detection)")
        total_hard += hard
        total_kept += kept
    print(f"Total: kept {total_kept} frames ({total_hard} hard) in {args.out}")


if __name__ == "__main__":
    main()
