"""Fine-tune COCO yolov8n on the single-class VisDrone "person" set.

Sized for a 4 GB GPU (GTX 1650 Ti). Do not run it while Gazebo is open.

    python training/train.py                         # full run
    python training/train.py --epochs 1 --fraction 0.05 --name smoke   # quick check
"""
import argparse
from pathlib import Path

from ultralytics import YOLO


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data", default=str(Path.home() / "datasets/visdrone_person/data.yaml"))
    parser.add_argument("--model", default="yolov8n.pt", help="start from COCO weights")
    parser.add_argument("--epochs", type=int, default=50)
    # VisDrone persons are only a few pixels tall; 960 keeps them visible
    # while a batch of 8 still fits in 4 GB.
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--fraction", type=float, default=1.0, help="share of train images to use")
    parser.add_argument("--name", default="visdrone_person")
    args = parser.parse_args()

    model = YOLO(args.model)
    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        fraction=args.fraction,
        workers=4,           # WSL has 10 GB RAM
        patience=15,         # stop early when val mAP stops improving
        project="runs/train",
        name=args.name,
        exist_ok=True,
    )
    metrics = model.val(data=args.data, split="test", imgsz=args.imgsz, batch=args.batch)
    print(f"test-dev mAP50={metrics.box.map50:.3f} mAP50-95={metrics.box.map:.3f}")


if __name__ == "__main__":
    main()
