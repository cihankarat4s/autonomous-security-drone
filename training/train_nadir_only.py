"""Fine-tune a person model on nadir images only (no VisDrone in the training set).

A nadir view has no "up", so vertical flips and full rotation are realistic here.
The learning rate is low because the small set would otherwise overwrite what the
starting weights learned. Best epoch is picked on the nadir validation images.

    python training/train_nadir_only.py --model best.pt --data ~/datasets/semantic_drone_person/data.yaml
"""
import argparse

from ultralytics import YOLO


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", required=True, help="starting weights")
    parser.add_argument("--data", required=True)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr0", type=float, default=0.0005)
    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--project", default="runs/train")
    parser.add_argument("--name", default="nadir_only")
    args = parser.parse_args()

    YOLO(args.model).train(
        data=args.data, epochs=args.epochs, imgsz=args.imgsz, batch=args.batch, workers=args.workers,
        optimizer="AdamW", lr0=args.lr0, lrf=0.01, flipud=0.5, degrees=180.0, patience=15,
        project=args.project, name=args.name, exist_ok=True,
    )


if __name__ == "__main__":
    main()
