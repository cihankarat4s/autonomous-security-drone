"""Fine-tune COCO yolov8n on the single-class VisDrone "person" set.

The defaults fit a 4 GB GPU (GTX 1650 Ti), but there a batch of 8 at 960 px
already spills into shared memory (~28 min/epoch). Full runs go to Colab
(training/colab_train.ipynb). Do not train locally while Gazebo is open.

    python training/train.py --epochs 1 --fraction 0.05 --name smoke   # quick check
    python training/train.py --imgsz 1280 --batch 16 --project /content/drive/MyDrive/security_drone/runs
    python training/train.py --resume --project ...                    # after a Colab disconnect
"""
import argparse
from pathlib import Path

from ultralytics import YOLO


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data", default=str(Path.home() / "datasets/visdrone_person/data.yaml"))
    parser.add_argument("--model", default="yolov8n.pt", help="start from COCO weights")
    parser.add_argument("--epochs", type=int, default=50)
    # VisDrone persons are only a few pixels tall, so the input stays large.
    # The companion runs at 1280 px; use that when the GPU has the memory.
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--fraction", type=float, default=1.0, help="share of train images to use")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--project", default="runs/train", help="output folder (a Drive path on Colab)")
    parser.add_argument("--name", default="visdrone_person")
    parser.add_argument("--resume", action="store_true", help="continue from <project>/<name>/weights/last.pt")
    args = parser.parse_args()

    run_dir = Path(args.project) / args.name
    if args.resume:
        model = YOLO(run_dir / "weights" / "last.pt")
        model.train(resume=True)
    else:
        model = YOLO(args.model)
        model.train(
            data=args.data,
            epochs=args.epochs,
            imgsz=args.imgsz,
            batch=args.batch,
            fraction=args.fraction,
            workers=args.workers,
            patience=15,         # stop early when val mAP stops improving
            project=args.project,
            name=args.name,
            exist_ok=True,
        )

    # Final score on test-dev, which training never saw, with the best weights.
    best = YOLO(run_dir / "weights" / "best.pt")
    metrics = best.val(
        data=args.data, split="test", imgsz=args.imgsz, batch=args.batch,
        project=args.project, name=f"{args.name}_test", exist_ok=True,
    )
    print(f"test-dev P={metrics.box.mp:.3f} R={metrics.box.mr:.3f} "
          f"mAP50={metrics.box.map50:.3f} mAP50-95={metrics.box.map:.3f}")


if __name__ == "__main__":
    main()
