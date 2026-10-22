"""Score detection models on the VisDrone person test-dev split.

Every model is run with the same data, input sizes and code, so the rows of
the output table can be compared directly. Only the person class is scored:
for the COCO model, predictions of its other 79 classes are filtered out,
as in the companion (detection.classes: [0]).

    python training/evaluate.py --models yolov8n.pt
    python training/evaluate.py --models yolov8n.pt best.pt --imgsz 640 1280
"""
import argparse
import csv
from pathlib import Path

from ultralytics import YOLO

FIELDS = ["model", "imgsz", "precision", "recall", "mAP50", "mAP50-95", "ms_per_image"]


def evaluate(model_path, data, imgsz, batch, split):
    model = YOLO(model_path)
    metrics = model.val(
        data=data, split=split, imgsz=imgsz, batch=batch, classes=[0],
        project="runs/evaluate", name=f"{Path(model_path).stem}_{imgsz}", exist_ok=True,
        plots=False, verbose=False,
    )
    return {
        "model": Path(model_path).name,
        "imgsz": imgsz,
        "precision": round(metrics.box.mp, 3),
        "recall": round(metrics.box.mr, 3),
        "mAP50": round(metrics.box.map50, 3),
        "mAP50-95": round(metrics.box.map, 3),
        "ms_per_image": round(metrics.speed["inference"], 1),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--models", nargs="+", required=True)
    parser.add_argument("--data", default=str(Path.home() / "datasets/visdrone_person/data.yaml"))
    parser.add_argument("--imgsz", type=int, nargs="+", default=[640, 1280])
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--split", default="test")
    parser.add_argument("--out", default="docs/results/visdrone_test.csv",
                        help="rows are appended; an existing row for the same model and size is replaced")
    args = parser.parse_args()

    out = Path(args.out)
    rows = []
    if out.exists():
        with out.open(newline="") as f:
            rows = list(csv.DictReader(f))

    for model_path in args.models:
        for imgsz in args.imgsz:
            row = evaluate(model_path, args.data, imgsz, args.batch, args.split)
            rows = [r for r in rows if (r["model"], str(r["imgsz"])) != (row["model"], str(imgsz))]
            rows.append(row)

    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    print(" | ".join(FIELDS))
    for r in rows:
        print(" | ".join(str(r[k]) for k in FIELDS))
    print(f"Saved to {out}")


if __name__ == "__main__":
    main()
