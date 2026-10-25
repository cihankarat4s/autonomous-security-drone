"""Measure CPU detection latency with PyTorch and NCNN.

The Raspberry Pi is not available, so this runs on the PC's CPU. Run it pinned
to 4 cores (the Pi 5 has 4) so the backends and models compare fairly:

    taskset -c 0-3 python training/benchmark_cpu.py \
        --models yolov8n_visdrone_person_neg.pt yolo26n_visdrone_person_neg.pt \
        --images ~/datasets/gazebo_urban/patrol_2 --glob '*_seq3.jpg'

The numbers are relative: a Pi 5 is several times slower than a laptop CPU,
so they show which model/backend/input size is faster, not Pi frame rates.
Latency is per frame, preprocessing + inference + postprocessing (as reported
by Ultralytics), median over the measured frames after a warm-up.
"""
import argparse
import csv
import statistics
from pathlib import Path

import torch
from ultralytics import YOLO

FIELDS = ["model", "backend", "imgsz", "frames", "median_ms", "fps"]


def median_latency(model, images, imgsz, warmup):
    for image in images[:warmup]:
        model.predict(image, imgsz=imgsz, device="cpu", verbose=False)
    times = []
    for image in images:
        speed = model.predict(image, imgsz=imgsz, device="cpu", verbose=False)[0].speed
        times.append(speed["preprocess"] + speed["inference"] + speed["postprocess"])
    return statistics.median(times)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--models", nargs="+", required=True)
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--glob", default="*.jpg")
    parser.add_argument("--frames", type=int, default=30)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--imgsz", type=int, nargs="+", default=[640, 960, 1280])
    parser.add_argument("--backends", nargs="+", default=["pytorch", "ncnn"])
    parser.add_argument("--threads", type=int, default=4, help="PyTorch CPU threads")
    parser.add_argument("--out", default="docs/results/cpu_speed.csv")
    args = parser.parse_args()

    torch.set_num_threads(args.threads)
    images = [str(p) for p in sorted(args.images.expanduser().glob(args.glob))[:args.frames]]
    rows = []
    for model_path in args.models:
        for imgsz in args.imgsz:
            for backend in args.backends:
                if backend == "pytorch":
                    model = YOLO(model_path)
                else:
                    # NCNN graphs have a fixed input size, so export once per size.
                    model = YOLO(YOLO(model_path).export(format="ncnn", imgsz=imgsz))
                ms = median_latency(model, images, imgsz, args.warmup)
                rows.append({"model": Path(model_path).name, "backend": backend,
                             "imgsz": imgsz, "frames": len(images),
                             "median_ms": round(ms, 1), "fps": round(1000 / ms, 1)})
                print(" | ".join(str(rows[-1][k]) for k in FIELDS), flush=True)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved to {out}")


if __name__ == "__main__":
    main()
