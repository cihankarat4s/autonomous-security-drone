"""Build a five-class copy of VisDrone: person, motor, bicycle, vehicle, dog.

The person-only sets dropped every other class, so a parked motorbike was just
background that happened to look a bit like a person. Here it gets its own class,
and the alarm still listens to "person" only. VisDrone has no dogs; that class comes
from the Semantic Drone Dataset. Riders keep their own person box beside the motor.

Images are hard-linked, only labels are rewritten. Run prepare_visdrone.py first:
it downloads and converts VisDrone.

    python training/prepare_visdrone_multiclass.py --root ~/datasets
"""
import argparse
import os
from pathlib import Path

import yaml

# VisDrone: 0 pedestrian, 1 people, 2 bicycle, 3 car, 4 van, 5 truck, 6 tricycle,
# 7 awning-tricycle, 8 bus, 9 motor
CLASS_MAP = {0: 0, 1: 0, 9: 1, 2: 2, 3: 3, 4: 3, 5: 3, 6: 3, 7: 3, 8: 3}
NAMES = {0: "person", 1: "motor", 2: "bicycle", 3: "vehicle", 4: "dog"}
SPLITS = ("train", "val", "test")


def remap_labels(text):
    """Rewrite the class of every row of a YOLO label file."""
    rows = []
    for line in text.splitlines():
        parts = line.split()
        if parts:
            rows.append(" ".join([str(CLASS_MAP[int(parts[0])]), *parts[1:]]))
    return "".join(row + "\n" for row in rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", type=Path, default=Path.home() / "datasets")
    args = parser.parse_args()

    source, target = args.root / "VisDrone", args.root / "visdrone_multi"
    for split in SPLITS:
        images, labels = target / "images" / split, target / "labels" / split
        images.mkdir(parents=True, exist_ok=True)
        labels.mkdir(parents=True, exist_ok=True)
        for image in (source / "images" / split).glob("*.jpg"):
            link = images / image.name
            if not link.exists():
                os.link(image, link)
            text = (source / "labels" / split / f"{image.stem}.txt").read_text()
            (labels / f"{image.stem}.txt").write_text(remap_labels(text))
    (target / "data.yaml").write_text(yaml.safe_dump(
        {"path": str(target), "train": "images/train", "val": "images/val",
         "test": "images/test", "names": NAMES}, sort_keys=False))
    print(f"Dataset file: {target / 'data.yaml'}")


if __name__ == "__main__":
    main()
