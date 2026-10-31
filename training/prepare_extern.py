"""Add chosen JLU_DD and Stanford Drone frames to the five-class training set.

Both are nadir videos with thousands of people. training/select_scenes.py picks a few
frames per background; this script rewrites their labels to our classes and packs
them. A cyclist box stays a person: if it were dropped, the frame would teach that a
rider is background.

    python training/prepare_extern.py --jlu chosen_jlu.txt --stanford chosen_stanford.txt \
        --target ~/datasets/extern_multi

Stanford Drone Dataset is CC BY-NC-SA 3.0: non-commercial, cite the ECCV 2016 paper.
"""
import argparse
import shutil
from pathlib import Path

JLU_MAP = {0: 3, 1: 0, 2: 0}         # car, cyclist, pedestrian -> vehicle, person, person
STANFORD_MAP = {0: 3, 1: 0}          # checked by eye: 0 is cars, 1 is people


def remap(text, class_map):
    """Rewrite the class of every row of a YOLO label file."""
    rows = []
    for line in text.splitlines():
        parts = line.split()
        if parts:
            rows.append(" ".join([str(class_map[int(parts[0])]), *parts[1:5]]))
    return "".join(row + "\n" for row in rows)


def add_frames(list_file, class_map, prefix, target):
    boxes = 0
    paths = [Path(p) for p in list_file.read_text().splitlines()]
    for image in paths:
        name = f"{prefix}_{image.stem}"
        shutil.copy(image, target / "images/train" / f"{name}{image.suffix}")
        text = remap((image.parent.parent / "labels" / f"{image.stem}.txt").read_text(), class_map)
        (target / "labels/train" / f"{name}.txt").write_text(text)
        boxes += len(text.splitlines())
    return len(paths), boxes


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--jlu", type=Path, required=True)
    parser.add_argument("--stanford", type=Path, required=True)
    parser.add_argument("--target", type=Path, required=True)
    args = parser.parse_args()

    shutil.rmtree(args.target, ignore_errors=True)
    (args.target / "images/train").mkdir(parents=True)
    (args.target / "labels/train").mkdir(parents=True)
    print("JLU_DD", add_frames(args.jlu, JLU_MAP, "jlu", args.target))
    print("Stanford", add_frames(args.stanford, STANFORD_MAP, "sdd80", args.target))


if __name__ == "__main__":
    main()
