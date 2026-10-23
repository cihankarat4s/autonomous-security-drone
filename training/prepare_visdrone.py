"""Download VisDrone2019-DET and build a single-class "person" copy of it.

VisDrone has 10 classes. Two of them are humans: "pedestrian" (walking or
standing) and "people" (sitting, lying, riding). Both are merged into class 0
"person"; vehicles are dropped. Images are hard-linked (no extra disk space),
only labels are rewritten. A symlinked image folder does not work: Ultralytics
resolves it and then reads the original 10-class labels next to the target.
Images without any person keep an empty label file, so they act as background
(negative) images during training.

    python training/prepare_visdrone.py                     # -> ~/datasets
    python training/prepare_visdrone.py --root /data/sets
    python training/prepare_visdrone.py --negatives ~/datasets/gazebo_negatives/sun_a ...

--negatives adds person-free images (e.g. Gazebo frames of the drone's own
shadow) to the train split with empty labels, as hard negatives.
"""
import argparse
import shutil
from pathlib import Path

import yaml

VISDRONE_PERSON_CLASSES = {0, 1}  # pedestrian, people
SPLITS = ("train", "val", "test")


def to_person_labels(text):
    """Keep pedestrian/people rows of a YOLO label file and set their class to 0."""
    rows = []
    for line in text.splitlines():
        parts = line.split()
        if parts and int(parts[0]) in VISDRONE_PERSON_CLASSES:
            rows.append(" ".join(["0", *parts[1:]]))
    return "".join(row + "\n" for row in rows)


def download_visdrone(root):
    """Let Ultralytics download and convert VisDrone into root/VisDrone."""
    from ultralytics.data.utils import check_det_dataset
    from ultralytics.utils import ROOT

    data = yaml.safe_load((ROOT / "cfg/datasets/VisDrone.yaml").read_text())
    data["path"] = str(root / "VisDrone")
    data_file = root / "VisDrone.yaml"
    data_file.write_text(yaml.safe_dump(data, sort_keys=False))
    check_det_dataset(str(data_file), autodownload=True)
    return root / "VisDrone"


def build_person_dataset(source, target):
    stats = {}
    for split in SPLITS:
        src_images = source / "images" / split
        dst_images = target / "images" / split
        dst_labels = target / "labels" / split
        dst_labels.mkdir(parents=True, exist_ok=True)
        if dst_images.is_symlink():
            dst_images.unlink()  # left by an earlier version of this script
        dst_images.mkdir(parents=True, exist_ok=True)

        images = boxes = empty = 0
        for image in sorted(src_images.glob("*.jpg")):
            linked = dst_images / image.name
            if not linked.exists():
                linked.hardlink_to(image)
            src_label = source / "labels" / split / (image.stem + ".txt")
            text = src_label.read_text() if src_label.exists() else ""
            labels = to_person_labels(text)
            (dst_labels / src_label.name).write_text(labels)
            images += 1
            boxes += labels.count("\n")
            empty += labels == ""
        stats[split] = (images, boxes, empty)

    data = {
        "path": str(target),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "names": {0: "person"},
    }
    (target / "data.yaml").write_text(yaml.safe_dump(data, sort_keys=False))
    return stats


def add_negatives(target, folders):
    """Copy person-free images into the train split with empty label files."""
    added = 0
    for folder in folders:
        for image in sorted(folder.glob("*.jpg")):
            # Copied, not linked: on Colab the source is on Drive, another filesystem.
            shutil.copy2(image, target / "images" / "train" / image.name)
            (target / "labels" / "train" / (image.stem + ".txt")).write_text("")
            added += 1
    return added


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", type=Path, default=Path.home() / "datasets")
    parser.add_argument("--negatives", type=Path, nargs="*", default=[],
                        help="folders of person-free images to add to train")
    args = parser.parse_args()

    root = args.root.expanduser()
    root.mkdir(parents=True, exist_ok=True)
    source = download_visdrone(root)
    target = root / ("visdrone_person_neg" if args.negatives else "visdrone_person")
    stats = build_person_dataset(source, target)

    for split, (images, boxes, empty) in stats.items():
        print(f"{split:5s}: {images} images, {boxes} person boxes, {empty} without person")
    if args.negatives:
        added = add_negatives(target, [f.expanduser() for f in args.negatives])
        print(f"train: added {added} hard negatives")
    print(f"Dataset file: {target / 'data.yaml'}")


if __name__ == "__main__":
    main()
