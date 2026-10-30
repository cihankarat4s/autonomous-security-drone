"""Turn the Semantic Drone Dataset (TU Graz) into YOLO person data.

These are real nadir images from 5-30 m, the view the drone's camera has and
VisDrone mostly lacks. The dataset allows trained models but not redistribution,
so it stays out of the repository.

Images are numbered in flight order, so neighbours show the same house. Blocks of
consecutive images are therefore held out, not single images: every fourth block
(~100 images) is the test set and is never trained on, and one block in eight
(~60 images) is validation, so the best epoch is not picked on VisDrone alone.

Training images are cut into overlapping tiles as well as used whole, which gives
two person sizes. Validation and test images stay whole, like a camera frame.

    python training/prepare_semantic_drone.py --source ~/datasets/semantic_drone/semantic_drone_dataset \
        --target ~/datasets/semantic_drone_person --base ~/datasets/visdrone_person/data.yaml

--base merges the set into another dataset's train and val lists. It writes
data.yaml (train and val on both) and test.yaml (the held-out images only).

--multiclass also labels vehicles, bicycles and dogs. The dataset has person boxes
but only painted masks for these, so their boxes are the connected regions of the
mask. Without them a frame would teach the model that a car is background.
"""
import argparse
import csv
import random
import xml.etree.ElementTree as ET
from pathlib import Path

import cv2
import numpy as np
import yaml

BLOCK = 20                       # consecutive images, about one house
TILE_W, TILE_H = 3000, 2000      # px of the 6000x4000 original
OUT_W = 1280                     # the companion's inference size
MIN_VISIBLE = 0.5                # drop a box when a tile cuts off more than this
EMPTY_TILE_RATE = 0.25           # share of person-free tiles kept as background
NAMES = {0: "person", 1: "motor", 2: "bicycle", 3: "vehicle", 4: "dog"}
MASK_CLASSES = {"car": 3, "bicycle": 2, "dog": 4}   # semantic class -> our class
MASK_SCALE = 4                   # masks are read at 1/4 size
MIN_MASK_AREA = 4000             # px of the original; smaller regions are label noise


def read_boxes(xml_path):
    """Person boxes (x1, y1, x2, y2) of a LabelMe file."""
    boxes = []
    for obj in ET.parse(xml_path).getroot().iter("object"):
        if obj.findtext("name") != "person" or obj.findtext("deleted") == "1":
            continue
        xs = [float(p.findtext("x")) for p in obj.iter("pt")]
        ys = [float(p.findtext("y")) for p in obj.iter("pt")]
        boxes.append((min(xs), min(ys), max(xs), max(ys)))
    return boxes


def mask_colors(source):
    """Semantic class name -> BGR colour of the label images."""
    rows = list(csv.reader(next(source.rglob("*.csv")).open()))[1:]
    return {r[0].strip(): (int(r[3]), int(r[2]), int(r[1])) for r in rows}


def mask_boxes(label_image, colors):
    """{our class: [(x1, y1, x2, y2), ...]} from a semantic label image."""
    image = cv2.imread(str(label_image))[::MASK_SCALE, ::MASK_SCALE]
    found = {}
    for name, cls in MASK_CLASSES.items():
        mask = np.all(image == colors[name], axis=2).astype(np.uint8)
        if not mask.any():
            continue
        # Close small gaps: the wheels and frame of a bicycle are separate regions.
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
        count, _, stats, _ = cv2.connectedComponentsWithStats(mask)
        boxes = [(x * MASK_SCALE, y * MASK_SCALE, (x + w) * MASK_SCALE, (y + h) * MASK_SCALE)
                 for x, y, w, h, area in stats[1:] if area * MASK_SCALE ** 2 >= MIN_MASK_AREA]
        if boxes:
            found[cls] = boxes
    return found


def split_blocks(stems):
    """Sorted image ids -> {train, val, test} of whole blocks."""
    splits = {"train": [], "val": [], "test": []}
    for i in range(0, len(stems), BLOCK):
        block = i // BLOCK
        split = "test" if block % 4 == 3 else "val" if block % 8 == 1 else "train"
        splits[split].extend(stems[i:i + BLOCK])
    return splits


def tile_origins(width, height):
    """Top-left corners of half-overlapping tiles covering the image."""
    return [(x, y) for y in range(0, height - TILE_H + 1, TILE_H // 2)
            for x in range(0, width - TILE_W + 1, TILE_W // 2)]


def crop_labels(boxes, x0, y0, w, h, cls=0):
    """YOLO label rows of the boxes inside a crop."""
    rows = []
    for x1, y1, x2, y2 in boxes:
        cx1, cy1 = max(x1, x0), max(y1, y0)
        cx2, cy2 = min(x2, x0 + w), min(y2, y0 + h)
        if cx2 <= cx1 or cy2 <= cy1:
            continue
        if (cx2 - cx1) * (cy2 - cy1) < MIN_VISIBLE * (x2 - x1) * (y2 - y1):
            continue
        rows.append(f"{cls} {((cx1 + cx2) / 2 - x0) / w:.6f} {((cy1 + cy2) / 2 - y0) / h:.6f} "
                    f"{(cx2 - cx1) / w:.6f} {(cy2 - cy1) / h:.6f}\n")
    return rows


def save(image, rows, stem, split, target):
    height = round(image.shape[0] * OUT_W / image.shape[1])
    small = cv2.resize(image, (OUT_W, height), interpolation=cv2.INTER_AREA)
    cv2.imwrite(str(target / "images" / split / f"{stem}.jpg"), small, [cv2.IMWRITE_JPEG_QUALITY, 92])
    (target / "labels" / split / f"{stem}.txt").write_text("".join(rows))


def build(source, target, seed=0, multiclass=False):
    root = source / "training_set"
    stems = sorted(p.stem for p in (root / "images").glob("*.jpg"))
    splits = split_blocks(stems)
    colors = mask_colors(source) if multiclass else None
    rng = random.Random(seed)
    for split in splits:
        (target / "images" / split).mkdir(parents=True, exist_ok=True)
        (target / "labels" / split).mkdir(parents=True, exist_ok=True)

    counts = {f"{split}{kind}": 0 for split in splits for kind in ("", "_boxes")}
    for split, ids in splits.items():
        for stem in ids:
            image = cv2.imread(str(root / "images" / f"{stem}.jpg"))
            h, w = image.shape[:2]
            boxes = read_boxes(root / "gt/bounding_box/label_me_xml" / f"{stem}.xml")
            labelled = {0: boxes}
            if multiclass:
                labelled.update(mask_boxes(root / "gt/semantic/label_images" / f"{stem}.png", colors))
            rows = [r for c, b in labelled.items() for r in crop_labels(b, 0, 0, w, h, c)]
            save(image, rows, f"sdd_{stem}", split, target)
            counts[split] += 1
            counts[f"{split}_boxes"] += len(rows)
            if split != "train":
                continue
            for x0, y0 in tile_origins(w, h):
                rows = [r for c, b in labelled.items() for r in crop_labels(b, x0, y0, TILE_W, TILE_H, c)]
                if not rows and rng.random() > EMPTY_TILE_RATE:
                    continue
                save(image[y0:y0 + TILE_H, x0:x0 + TILE_W], rows, f"sdd_{stem}_{x0}_{y0}", split, target)
                counts[split] += 1
                counts[f"{split}_boxes"] += len(rows)
    return counts


def write_yamls(target, base_yaml, multiclass=False):
    """data.yaml (base + this set) and test.yaml (held-out nadir images only)."""
    names = NAMES if multiclass else {0: "person"}
    train, val = [str(target / "images/train")], [str(target / "images/val")]
    if base_yaml:
        base = yaml.safe_load(base_yaml.read_text())
        base_root = Path(base["path"])
        train.insert(0, str(base_root / base["train"]))
        val.insert(0, str(base_root / base["val"]))
    data = {"path": str(target), "train": train, "val": val,
            "test": str(target / "images/test"), "names": names}
    (target / "data.yaml").write_text(yaml.safe_dump(data, sort_keys=False))
    test = {"path": str(target), "train": "images/train", "val": "images/test",
            "test": "images/test", "names": names}
    (target / "test.yaml").write_text(yaml.safe_dump(test, sort_keys=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--source", type=Path, required=True, help="the unzipped semantic_drone_dataset folder")
    parser.add_argument("--target", type=Path, required=True)
    parser.add_argument("--base", type=Path, help="data.yaml of the dataset to merge with")
    parser.add_argument("--multiclass", action="store_true", help="also label vehicle, bicycle and dog")
    args = parser.parse_args()
    print(build(args.source, args.target, multiclass=args.multiclass))
    write_yamls(args.target, args.base, args.multiclass)


if __name__ == "__main__":
    main()
