"""Split detections on the orthophoto flight into the photo's real person and false positives.

The orthophoto (sim/make_ortho_ground.py) contains one real pedestrian, a
dark figure on the road west of the park. Boxes on that figure are correct
detections, not false positives. Each box is compared with the figure as it
appears in the ground texture: the frame around the box is scaled to the
texture's 4 cm/px from the altitude in the file name and matched at every
rotation (the drone's heading is unknown). A box counts as the pedestrian if
it matches and is small; car bodies in shadow also match.

This is a sorting aid, not ground truth: check the crops in --save-dir by
eye. On ortho_c one VisDrone box at 0.26 on a dark roof gap matched (0.72)
and was moved to the false positives by hand (docs/results/ortho_negatives.csv).

    python training/split_photo_person.py --models a.pt b.pt \
        --folder ~/datasets/gazebo_negatives/ortho_c --conf 0.25 --save-dir logs/fp_split
"""
import argparse
import math
import re
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

TEXTURE = Path(__file__).resolve().parents[1] / "sim/models/ortho_ground/materials/textures/ground.jpg"
TEXTURE_M_PER_PX = 0.0399
# The pedestrian in world metres (home = origin), found by eye in the texture.
PERSON_XY = (-76.9, 60.4)
PATCH_M = 2.4                     # compared window around the box centre
# Camera after the 1.8x zoom: tan(half horizontal FOV), image width.
TAN_HALF_FOV = math.tan(1.0) / 1.8
IMAGE_W = 640
MATCH = 0.6                       # correlation above this = the pedestrian
# Shaded car bodies also correlate with the dark figure (0.6-0.7 for COCO's
# boxes), but a box on the pedestrian is small.
MAX_BOX_M = 1.5


def person_template(texture):
    col = (PERSON_XY[0] + 100) / TEXTURE_M_PER_PX
    row = (130 - PERSON_XY[1]) / TEXTURE_M_PER_PX
    half = round(PATCH_M / TEXTURE_M_PER_PX / 2)
    row, col = round(row), round(col)
    patch = texture[row - half:row + half, col - half:col + half]
    return cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY).astype(np.float32)


def metres_per_px(alt_m):
    return 2 * alt_m * TAN_HALF_FOV / IMAGE_W


def box_patch(frame, box, alt_m, size):
    """Frame window of PATCH_M around the box centre, at texture resolution."""
    m_per_px = metres_per_px(alt_m)
    half = int(PATCH_M / m_per_px / 2) + 1
    cx, cy = int((box[0] + box[2]) / 2), int((box[1] + box[3]) / 2)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    padded = cv2.copyMakeBorder(gray, half, half, half, half, cv2.BORDER_REPLICATE)
    window = padded[cy:cy + 2 * half, cx:cx + 2 * half]
    return cv2.resize(window, (size, size), interpolation=cv2.INTER_AREA).astype(np.float32)


def similarity(patch, template):
    """Best normalised correlation over rotations in 10 degree steps."""
    size = template.shape[0]
    centre = (size / 2, size / 2)
    best = -1.0
    for angle in range(0, 360, 10):
        rot = cv2.warpAffine(patch, cv2.getRotationMatrix2D(centre, angle, 1), (size, size),
                             borderMode=cv2.BORDER_REPLICATE)
        best = max(best, float(cv2.matchTemplate(rot, template, cv2.TM_CCOEFF_NORMED)[0, 0]))
    return best


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--models", nargs="+", required=True)
    parser.add_argument("--folder", type=Path, required=True)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument("--save-dir", type=Path, help="crops of each box, sorted by verdict")
    args = parser.parse_args()

    template = person_template(cv2.imread(str(TEXTURE)))
    images = sorted(args.folder.expanduser().glob("*.jpg"))
    print(f"conf {args.conf}, {len(images)} images")
    print("model | frames with the real person | frames with a false positive | false-positive boxes")
    for model_path in args.models:
        model = YOLO(model_path)
        person_frames = fp_frames = fp_boxes = 0
        for image in images:
            frame = cv2.imread(str(image))
            alt = float(re.search(r"_alt([\d.]+)_", image.name).group(1))
            result = model.predict(frame, conf=args.conf, imgsz=args.imgsz, classes=[0],
                                   verbose=False)[0]
            on_person = fp = 0
            for box, conf in zip(result.boxes.xyxy.tolist(), result.boxes.conf.tolist()):
                score = similarity(box_patch(frame, box, alt, template.shape[0]), template)
                size_m = max(box[2] - box[0], box[3] - box[1]) * metres_per_px(alt)
                verdict = "person" if score > MATCH and size_m < MAX_BOX_M else "fp"
                on_person += verdict == "person"
                fp += verdict == "fp"
                if args.save_dir:
                    out = args.save_dir / Path(model_path).stem / verdict
                    out.mkdir(parents=True, exist_ok=True)
                    x1, y1, x2, y2 = (int(v) for v in box)
                    crop = frame[max(0, y1 - 40):y2 + 40, max(0, x1 - 40):x2 + 40]
                    cv2.imwrite(str(out / f"{image.stem}_{conf:.2f}_{size_m:.1f}m_{score:.2f}.png"),
                                cv2.resize(crop, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC))
            person_frames += on_person > 0
            fp_frames += fp > 0
            fp_boxes += fp
        n = len(images)
        print(f"{Path(model_path).name} | {person_frames} | {fp_frames} ({fp_frames / n:.1%}) | {fp_boxes}")


if __name__ == "__main__":
    main()
