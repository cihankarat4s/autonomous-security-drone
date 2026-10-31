"""Group frames of a video-like dataset by background and keep a few from each group.

Video datasets repeat one background hundreds of times, and a model then memorises
that background. Frames are compared as small gray thumbnails, which the moving
people barely change; a frame joins the first group it is close to. Up to --per-scene
frames, evenly spaced, are kept from each group.

    python training/select_scenes.py ~/datasets/extern/jlu_dd/raw --per-scene 30 --sheet logs/jlu.jpg
"""
import argparse
from pathlib import Path

import cv2
import numpy as np

THUMB = 48
SAME_SCENE = 14.0                # mean gray difference of two thumbnails


def thumbnail(path):
    return cv2.resize(cv2.imread(str(path), cv2.IMREAD_GRAYSCALE), (THUMB, THUMB), interpolation=cv2.INTER_AREA).astype(np.float32)


def group(paths, same_scene=SAME_SCENE):
    """Lists of paths with a similar background; the first frame stands for a group."""
    groups, centers = [], []
    for path in paths:
        t = thumbnail(path)
        for g, center in zip(groups, centers):
            if np.abs(t - center).mean() < same_scene:
                g.append(path)
                break
        else:
            groups.append([path])
            centers.append(t)
    return groups


def pick(groups, per_scene):
    """Evenly spaced frames from each group."""
    chosen = []
    for g in groups:
        step = max(len(g) / per_scene, 1)
        chosen += [g[int(i * step)] for i in range(min(per_scene, len(g)))]
    return chosen


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("root", type=Path, help="folder holding */images/*")
    parser.add_argument("--per-scene", type=int, default=30)
    parser.add_argument("--same", type=float, default=SAME_SCENE, help="thumbnail difference within a scene")
    parser.add_argument("--list", type=Path, help="write the chosen paths here")
    parser.add_argument("--sheet", type=Path, help="one frame of each group, for a look")
    args = parser.parse_args()

    paths = sorted(args.root.glob("*/images/*"), key=lambda p: p.name)
    groups = sorted(group(paths, args.same), key=len, reverse=True)
    chosen = pick(groups, args.per_scene)
    print(f"{len(paths)} frames, {len(groups)} groups, sizes {[len(g) for g in groups[:30]]}, chosen {len(chosen)}")
    if args.list:
        args.list.write_text("".join(f"{p}\n" for p in chosen))
    if args.sheet:
        tiles = [cv2.resize(cv2.imread(str(g[0])), (160, 160)) for g in groups[:60]]
        while len(tiles) % 10:
            tiles.append(np.zeros((160, 160, 3), np.uint8))
        cv2.imwrite(str(args.sheet), np.vstack([np.hstack(tiles[i:i + 10]) for i in range(0, len(tiles), 10)]))


if __name__ == "__main__":
    main()
