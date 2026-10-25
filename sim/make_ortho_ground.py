"""Make the ground texture of the orthophoto world (sim/models/ortho_ground).

The ground is a real drone orthophoto from OpenAerialMap (CC-BY 4.0), not a
generated texture: "Perivoj Ane Roje", Split, Croatia, 4 May 2022, 4 cm/px,
provider WebODM / om1s186. It is downloaded on first use (2.4 MB, outside the
repository) and rotated so that the patrol mission fits it: home on the
parking lot, the protected area on the park 60 m away, straight north in the
world. Areas the photo does not cover are filled with its mean colour.

    python sim/make_ortho_ground.py
"""
import math
import urllib.request
from pathlib import Path

import cv2
import numpy as np

URL = ("https://oin-hotosm-temp.s3.amazonaws.com/6278047b3945af0007f9c581/0/"
       "6278047b3945af0007f9c582.tif")
SOURCE = Path.home() / "datasets/ortho/perivoj_ane_roje_2022.tif"
M_PER_PX = 0.0399   # ground sample distance of the source

# Spots in the source image, metres from its top-left corner (right, down).
AREA_CENTRE = (95.0, 35.0)    # grass park with paths, no people in the photo
AREA_NORTH_M = 60.0           # same as the patrol mission
# Home on the parking lot, in the direction of the lot from the park.
_dir = (35.0, 50.0)
_len = math.hypot(*_dir)
HOME = (AREA_CENTRE[0] + AREA_NORTH_M * _dir[0] / _len,
        AREA_CENTRE[1] + AREA_NORTH_M * _dir[1] / _len)

# World area covered by the texture (x east, y north), as in model.sdf.
X_MIN, X_MAX = -100.0, 100.0
Y_MIN, Y_MAX = -70.0, 130.0

OUT = Path(__file__).resolve().parent / "models/ortho_ground/materials/textures/ground.jpg"


def rotation():
    """Angle (rad, counter-clockwise) that turns home -> area into world north."""
    east = AREA_CENTRE[0] - HOME[0]
    north = HOME[1] - AREA_CENTRE[1]   # image rows grow downwards
    return math.atan2(east, north)


def world_to_source(x, y):
    """World metres (home = origin) -> source pixel (column, row)."""
    a = -rotation()
    east = math.cos(a) * x - math.sin(a) * y
    north = math.sin(a) * x + math.cos(a) * y
    return (HOME[0] + east) / M_PER_PX, (HOME[1] - north) / M_PER_PX


def output_to_source_matrix():
    """Affine map from output texture pixels to source pixels (for warpAffine)."""
    # world_to_source is affine in (x, y), and x, y are affine in (col, row).
    def src(col, row):
        return world_to_source(X_MIN + col * M_PER_PX, Y_MAX - row * M_PER_PX)
    o, c, r = src(0, 0), src(1, 0), src(0, 1)
    return np.array([[c[0] - o[0], r[0] - o[0], o[0]],
                     [c[1] - o[1], r[1] - o[1], o[1]]], np.float64)


def download():
    if SOURCE.exists():
        return
    SOURCE.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {URL}")
    urllib.request.urlretrieve(URL, SOURCE)


def main():
    download()
    src = cv2.imread(str(SOURCE))
    valid = (src.max(axis=2) > 0).astype(np.uint8) * 255   # black = no data
    # The photo's edge is a ragged dark seam; drop a few pixels of it.
    valid = cv2.erode(valid, np.ones((9, 9), np.uint8))
    fill = [int(v) for v in cv2.mean(src, mask=valid)[:3]]

    size = (round((X_MAX - X_MIN) / M_PER_PX), round((Y_MAX - Y_MIN) / M_PER_PX))
    m = output_to_source_matrix()
    flags = cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP
    out = cv2.warpAffine(src, m, size, flags=flags, borderValue=fill)
    covered = cv2.warpAffine(valid, m, size, flags=flags, borderValue=0)
    out[covered < 128] = fill

    OUT.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(OUT), out, [cv2.IMWRITE_JPEG_QUALITY, 92])
    print(f"Saved {OUT} ({size[0]} x {size[1]} px), rotated "
          f"{math.degrees(rotation()):.1f} deg, fill colour {fill}")


if __name__ == "__main__":
    main()
