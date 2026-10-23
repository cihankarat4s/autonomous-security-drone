"""Generate the ground texture of the urban world (sim/models/urban_ground).

One large image covers the whole area at 4 cm/px, so the top-down camera sees
real texture (grass variation, dirt, asphalt, paving) instead of a flat colour
or an obviously repeating tile. The seed is fixed, so the image is the same on
every machine and does not need to be in git.

    python sim/make_ground_texture.py
"""
from pathlib import Path

import cv2
import numpy as np

# World area covered by the ground, metres (x east, y north), and resolution.
X_MIN, X_MAX = -80.0, 80.0
Y_MIN, Y_MAX = -90.0, 110.0
M_PER_PX = 0.04
SEED = 2026

OUT = Path(__file__).resolve().parent / "models/urban_ground/materials/textures/ground.jpg"


def px(x, y):
    """World metres -> image pixel (column, row); north is up in the image."""
    return int(round((x - X_MIN) / M_PER_PX)), int(round((Y_MAX - y) / M_PER_PX))


def noise(rng, h, w, scales=(8, 32, 128, 512), weights=(0.4, 0.3, 0.2, 0.1)):
    """Fractal value noise in [0, 1]: random grids upsampled and summed."""
    out = np.zeros((h, w), np.float32)
    for cells, weight in zip(scales, weights):
        grid = rng.random((max(2, h * cells // w), cells)).astype(np.float32)
        out += weight * cv2.resize(grid, (w, h), interpolation=cv2.INTER_CUBIC)
    out -= out.min()
    return out / out.max()


def blend(img, mask, color, n=None, amount=0.25):
    """Paint color (BGR) where mask > 0, with brightness varied by noise n."""
    color = np.array(color, np.float32)
    shade = 1.0 if n is None else (1 - amount / 2 + amount * n)[..., None]
    m = (mask.astype(np.float32) / 255.0)[..., None]
    img[:] = img * (1 - m) + color * shade * m


def rect(mask, x0, y0, x1, y1):
    c0, r0 = px(x0, y1)
    c1, r1 = px(x1, y0)
    cv2.rectangle(mask, (c0, r0), (c1, r1), 255, -1)


def main():
    rng = np.random.default_rng(SEED)
    w, h = px(X_MAX, Y_MIN)
    img = np.zeros((h, w, 3), np.float32)

    # Grass: green with patchy tone and colour shifts.
    n1, n2 = noise(rng, h, w), noise(rng, h, w, scales=(64, 256, 1024), weights=(0.3, 0.4, 0.3))
    img[:] = np.array([60, 115, 85], np.float32)
    img *= (0.75 + 0.5 * n1)[..., None]
    img[..., 2] += 25 * (n2 - 0.5)        # some patches drier (redder)

    # Dirt patches where a low-frequency noise is high.
    dirt = ((noise(rng, h, w, scales=(6, 20), weights=(0.6, 0.4)) > 0.72) * 255).astype(np.uint8)
    dirt = cv2.GaussianBlur(dirt, (0, 0), 25)
    blend(img, dirt, (70, 100, 125), n1, 0.4)

    def paint(shape_fn, color, amount=0.2):
        mask = np.zeros((h, w), np.uint8)
        shape_fn(mask)
        mask = cv2.GaussianBlur(mask, (0, 0), 1.5)
        blend(img, mask, color, n2, amount)

    # North-south road east of the patrol line: sidewalks, asphalt, markings.
    paint(lambda m: rect(m, 9.5, Y_MIN, 21.5, Y_MAX), (165, 168, 170))
    paint(lambda m: rect(m, 11.5, Y_MIN, 19.5, Y_MAX), (72, 74, 76), 0.15)
    for y in np.arange(Y_MIN, Y_MAX, 6.0):
        paint(lambda m, y=y: rect(m, 15.4, y, 15.6, y + 3), (225, 230, 230))
    # East-west road in the south (training zone).
    paint(lambda m: rect(m, X_MIN, -27.5, 9.5, -19.5), (72, 74, 76), 0.15)
    for x in np.arange(X_MIN, 9.5, 6.0):
        paint(lambda m, x=x: rect(m, x, -23.6, x + 3, -23.4), (225, 230, 230))

    # Paved plaza around the protected area, tile joints, red border.
    paint(lambda m: rect(m, -14, 46, 14, 74), (150, 160, 170), 0.2)
    joints = np.zeros((h, w), np.uint8)
    for v in np.arange(-14, 14.1, 1.0):
        cv2.line(joints, px(v, 46), px(v, 74), 255, 1)
        cv2.line(joints, px(-14, v + 60), px(14, v + 60), 255, 1)
    blend(img, joints, (115, 120, 125))
    border = np.zeros((h, w), np.uint8)
    rect(border, -10, 49.7, 10, 70.3)
    inner = np.zeros((h, w), np.uint8)
    rect(inner, -9.7, 50, 9.7, 70)
    blend(img, cv2.subtract(border, inner), (40, 40, 190))

    # Parking lot in the training zone with bay lines.
    paint(lambda m: rect(m, -65, -70, -30, -45), (78, 80, 82), 0.15)
    for x in np.arange(-64, -30, 2.7):
        paint(lambda m, x=x: rect(m, x, -70, x + 0.12, -64), (220, 225, 225))
        paint(lambda m, x=x: rect(m, x, -51, x + 0.12, -45), (220, 225, 225))

    # Gravel footpath from home to the plaza, and the home pad.
    path = np.zeros((h, w), np.uint8)
    pts = np.array([px(0, 4), px(-4, 20), px(-6, 32), px(-2, 46)], np.int32)
    cv2.polylines(path, [pts], False, 255, int(2.0 / M_PER_PX))
    blend(img, cv2.GaussianBlur(path, (0, 0), 3), (120, 140, 150), n1, 0.4)
    paint(lambda m: rect(m, -3, -3, 3, 3), (175, 180, 182))

    # Fine grain everywhere so close-up frames are not perfectly smooth.
    img += rng.normal(0, 4, img.shape[:2]).astype(np.float32)[..., None]

    OUT.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(OUT), np.clip(img, 0, 255).astype(np.uint8), [cv2.IMWRITE_JPEG_QUALITY, 90])
    print(f"Saved {OUT} ({w}x{h} px, {M_PER_PX * 100:.0f} cm/px)")


if __name__ == "__main__":
    main()
