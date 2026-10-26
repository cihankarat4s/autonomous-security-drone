"""Generate the neighborhood patrol world: ground texture and world SDF.

A suburban neighborhood north of home, laid out on a street grid:

    E-W streets at y = 30, 90, 150 and N-S streets at x = -60, 0, 60
    (8 m asphalt, curbs, 2.5 m sidewalks), 24 houses on lots facing the
    streets, a park in the north-east block, and a paved plaza with a fire
    station and a landing pad at home (0, 0).

The same layout drives both outputs, so the houses, driveways and trees in
the SDF line up with the lawns, driveways and paths painted on the ground:

    sim/models/neighborhood_ground/materials/textures/ground.jpg  (not in git)
    sim/worlds/security_patrol_neighborhood.sdf

Models are Gazebo Fuel models (OpenRobotics), downloaded on first use. The
seed is fixed, so every run produces the same neighborhood.

    python sim/make_neighborhood.py
"""
import math
import random
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

SIM = Path(__file__).resolve().parent
TEXTURE = SIM / "models/neighborhood_ground/materials/textures/ground.jpg"
WORLD = SIM / "worlds/security_patrol_neighborhood.sdf"
SEED = 2026

# Ground texture area (x east, y north), as in the ground model.sdf.
X_MIN, X_MAX = -100.0, 100.0
Y_MIN, Y_MAX = -30.0, 195.0
M_PER_PX = 0.04

STREETS_Y = (30.0, 90.0, 150.0)      # east-west streets, the full width
STREETS_X = (-60.0, 0.0, 60.0)       # north-south streets, from y = 30 north
HALF_ROAD = 4.0
CURB = 0.2
SIDEWALK_OUT = HALF_ROAD + CURB + 2.5     # outer edge of the sidewalk
SETBACK = 5.0                             # front lawn between sidewalk and house
PARK = (0.0, 90.0, 60.0, 150.0)           # x0, y0, x1, y1 of the park block
PLAZA = (-45.0, -12.0, 45.0, 30.0 - SIDEWALK_OUT)  # up to the first street's sidewalk

FUEL = "https://fuel.gazebosim.org/1.0/OpenRobotics/models/"

# Footprints in metres (model frame), measured from the meshes: x min/max, y min/max.
# The houses are Fuel's House 1-3 wrapped in sim/models/neighborhood_house_*
# with PBR materials (Fuel's Ogre 1 material scripts render black in Harmonic).
HOUSES = {
    "model://neighborhood_house_1": (-8.64, 7.85, -6.84, 6.09),
    "model://neighborhood_house_2": (-5.15, 7.56, -4.11, 5.66),
    "model://neighborhood_house_3": (-7.08, 6.09, -4.37, 4.34),
}
FIRE_STATION = (-10.74, 10.73, -5.74, 6.12)
# Cars and the yaw that turns each model's long axis onto its x axis.
# ("Hatchback blue" is left out: its material file points at the red one's texture.)
CARS = {"SUV": 0.0, "Prius Hybrid": math.pi / 2, "Hatchback": 0.0,
        "Hatchback red": 0.0, "Pickup": 0.0}
PEOPLE = ("Standing person", "Casual female", "Walking person")


@dataclass
class Item:
    model: str
    x: float
    y: float
    yaw: float = 0.0


# --------------------------------------------------------------------------
# Layout


def rotate(x, y, yaw):
    return x * math.cos(yaw) - y * math.sin(yaw), x * math.sin(yaw) + y * math.cos(yaw)


def place_footprint(model, box, cx, cy, yaw):
    """Model origin that puts the footprint's centre at (cx, cy)."""
    ox, oy = rotate((box[0] + box[1]) / 2, (box[2] + box[3]) / 2, yaw)
    return Item(model, cx - ox, cy - oy, yaw)


@dataclass
class Lot:
    """A house lot: the street it faces and where along that street."""
    street_axis: str      # "x": the street runs east-west; "y": north-south
    street_pos: float     # y of an E-W street, or x of an N-S street
    along: float          # position along the street
    side: int             # +1: the lot is on the north/east side of the street


def lots():
    out = []
    blocks = [(-60.0, 30.0), (0.0, 30.0), (-60.0, 90.0)]       # inner blocks, (0, 90) is the park
    for x0, y0 in blocks:
        for along in (x0 + 18.35, x0 + 41.65):
            out.append(Lot("x", y0, along, +1))
            out.append(Lot("x", y0 + 60.0, along, -1))
    for x0 in (-60.0, 0.0):                                      # north of the last street
        for along in (x0 + 18.35, x0 + 41.65):
            out.append(Lot("x", 150.0, along, +1))
    for y0 in (30.0, 90.0):                                      # west and east edges
        for along in (y0 + 18.35, y0 + 41.65):
            out.append(Lot("y", -60.0, along, -1))
            out.append(Lot("y", 60.0, along, +1))
    return out


def lot_frame(lot):
    """Unit vectors along the street and away from it (toward the lot)."""
    if lot.street_axis == "x":
        return (1.0, 0.0), (0.0, float(lot.side))
    return (0.0, 1.0), (float(lot.side), 0.0)


def lot_point(lot, along_offset, depth):
    """World point at along_offset along the street and depth from its centreline."""
    t, n = lot_frame(lot)
    if lot.street_axis == "x":
        base = (lot.along, lot.street_pos)
    else:
        base = (lot.street_pos, lot.along)
    return (base[0] + t[0] * along_offset + n[0] * depth,
            base[1] + t[1] * along_offset + n[1] * depth)


def build_layout(rng):
    houses, driveways, items = [], [], []
    for lot in lots():
        model = rng.choice(sorted(HOUSES))
        box = HOUSES[model]
        width, depth = box[1] - box[0], box[3] - box[2]
        t, n = lot_frame(lot)
        # Long side along the street; flip half of them for variety.
        yaw = math.atan2(t[1], t[0]) + (math.pi if rng.random() < 0.5 else 0.0)
        side = rng.choice((-1, 1))               # driveway left or right of the house
        cx, cy = lot_point(lot, -side * 2.0, SIDEWALK_OUT + SETBACK + depth / 2)
        houses.append(place_footprint(model, box, cx, cy, yaw))
        drive_along = -side * 2.0 + side * (width / 2 + 2.3)
        driveways.append((lot, drive_along, SIDEWALK_OUT + SETBACK + min(depth, 7.0)))
        street_yaw = math.atan2(n[1], n[0])      # cars in driveways face the street
        if rng.random() < 0.55:
            x, y = lot_point(lot, drive_along, SIDEWALK_OUT + 3.2)
            car = rng.choice(sorted(CARS))
            items.append(Item(FUEL + car, x, y, street_yaw + CARS[car] + rng.choice((0, math.pi))))
        if rng.random() < 0.5:
            x, y = lot_point(lot, drive_along + side * 2.2, SIDEWALK_OUT + 0.4)
            items.append(Item(FUEL + "Mailbox", x, y, street_yaw))
        if rng.random() < 0.35:
            x, y = lot_point(lot, drive_along - side * 2.2, HALF_ROAD + CURB + 0.4)
            items.append(Item(FUEL + "TrashBin", x, y, rng.uniform(0, math.pi)))
        # A pine in the front lawn on the far side from the driveway.
        if rng.random() < 0.7:
            x, y = lot_point(lot, -side * (width / 2 + 1.0), SIDEWALK_OUT + 2.5)
            items.append(Item(FUEL + "Pine Tree", x, y, rng.uniform(0, math.pi)))
        # An oak in the back yard.
        if rng.random() < 0.6:
            x, y = lot_point(lot, rng.uniform(-6, 6), SIDEWALK_OUT + SETBACK + depth + 4.5)
            items.append(Item(FUEL + "Oak tree", x, y, rng.uniform(0, 2 * math.pi)))
    return houses, driveways, items


def street_items(rng):
    items = []
    # Parked cars along the curbs, away from intersections.
    for y in STREETS_Y:
        for x0 in (-95.0, -55.0, 5.0, 65.0):
            if rng.random() < 0.8:
                car = rng.choice(sorted(CARS))
                side = rng.choice((-1, 1))
                x = x0 + rng.uniform(4, 26)
                items.append(Item(FUEL + car, x, y + side * 2.9, CARS[car] + (0 if side < 0 else math.pi)))
    for x in STREETS_X:
        for y0 in (35.0, 95.0, 155.0):
            if rng.random() < 0.7:
                car = rng.choice(sorted(CARS))
                side = rng.choice((-1, 1))
                y = y0 + rng.uniform(8, 40) if y0 < 150 else y0 + rng.uniform(8, 30)
                items.append(Item(FUEL + car, x + side * 2.9, y,
                                  math.pi / 2 + CARS[car] + (0 if side > 0 else math.pi)))
    # Street lamps every 40 m, arm over the road, alternating sides.
    for y in STREETS_Y:
        for i, x in enumerate(np.arange(-80.0, 81.0, 40.0)):
            side = 1 if i % 2 else -1
            items.append(lamp(x + 10.0, y + side * (SIDEWALK_OUT - 0.8), (0.0, float(side))))
    for x in STREETS_X:
        for i, y in enumerate(np.arange(50.0, 190.0, 40.0)):
            side = 1 if i % 2 else -1
            items.append(lamp(x + side * (SIDEWALK_OUT - 0.8), y + 10.0, (float(side), 0.0)))
    # Fire hydrants near some corners.
    for x in STREETS_X:
        for y in STREETS_Y:
            if rng.random() < 0.5:
                items.append(Item(FUEL + "Fire hydrant", x + 9.0, y - SIDEWALK_OUT + 0.6))
    return items


def lamp(x, y, away):
    """A lamp post whose arm (model -y) reaches over the road (opposite to away)."""
    return Item(FUEL + "Lamp Post", x, y, math.atan2(-away[0], away[1]))


def park_items(rng):
    x0, y0, x1, y1 = PARK
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    items = []
    for a in np.arange(0, 2 * math.pi, math.pi / 3):
        bx, by = cx + 12.5 * math.cos(a + 0.3), cy + 12.5 * math.sin(a + 0.3)
        items.append(Item(FUEL + "FoodCourtBenchLong", bx, by, a + 0.3))
    for _ in range(9):
        while True:
            x, y = rng.uniform(x0 + 10, x1 - 10), rng.uniform(y0 + 10, y1 - 10)
            if abs(math.hypot(x - cx, y - cy) - 12) > 5 and abs((x - cx) - (y - cy)) > 5 \
                    and abs((x - cx) + (y - cy)) > 5:
                break
        items.append(Item(FUEL + rng.choice(("Oak tree", "Pine Tree")), x, y, rng.uniform(0, 6.28)))
    items.append(Item(FUEL + "TrashBin", cx + 15.0, cy - 3.0))
    return items


def plaza_items():
    fx, fy = -28.0, 6.0
    items = [place_footprint(FUEL + "Fire station", FIRE_STATION, fx, fy, 0.0),
             Item(FUEL + "Dumpster", fx - 14.0, fy - 2.0, math.pi / 2)]
    for i, car in enumerate(("SUV", "Hatchback", "Prius Hybrid", "Hatchback red")):
        items.append(Item(FUEL + car, 20.0 + i * 5.4, 8.0, math.pi / 2 + CARS[car]))
    return items


def static_people(rng, driveways):
    people = []
    for lot, drive_along, _ in rng.sample(driveways, 5):
        x, y = lot_point(lot, drive_along + rng.uniform(-2, 2), SIDEWALK_OUT + rng.uniform(1.5, 4.0))
        people.append(Item(FUEL + rng.choice(PEOPLE), x, y, rng.uniform(0, 2 * math.pi)))
    # On sidewalks.
    people.append(Item(FUEL + "Standing person", -30.0, 90.0 + 5.4, 1.2))
    people.append(Item(FUEL + "Casual female", 60.0 - 5.4, 120.0, -2.0))
    # In the park.
    x0, y0, x1, y1 = PARK
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    people.append(Item(FUEL + "Casual female", cx + 3.0, cy + 8.0, 0.5))
    people.append(Item(FUEL + "Standing person", cx + 4.0, cy + 8.6, -2.6))
    people.append(Item(FUEL + "Walking person", cx - 9.0, cy - 6.0, 2.2))
    # On the plaza, by the fire station.
    people.append(Item(FUEL + "Standing person", -14.0, 4.0, 3.0))
    return people


def dogs(rng, people, driveways):
    """Labradors (sim/models/labrador_dog): some next to people, some alone.

    The lone dogs are on purpose: a detector should not call a dog a person.
    Named dog_*, so run_gazebo.sh --no-people keeps them.
    """
    out = []
    for person in people[:3] + people[-4:-2]:          # yard and park people
        a = rng.uniform(0, 2 * math.pi)
        out.append(Item("model://labrador_dog", person.x + 1.2 * math.cos(a),
                        person.y + 1.2 * math.sin(a), rng.uniform(0, 2 * math.pi)))
    d = HALF_ROAD + CURB + 1.25
    out.append(Item("model://labrador_dog", -24.0, 30.0 + d, 0.0))          # alone on a sidewalk
    out.append(Item("model://labrador_dog", 60.0 + d, 70.0, math.pi / 2))
    for lot, along, _ in rng.sample(driveways, 2):                           # alone on front lawns
        x, y = lot_point(lot, along - 4.0, SIDEWALK_OUT + 2.0)
        out.append(Item("model://labrador_dog", x, y, rng.uniform(0, 2 * math.pi)))
    out.append(Item("model://labrador_dog", 8.0, -6.0, 2.5))                 # on the plaza
    return out


def walkers():
    """Sidewalk routes of the walking actors: (name, [(x, y), ...], closed)."""
    d = HALF_ROAD + CURB + 1.25          # middle of the sidewalk
    x0, y0, x1, y1 = PARK
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    ring = [(cx + 12 * math.cos(a), cy + 12 * math.sin(a)) for a in np.arange(0, 2 * math.pi, math.pi / 6)]
    return [
        ("walker_1", [(-52.0, 30.0 + d), (-8.0, 30.0 + d)], False),
        ("walker_2", [(8.0, 90.0 - d), (52.0, 90.0 - d)], False),
        ("walker_3", [(-60.0 + d, 98.0), (-60.0 + d, 142.0)], False),
        ("walker_4", [(0.0 - d, 40.0), (0.0 - d, 82.0)], False),
        ("walker_5", [(-50.0, 150.0 + d), (-6.0, 150.0 + d)], False),
        ("walker_park", ring, True),
    ]


# --------------------------------------------------------------------------
# Ground texture


def px(x, y):
    return int(round((x - X_MIN) / M_PER_PX)), int(round((Y_MAX - y) / M_PER_PX))


def noise(rng, h, w, scales=(8, 32, 128, 512), weights=(0.4, 0.3, 0.2, 0.1)):
    """Fractal value noise in [0, 1]: random grids upsampled and summed."""
    out = np.zeros((h, w), np.float32)
    for cells, weight in zip(scales, weights):
        grid = rng.random((max(2, h * cells // w), cells)).astype(np.float32)
        out += weight * cv2.resize(grid, (w, h), interpolation=cv2.INTER_CUBIC)
    out -= out.min()
    return out / out.max()


class Painter:
    """Paints the ground in layers: every shape of a layer is drawn into one
    mask, and each layer is blended onto the image once, in order (blending
    the 5000 px image once per dash would take minutes)."""

    def __init__(self, rng):
        self.w, self.h = px(X_MAX, Y_MIN)
        self.img = np.zeros((self.h, self.w, 3), np.float32)
        self.n1 = noise(rng, self.h, self.w)
        self.n2 = noise(rng, self.h, self.w, scales=(64, 256, 1024), weights=(0.3, 0.4, 0.3))

    def layer(self, color, amount=0.2, soft=1.5):
        """Draw the layer's shapes into the returned Layer, then call .blend()."""
        return Layer(self, color, amount, soft)


class Layer:
    def __init__(self, painter, color, amount, soft):
        self.p = painter
        self.color, self.amount, self.soft = np.array(color, np.float32), amount, soft
        self.m = np.zeros((painter.h, painter.w), np.uint8)

    def rect(self, x0, y0, x1, y1):
        cv2.rectangle(self.m, px(min(x0, x1), max(y0, y1)), px(max(x0, x1), min(y0, y1)), 255, -1)

    def poly(self, points):
        cv2.fillPoly(self.m, [np.array([px(x, y) for x, y in points], np.int32)], 255)

    def line(self, points, width_m, closed=False):
        cv2.polylines(self.m, [np.array([px(x, y) for x, y in points], np.int32)], closed, 255,
                      max(1, int(width_m / M_PER_PX)))

    def blend(self):
        mask = cv2.GaussianBlur(self.m, (0, 0), self.soft) if self.soft else self.m
        shade = (1 - self.amount / 2 + self.amount * self.p.n2)[..., None]
        a = (mask.astype(np.float32) / 255.0)[..., None]
        self.p.img[:] = self.p.img * (1 - a) + self.color * shade * a


ASPHALT = (68, 70, 72)
CONCRETE = (172, 176, 178)
CURB_C = (195, 198, 200)
WHITE = (228, 232, 232)
YELLOW = (40, 185, 225)
GRAVEL = (135, 160, 175)


def road_spans():
    """(axis, position, start, end) of every street segment."""
    spans = [("x", y, X_MIN, X_MAX) for y in STREETS_Y]
    spans += [("y", x, STREETS_Y[0], Y_MAX) for x in STREETS_X]
    return spans


def span_rect(axis, pos, a, b, half):
    if axis == "x":
        return a, pos - half, b, pos + half
    return pos - half, a, pos + half, b


def paint_ground(rng, driveways):
    p = Painter(rng)
    img = p.img
    # Grass with patchy tone and drier patches.
    img[:] = np.array([62, 118, 88], np.float32)
    img *= (0.78 + 0.45 * p.n1)[..., None]
    img[..., 2] += 22 * (p.n2 - 0.5)

    crossings = [(x, y) for x in STREETS_X for y in STREETS_Y]

    def near_crossing(x, y, r):
        return any(abs(x - cx) < r and abs(y - cy) < r for cx, cy in crossings)

    # Park lawn and paths, and the sandy playground.
    x0, y0, x1, y1 = PARK
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    lawn = p.layer((55, 125, 80), 0.25)
    lawn.rect(x0 + SIDEWALK_OUT, y0 + SIDEWALK_OUT, x1 - SIDEWALK_OUT, y1 - SIDEWALK_OUT)
    lawn.blend()
    paths = p.layer(GRAVEL, 0.3)
    paths.line([(cx + 12 * math.cos(a), cy + 12 * math.sin(a)) for a in np.linspace(0, 2 * math.pi, 72)],
               2.2, closed=True)
    for sx, sy in ((1, 1), (1, -1)):
        paths.line([(cx - sx * 23, cy - sy * 23), (cx + sx * 23, cy + sy * 23)], 2.0)
    paths.blend()
    sand = p.layer((120, 175, 205), 0.25)
    sand.rect(cx + 14, cy - 22, cx + 23, cy - 13)
    sand.blend()

    # Plaza at home.
    px0, py0, px1, py1 = PLAZA
    plaza = p.layer((150, 158, 164), 0.15)
    plaza.rect(px0, py0, px1, py1)
    plaza.blend()
    tiles = p.layer((118, 124, 128), 0.1, soft=0)
    for v in np.arange(px0, px1, 2.0):
        tiles.line([(v, py0), (v, py1)], 0.03)
    for v in np.arange(py0, py1, 2.0):
        tiles.line([(px0, v), (px1, v)], 0.03)
    tiles.blend()

    # Driveways from the sidewalk to the end of each house.
    drives = p.layer((160, 166, 170), 0.2)
    for lot, along, depth in driveways:
        drives.poly([lot_point(lot, along - 1.6, SIDEWALK_OUT - 0.1), lot_point(lot, along + 1.6, SIDEWALK_OUT - 0.1),
                     lot_point(lot, along + 1.6, depth), lot_point(lot, along - 1.6, depth)])
    drives.blend()

    # Sidewalks with joints every 1.5 m, then curbs and asphalt over them, so
    # crossings stay clean (the plaza's car park and lane are asphalt too).
    sidewalk = p.layer(CONCRETE, 0.15)
    joints = p.layer((130, 134, 136), 0.1, soft=0)
    for axis, pos, a, b in road_spans():
        sidewalk.rect(*span_rect(axis, pos, a, b, SIDEWALK_OUT))
        for s_ in np.arange(a, b, 1.5):
            for side in (-1, 1):
                d0, d1 = side * (HALF_ROAD + CURB), side * SIDEWALK_OUT
                if axis == "x":
                    joints.line([(s_, pos + d0), (s_, pos + d1)], 0.03)
                else:
                    joints.line([(pos + d0, s_), (pos + d1, s_)], 0.03)
    sidewalk.blend()
    joints.blend()
    for color, amount, half in ((CURB_C, 0.1, HALF_ROAD + CURB), (ASPHALT, 0.18, HALF_ROAD)):
        layer = p.layer(color, amount)
        for axis, pos, a, b in road_spans():
            layer.rect(*span_rect(axis, pos, a, b, half))
        if color == ASPHALT:
            layer.rect(15, 3, 42, 13)
            layer.rect(-3.0, 0.0, 3.0, 30.0)     # lane from the plaza to the first street
        layer.blend()

    # Dashed yellow centre lines; white edge lines, zebra crossings, bays, pad.
    yellow = p.layer(YELLOW, 0.1)
    white = p.layer(WHITE, 0.1)
    for axis, pos, a, b in road_spans():
        for s_ in np.arange(a, b, 6.0):
            x, y = (s_, pos) if axis == "x" else (pos, s_)
            if not near_crossing(x, y, HALF_ROAD + 4):
                yellow.rect(*span_rect(axis, pos, s_, s_ + 3.0, 0.08))
        for side in (-1, 1):
            off = pos + side * (HALF_ROAD - 0.35)
            for s_ in np.arange(a, b, 1.0):
                x, y = (s_, pos) if axis == "x" else (pos, s_)
                if not near_crossing(x, y, HALF_ROAD + 2):
                    white.rect(*span_rect(axis, off, s_, s_ + 1.0, 0.06))
    for cx_, cy_ in crossings:
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            if dy < 0 and cy_ == STREETS_Y[0]:
                continue                          # no street south of the first one
            near, far = HALF_ROAD + CURB + 1.0, HALF_ROAD + CURB + 4.0
            for k in np.arange(-HALF_ROAD + 0.4, HALF_ROAD - 0.3, 1.0):
                if dx:
                    white.rect(cx_ + dx * near, cy_ + k, cx_ + dx * far, cy_ + k + 0.5)
                else:
                    white.rect(cx_ + k, cy_ + dy * near, cx_ + k + 0.5, cy_ + dy * far)
    for x in np.arange(17.3, 42, 5.4):
        white.rect(x, 3.5, x + 0.12, 12.5)
    cv2.circle(white.m, px(0, 0), int(3.0 / M_PER_PX), 255, int(0.3 / M_PER_PX))
    for seg in (((-1.0, -1.4), (-1.0, 1.4)), ((1.0, -1.4), (1.0, 1.4)), ((-1.0, 0.0), (1.0, 0.0))):
        white.line(list(seg), 0.3)
    yellow.blend()
    white.blend()

    img += rng.normal(0, 4, img.shape[:2]).astype(np.float32)[..., None]
    return np.clip(img, 0, 255).astype(np.uint8)


# --------------------------------------------------------------------------
# World SDF

HEADER = """<?xml version="1.0" ?>
<!--
  Neighborhood security patrol world, generated by sim/make_neighborhood.py
  (do not edit by hand; change the script and run it again).

  A suburban street grid north of home: E-W streets at y = 30, 90, 150 and
  N-S streets at x = -60, 0, 60, 24 houses, a park in the north-east block,
  parked cars, street lamps, trees, standing people, people walking on the
  sidewalks and dogs ("Labrador Dog" by kenchoo, Sketchfab, CC-BY-4.0).
  Home (0, 0) is a landing pad on a plaza with a fire station. The patrol
  mission (sim/fly_patrol_mission.py --route streets) flies along
  the streets at 15 m; every model is lower than that.

  Frame: +X = east, +Y = north. World name stays "security_patrol" so the
  camera topic in config/default.yaml is unchanged. Fuel models download on
  the first run.
-->
<sdf version="1.9">
  <world name="security_patrol">
    <physics name="1ms" type="ignore">
      <max_step_size>0.001</max_step_size>
      <real_time_factor>1.0</real_time_factor>
    </physics>

    <plugin filename="gz-sim-physics-system" name="gz::sim::systems::Physics"/>
    <plugin filename="gz-sim-sensors-system" name="gz::sim::systems::Sensors">
      <render_engine>ogre2</render_engine>
    </plugin>
    <plugin filename="gz-sim-user-commands-system" name="gz::sim::systems::UserCommands"/>
    <plugin filename="gz-sim-scene-broadcaster-system" name="gz::sim::systems::SceneBroadcaster"/>
    <plugin filename="gz-sim-imu-system" name="gz::sim::systems::Imu"/>
    <plugin filename="gz-sim-navsat-system" name="gz::sim::systems::NavSat"/>

    <scene>
      <ambient>0.6 0.6 0.6</ambient>
      <background>0.6 0.75 0.9</background>
      <sky></sky>
    </scene>

    <!-- Same home as iris_runway, so SITL's home matches. -->
    <spherical_coordinates>
      <latitude_deg>-35.363262</latitude_deg>
      <longitude_deg>149.165237</longitude_deg>
      <elevation>584</elevation>
      <heading_deg>0</heading_deg>
      <surface_model>EARTH_WGS84</surface_model>
    </spherical_coordinates>

    <light type="directional" name="sun">
      <cast_shadows>true</cast_shadows>
      <pose>0 0 10 0 0 0</pose>
      <diffuse>0.8 0.8 0.8 1</diffuse>
      <specular>0.2 0.2 0.2 1</specular>
      <direction>-0.4 0.3 -0.85</direction>
    </light>

    <include><uri>model://neighborhood_ground</uri></include>
"""

FOOTER = """
    <include>
      <uri>model://iris_with_gimbal</uri>
      <pose degrees="true">0 0 0.195 0 0 90</pose>
    </include>

  </world>
</sdf>
"""

ACTOR_SKIN = "https://fuel.gazebosim.org/1.0/Mingfei/models/actor/tip/files/meshes/walk.dae"
WALK_SPEED = 1.1     # m/s


def include(name, item):
    return (f"    <include>\n      <name>{name}</name>\n      <uri>{item.model}</uri>\n"
            f"      <pose>{item.x:.2f} {item.y:.2f} 0 0 0 {item.yaw:.3f}</pose>\n    </include>\n")


def actor(name, route, closed):
    """Walking actor: there and back along route (or round it if closed)."""
    points = route + [route[0]] if closed else route + route[-2::-1]
    lines, t = [], 0.0
    for i, (x, y) in enumerate(points):
        if i:
            px_, py_ = points[i - 1]
            yaw = math.atan2(y - py_, x - px_)
            t += math.hypot(x - px_, y - py_) / WALK_SPEED
            lines.append(f"<waypoint><time>{t:.1f}</time><pose>{x:.2f} {y:.2f} 1.0 0 0 {yaw:.3f}</pose></waypoint>")
            if i + 1 < len(points):
                nx, ny = points[i + 1]
                t += 1.0     # turn on the spot
                lines.append(f"<waypoint><time>{t:.1f}</time><pose>{x:.2f} {y:.2f} 1.0 0 0 "
                             f"{math.atan2(ny - y, nx - x):.3f}</pose></waypoint>")
        else:
            nx, ny = points[1]
            lines.append(f"<waypoint><time>0</time><pose>{x:.2f} {y:.2f} 1.0 0 0 "
                         f"{math.atan2(ny - y, nx - x):.3f}</pose></waypoint>")
    body = "\n          ".join(lines)
    return f"""    <actor name="{name}">
      <skin><filename>{ACTOR_SKIN}</filename><scale>1.0</scale></skin>
      <animation name="walking">
        <filename>{ACTOR_SKIN}</filename>
        <interpolate_x>true</interpolate_x>
      </animation>
      <script>
        <loop>true</loop>
        <auto_start>true</auto_start>
        <trajectory id="0" type="walking">
          {body}
        </trajectory>
      </script>
    </actor>
"""


def write_world(houses, items, people, dog_items):
    parts = [HEADER, "\n    <!-- Houses and the fire station -->\n"]
    parts += [include(f"house_{i}", h) for i, h in enumerate(houses)]
    parts.append("\n    <!-- Cars, trees, lamps, street furniture -->\n")
    counts = {}
    for item in items:
        stem = item.model.rsplit("/", 1)[-1].lower().replace(" ", "_")
        counts[stem] = counts.get(stem, 0) + 1
        parts.append(include(f"{stem}_{counts[stem]}", item))
    parts.append("\n    <!-- Standing people (person_* so run_gazebo.sh --no-people removes them) -->\n")
    parts += [include(f"person_{i}", p) for i, p in enumerate(people)]
    parts.append("\n    <!-- Dogs (named dog_*, kept by --no-people) -->\n")
    parts += [include(f"dog_{i}", d) for i, d in enumerate(dog_items)]
    parts.append("\n    <!-- People walking on the sidewalks and round the park -->\n")
    parts += [actor(name, route, closed) for name, route, closed in walkers()]
    parts.append(FOOTER)
    WORLD.write_text("".join(parts))


def main():
    rng = random.Random(SEED)
    houses, driveways, items = build_layout(rng)
    plaza = plaza_items()
    houses.append(plaza[0])
    items += plaza[1:] + street_items(rng) + park_items(rng)
    people = static_people(rng, driveways)
    dog_items = dogs(rng, people, driveways)
    write_world(houses, items, people, dog_items)
    print(f"Saved {WORLD}: {len(houses)} buildings, {len(items)} objects, "
          f"{len(people)} standing people, {len(walkers())} walkers, {len(dog_items)} dogs")

    ground = paint_ground(np.random.default_rng(SEED), driveways)
    TEXTURE.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(TEXTURE), ground, [cv2.IMWRITE_JPEG_QUALITY, 90])
    print(f"Saved {TEXTURE} ({ground.shape[1]}x{ground.shape[0]} px)")


if __name__ == "__main__":
    main()
