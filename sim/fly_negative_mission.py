"""Fly a varied mission in the person-free world to collect hard negatives.

The drone's own shadow is what the VisDrone model mistook for a person, so
this mission shows it in many ways: several altitudes (6-25 m), headings in
all four directions, grass and the tan protected area, short hovers, and the
landing. Run sim/capture_negatives.py alongside it.

    sim/run_gazebo.sh --headless --no-intruder        # then sim/run_sitl.sh
    python sim/capture_negatives.py --name sun_default
    python sim/fly_negative_mission.py                  # --zone urban with --world urban
"""
import argparse

from pymavlink import mavutil

from fly_patrol_mission import (arm_when_ready, offset, set_mode, upload_mission,
                                wait_until_disarmed)
from security_drone.mavlink.commands import send_command, set_message_interval
from security_drone.mavlink.connection import connect

mavlink = mavutil.mavlink

# (north m, east m, altitude m, hover s) of each leg's end point.
LEGS = [
    (0, 30, 6, 5),
    (30, 30, 10, 0),
    (60, 30, 14, 5),     # over the edge of the protected area
    (60, -10, 18, 0),    # across the tan patch
    (30, -30, 22, 5),
    (0, -30, 25, 0),
    (-20, 0, 12, 5),
    (0, 0, 8, 0),
]

# Urban world (run_gazebo.sh --world urban): stay in the person-free training
# zone south of home (houses, bus, parking lot, barriers), never over the test
# zone around the protected area. Altitudes clear the houses and trees.
URBAN_LEGS = [
    (-20, 0, 8, 5),
    (-45, 2, 10, 0),
    (-58, -45, 14, 5),   # parking lot
    (-30, -60, 18, 0),
    (-8, -40, 22, 5),    # house
    (-45, 30, 25, 0),    # house across the road
    (-40, 14, 12, 5),    # bus
    (-10, 4, 10, 0),
]

# Orthophoto world with --no-people (run_gazebo.sh --world ortho --no-people):
# a false-positive test on real photo texture, never used for training. Visits
# the parking lot, the school roof and courtyard, a house, the hill, the road
# with parked cars, the park and the houses east of it.
ORTHO_LEGS = [
    (-14, 4, 8, 5),      # parking lot
    (19, -30, 12, 0),
    (19, -42, 15, 5),    # school roof
    (-3, -56, 10, 5),    # courtyard and playground
    (48, -73, 18, 0),    # red-roofed house
    (88, -51, 20, 5),    # hill
    (63, -27, 12, 5),    # road with parked cars
    (60, 0, 15, 5),      # park (protected area)
    (57, 20, 18, 0),     # houses east
    (0, 0, 8, 0),
]


def build_mission(home_lat, home_lon, legs=LEGS):
    rel = mavlink.MAV_FRAME_GLOBAL_RELATIVE_ALT_INT
    items = [
        (mavlink.MAV_FRAME_GLOBAL_INT, mavlink.MAV_CMD_NAV_WAYPOINT, 0, 0, home_lat, home_lon, 0),
        (rel, mavlink.MAV_CMD_NAV_TAKEOFF, 0, 0, 0, 0, legs[0][2]),
    ]
    for north, east, alt, hover_s in legs:
        lat, lon = offset(home_lat, home_lon, north, east)
        items.append((rel, mavlink.MAV_CMD_NAV_WAYPOINT, 0, 0, lat, lon, alt))
        if hover_s:
            items.append((rel, mavlink.MAV_CMD_NAV_LOITER_TIME, hover_s, 0, lat, lon, alt))
    items.append((rel, mavlink.MAV_CMD_NAV_LAND, 0, 0, home_lat, home_lon, 0))
    return items


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--connection", default="tcp:127.0.0.1:5763")
    parser.add_argument("--zone", choices=["flat", "urban", "ortho"], default="flat",
                        help="urban: training zone of the urban world; ortho: orthophoto world")
    args = parser.parse_args()

    master = connect(args.connection, 30, source_system=255,
                     source_component=mavlink.MAV_COMP_ID_MISSIONPLANNER)
    set_message_interval(master, mavlink.MAVLINK_MSG_ID_GLOBAL_POSITION_INT, 2)
    pos = master.recv_match(type="GLOBAL_POSITION_INT", blocking=True, timeout=10)
    legs = {"urban": URBAN_LEGS, "ortho": ORTHO_LEGS}.get(args.zone, LEGS)
    items = build_mission(pos.lat / 1e7, pos.lon / 1e7, legs)
    upload_mission(master, items)
    print(f"Mission uploaded ({len(items)} items)")

    set_mode(master, "GUIDED")
    arm_when_ready(master)
    set_mode(master, "AUTO")
    send_command(master, mavlink.MAV_CMD_MISSION_START, (0, 0))
    print("Mission started")
    wait_until_disarmed(master)


if __name__ == "__main__":
    main()
