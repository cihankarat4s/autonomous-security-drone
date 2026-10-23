"""Fly a varied mission in the person-free world to collect hard negatives.

The drone's own shadow is what the VisDrone model mistook for a person, so
this mission shows it in many ways: several altitudes (6-25 m), headings in
all four directions, grass and the tan protected area, short hovers, and the
landing. Run sim/capture_negatives.py alongside it.

    sim/run_gazebo.sh --headless --no-intruder        # then sim/run_sitl.sh
    python sim/capture_negatives.py --name sun_default
    python sim/fly_negative_mission.py
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


def build_mission(home_lat, home_lon):
    rel = mavlink.MAV_FRAME_GLOBAL_RELATIVE_ALT_INT
    items = [
        (mavlink.MAV_FRAME_GLOBAL_INT, mavlink.MAV_CMD_NAV_WAYPOINT, 0, home_lat, home_lon, 0),
        (rel, mavlink.MAV_CMD_NAV_TAKEOFF, 0, 0, 0, LEGS[0][2]),
    ]
    for north, east, alt, hover_s in LEGS:
        lat, lon = offset(home_lat, home_lon, north, east)
        items.append((rel, mavlink.MAV_CMD_NAV_WAYPOINT, 0, lat, lon, alt))
        if hover_s:
            items.append((rel, mavlink.MAV_CMD_NAV_LOITER_TIME, hover_s, lat, lon, alt))
    items.append((rel, mavlink.MAV_CMD_NAV_LAND, 0, home_lat, home_lon, 0))
    return items


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--connection", default="tcp:127.0.0.1:5763")
    args = parser.parse_args()

    master = connect(args.connection, 30, source_system=255,
                     source_component=mavlink.MAV_COMP_ID_MISSIONPLANNER)
    set_message_interval(master, mavlink.MAVLINK_MSG_ID_GLOBAL_POSITION_INT, 2)
    pos = master.recv_match(type="GLOBAL_POSITION_INT", blocking=True, timeout=10)
    items = build_mission(pos.lat / 1e7, pos.lon / 1e7)
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
