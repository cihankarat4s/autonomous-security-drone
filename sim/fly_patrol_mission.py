"""Upload a patrol mission to SITL and fly it in AUTO.

Stands in for Mission Planner during tests, so the companion app can be
checked against a real mission. Mission layout (same as the thesis tests):

    seq 0  home (ArduPilot overwrites it)
    seq 1  TAKEOFF 15 m
    seq 2  WAYPOINT 60 m north        <- protected area, trigger_seq: 2
    seq 3  WAYPOINT 60 m north-east
    seq 4  RTL

Usage (inside WSL, SITL running):
    python sim/fly_patrol_mission.py --connection tcp:127.0.0.1:5763
"""
import argparse
import math
import time

from pymavlink import mavutil

from security_drone.mavlink.commands import CommandFailed, send_command, set_message_interval
from security_drone.mavlink.connection import connect

mavlink = mavutil.mavlink

TAKEOFF_ALT_M = 15
EARTH_M_PER_DEG = 111_320


def offset(lat, lon, north_m, east_m):
    """Shift a lat/lon by metres (flat-earth approximation, fine for < 1 km)."""
    dlat = north_m / EARTH_M_PER_DEG
    dlon = east_m / (EARTH_M_PER_DEG * math.cos(math.radians(lat)))
    return lat + dlat, lon + dlon


def build_mission(home_lat, home_lon):
    rel = mavlink.MAV_FRAME_GLOBAL_RELATIVE_ALT_INT
    wp1 = offset(home_lat, home_lon, 60, 0)
    wp2 = offset(home_lat, home_lon, 60, 60)
    # (frame, command, lat, lon, alt)
    return [
        (mavlink.MAV_FRAME_GLOBAL_INT, mavlink.MAV_CMD_NAV_WAYPOINT, home_lat, home_lon, 0),
        (rel, mavlink.MAV_CMD_NAV_TAKEOFF, 0, 0, TAKEOFF_ALT_M),
        (rel, mavlink.MAV_CMD_NAV_WAYPOINT, *wp1, TAKEOFF_ALT_M),
        (rel, mavlink.MAV_CMD_NAV_WAYPOINT, *wp2, TAKEOFF_ALT_M),
        (rel, mavlink.MAV_CMD_NAV_RETURN_TO_LAUNCH, 0, 0, 0),
    ]


def upload_mission(master, items):
    """MAVLink mission upload: send COUNT, answer each REQUEST, expect ACK."""
    master.mav.mission_count_send(
        master.target_system, master.target_component, len(items),
        mavlink.MAV_MISSION_TYPE_MISSION,
    )
    while True:
        msg = master.recv_match(
            type=["MISSION_REQUEST_INT", "MISSION_REQUEST", "MISSION_ACK"],
            blocking=True, timeout=5,
        )
        if msg is None:
            raise RuntimeError("Mission upload timed out")
        if msg.get_type() == "MISSION_ACK":
            if msg.type != mavlink.MAV_MISSION_ACCEPTED:
                raise RuntimeError(f"Mission rejected: {msg.type}")
            return
        frame, command, lat, lon, alt = items[msg.seq]
        master.mav.mission_item_int_send(
            master.target_system, master.target_component, msg.seq,
            frame, command,
            0, 1,  # current, autocontinue
            0, 0, 0, 0,  # param1-4
            int(lat * 1e7), int(lon * 1e7), alt,
            mavlink.MAV_MISSION_TYPE_MISSION,
        )


def set_mode(master, name):
    send_command(master, mavlink.MAV_CMD_DO_SET_MODE, (
        mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED,
        master.mode_mapping()[name],
    ))


def arm_when_ready(master, timeout_s=120):
    """Retry arming until the pre-arm checks (GPS, EKF) pass."""
    deadline = time.monotonic() + timeout_s
    while True:
        try:
            send_command(master, mavlink.MAV_CMD_COMPONENT_ARM_DISARM, (1,))
            return
        except CommandFailed as exc:
            if time.monotonic() > deadline:
                raise
            print(f"Not ready to arm yet ({exc}), retrying")
            time.sleep(3)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--connection", default="tcp:127.0.0.1:5763")
    args = parser.parse_args()

    # Acts like a ground station: system 255, Mission Planner component id.
    master = connect(args.connection, 30, source_system=255,
                     source_component=mavlink.MAV_COMP_ID_MISSIONPLANNER)
    print(f"Connected to system {master.target_system}")

    set_message_interval(master, mavlink.MAVLINK_MSG_ID_GLOBAL_POSITION_INT, 2)
    pos = master.recv_match(type="GLOBAL_POSITION_INT", blocking=True, timeout=10)
    home_lat, home_lon = pos.lat / 1e7, pos.lon / 1e7

    upload_mission(master, build_mission(home_lat, home_lon))
    print("Mission uploaded (5 items)")

    set_mode(master, "GUIDED")
    arm_when_ready(master)
    print("Armed")
    # Start right away: the vehicle disarms itself after DISARM_DELAY (10 s).
    set_mode(master, "AUTO")
    send_command(master, mavlink.MAV_CMD_MISSION_START, (0, 0))
    print("Mission started")

    # Follow the flight until the vehicle disarms after RTL.
    while True:
        msg = master.recv_match(type=["MISSION_ITEM_REACHED", "HEARTBEAT", "STATUSTEXT"],
                                blocking=True, timeout=5)
        if msg is None or msg.get_srcSystem() != master.target_system:
            continue
        if msg.get_type() == "MISSION_ITEM_REACHED":
            print(f"Reached item {msg.seq}")
        elif msg.get_type() == "STATUSTEXT":
            # Includes the companion's threat messages, routed by the autopilot.
            print(f"STATUSTEXT from component {msg.get_srcComponent()}: {msg.text}")
        elif (msg.get_srcComponent() == master.target_component
              and not msg.base_mode & mavlink.MAV_MODE_FLAG_SAFETY_ARMED):
            print("Disarmed, mission finished")
            return


if __name__ == "__main__":
    main()
