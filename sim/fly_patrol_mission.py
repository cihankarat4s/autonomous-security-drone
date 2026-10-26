"""Upload a patrol mission to SITL and fly it in AUTO.

Stands in for Mission Planner during tests. The mission follows the thesis:
take off, fly to the protected area, patrol it until the mission time is up,
return home. The companion app ends the patrol by sending RTL.

    seq 0  home (ArduPilot overwrites it)
    seq 1  TAKEOFF 15 m
    seq 2  WAYPOINT area centre, 60 m north    <- trigger_seq: 2
    seq 3-6  WAYPOINTs: square lap around the centre
    seq 7  DO_JUMP to seq 3, --max-laps times
    seq 8  RTL, only if the companion never sent one

The trigger waypoint must be followed by a NAV item: with a DO_CHANGE_SPEED
right after it, ArduPilot sent MISSION_ITEM_REACHED with seq 3 for it (while
its own STATUSTEXT said "Reached command #2"), and detection never started.
The corners are only 10 m apart, so the vehicle slows down at each of them
anyway (~6 s per leg) and needs no speed limit.

--hover S flies the earlier test layout instead (WAYPOINT, LOITER_TIME S,
WAYPOINT north-east, RTL), which sim/collect_patrol_flight.sh used to record
the hover frames (*_seq3.jpg) of the datasets in docs/results/.

The camera sees ~26 x 19.5 m from 15 m and the area is 20 x 20 m. Flying the
square 5 m off the centre keeps the centre in view on every leg and each edge
in view on at least one leg, and the area is covered once per lap (~25 s).

--route streets flies the neighborhood world instead
(sim/worlds/security_patrol_neighborhood.sdf): the trigger waypoint is the
first crossing, 30 m north of home, and each lap follows the streets past
houses, parked cars and the park (~480 m, ~2.5 min).

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
AREA_NORTH_M = 60     # centre of the protected area, from home
LAP_OFFSET_M = 5      # half side of the patrol square
# Upper bound on laps if the companion never ends the patrol: ~12 min at
# ~25 s a lap, longer than the battery. The Pixhawk's battery failsafe and
# this final RTL are the backups; the companion's RTL is the normal end.
MAX_LAPS = 30
# Neighborhood streets as (north, east) from home: the first crossing, then a
# lap along the streets that ends back at it.
STREET_ENTRY = (30, 0)
STREET_LAP = [(30, 60), (90, 60), (90, 0), (150, 0), (150, -60), (30, -60), (30, 0)]
STREET_MAX_LAPS = 6     # ~15 min, longer than the battery
HEARTBEAT_LOST_S = 15
EARTH_M_PER_DEG = 111_320


def offset(lat, lon, north_m, east_m):
    """Shift a lat/lon by metres (flat-earth approximation, fine for < 1 km)."""
    dlat = north_m / EARTH_M_PER_DEG
    dlon = east_m / (EARTH_M_PER_DEG * math.cos(math.radians(lat)))
    return lat + dlat, lon + dlon


def build_mission(home_lat, home_lon, max_laps=MAX_LAPS, route="square"):
    rel = mavlink.MAV_FRAME_GLOBAL_RELATIVE_ALT_INT
    if route == "streets":
        centre = offset(home_lat, home_lon, *STREET_ENTRY)
        corners = [offset(home_lat, home_lon, n, e) for n, e in STREET_LAP]
    else:
        centre = offset(home_lat, home_lon, AREA_NORTH_M, 0)
        d = LAP_OFFSET_M
        # North-east, north-west, south-west, south-east corners.
        corners = [offset(*centre, n, e) for n, e in ((d, d), (d, -d), (-d, -d), (-d, d))]
    first_corner = 3
    # (frame, command, param1, param2, lat, lon, alt)
    return [
        (mavlink.MAV_FRAME_GLOBAL_INT, mavlink.MAV_CMD_NAV_WAYPOINT, 0, 0, home_lat, home_lon, 0),
        (rel, mavlink.MAV_CMD_NAV_TAKEOFF, 0, 0, 0, 0, TAKEOFF_ALT_M),
        (rel, mavlink.MAV_CMD_NAV_WAYPOINT, 0, 0, *centre, TAKEOFF_ALT_M),
        *[(rel, mavlink.MAV_CMD_NAV_WAYPOINT, 0, 0, *c, TAKEOFF_ALT_M) for c in corners],
        # param1 = item to jump to, param2 = how many more times.
        (rel, mavlink.MAV_CMD_DO_JUMP, first_corner, max_laps - 1, 0, 0, 0),
        (rel, mavlink.MAV_CMD_NAV_RETURN_TO_LAUNCH, 0, 0, 0, 0, 0),
    ]


def build_hover_mission(home_lat, home_lon, hover_s):
    rel = mavlink.MAV_FRAME_GLOBAL_RELATIVE_ALT_INT
    wp1 = offset(home_lat, home_lon, AREA_NORTH_M, 0)
    wp2 = offset(home_lat, home_lon, AREA_NORTH_M, 60)
    # The hold is a separate item: a waypoint with its own hold time only
    # reports MISSION_ITEM_REACHED when the hold ends, too late to start
    # detection.
    return [
        (mavlink.MAV_FRAME_GLOBAL_INT, mavlink.MAV_CMD_NAV_WAYPOINT, 0, 0, home_lat, home_lon, 0),
        (rel, mavlink.MAV_CMD_NAV_TAKEOFF, 0, 0, 0, 0, TAKEOFF_ALT_M),
        (rel, mavlink.MAV_CMD_NAV_WAYPOINT, 0, 0, *wp1, TAKEOFF_ALT_M),
        (rel, mavlink.MAV_CMD_NAV_LOITER_TIME, hover_s, 0, *wp1, TAKEOFF_ALT_M),
        (rel, mavlink.MAV_CMD_NAV_WAYPOINT, 0, 0, *wp2, TAKEOFF_ALT_M),
        (rel, mavlink.MAV_CMD_NAV_RETURN_TO_LAUNCH, 0, 0, 0, 0, 0),
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
        frame, command, param1, param2, lat, lon, alt = items[msg.seq]
        master.mav.mission_item_int_send(
            master.target_system, master.target_component, msg.seq,
            frame, command,
            0, 1,  # current, autocontinue
            param1, param2, 0, 0,  # param1-4
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
    parser.add_argument("--route", choices=["square", "streets"], default="square",
                        help="streets: the neighborhood world's street lap")
    parser.add_argument("--max-laps", type=int, help=f"default {MAX_LAPS} (streets: {STREET_MAX_LAPS})")
    parser.add_argument("--hover", type=float, metavar="S",
                        help="earlier test layout: hover S seconds, then RTL")
    args = parser.parse_args()

    # Acts like a ground station: system 255, Mission Planner component id.
    master = connect(args.connection, 30, source_system=255,
                     source_component=mavlink.MAV_COMP_ID_MISSIONPLANNER)
    print(f"Connected to system {master.target_system}")

    set_message_interval(master, mavlink.MAVLINK_MSG_ID_GLOBAL_POSITION_INT, 2)
    pos = master.recv_match(type="GLOBAL_POSITION_INT", blocking=True, timeout=10)
    home_lat, home_lon = pos.lat / 1e7, pos.lon / 1e7

    if args.hover:
        items = build_hover_mission(home_lat, home_lon, args.hover)
    else:
        laps = args.max_laps or (STREET_MAX_LAPS if args.route == "streets" else MAX_LAPS)
        items = build_mission(home_lat, home_lon, laps, args.route)
    upload_mission(master, items)
    print(f"Mission uploaded ({len(items)} items)")

    set_mode(master, "GUIDED")
    arm_when_ready(master)
    print("Armed")
    # Start right away: the vehicle disarms itself after DISARM_DELAY (10 s).
    set_mode(master, "AUTO")
    send_command(master, mavlink.MAV_CMD_MISSION_START, (0, 0))
    print("Mission started")
    wait_until_disarmed(master)


def wait_until_disarmed(master, timeout_s=1200):
    """Print mission progress until the vehicle disarms after landing.

    Gives up after timeout_s, so a stuck simulation does not hang a run.
    """
    deadline = time.monotonic() + timeout_s
    last_heartbeat = time.monotonic()
    while True:
        now = time.monotonic()
        if now > deadline:
            raise RuntimeError(f"Vehicle still armed after {timeout_s} s")
        # A closed socket makes recv_match return at once, forever; without
        # this check the loop spun on "EOF on TCP socket" until the timeout.
        if now - last_heartbeat > HEARTBEAT_LOST_S:
            raise RuntimeError(f"No heartbeat for {HEARTBEAT_LOST_S} s, link lost")
        msg = master.recv_match(type=["MISSION_ITEM_REACHED", "HEARTBEAT", "STATUSTEXT"],
                                blocking=True, timeout=5)
        if msg is None or msg.get_srcSystem() != master.target_system:
            time.sleep(0.1)
            continue
        if msg.get_type() == "HEARTBEAT" and msg.get_srcComponent() == master.target_component:
            last_heartbeat = time.monotonic()
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
