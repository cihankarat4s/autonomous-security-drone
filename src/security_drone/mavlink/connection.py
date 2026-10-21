import time

from pymavlink import mavutil


def is_autopilot_heartbeat(msg):
    """True for a vehicle's heartbeat, False for one from a ground station.

    The autopilot forwards GCS heartbeats (e.g. Mission Planner's) to the
    other links, so not every HEARTBEAT on this link is from the vehicle.
    """
    return (msg.type != mavutil.mavlink.MAV_TYPE_GCS
            and msg.autopilot != mavutil.mavlink.MAV_AUTOPILOT_INVALID)


def connect(url, timeout_s, source_system=1,
            source_component=mavutil.mavlink.MAV_COMP_ID_ONBOARD_COMPUTER):
    """Open a MAVLink connection and wait for the autopilot's heartbeat."""
    # By default we are the companion computer: part of the vehicle, so it
    # shares system id 1 and identifies itself as the onboard computer.
    master = mavutil.mavlink_connection(
        url,
        source_system=source_system,
        source_component=source_component,
    )
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        # wait_heartbeat returns None on timeout instead of raising.
        heartbeat = master.wait_heartbeat(timeout=deadline - time.monotonic())
        if heartbeat is not None and is_autopilot_heartbeat(heartbeat):
            # pymavlink only fills in target_system and leaves
            # target_component at 0 (broadcast), so set both explicitly.
            master.target_system = heartbeat.get_srcSystem()
            master.target_component = heartbeat.get_srcComponent()
            return master
    raise TimeoutError(f"No autopilot heartbeat from {url} within {timeout_s} s")
