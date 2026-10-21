from pymavlink import mavutil


def connect(url, timeout_s):
    """Open a MAVLink connection and wait for the autopilot's heartbeat."""
    # The companion computer is part of the vehicle, so it shares system id 1
    # and identifies itself as the onboard computer component.
    master = mavutil.mavlink_connection(
        url,
        source_system=1,
        source_component=mavutil.mavlink.MAV_COMP_ID_ONBOARD_COMPUTER,
    )
    # wait_heartbeat returns None on timeout instead of raising.
    # It also sets master.target_system / target_component for later commands.
    heartbeat = master.wait_heartbeat(timeout=timeout_s)
    if heartbeat is None:
        raise TimeoutError(f"No heartbeat from {url} within {timeout_s} s")
    return master
