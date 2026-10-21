from pymavlink import mavutil

from security_drone.mavlink.telemetry import parse_heartbeat, parse_position

mavlink = mavutil.mavlink

COPTER_AUTO = 3  # ArduCopter custom_mode number for AUTO


def make_heartbeat(base_mode, custom_mode):
    return mavlink.MAVLink_heartbeat_message(
        type=mavlink.MAV_TYPE_QUADROTOR,
        autopilot=mavlink.MAV_AUTOPILOT_ARDUPILOTMEGA,
        base_mode=base_mode,
        custom_mode=custom_mode,
        system_status=mavlink.MAV_STATE_ACTIVE,
        mavlink_version=3,
    )


def test_armed_auto_heartbeat():
    base_mode = (
        mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED
        | mavlink.MAV_MODE_FLAG_SAFETY_ARMED
    )
    state = parse_heartbeat(make_heartbeat(base_mode, COPTER_AUTO))
    assert state.mode == "AUTO"
    assert state.armed is True


def test_disarmed_heartbeat():
    state = parse_heartbeat(
        make_heartbeat(mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED, COPTER_AUTO)
    )
    assert state.armed is False


def test_position_units():
    msg = mavlink.MAVLink_global_position_int_message(
        time_boot_ms=0,
        lat=-353632610,
        lon=1491652300,
        alt=599000,
        relative_alt=15000,
        vx=0,
        vy=0,
        vz=0,
        hdg=0,
    )
    pos = parse_position(msg)
    assert pos.lat_deg == -35.363261
    assert pos.lon_deg == 149.16523
    assert pos.relative_alt_m == 15.0
