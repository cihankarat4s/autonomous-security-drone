from pymavlink import mavutil

from security_drone.mavlink.connection import is_autopilot_heartbeat

mavlink = mavutil.mavlink


def make_heartbeat(mav_type, autopilot):
    return mavlink.MAVLink_heartbeat_message(
        type=mav_type, autopilot=autopilot, base_mode=0, custom_mode=0,
        system_status=0, mavlink_version=3,
    )


def test_copter_heartbeat_is_autopilot():
    msg = make_heartbeat(mavlink.MAV_TYPE_QUADROTOR, mavlink.MAV_AUTOPILOT_ARDUPILOTMEGA)
    assert is_autopilot_heartbeat(msg)


def test_ground_station_heartbeat_is_ignored():
    msg = make_heartbeat(mavlink.MAV_TYPE_GCS, mavlink.MAV_AUTOPILOT_INVALID)
    assert not is_autopilot_heartbeat(msg)
