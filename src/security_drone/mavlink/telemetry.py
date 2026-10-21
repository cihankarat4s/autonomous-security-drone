from dataclasses import dataclass

from pymavlink import mavutil

mavlink = mavutil.mavlink


@dataclass(frozen=True)
class VehicleState:
    mode: str
    armed: bool


@dataclass(frozen=True)
class Position:
    lat_deg: float
    lon_deg: float
    relative_alt_m: float


def parse_heartbeat(msg):
    """Read flight mode and armed flag from a HEARTBEAT message."""
    # Armed is one bit of base_mode; the flight mode itself is in custom_mode,
    # whose numbers depend on the autopilot and vehicle type.
    armed = bool(msg.base_mode & mavlink.MAV_MODE_FLAG_SAFETY_ARMED)
    mode = mavutil.mode_string_v10(msg)
    return VehicleState(mode=mode, armed=armed)


def parse_position(msg):
    """Convert GLOBAL_POSITION_INT's integer units to degrees and metres."""
    # lat/lon are sent as degrees * 1e7, relative_alt as millimetres.
    return Position(
        lat_deg=msg.lat / 1e7,
        lon_deg=msg.lon / 1e7,
        relative_alt_m=msg.relative_alt / 1000,
    )
