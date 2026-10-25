import importlib.util
from pathlib import Path

from pymavlink import mavutil

from security_drone.config import DEFAULT_CONFIG_PATH, load_config

mavlink = mavutil.mavlink

_SCRIPT = Path(__file__).resolve().parents[1] / "sim" / "fly_patrol_mission.py"
_spec = importlib.util.spec_from_file_location("fly_patrol_mission", _SCRIPT)
fly_patrol_mission = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(fly_patrol_mission)

HOME = (-35.363262, 149.165237)


def commands(items):
    return [item[1] for item in items]


def test_trigger_seq_is_the_area_centre():
    items = fly_patrol_mission.build_mission(*HOME)
    trigger = items[load_config(DEFAULT_CONFIG_PATH)["mission"]["trigger_seq"]]
    centre = fly_patrol_mission.offset(*HOME, fly_patrol_mission.AREA_NORTH_M, 0)
    assert trigger[1] == mavlink.MAV_CMD_NAV_WAYPOINT
    assert trigger[4:6] == centre


def test_jump_repeats_the_lap_then_returns():
    items = fly_patrol_mission.build_mission(*HOME, max_laps=5)
    jump_seq = commands(items).index(mavlink.MAV_CMD_DO_JUMP)
    _, _, target, repeats, *_ = items[jump_seq]
    # The jump goes back to the first corner, and every item of the lap is a waypoint.
    assert commands(items)[target:jump_seq] == [mavlink.MAV_CMD_NAV_WAYPOINT] * 4
    assert repeats == 4   # 1 lap + 4 repeats = 5 laps
    assert commands(items)[-1] == mavlink.MAV_CMD_NAV_RETURN_TO_LAUNCH


def test_lap_stays_inside_the_protected_area():
    items = fly_patrol_mission.build_mission(*HOME)
    centre = fly_patrol_mission.offset(*HOME, fly_patrol_mission.AREA_NORTH_M, 0)
    jump_seq = commands(items).index(mavlink.MAV_CMD_DO_JUMP)
    for item in items[jump_seq - 4:jump_seq]:
        north_m = (item[4] - centre[0]) * fly_patrol_mission.EARTH_M_PER_DEG
        assert abs(north_m) <= 10   # the area is 20 x 20 m


def test_trigger_is_followed_by_a_nav_item():
    # A DO item right after the trigger waypoint made ArduPilot report its
    # MISSION_ITEM_REACHED with the next seq, so detection never started.
    items = fly_patrol_mission.build_mission(*HOME)
    after = items[load_config(DEFAULT_CONFIG_PATH)["mission"]["trigger_seq"] + 1]
    assert after[1] == mavlink.MAV_CMD_NAV_WAYPOINT
