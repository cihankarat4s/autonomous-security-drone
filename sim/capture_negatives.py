"""Save camera frames during a flight in the person-free world.

Every saved frame is a negative: no person exists in the world. Frames are
taken at a fixed interval while the vehicle is armed and above --min-alt, so
consecutive images are not near-duplicates. Uses the companion's port (5762),
so do not run the companion at the same time.

    python sim/capture_negatives.py --name sun_default
"""
import argparse
import time
from pathlib import Path

import cv2
from pymavlink import mavutil

from security_drone.camera.gazebo_camera import GazeboCamera
from security_drone.config import DEFAULT_CONFIG_PATH, load_config
from security_drone.mavlink.commands import set_message_interval
from security_drone.mavlink.connection import connect
from security_drone.mavlink.telemetry import parse_heartbeat, parse_position

mavlink = mavutil.mavlink


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--name", required=True, help="flight name, used as sub-folder and prefix")
    parser.add_argument("--root", type=Path, default=Path.home() / "datasets/gazebo_negatives")
    parser.add_argument("--period", type=float, default=0.5, help="seconds between saved frames")
    parser.add_argument("--min-alt", type=float, default=0.5, help="skip frames on the ground")
    parser.add_argument("--connection", default="tcp:127.0.0.1:5762")
    args = parser.parse_args()

    config = load_config(DEFAULT_CONFIG_PATH)
    camera = GazeboCamera(config["camera"]["gazebo_topic"])
    camera.start()
    master = connect(args.connection, 30)
    set_message_interval(master, mavlink.MAVLINK_MSG_ID_GLOBAL_POSITION_INT, 5)

    out_dir = args.root / args.name
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Saving to {out_dir}; waiting for the vehicle to arm")

    armed = was_armed = False
    alt = 0.0
    last_save = 0.0
    saved = 0
    while not (was_armed and not armed):
        msg = master.recv_match(type=["HEARTBEAT", "GLOBAL_POSITION_INT"],
                                blocking=True, timeout=0.1)
        if msg is not None and msg.get_srcSystem() == master.target_system \
                and msg.get_srcComponent() == master.target_component:
            if msg.get_type() == "HEARTBEAT":
                armed = parse_heartbeat(msg).armed
                was_armed = was_armed or armed
            else:
                alt = parse_position(msg).relative_alt_m

        now = time.monotonic()
        if armed and alt >= args.min_alt and now - last_save >= args.period:
            frame = camera.read()
            if frame is not None:
                path = out_dir / f"{args.name}_{saved:04d}_alt{alt:04.1f}.jpg"
                cv2.imwrite(str(path), frame)
                saved += 1
                last_save = now

    print(f"Disarmed: saved {saved} frames to {out_dir}")


if __name__ == "__main__":
    main()
