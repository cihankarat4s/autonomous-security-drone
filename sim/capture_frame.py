"""Save a Gazebo camera frame, by default when the trigger waypoint is reached.

Run it before starting the mission (SITL and Gazebo running):
    python sim/capture_frame.py --out logs/camera_wp2.png
    python sim/capture_frame.py --now --out /tmp/test.png   # grab a frame right away
"""
import argparse
import time
from pathlib import Path

import cv2
from pymavlink import mavutil

from security_drone.camera.gazebo_camera import GazeboCamera
from security_drone.config import DEFAULT_CONFIG_PATH, load_config
from security_drone.mavlink.connection import connect

mavlink = mavutil.mavlink


def wait_for_frame(camera, timeout_s=10):
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        frame = camera.read()
        if frame is not None:
            return frame
        time.sleep(0.1)
    raise RuntimeError("No camera frame received")


def wait_for_trigger(master, trigger_seq):
    while True:
        msg = master.recv_match(type="MISSION_ITEM_REACHED", blocking=True, timeout=5)
        if msg is not None and msg.seq == trigger_seq:
            return


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--connection", help="override mavlink.connection")
    parser.add_argument("--out", default="logs/camera_wp2.png")
    parser.add_argument("--now", action="store_true", help="do not wait for the trigger")
    args = parser.parse_args()

    config = load_config(args.config)
    camera = GazeboCamera(config["camera"]["gazebo_topic"])
    camera.start()
    wait_for_frame(camera)

    if not args.now:
        master = connect(args.connection or config["mavlink"]["connection"],
                         config["mavlink"]["heartbeat_timeout_s"])
        print("Waiting for the trigger waypoint")
        wait_for_trigger(master, config["mission"]["trigger_seq"])

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(args.out, camera.read())
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
