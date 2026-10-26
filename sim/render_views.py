"""Render overview pictures of a world from fixed cameras, without the GUI.

Writes a copy of the world with static camera models added, runs Gazebo
headless on it, saves one frame per camera and stops Gazebo. Used for the
README figures and to look at a world when the GUI window is not available.

    python sim/render_views.py --world neighborhood --out logs/views
"""
import argparse
import math
import subprocess
import time
from pathlib import Path

import cv2

from security_drone.camera.gazebo_camera import GazeboCamera

SIM = Path(__file__).resolve().parent

# name: (x, y, z, pitch down in degrees, yaw in degrees from east, horizontal FOV in degrees)
VIEWS = {
    "neighborhood": {
        "overview": (0, -75, 95, 42, 90, 70),
        "corner": (-95, 15, 40, 28, 40, 60),
        "park": (30, 120, 70, 90, 90, 60),
        "street": (-38, 52, 14, 22, 0, 75),
        "plaza": (12, -22, 22, 35, 125, 75),
    },
}
WIDTH, HEIGHT = 1600, 900


def camera_model(name, x, y, z, pitch, yaw, fov):
    return f"""
    <model name="view_{name}">
      <static>true</static>
      <pose>{x} {y} {z} 0 {math.radians(pitch):.4f} {math.radians(yaw):.4f}</pose>
      <link name="link">
        <sensor name="camera" type="camera">
          <camera>
            <horizontal_fov>{math.radians(fov):.4f}</horizontal_fov>
            <image><width>{WIDTH}</width><height>{HEIGHT}</height></image>
            <clip><near>0.5</near><far>1500</far></clip>
          </camera>
          <always_on>1</always_on>
          <update_rate>2</update_rate>
          <topic>view_{name}</topic>
        </sensor>
      </link>
    </model>
"""


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--world", default="neighborhood", choices=sorted(VIEWS))
    parser.add_argument("--out", type=Path, default=Path("logs/views"))
    parser.add_argument("--settle", type=float, default=40.0,
                        help="seconds to let Gazebo load the models before capturing")
    args = parser.parse_args()

    views = VIEWS[args.world]
    world = (SIM / f"worlds/security_patrol_{args.world}.sdf").read_text()
    cameras = "".join(camera_model(n, *v) for n, v in views.items())
    variant = SIM / "worlds/.security_patrol_variant.sdf"
    variant.write_text(world.replace("  </world>", cameras + "  </world>"))

    # Same model paths and GPU choice as run_gazebo.sh.
    env_cmd = (f'export GZ_SIM_RESOURCE_PATH="{SIM}/models:$HOME/ardupilot_gazebo/models:'
               f'$HOME/ardupilot_gazebo/worlds:${{GZ_SIM_RESOURCE_PATH:-}}"; '
               f'export GZ_SIM_SYSTEM_PLUGIN_PATH="$HOME/ardupilot_gazebo/build"; '
               f'export MESA_D3D12_DEFAULT_ADAPTER_NAME=NVIDIA; '
               f'exec gz sim -v2 -s -r --headless-rendering {variant}')
    gz = subprocess.Popen(["bash", "-c", env_cmd])
    try:
        time.sleep(args.settle)
        args.out.mkdir(parents=True, exist_ok=True)
        for name in views:
            cam = GazeboCamera(f"/view_{name}")
            cam.start()
            frame = None
            deadline = time.monotonic() + 60
            while frame is None and time.monotonic() < deadline:
                time.sleep(0.5)
                frame = cam.read()
            if frame is None:
                print(f"{name}: no image")
                continue
            path = args.out / f"{args.world}_{name}.jpg"
            cv2.imwrite(str(path), frame, [cv2.IMWRITE_JPEG_QUALITY, 92])
            print(f"Saved {path}")
    finally:
        gz.terminate()
        try:
            gz.wait(timeout=10)
        except subprocess.TimeoutExpired:
            gz.kill()
        variant.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
