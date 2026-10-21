#!/usr/bin/env bash
# Start Gazebo with the patrol world (run inside WSL). Start this first, then SITL.
#
#   sim/run_gazebo.sh              with GUI
#   sim/run_gazebo.sh --headless   server only, the camera still renders
#
# SITL (second terminal, from ~/ardupilot/ArduCopter; drop --console for headless):
#   sim_vehicle.py -v ArduCopter -f gazebo-iris --model JSON --console \
#     --add-param-file=Tools/autotest/default_params/copter.parm \
#     --add-param-file=Tools/autotest/default_params/gazebo-iris.parm
# Restart Gazebo and SITL together: a lone SITL restart can hang the plugin.
set -euo pipefail

WORLD="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/worlds/security_patrol.sdf"

# Finds model://iris_with_gimbal and the ArduPilot plugin.
export GZ_SIM_RESOURCE_PATH="$HOME/ardupilot_gazebo/models:$HOME/ardupilot_gazebo/worlds:${GZ_SIM_RESOURCE_PATH:-}"
export GZ_SIM_SYSTEM_PLUGIN_PATH="$HOME/ardupilot_gazebo/build:${GZ_SIM_SYSTEM_PLUGIN_PATH:-}"
# Render on the NVIDIA GPU, not the Intel iGPU.
export MESA_D3D12_DEFAULT_ADAPTER_NAME=NVIDIA

# The gimbal starts at the horizon. Point it straight down (like the real drone's
# overhead camera) and zoom in to a Pi-camera-like field of view (~80 deg instead
# of 115 deg), so a person is about 40 px tall from 15 m. The joint
# controller keeps the last command, so this only has to be sent once.
(
    until gz topic -l 2>/dev/null | grep -q "^/gimbal/cmd_pitch$"; do sleep 1; done
    sleep 2
    gz topic -t /gimbal/cmd_pitch -m gz.msgs.Double -p "data: 1.5708"
    gz topic -t /model/gimbal/sensor/camera/zoom/cmd_zoom -m gz.msgs.Double -p "data: 1.8"
) &

if [[ "${1:-}" == "--headless" ]]; then
    exec gz sim -v3 -s -r --headless-rendering "$WORLD"
fi
exec gz sim -v3 -r "$WORLD"
