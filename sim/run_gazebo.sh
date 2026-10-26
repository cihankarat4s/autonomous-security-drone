#!/usr/bin/env bash
# Start Gazebo with the patrol world (run inside WSL). Start this first, then SITL.
#
#   sim/run_gazebo.sh                         with GUI
#   sim/run_gazebo.sh --headless              server only, the camera still renders
#   sim/run_gazebo.sh --headless --no-intruder --sun 0.4,-0.6,-0.7
#   sim/run_gazebo.sh --headless --world urban
#   sim/run_gazebo.sh --world ortho
#   sim/run_gazebo.sh --world neighborhood
#   sim/run_gazebo.sh --headless --world ortho --no-people
#
# --world urban loads security_patrol_urban.sdf (textured ground, trees,
# buildings, vehicles, several people) instead of the flat default world.
# --world ortho loads security_patrol_ortho.sdf: a real drone orthophoto as
# the ground (sim/make_ortho_ground.py) with people in the protected area.
# --world neighborhood loads security_patrol_neighborhood.sdf, a street grid
# with houses, cars, trees and walking people (sim/make_neighborhood.py); fly
# it with fly_patrol_mission.py --route streets.
#
# --no-intruder removes the walking person, so every camera frame is a
# guaranteed negative (used to collect hard negatives such as the drone's own
# shadow). --no-people also removes the standing people (person_* models).
# --sun sets the sun direction, which moves and stretches shadows.
#
# Then start SITL in a second terminal with sim/run_sitl.sh.
# Restart Gazebo and SITL together: a lone SITL restart can hang the plugin.
set -euo pipefail

SIM_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORLD="$SIM_DIR/worlds/security_patrol.sdf"

HEADLESS=0
NO_INTRUDER=0
NO_PEOPLE=0
SUN=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --headless) HEADLESS=1 ;;
        --world) WORLD="$SIM_DIR/worlds/security_patrol_$2.sdf"; shift ;;
        --no-intruder) NO_INTRUDER=1 ;;
        --no-people) NO_INTRUDER=1; NO_PEOPLE=1 ;;
        # Comma-separated, so the value survives Windows -> WSL quoting.
        --sun) SUN="${2//,/ }"; shift ;;
        *) echo "Unknown option $1" >&2; exit 1 ;;
    esac
    shift
done

# The orthophoto ground is downloaded and rotated once, not stored in git.
if [[ "$WORLD" == *ortho* && ! -f "$SIM_DIR/models/ortho_ground/materials/textures/ground.jpg" ]]; then
    "$HOME/venvs/security-drone/bin/python" "$SIM_DIR/make_ortho_ground.py"
fi

# Variants are written next to the original so model:// lookups stay the same.
if [[ $NO_INTRUDER == 1 || -n "$SUN" ]]; then
    VARIANT="$(dirname "$WORLD")/.security_patrol_variant.sdf"
    cp "$WORLD" "$VARIANT"
    if [[ $NO_INTRUDER == 1 ]]; then
        sed -i '/<actor name="intruder">/,/<\/actor>/d' "$VARIANT"
    fi
    if [[ $NO_PEOPLE == 1 ]]; then
        perl -0pi -e 's|\s*<include>\s*<name>person_.*?</include>||gs' "$VARIANT"
        sed -i '/<actor name=/,/<\/actor>/d' "$VARIANT"
    fi
    if [[ -n "$SUN" ]]; then
        sed -i "s|<direction>.*</direction>|<direction>$SUN</direction>|" "$VARIANT"
    fi
    WORLD="$VARIANT"
fi

# The urban ground texture is generated, not stored in git.
if [[ ! -f "$SIM_DIR/models/urban_ground/materials/textures/ground.jpg" ]]; then
    "$HOME/venvs/security-drone/bin/python" "$SIM_DIR/make_ground_texture.py"
fi
if [[ ! -f "$SIM_DIR/models/neighborhood_ground/materials/textures/ground.jpg" ]]; then
    "$HOME/venvs/security-drone/bin/python" "$SIM_DIR/make_neighborhood.py"
fi

# Finds model://iris_with_gimbal, model://urban_ground and the ArduPilot plugin.
export GZ_SIM_RESOURCE_PATH="$SIM_DIR/models:$HOME/ardupilot_gazebo/models:$HOME/ardupilot_gazebo/worlds:${GZ_SIM_RESOURCE_PATH:-}"
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

if [[ $HEADLESS == 1 ]]; then
    exec gz sim -v3 -s -r --headless-rendering "$WORLD"
fi
exec gz sim -v3 -r "$WORLD"
