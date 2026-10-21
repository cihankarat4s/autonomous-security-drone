#!/usr/bin/env bash
# Start ArduCopter SITL for the Gazebo world (run inside WSL after sim/run_gazebo.sh).
#
#   sim/run_sitl.sh                    MAVProxy prompt in this terminal
#   sim/run_sitl.sh --console --map    with MAVProxy console and map windows
#
# The two param files set the Gazebo iris frame; without them FRAME_CLASS stays 0
# and pre-arm fails. Absolute paths, because sim_vehicle.py resolves relative
# ones from a different directory.
set -euo pipefail

PARAMS="$HOME/ardupilot/Tools/autotest/default_params"
cd "$HOME/ardupilot/ArduCopter"
exec sim_vehicle.py -v ArduCopter -f gazebo-iris --model JSON \
    --add-param-file="$PARAMS/copter.parm" \
    --add-param-file="$PARAMS/gazebo-iris.parm" "$@"
