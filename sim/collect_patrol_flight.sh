#!/usr/bin/env bash
# One patrol flight over the urban world with people in it, frames saved for
# offline scoring (run inside WSL):
# Gazebo (headless, urban, intruder) -> SITL -> frame capture -> patrol mission
# -> shut everything down. Frames go to ~/datasets/gazebo_urban/<name>; the
# hover over the plaza is the frames named *_seq3.jpg (LOITER_TIME).
#
#   bash sim/collect_patrol_flight.sh <name> [sun x,y,z]
#   bash sim/collect_patrol_flight.sh patrol_2 0.5,0.5,-0.7
set -euo pipefail

NAME="$1"
SUN="${2:-}"
cd "$(dirname "${BASH_SOURCE[0]}")/.."
PY="$HOME/venvs/security-drone/bin/python"
mkdir -p logs

GZ_ARGS=(--headless --world urban)
[[ -n "$SUN" ]] && GZ_ARGS+=(--sun "$SUN")

cleanup() {
    pkill -f "[c]apture_negatives.py" || true
    pkill -f "[s]im_vehicle.py" || true
    pkill -f "[b]in/arducopter" || true
    pkill -f "[m]avproxy" || true
    pkill -9 -f "[g]z sim -v3" || true
}
trap cleanup EXIT

echo "Starting Gazebo (${GZ_ARGS[*]})"
bash sim/run_gazebo.sh "${GZ_ARGS[@]}" > "logs/collect_${NAME}_gazebo.log" 2>&1 &
until gz topic -l 2>/dev/null | grep -q "^/gimbal/cmd_pitch$"; do sleep 2; done
sleep 5   # let the gimbal command go out and the models load

echo "Starting SITL"
# No terminal here: MAVProxy needs --daemon, or it exits at once.
bash sim/run_sitl.sh --mavproxy-args="--daemon" < /dev/null > "logs/collect_${NAME}_sitl.log" 2>&1 &
until (echo > /dev/tcp/127.0.0.1/5762) 2>/dev/null; do sleep 2; done
sleep 3

"$PY" sim/capture_negatives.py --name "$NAME" --root "$HOME/datasets/gazebo_urban" --period 0.25 &
CAPTURE=$!
sleep 8
"$PY" sim/fly_patrol_mission.py
wait "$CAPTURE"
