#!/usr/bin/env bash
# One hard-negative collection flight, start to finish (run inside WSL):
# Gazebo (headless, no intruder) -> SITL -> frame capture -> negative mission
# -> shut everything down. Gazebo and SITL output go to logs/collect_<name>_*.log.
#
#   bash sim/collect_negative_flight.sh <name> <flat|urban> [sun x,y,z]
#   bash sim/collect_negative_flight.sh urban_a urban
#   bash sim/collect_negative_flight.sh urban_c urban 0.15,0.1,-1
set -euo pipefail

NAME="$1"
ZONE="$2"
SUN="${3:-}"
cd "$(dirname "${BASH_SOURCE[0]}")/.."
PY="$HOME/venvs/security-drone/bin/python"
mkdir -p logs

GZ_ARGS=(--headless --no-intruder)
[[ "$ZONE" == urban ]] && GZ_ARGS+=(--world urban)
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

"$PY" sim/capture_negatives.py --name "$NAME" &
CAPTURE=$!
sleep 8
"$PY" sim/fly_negative_mission.py --zone "$ZONE"
wait "$CAPTURE"
