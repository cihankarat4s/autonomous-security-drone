#!/usr/bin/env bash
# End-to-end run, unattended (inside WSL): Gazebo -> SITL -> companion app on
# its default config -> patrol mission -> shut everything down. The companion
# logs every processed frame and saves every detection frame under logs/<name>.
#
#   bash sim/fly_companion_patrol.sh <name> [flat|urban]
#   bash sim/fly_companion_patrol.sh urban_26n urban
#
# Afterwards: python sim/summarize_flight_logs.py logs/<name>.csv
set -euo pipefail

NAME="$1"
WORLD="${2:-urban}"
cd "$(dirname "${BASH_SOURCE[0]}")/.."
source sim/lib.sh
PY="$HOME/venvs/security-drone/bin/python"
mkdir -p logs
rm -rf "logs/$NAME" "logs/$NAME.csv"

GZ_ARGS=(--headless)
[[ "$WORLD" == urban ]] && GZ_ARGS+=(--world urban)

cleanup() {
    pkill -INT -f "[s]ecurity_drone.main" || true
    sleep 2
    # pymavlink may be stuck reconnecting and ignore Ctrl-C.
    pkill -9 -f "[s]ecurity_drone.main" || true
    pkill -f "[s]im_vehicle.py" || true
    pkill -f "[b]in/arducopter" || true
    pkill -f "[m]avproxy" || true
    pkill -9 -f "[g]z sim -v3" || true
}
trap cleanup EXIT

echo "Starting Gazebo (${GZ_ARGS[*]})"
bash sim/run_gazebo.sh "${GZ_ARGS[@]}" > "logs/${NAME}_gazebo.log" 2>&1 &
wait_for "Gazebo start" 180 $! gimbal_topic_up
sleep 5   # let the gimbal command go out and the models load

echo "Starting SITL"
# No terminal here: MAVProxy needs --daemon, or it exits at once.
bash sim/run_sitl.sh --mavproxy-args="--daemon" < /dev/null > "logs/${NAME}_sitl.log" 2>&1 &
wait_for "SITL start" 240 $! port_open 5762
sleep 3

echo "Starting companion"
"$PY" -m security_drone.main --snapshot-dir "logs/$NAME" --frame-log "logs/$NAME.csv" \
    --save-all-detections > "logs/${NAME}_companion.log" 2>&1 &
wait_for "companion start" 120 $! grep -q "GLOBAL_POSITION_INT stream requested" "logs/${NAME}_companion.log"
grep -E "Loaded|Heartbeat" "logs/${NAME}_companion.log"

echo "Flying the patrol mission"
timeout 900 "$PY" sim/fly_patrol_mission.py
sleep 3
echo "== companion alarms"
grep -E "Trigger|Sent STATUSTEXT" "logs/${NAME}_companion.log" || true
