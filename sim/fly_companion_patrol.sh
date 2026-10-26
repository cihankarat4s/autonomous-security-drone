#!/usr/bin/env bash
# End-to-end run, unattended (inside WSL): Gazebo -> SITL -> companion app on
# its default config -> patrol mission -> shut everything down. The companion
# logs every processed frame and saves every detection frame under logs/<name>.
# The drone patrols the area until the companion's patrol time is up (config
# mission.patrol_time_s, or the third argument in seconds) and it sends RTL.
#
#   bash sim/fly_companion_patrol.sh <name> [flat|urban|ortho|neighborhood] [patrol seconds]
#   bash sim/fly_companion_patrol.sh patrol_test urban 120
#   GZ_EXTRA="--no-people" bash sim/fly_companion_patrol.sh dogs_only neighborhood 150
#
# Afterwards: python sim/summarize_flight_logs.py logs/<name>.csv
set -euo pipefail

NAME="$1"
WORLD="${2:-urban}"
PATROL_S="${3:-}"
cd "$(dirname "${BASH_SOURCE[0]}")/.."
source sim/lib.sh
PY="$HOME/venvs/security-drone/bin/python"
mkdir -p logs
rm -rf "logs/$NAME" "logs/$NAME.csv"

COMPANION_ARGS=()
[[ -n "$PATROL_S" ]] && COMPANION_ARGS+=(--patrol-time "$PATROL_S")
GZ_ARGS=(--headless)
# Extra run_gazebo.sh options, e.g. GZ_EXTRA="--no-people" for a false-positive run.
read -r -a EXTRA <<< "${GZ_EXTRA:-}"
GZ_ARGS+=("${EXTRA[@]}")
[[ "$WORLD" != flat ]] && GZ_ARGS+=(--world "$WORLD")

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
    --save-all-detections "${COMPANION_ARGS[@]}" > "logs/${NAME}_companion.log" 2>&1 &
wait_for "companion start" 120 $! grep -q "GLOBAL_POSITION_INT stream requested" "logs/${NAME}_companion.log"
grep -E "Loaded|Heartbeat" "logs/${NAME}_companion.log"

echo "Flying the patrol mission"
ROUTE=square
[[ "$WORLD" == neighborhood ]] && ROUTE=streets
timeout 1500 "$PY" sim/fly_patrol_mission.py --route "$ROUTE"
sleep 3
echo "== companion"
grep -E "Trigger|Patrol|patrol|RTL|Sent STATUSTEXT|Detection stopped" "logs/${NAME}_companion.log" || true
