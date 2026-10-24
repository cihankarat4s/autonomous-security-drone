#!/usr/bin/env bash
# Held-out data for the confidence threshold: one person-free flat flight and
# one urban patrol with people, both under a sun direction no training or
# threshold-sweep flight used.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
SUN=0.6,0.4,-0.7
bash sim/collect_negative_flight.sh val_e flat "$SUN"
bash sim/collect_patrol_flight.sh patrol_2 "$SUN"
