#!/usr/bin/env bash
# The Linux demo: AAS/Gazebo footage of the SAR mission, one drone and a swarm,
# plus QGroundControl logs. Run INSIDE the avians container, with AAS running:
#
#   ./scripts/shell.sh                 # host: enter the container
#   ./scripts/linux_demo.sh            # container: replays only (safe, ~10 min)
#   LIVE=1 ./scripts/linux_demo.sh     # also a live PX4 flight in Gazebo (one drone)
#   SPEED=1 ./scripts/linux_demo.sh    # replay in real time (default 3x faster)
#
# Stages (each logged; a failing stage does not stop the next):
#   1  checks       avians doctor, docker socket, AAS simulation container, Gazebo world
#   2  replays      the committed example episodes (examples/episodes: one drone in
#                   a storm; two Haiku pilots in a storm) re-flown in AAS's Gazebo as
#                   visual drones, recorded: gazebo.mp4 (chase) + gazebo_overview.mp4
#   3  qgc          flight.tlog + world.kml per episode (QGC: Analyze > Log Replay)
#   4  2d           the 2D replay of each episode (replay.mp4), for side-by-side
#   5  live         (LIVE=1) the scripted pilot flies sar_supply through PX4 in Gazebo
#   6  pack         everything into data/linux_demo/<stamp>.tar.gz: send that back
set -uo pipefail
cd "$(dirname "$0")/.."
STAMP=$(date +%Y%m%d_%H%M%S)
OUT=data/linux_demo/$STAMP
SPEED=${SPEED:-3}
mkdir -p "$OUT"
LOG="$OUT/demo.log"
ok=(); bad=()
stage() { echo; echo "=================== $* ===================" | tee -a "$LOG"; }
run() {   # run <label> <command...>: log it, record pass/fail, never exit
  local label=$1; shift
  echo "\$ $*" | tee -a "$LOG"
  if "$@" 2>&1 | tee -a "$LOG"; then ok+=("$label"); else bad+=("$label"); fi
}

stage "1  checks"
run doctor avians doctor
run docker-socket bash -c 'docker ps >/dev/null && echo "docker socket ok"'
run aas-running bash -c 'docker ps --format "{{.Names}}" | grep -q simulation-container-inst0 && echo "AAS simulation container is up" || { echo "AAS is not running: start sim_run.sh on the host (docs/linux_test.md, step 3)"; exit 1; }'
run gazebo-world bash -c 'docker exec simulation-container-inst0 bash -lc "gz topic -l" | grep -m1 "^/world/" || { echo "no Gazebo world topics"; exit 1; }'

stage "2  replays in Gazebo (speed ${SPEED}x)"
for ep in examples/episodes/*/; do
  name=$(basename "$ep")
  cp -r "$ep" "$OUT/$name"
  run "replay:$name" avians gazebo-replay "$OUT/$name" --speed "$SPEED"
  ls -la "$OUT/$name"/gazebo*.mp4 2>&1 | tee -a "$LOG"
done

stage "3  QGroundControl logs"
for ep in "$OUT"/*/; do run "qgc:$(basename "$ep")" avians qgc "$ep"; done

stage "4  2D replays"
for ep in "$OUT"/*/; do run "2d:$(basename "$ep")" avians video "$ep" --speed 25; done

if [ "${LIVE:-0}" = "1" ]; then
  stage "5  live flight through PX4 in Gazebo (one drone, scripted pilot)"
  run live avians fly missions/sar_supply.yaml --tier gazebo --seed 0 --out "$OUT/live" --qgc
fi

stage "6  pack"
tar -czf "data/linux_demo/$STAMP.tar.gz" -C data/linux_demo "$STAMP" 2>&1 | tee -a "$LOG"
echo | tee -a "$LOG"
echo "passed: ${ok[*]:-none}" | tee -a "$LOG"
echo "failed: ${bad[*]:-none}" | tee -a "$LOG"
echo "videos:" | tee -a "$LOG"
find "$OUT" -name "*.mp4" -size +10k -printf "  %p (%k KB)\n" 2>/dev/null | tee -a "$LOG"
echo "send back: data/linux_demo/$STAMP.tar.gz" | tee -a "$LOG"
