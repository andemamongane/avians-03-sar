#!/usr/bin/env bash
# Linux machine: start the avians container next to a running aerial-autonomy-stack
# (docker/compose.remote.yml). Start AAS first (README.md, "Gazebo").
set -euo pipefail
cd "$(dirname "$0")/.."
[ -f .env ] || { cp .env.example .env; echo "Created .env; add your ANTHROPIC_API_KEY to it."; }
docker network inspect aas-sim-network-inst0 >/dev/null 2>&1 || {
  echo "AAS is not running (no aas-sim-network-inst0). Start it first:"
  echo "  cd external/aerial-autonomy-stack/tools_and_docs && NUM_QUADS=1 WORLD=impalpable_greyness ./sim_run.sh"
  exit 1; }
docker compose -f docker/compose.yml up -d --build
docker compose -f docker/compose.yml exec avians bash -lc "avians doctor"
