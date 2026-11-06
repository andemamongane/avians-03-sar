#!/usr/bin/env bash
# Enter the running avians container (prompt becomes root@...:/workspace#).
set -euo pipefail
cd "$(dirname "$0")/.."
docker compose -f docker/compose.yml exec avians bash
