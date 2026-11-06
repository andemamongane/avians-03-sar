#!/usr/bin/env bash
# Install every avians package into the active Python (>= 3.10), editable, so
# code changes apply without reinstalling. Use a virtualenv:
#   python3 -m venv .venv && source .venv/bin/activate && ./scripts/install.sh
set -euo pipefail
cd "$(dirname "$0")/.."
python3 -m pip install --upgrade pip setuptools
args=()
for p in env spec perception skills safety sim adp viz agent; do args+=(-e "packages/avians_$p"); done
python3 -m pip install "${args[@]}" pytest "anthropic>=1"
echo "installed; try: avians doctor && avians fly missions/sar_supply.yaml --video"
