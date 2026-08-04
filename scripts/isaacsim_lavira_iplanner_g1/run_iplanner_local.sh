#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECTS_DIR="$(cd "$SCRIPT_DIR/../../.." && pwd)"
IPLANNER_DIR="${UNILAVIRA_IPLANNER_DIR:-$PROJECTS_DIR/uni-lavira-code/real-world-code/unitree_g1/iplanner}"
CHECKPOINT="${IPLANNER_CHECKPOINT:-$SCRIPT_DIR/checkpoints/iplanner.pth}"
PYTHON_BIN="${IPLANNER_PYTHON:-python}"
DEVICE="${IPLANNER_DEVICE:-cuda}"
PORT="${IPLANNER_PORT:-8888}"

if [[ ! -f "$CHECKPOINT" ]]; then
    echo "ERROR: converted iPlanner checkpoint is missing: $CHECKPOINT" >&2
    exit 1
fi
if [[ ! -f "$IPLANNER_DIR/iplanner_server.py" ]]; then
    echo "ERROR: Uni-LaViRA iPlanner server is missing: $IPLANNER_DIR" >&2
    exit 1
fi

exec "$PYTHON_BIN" "$IPLANNER_DIR/iplanner_server.py" \
    --config "$IPLANNER_DIR/configs/iplanner.yaml" \
    --checkpoint "$CHECKPOINT" \
    --device "$DEVICE" \
    --port "$PORT"
