#!/usr/bin/env bash
set -euo pipefail

# 本脚本只启动本机路径规划服务，不连接 G1，也不启动远端 G3 服务。
# 先确认转换后的权重存在，再把脚本进程替换成 Python 服务进程。
# IPLANNER_* 环境变量用于覆盖权重、解释器、设备和端口，不修改 Python 源码。

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
