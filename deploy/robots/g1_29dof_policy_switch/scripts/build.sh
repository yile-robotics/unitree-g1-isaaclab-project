#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cmake -S "${PROJECT_DIR}" -B "${PROJECT_DIR}/build"
cmake --build "${PROJECT_DIR}/build" -j"$(nproc)"

