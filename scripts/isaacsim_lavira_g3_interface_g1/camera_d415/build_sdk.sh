#!/usr/bin/env bash
# 在运行此脚本的机器上编译；笔记本产物不能复制到 ARM64 机器人运行。
# 参数：官方 v2.54.1 源码目录、自己的构建/安装根目录（两个均使用绝对路径）。
set -euo pipefail
sdk_source=${1:?请提供官方 librealsense-v2.54.1 源码目录}
work_root=${2:?请提供自己的构建和安装目录}
[[ "$sdk_source" = /* && "$work_root" = /* ]] || { echo '请使用绝对路径'; exit 2; }
expected_commit=8ffb17b027e100c2a14fa21f01f97a1921ec1e1b
[[ $(git -C "$sdk_source" rev-parse HEAD) = "$expected_commit" ]] || {
  echo '源码提交与官方 v2.54.1 不一致，停止构建。'; exit 2;
}
[[ -z $(git -C "$sdk_source" status --porcelain) ]] || {
  echo 'SDK 源码存在修改，停止构建以保留可追溯的官方基线。'; exit 2;
}
arch=$(uname -m)
sdk_build="$work_root/build-sdk-$arch"
sdk_prefix="$work_root/install-$arch"
probe_source=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
# RSUSB 是官方提供的用户态 USB 后端，不需要运行内核补丁脚本。
# 不启用 CUDA、图形示例或固件下载；构建到用户目录，不调用 sudo。
cmake -S "$sdk_source" -B "$sdk_build" \
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX="$sdk_prefix" \
  -DFORCE_RSUSB_BACKEND=ON -DBUILD_WITH_CUDA=OFF \
  -DBUILD_GRAPHICAL_EXAMPLES=OFF -DBUILD_GLSL_EXTENSIONS=OFF \
  -DBUILD_EXAMPLES=ON -DBUILD_PYTHON_BINDINGS=OFF \
  -DBUILD_UNIT_TESTS=OFF -DIMPORT_DEPTH_CAM_FW=OFF -DCHECK_FOR_UPDATES=OFF
cmake --build "$sdk_build" --parallel "${JOBS:-2}"
cmake --install "$sdk_build"
cmake -S "$probe_source" -B "$work_root/build-probe-$arch" \
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH="$sdk_prefix"
cmake --build "$work_root/build-probe-$arch" --parallel "${JOBS:-2}"
echo "完成：$work_root/build-probe-$arch/g1_d435i_probe"
