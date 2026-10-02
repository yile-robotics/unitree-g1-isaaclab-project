#!/usr/bin/env python3
"""用VLN DDS后端前进0.5m/s（最多3秒），测试Ctrl+C触发stop()/close()。

只加载运动后端；现场保留前方空间和遥控器停车手段。
"""

from __future__ import annotations

import argparse
import math
import os
import signal
import time

from g1_real_smoke import StateMonitor
from unified_vln.g1_dds_backend import UnitreeG1DDSBackend


_active_dds: UnitreeG1DDSBackend | None = None
_active_subscriber = None
_cleaned_up = False


def _cleanup() -> None:
    """与 VLN 入口一致：先清零并 StopMove，再停止发送线程。"""

    global _active_dds, _active_subscriber, _cleaned_up
    if _cleaned_up:
        return
    _cleaned_up = True
    dds = _active_dds
    if dds is not None:
        try:
            dds.stop()
        except Exception as exc:
            print(f"[CTRL-C PROBE] stop error: {exc}", flush=True)
        try:
            dds.close()
        except Exception as exc:
            print(f"[CTRL-C PROBE] close error: {exc}", flush=True)
        _active_dds = None
    if _active_subscriber is not None:
        try:
            _active_subscriber.Close()
        except Exception as exc:
            print(f"[CTRL-C PROBE] subscriber close error: {exc}", flush=True)
        _active_subscriber = None


def _on_sigint(_signum, _frame) -> None:
    """仿照 VLN 的 SIGINT handler，清理后直接退出整个进程。"""

    print("\n[CTRL-C PROBE] Ctrl+C received; stopping VLN DDS backend...", flush=True)
    _cleanup()
    print("[CTRL-C PROBE] cleanup returned; confirm robot stopped visually", flush=True)
    os._exit(0)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--network-interface", required=True)
    parser.add_argument("--seconds", type=float, default=2.0, help="motion window, at most 3.0 s")
    parser.add_argument("--execute", action="store_true", help="required to send motion commands")
    args = parser.parse_args()
    if not math.isfinite(args.seconds) or not 0 < args.seconds <= 3.0:
        parser.error("--seconds must be in (0, 3.0]")
    if not args.execute:
        parser.error("--execute is required for real robot motion")

    signal.signal(signal.SIGINT, _on_sigint)

    # 与 VLN 主程序使用同一个 DDS 后端，包含持续发送速度的后台线程。
    from unitree_sdk2py.core.channel import ChannelSubscriber
    from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowState_

    global _active_dds, _active_subscriber
    try:
        _active_dds = UnitreeG1DDSBackend(args.network_interface)
        monitor = StateMonitor()
        _active_subscriber = ChannelSubscriber("rt/lowstate", LowState_)
        _active_subscriber.Init(monitor.on_state)

        # 收不到真机状态就不下发非零速度；与现有 smoke test 保持一致。
        wait_deadline = time.monotonic() + 3.0
        while time.monotonic() < wait_deadline:
            try:
                monitor.require_fresh()
                break
            except RuntimeError:
                time.sleep(0.05)
        else:
            raise RuntimeError("no fresh G1 lowstate within 3 seconds; refusing motion")

        print("[CTRL-C PROBE] lowstate fresh; motion starts after countdown", flush=True)
        for remaining in (3, 2, 1):
            monitor.require_fresh()
            print(f"[CTRL-C PROBE] starting in {remaining}...", flush=True)
            time.sleep(1.0)
        monitor.require_fresh()
        print(f"[CTRL-C PROBE] moving at vx=0.5 m/s for at most {args.seconds:.1f} s", flush=True)
        print("[CTRL-C PROBE] press Ctrl+C while the robot is moving", flush=True)
        _active_dds.set_velocity(0.5, 0.0, 0.0)
        deadline = time.monotonic() + args.seconds
        while time.monotonic() < deadline:
            monitor.require_fresh()
            time.sleep(min(0.05, max(0.0, deadline - time.monotonic())))
        print("[CTRL-C PROBE] time limit reached without Ctrl+C; stopping", flush=True)
        return 0
    finally:
        _cleanup()


if __name__ == "__main__":
    raise SystemExit(main())
