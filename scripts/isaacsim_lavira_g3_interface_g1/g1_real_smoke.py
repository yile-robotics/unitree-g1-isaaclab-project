#!/usr/bin/env python3
"""G1 真机高层 DDS 接口的独立冒烟测试，不启动 VLN 导航状态机。

建议现场依次运行 status、stop、stand、move，每次只测试一种动作。
运行前须按当前机器人和遥控器的说明确认人工停止方式，并安排专人操作。
StopMove 只是软件零速度请求，既不能替代人工急停，也不能证明机器人已停止。
HighStand 只请求内置运动控制器调整站立高度，不负责从倒地状态起身，
也不会部署 Isaac Sim 中训练的站立或行走策略。
"""

from __future__ import annotations

import argparse
import math
import signal
import threading
import time


# 冒烟测试的硬上限；不提供命令行覆盖，避免误输大速度。
MAX_LINEAR_M_S = 0.15
MAX_YAW_RAD_S = 0.20
MAX_MOVE_SECONDS = 1.0
# 单独启用的前向速度试验：仅允许 0.5 m/s，最长 1.0 秒。
FAST_FORWARD_M_S = 0.5
FAST_FORWARD_MAX_SECONDS = 1.0
# 只有显式 Ctrl+C 测试才放宽至 2 秒；忘记按键时也会到时停车。
CTRL_C_TEST_MAX_SECONDS = 2.0
# 与 SDK HighStand() 使用同一个高度值；直接调底层包装以取得 RPC 返回码。
HIGH_STAND_HEIGHT_SENTINEL = (1 << 32) - 1
# 超过这段时间收不到有效状态，就拒绝继续发送非零速度。
STATE_MAX_AGE_SECONDS = 0.5
# 单条速度 RPC 请求的有效时长；实际失联停车行为仍须在真机验证。
VELOCITY_COMMAND_LIFETIME_SECONDS = 0.3
# StopMove 对照测试：最后一条前进命令在观察窗口后才会自然到期。
STOPMOVE_PROBE_FORWARD_SECONDS = 0.8
STOPMOVE_PROBE_COMMAND_LIFETIME_SECONDS = 1.5
STOPMOVE_PROBE_OBSERVE_SECONDS = 0.6


class StateMonitor:
    """监听 G1 的 rt/lowstate，仅用 IMU yaw 和接收时间判断状态是否持续到达。"""

    def __init__(self) -> None:
        # DDS 回调与主线程可能并发访问状态，用锁保护接收时间和 yaw。
        self._lock = threading.Lock()
        self._last_time: float | None = None
        self._yaw: float | None = None

    def on_state(self, message) -> None:
        """丢弃损坏消息；不把 LowState 中的状态信息当作 SLAM 世界位姿。"""
        try:
            yaw = float(message.imu_state.rpy[2])
        except (AttributeError, IndexError, TypeError, ValueError):
            return
        if not math.isfinite(yaw):
            return
        with self._lock:
            self._yaw = yaw
            self._last_time = time.monotonic()

    def snapshot(self) -> tuple[float | None, float | None]:
        """原子读取最近一次有效状态的本机接收时间与 yaw。"""
        with self._lock:
            return self._last_time, self._yaw

    def require_fresh(self) -> float:
        """消息超过 0.5 秒未更新时失败，供动作前和动作期间检查。"""
        last_time, yaw = self.snapshot()
        if last_time is None or yaw is None or time.monotonic() - last_time > STATE_MAX_AGE_SECONDS:
            raise RuntimeError("G1 lowstate missing or stale; refusing motion")
        return yaw


def validate_move(
    vx: float, vy: float, wz: float, seconds: float, *,
    fast_forward_test: bool = False, ctrl_c_test: bool = False,
) -> None:
    """普通测试保持低速限制；Ctrl+C 试验才允许前进最多 2 秒。"""
    values = (vx, vy, wz, seconds)
    if not all(math.isfinite(value) for value in values):
        raise ValueError("all motion values must be finite")
    if ctrl_c_test and not fast_forward_test:
        raise ValueError("--ctrl-c-test requires --fast-forward-test")
    if fast_forward_test:
        # 只放宽显式 Ctrl+C 试验的时长，不放宽速度、方向或轴数。
        if (vx, vy, wz) != (FAST_FORWARD_M_S, 0.0, 0.0):
            raise ValueError("fast-forward test requires vx=0.5, vy=0, wz=0")
        limit = CTRL_C_TEST_MAX_SECONDS if ctrl_c_test else FAST_FORWARD_MAX_SECONDS
        if not 0 < seconds <= limit:
            raise ValueError(f"fast-forward duration must be in (0, {limit}] seconds")
        return
    if not 0 < seconds <= MAX_MOVE_SECONDS:
        raise ValueError(f"duration must be in (0, {MAX_MOVE_SECONDS}] seconds")
    if abs(vx) > MAX_LINEAR_M_S or abs(vy) > MAX_LINEAR_M_S or abs(wz) > MAX_YAW_RAD_S:
        raise ValueError("velocity exceeds the fixed smoke-test limit")
    if sum(value != 0.0 for value in (vx, vy, wz)) != 1:
        raise ValueError("give exactly one nonzero axis per test")


def request_high_stand(client) -> None:
    """发送 SDK HighStand 对应的高度 RPC，并检查包装函数原本丢弃的返回码。"""
    code = client.SetStandHeight(HIGH_STAND_HEIGHT_SENTINEL)
    if code != 0:
        raise RuntimeError(f"HighStand height RPC failed: {code}")


def stop_move(client) -> None:
    """先调用 SDK StopMove，再发一次可检查返回码的零速度 RPC。"""
    # Python SDK 的 StopMove 包装函数不返回 RPC 状态码。即使它抛异常，
    # finally 仍尽力再发零速度；请求成功也不等于物理急停得到验证。
    try:
        client.StopMove()
    finally:
        code = client.SetVelocity(0.0, 0.0, 0.0, VELOCITY_COMMAND_LIFETIME_SECONDS)
        if code != 0:
            raise RuntimeError(f"zero-velocity RPC failed: {code}")


def move_for(
    client, monitor: StateMonitor, vx: float, vy: float, wz: float, seconds: float,
    *, fast_forward_test: bool = False, ctrl_c_test: bool = False,
) -> None:
    """限时重复发送单轴速度，正常结束或异常时都尝试软件停车。"""
    validate_move(vx, vy, wz, seconds, fast_forward_test=fast_forward_test, ctrl_c_test=ctrl_c_test)
    # 没有实时状态就不开始运动；循环内继续复查状态是否过期。
    monitor.require_fresh()
    deadline = time.monotonic() + seconds
    try:
        while time.monotonic() < deadline:
            monitor.require_fresh()
            # SetVelocity 是 LocoClient.Move 底层使用的同一个运动 RPC；
            # 这里缩短单条指令有效期，并检查 SDK 返回码。
            code = client.SetVelocity(vx, vy, wz, VELOCITY_COMMAND_LIFETIME_SECONDS)
            if code != 0:
                raise RuntimeError(f"velocity RPC failed: {code}")
            # 约每 0.1 秒续发一次，同时不能睡过本次测试的结束时刻。
            time.sleep(min(0.1, max(0.0, deadline - time.monotonic())))
    finally:
        # Ctrl+C、RPC 错误和状态过期都会请求停车；SIGKILL/断电不会执行 finally。
        stop_move(client)


def probe_stopmove(client, monitor: StateMonitor) -> None:
    """先短时前进，仅调用一次 StopMove，再在观察窗口结束后发备用零速度。"""
    monitor.require_fresh()
    forward_deadline = time.monotonic() + STOPMOVE_PROBE_FORWARD_SECONDS
    try:
        # 约 10 Hz 续发，使动作起步；较长的单条有效期用于区分自然到期。
        while time.monotonic() < forward_deadline:
            monitor.require_fresh()
            code = client.SetVelocity(
                FAST_FORWARD_M_S, 0.0, 0.0,
                STOPMOVE_PROBE_COMMAND_LIFETIME_SECONDS,
            )
            if code != 0:
                raise RuntimeError(f"probe forward RPC failed: {code}")
            time.sleep(min(0.1, max(0.0, forward_deadline - time.monotonic())))

        # SDK StopMove() 内部就是 SetVelocity(0, 0, 0)，且不返回 RPC 状态码。
        # 在下方观察窗口结束之前，脚本不发送第二条显式零速度。
        print("StopMove request starting; observe robot before backup zero", flush=True)
        call_started = time.monotonic()
        client.StopMove()
        print(
            f"StopMove returned after {time.monotonic() - call_started:.3f} s; "
            "observe stopping now", flush=True,
        )
        observe_deadline = time.monotonic() + STOPMOVE_PROBE_OBSERVE_SECONDS
        while time.monotonic() < observe_deadline:
            monitor.require_fresh()
            time.sleep(min(0.05, max(0.0, observe_deadline - time.monotonic())))
    finally:
        # 窗口到时、RPC 异常、状态丢失或 Ctrl+C 都发备用零速度。
        # 这条备用请求不属于 StopMove 单独生效的观察区间。
        code = client.SetVelocity(0.0, 0.0, 0.0, VELOCITY_COMMAND_LIFETIME_SECONDS)
        if code != 0:
            raise RuntimeError(f"backup zero-velocity RPC failed: {code}")
        print("backup zero-velocity RPC code=0", flush=True)


def build_parser() -> argparse.ArgumentParser:
    """分开只读状态与物理动作，动作必须显式选择。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--network-interface", help="network interface connected to G1, e.g. enp4s0")
    parser.add_argument("--execute", action="store_true", help="required for stand, move and stopmove-probe")
    actions = parser.add_subparsers(dest="action", required=True)
    actions.add_parser("status", help="read G1 lowstate without sending control RPCs")
    actions.add_parser("stop", help="send software zero velocity and StopMove")
    actions.add_parser("stand", help="request HighStand; robot must already be in its normal stable control mode")
    actions.add_parser(
        "stopmove-probe",
        help="forward at 0.5 m/s for 0.8 s, call StopMove, observe 0.6 s, then send backup zero",
    )
    move = actions.add_parser("move", help="send one bounded velocity axis, then stop")
    move.add_argument("--vx", type=float, default=0.0)
    move.add_argument("--vy", type=float, default=0.0)
    move.add_argument("--wz", type=float, default=0.0)
    move.add_argument("--seconds", type=float, default=0.5)
    move.add_argument(
        "--fast-forward-test", action="store_true",
        help="explicitly permit only vx=0.5 m/s forward for at most 1.0 s",
    )
    move.add_argument(
        "--ctrl-c-test", action="store_true",
        help="with --fast-forward-test, permit at most 2.0 s to test Ctrl+C stop",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """先校验参数再初始化 DDS，避免无效命令触碰真机。"""
    args = build_parser().parse_args(argv)
    if not args.network_interface:
        raise SystemExit("--network-interface is required")
    if args.action in ("stand", "move", "stopmove-probe") and not args.execute:
        raise SystemExit("stand/move/stopmove-probe require --execute and a ready manual stop operator")
    if args.action == "move":
        validate_move(
            args.vx, args.vy, args.wz, args.seconds,
            fast_forward_test=args.fast_forward_test, ctrl_c_test=args.ctrl_c_test,
        )

    # 延迟导入 SDK：离线测试直接导入本文件时不会自动打开 DDS。
    from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelSubscriber
    from unitree_sdk2py.g1.loco.g1_loco_client import LocoClient
    from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowState_

    ChannelFactoryInitialize(0, args.network_interface)
    monitor = StateMonitor()
    # status 只建立此订阅器，不创建运动客户端。
    subscriber = ChannelSubscriber("rt/lowstate", LowState_)
    subscriber.Init(monitor.on_state)
    client = None
    try:
        if args.action != "status":
            # stop/stand/move/stopmove-probe 才建立高层运动 RPC 客户端，不涉及关节级控制。
            client = LocoClient()
            client.SetTimeout(2.0)
            client.Init()

        if args.action in ("status", "stand", "move", "stopmove-probe"):
            # 最多等 3 秒收有效状态；stop 无需等待，便于尽快请求停车。
            deadline = time.monotonic() + 3.0
            while time.monotonic() < deadline:
                try:
                    yaw = monitor.require_fresh()
                    print(f"lowstate received; IMU yaw={yaw:.3f} rad", flush=True)
                    break
                except RuntimeError:
                    time.sleep(0.05)
            else:
                raise RuntimeError("no valid G1 lowstate within 3 seconds")

        if args.action == "stop":
            stop_move(client)
            print("software stop RPC sent; confirm actual robot motion visually", flush=True)
        elif args.action == "stand":
            try:
                # 使用与 SDK HighStand() 相同的高度参数，但检查 RPC 返回码。
                # code=0 只说明请求未报错，不说明真机一定改变了姿态。
                request_high_stand(client)
                # 保留 1 秒观察窗口，并在结束时再次确认状态仍在更新。
                time.sleep(1.0)
                monitor.require_fresh()
                print("HighStand height RPC code=0; verify actual posture visually", flush=True)
            finally:
                stop_move(client)
        elif args.action == "stopmove-probe":
            print(
                "stopmove probe: vx=0.5 m/s for 0.8 s; first StopMove, "
                "then 0.6 s observation, then backup zero", flush=True,
            )
            probe_stopmove(client, monitor)
            print("stopmove probe complete; check observed stop timing", flush=True)
        elif args.action == "move":
            print(f"command: vx={args.vx}, vy={args.vy}, wz={args.wz}, seconds={args.seconds}", flush=True)
            move_for(
                client, monitor, args.vx, args.vy, args.wz, args.seconds,
                fast_forward_test=args.fast_forward_test, ctrl_c_test=args.ctrl_c_test,
            )
            print("bounded motion complete; software stop sent", flush=True)
        return 0
    finally:
        # 关闭本脚本建立的状态订阅器。
        subscriber.Close()


def _handle_sigterm(_sig, _frame) -> None:
    """将 SIGTERM 转成异常展开，以运行动作函数中的停车 finally。"""
    raise KeyboardInterrupt("SIGTERM received")


if __name__ == "__main__":
    # Ctrl+C 默认产生 KeyboardInterrupt；SIGTERM 走相同清理路径。
    # 软件清理不能处理 SIGKILL、断电或 DDS 失效，不能替代现场人工停止。
    signal.signal(signal.SIGTERM, _handle_sigterm)
    try:
        raise SystemExit(main())
    except (KeyboardInterrupt, RuntimeError, ValueError) as exc:
        print(f"G1 smoke test stopped: {exc}", flush=True)
        raise SystemExit(1) from exc
