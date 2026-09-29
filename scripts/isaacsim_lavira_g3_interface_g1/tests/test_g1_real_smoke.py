"""G1 冒烟脚本的纯离线测试：不初始化 DDS，也不向真实机器人发命令。"""

from __future__ import annotations

from pathlib import Path
import sys
import unittest
from unittest.mock import patch


# 直接导入同目录脚本，测试参数保护和控制函数，不连接真机。
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import g1_real_smoke as smoke  # noqa: E402


class FakeClient:
    """记录运动 RPC 调用顺序，并可模拟非零速度请求被拒绝。"""

    def __init__(self, first_velocity_code: int = 0) -> None:
        self.calls = []
        self.first_velocity_code = first_velocity_code

    def SetVelocity(self, vx, vy, wz, duration):
        # SDK 返回 0 表示请求未报错；非零值模拟 RPC 错误。
        self.calls.append(("velocity", vx, vy, wz, duration))
        if (vx, vy, wz) != (0.0, 0.0, 0.0):
            code = self.first_velocity_code
            self.first_velocity_code = 0
            return code
        return 0

    def StopMove(self):
        self.calls.append(("stop",))


class FakeMonitor:
    """模拟持续有效的状态，以便聚焦速度发送与停车逻辑。"""

    def require_fresh(self):
        return 0.0


class FakeClock:
    """让限时循环使用可推进的假时钟，不必真实等待。"""

    def __init__(self):
        self.now = 0.0

    def monotonic(self):
        return self.now

    def sleep(self, seconds):
        self.now += seconds


class G1RealSmokeTest(unittest.TestCase):
    def test_motion_requires_explicit_enable_before_dds_import(self):
        # 忘记 --execute 时，在 SDK/DDS 初始化之前就拒绝运动。
        with self.assertRaisesRegex(SystemExit, "require --execute"):
            smoke.main(["--network-interface", "lo", "move", "--vx", "0.05"])

    def test_rejects_combined_axes_and_unbounded_commands(self):
        # 首轮真机测试只允许单轴、小幅、限时指令；NaN 也必须拒绝。
        for command in ((0.05, 0.01, 0.0, 0.5), (0.16, 0.0, 0.0, 0.5),
                        (0.05, 0.0, 0.0, 1.1), (float("nan"), 0.0, 0.0, 0.5)):
            with self.subTest(command=command), self.assertRaises(ValueError):
                smoke.validate_move(*command)

    def test_fast_forward_requires_opt_in_and_exact_short_forward_command(self):
        # 默认仍拒绝 0.5 m/s；高速开关只接受指定的前向短时试验。
        with self.assertRaises(ValueError):
            smoke.validate_move(0.5, 0.0, 0.0, 0.5)
        smoke.validate_move(0.5, 0.0, 0.0, 1.0, fast_forward_test=True)
        for command in ((0.5, 0.0, 0.0, 1.01), (0.5, 0.01, 0.0, 0.5),
                        (-0.5, 0.0, 0.0, 0.5), (0.4, 0.0, 0.0, 0.5)):
            with self.subTest(command=command), self.assertRaises(ValueError):
                smoke.validate_move(*command, fast_forward_test=True)

    def test_fast_forward_cli_requires_flag_before_dds_import(self):
        # 漏掉高速开关时，命令行在导入 SDK 之前拒绝 0.5 m/s。
        command = ["--network-interface", "lo", "--execute", "move", "--vx", "0.5", "--seconds", "0.5"]
        with self.assertRaises(ValueError):
            smoke.main(command)
        args = smoke.build_parser().parse_args(command + ["--fast-forward-test"])
        self.assertTrue(args.fast_forward_test)

    def test_ctrl_c_two_second_mode_has_separate_limit(self):
        # 普通高速命令仍最多 1 秒，双开关才允许 2 秒，且不能改方向或速度。
        with self.assertRaises(ValueError):
            smoke.validate_move(0.5, 0.0, 0.0, 2.0, fast_forward_test=True)
        with self.assertRaises(ValueError):
            smoke.validate_move(0.5, 0.0, 0.0, 2.0, ctrl_c_test=True)
        smoke.validate_move(0.5, 0.0, 0.0, 2.0, fast_forward_test=True, ctrl_c_test=True)
        for command in ((0.5, 0.0, 0.0, 2.01), (0.4, 0.0, 0.0, 2.0),
                        (0.5, 0.01, 0.0, 2.0)):
            with self.subTest(command=command), self.assertRaises(ValueError):
                smoke.validate_move(*command, fast_forward_test=True, ctrl_c_test=True)

    def test_ctrl_c_two_second_mode_stops_after_interrupt(self):
        # 模拟机器人已经收到几次前进命令后按 Ctrl+C，验证仍请求零速度。
        class InterruptingClient(FakeClient):
            def __init__(self):
                super().__init__()
                self.forward_calls = 0

            def SetVelocity(self, vx, vy, wz, duration):
                result = super().SetVelocity(vx, vy, wz, duration)
                if vx == 0.5:
                    self.forward_calls += 1
                    if self.forward_calls == 3:
                        raise KeyboardInterrupt()
                return result

        client = InterruptingClient()
        clock = FakeClock()
        with patch.object(smoke.time, "monotonic", clock.monotonic), patch.object(
            smoke.time, "sleep", clock.sleep
        ), self.assertRaises(KeyboardInterrupt):
            smoke.move_for(
                client, FakeMonitor(), 0.5, 0.0, 0.0, 2.0,
                fast_forward_test=True, ctrl_c_test=True,
            )
        self.assertLess(clock.now, 2.0)
        self.assertEqual(client.calls[-2], ("stop",))
        self.assertEqual(client.calls[-1], ("velocity", 0.0, 0.0, 0.0, 0.3))

    def test_fast_forward_command_still_requests_stop(self):
        # 用假时钟验证显式高速试验走既有停车路径，不连接真实机器人。
        client = FakeClient()
        clock = FakeClock()
        with patch.object(smoke.time, "monotonic", clock.monotonic), patch.object(
            smoke.time, "sleep", clock.sleep
        ):
            smoke.move_for(client, FakeMonitor(), 0.5, 0.0, 0.0, 1.0, fast_forward_test=True)
        self.assertTrue(any(call[0] == "velocity" and call[1] == 0.5 for call in client.calls))
        self.assertEqual(client.calls[-2], ("stop",))
        self.assertEqual(client.calls[-1], ("velocity", 0.0, 0.0, 0.0, 0.3))

    def test_stopmove_probe_requires_execute_before_dds_import(self):
        # 专用动作必须显式启用；拒绝时不能触碰 DDS。
        with self.assertRaisesRegex(SystemExit, "require --execute"):
            smoke.main(["--network-interface", "lo", "stopmove-probe"])

    def test_stopmove_probe_forward_failure_still_sends_backup_zero(self):
        # 首条前进请求失败时仍发备用零速度，不等待观察窗口。
        client = FakeClient(first_velocity_code=9)
        clock = FakeClock()
        with patch.object(smoke.time, "monotonic", clock.monotonic), patch.object(
            smoke.time, "sleep", clock.sleep
        ), self.assertRaisesRegex(RuntimeError, "probe forward RPC failed: 9"):
            smoke.probe_stopmove(client, FakeMonitor())
        self.assertEqual(client.calls[-1], ("velocity", 0.0, 0.0, 0.0, 0.3))
        self.assertNotIn(("stop",), client.calls)

    def test_stopmove_probe_waits_before_backup_zero(self):
        # 仅一条 StopMove 先于备用零速度，观察区间没有第二条零速度。
        clock = FakeClock()

        class TimedClient(FakeClient):
            def __init__(self):
                super().__init__()
                self.events = []

            def SetVelocity(self, vx, vy, wz, duration):
                self.events.append(("velocity", clock.now, vx, duration))
                return super().SetVelocity(vx, vy, wz, duration)

            def StopMove(self):
                self.events.append(("stop", clock.now))
                super().StopMove()

        client = TimedClient()
        with patch.object(smoke.time, "monotonic", clock.monotonic), patch.object(
            smoke.time, "sleep", clock.sleep
        ):
            smoke.probe_stopmove(client, FakeMonitor())
        stops = [event for event in client.events if event[0] == "stop"]
        zeroes = [event for event in client.events if event[0] == "velocity" and event[2] == 0.0]
        self.assertEqual(len(stops), 1)
        self.assertEqual(len(zeroes), 1)
        self.assertAlmostEqual(
            zeroes[0][1] - stops[0][1], smoke.STOPMOVE_PROBE_OBSERVE_SECONDS,
        )
        self.assertTrue(any(event[2:] == (0.5, 1.5) for event in client.events if event[0] == "velocity"))

    def test_stopmove_probe_sends_backup_zero_if_stopmove_fails(self):
        class FailingStopClient(FakeClient):
            def StopMove(self):
                self.calls.append(("stop",))
                raise RuntimeError("StopMove RPC failed")

        client = FailingStopClient()
        clock = FakeClock()
        with patch.object(smoke.time, "monotonic", clock.monotonic), patch.object(
            smoke.time, "sleep", clock.sleep
        ), self.assertRaisesRegex(RuntimeError, "StopMove RPC failed"):
            smoke.probe_stopmove(client, FakeMonitor())
        self.assertEqual(client.calls[-1], ("velocity", 0.0, 0.0, 0.0, 0.3))

    def test_high_stand_checks_the_underlying_rpc_result(self):
        # Python SDK 的 HighStand() 丢弃返回码；改用同一高度参数的 RPC 并检查状态。
        class FakeStandClient:
            def __init__(self, code):
                self.code = code
                self.heights = []

            def SetStandHeight(self, height):
                self.heights.append(height)
                return self.code

        good = FakeStandClient(0)
        smoke.request_high_stand(good)
        self.assertEqual(good.heights, [(1 << 32) - 1])
        bad = FakeStandClient(9)
        with self.assertRaisesRegex(RuntimeError, "HighStand height RPC failed: 9"):
            smoke.request_high_stand(bad)

    def test_bounded_move_always_ends_in_zero_velocity(self):
        # 正常结束时，先 StopMove，再发可检查返回码的零速度。
        client = FakeClient()
        clock = FakeClock()
        with patch.object(smoke.time, "monotonic", clock.monotonic), patch.object(
            smoke.time, "sleep", clock.sleep
        ):
            smoke.move_for(client, FakeMonitor(), 0.05, 0.0, 0.0, 0.3)

        self.assertTrue(any(call[0] == "velocity" and call[1] == 0.05 for call in client.calls))
        self.assertEqual(client.calls[-2], ("stop",))
        self.assertEqual(client.calls[-1], ("velocity", 0.0, 0.0, 0.0, 0.3))

    def test_velocity_rpc_failure_still_requests_stop(self):
        # 非零速度 RPC 被拒绝，也必须经过 finally 请求停车。
        client = FakeClient(first_velocity_code=9)
        with self.assertRaisesRegex(RuntimeError, "velocity RPC failed"):
            smoke.move_for(client, FakeMonitor(), 0.05, 0.0, 0.0, 0.5)
        self.assertEqual(client.calls[-2], ("stop",))
        self.assertEqual(client.calls[-1][1:4], (0.0, 0.0, 0.0))


    def test_ctrl_c_during_motion_still_requests_stop(self):
        # 在第一条非零速度 RPC 期间模拟 Ctrl+C；KeyboardInterrupt 必须触发
        # move_for 的 finally，依次请求 StopMove 和显式零速度。
        class InterruptingClient(FakeClient):
            def SetVelocity(self, vx, vy, wz, duration):
                result = super().SetVelocity(vx, vy, wz, duration)
                if (vx, vy, wz) != (0.0, 0.0, 0.0):
                    raise KeyboardInterrupt()
                return result

        client = InterruptingClient()
        with self.assertRaises(KeyboardInterrupt):
            smoke.move_for(client, FakeMonitor(), 0.05, 0.0, 0.0, 1.0)
        self.assertEqual(client.calls[-2], ("stop",))
        self.assertEqual(client.calls[-1][1:4], (0.0, 0.0, 0.0))


if __name__ == "__main__":
    unittest.main()
