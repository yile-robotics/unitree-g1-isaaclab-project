from __future__ import annotations

from pathlib import Path
import signal
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import run_g1_real  # noqa: E402


class _Resource:
    def __init__(self, name: str, events: list[str]):
        self.name = name
        self.events = events

    def stop(self) -> None:
        self.events.append(f"{self.name}.stop")

    def close(self) -> None:
        self.events.append(f"{self.name}.close")


class RealG1SignalTest(unittest.TestCase):
    def tearDown(self) -> None:
        run_g1_real._active_dds = None
        run_g1_real._active_camera = None
        run_g1_real._active_odometry = None

    def test_sigint_stops_dds_before_closing_resources_and_force_exits(self):
        events: list[str] = []
        run_g1_real._active_dds = _Resource("dds", events)
        run_g1_real._active_camera = _Resource("camera", events)
        run_g1_real._active_odometry = _Resource("odometry", events)

        with patch.object(run_g1_real.os, "_exit") as force_exit:
            run_g1_real._signal_handler(signal.SIGINT, None)

        self.assertEqual(
            events,
            ["dds.stop", "dds.close", "camera.close", "odometry.close"],
        )
        force_exit.assert_called_once_with(0)
        self.assertIsNone(run_g1_real._active_dds)
        self.assertIsNone(run_g1_real._active_camera)
        self.assertIsNone(run_g1_real._active_odometry)

    def test_main_registers_sigint_before_running(self):
        parser = MagicMock()
        parser.parse_args.return_value = object()
        with (
            patch.object(run_g1_real.signal, "signal") as register,
            patch.object(run_g1_real, "build_parser", return_value=parser),
            patch.object(run_g1_real, "run", return_value=0) as run,
        ):
            result = run_g1_real.main()

        register.assert_called_once_with(signal.SIGINT, run_g1_real._signal_handler)
        run.assert_called_once_with(parser.parse_args.return_value)
        self.assertEqual(result, 0)

    def test_motion_to_stand_edge_calls_stopmove_once(self):
        dds = MagicMock()
        moving = SimpleNamespace(
            desired_mode="locomotion",
            command=[0.3, 0.0, 0.1],
        )
        standing = SimpleNamespace(
            desired_mode="stand",
            command=[9.0, 9.0, 9.0],
        )

        command, mode = run_g1_real._apply_episode_command(dds, moving, "stand")
        self.assertEqual(command.tolist(), [0.3, 0.0, 0.1])
        self.assertEqual(mode, "locomotion")
        dds.set_velocity.assert_called_once_with(0.3, 0.0, 0.1)
        dds.stop.assert_not_called()

        dds.reset_mock()
        command, mode = run_g1_real._apply_episode_command(
            dds, standing, "locomotion"
        )
        self.assertEqual(command.tolist(), [0.0, 0.0, 0.0])
        self.assertEqual(mode, "stand")
        dds.stop.assert_called_once_with()
        dds.set_velocity.assert_not_called()

        dds.reset_mock()
        run_g1_real._apply_episode_command(dds, standing, "stand")
        dds.stop.assert_not_called()
        dds.set_velocity.assert_called_once_with(0.0, 0.0, 0.0)


if __name__ == "__main__":
    unittest.main()
