"""用可控时钟与假硬件验证正式真机状态机，覆盖停车调用耗时和相机阻塞。"""
import json
import math
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from unified_vln.episode import EpisodeConfig, EpisodeState, LocalEndToEndEpisode
from unified_vln.local_trajectory import LocalFollowerConfig
from unified_vln.odometry import Pose2D, wrap_to_pi
from unified_vln.real_episode import RealG1Episode, RealPanoramaConfig
from unified_vln.types import DIRECTION_ORDER, ViewFrame


class RealPanoramaTest(unittest.TestCase):
    def fixture(self, *, camera_failure=None, map_scale=1.0, output=None):
        state = SimpleNamespace(now=0.0, yaw=math.radians(170), command=np.zeros(3),
                                stops=[], captures=[], starts=0, imu_live=True, map_live=True)
        state.initial_yaw = state.yaw

        def advance(dt):
            state.yaw += float(state.command[2]) * dt
            state.now += dt

        def stop():
            moving = state.command[2] != 0
            state.command.fill(0)
            advance(0.2)  # 对应正式DDS.stop()的调用耗时，不算入额外settle等待。
            if moving:
                state.yaw += math.radians(14)  # 模拟请求停车后的余动。
            state.stops.append(state.now)

        imu = SimpleNamespace(get_yaw=lambda: wrap_to_pi(state.yaw) if state.imu_live else None)
        def get_pose():
            if not state.map_live:
                return None
            yaw = state.initial_yaw + (state.yaw - state.initial_yaw) * map_scale
            return Pose2D(1, 2, wrap_to_pi(yaw), state.now)

        testcase = self
        class Camera:
            last_metadata = {"serial": "offline"}
            def capture_forward(self, sim_step, timestamp):
                testcase.assertTrue(np.all(state.command == 0), "相机请求期间必须已经实际停车")
                testcase.assertGreaterEqual(state.now - state.stops[-1], 0.5 - 1e-8)
                index = len(state.captures)
                if index == camera_failure:
                    raise RuntimeError("camera disconnected")
                state.captures.append((state.now, state.yaw))
                advance(0.12)  # 同步取图阻塞期间DDS也必须保持零速度。
                return ViewFrame("forward", index+1, sim_step, timestamp+0.12,
                                 np.full((8, 12, 3), index+10, np.uint8),
                                 np.ones((8, 12), np.float32),
                                 np.array([[10, 0, 6], [0, 10, 4], [0, 0, 1]], float))

        class Episode(RealG1Episode):
            def _start_decision_request(self, panorama, *, decision_pose):
                self.model_panorama = panorama
                self.model_pose = decision_pose
                self.state = EpisodeState.WAITING_DECISION

        episode = Episode(EpisodeConfig("offline_real", "go to the door", warmup_steps=0,
                                       single_forward_panorama=True, output_dir=output),
                          LocalFollowerConfig(), camera=Camera(), model=None, planner=None,
                          odometry=SimpleNamespace(get_pose=get_pose), panorama_imu=imu,
                          stop_robot=stop, clock=lambda: state.now)
        def tick():
            advance(0.05)
            old_command = state.command.copy()
            result = episode.update(completed_step=int(state.now*20), step_dt=0.05,
                                    timestamp=state.now, applied_command=old_command,
                                    stand_ready=True, locomotion_ready=True)
            if result.command[2] != 0 and state.command[2] == 0:
                state.starts += 1
            state.command = result.command.copy()
            return result
        return episode, state, tick

    def run_until_done(self, episode, tick):
        for _ in range(1000):
            result = tick()
            if episode.state in (EpisodeState.WAITING_DECISION, EpisodeState.FAILED):
                return result
        self.fail("state machine did not finish")

    def test_four_turns_stop_then_wait_then_capture_and_upload_original_four(self):
        with tempfile.TemporaryDirectory() as tmp:
            episode, state, tick = self.fixture(output=Path(tmp))
            self.run_until_done(episode, tick)
            self.assertEqual(episode.state, EpisodeState.WAITING_DECISION)
            self.assertEqual(state.starts, 4)
            self.assertEqual(len(state.stops), 5)  # 初始停车+四段停车。
            self.assertEqual(len(state.captures), 5)
            self.assertEqual(tuple(episode.model_panorama.views), DIRECTION_ORDER)
            self.assertEqual([v.frame_id for v in episode.model_panorama.views.values()], [1, 2, 3, 4])
            self.assertTrue(np.all(episode.model_panorama.views["forward"].rgb == 10))
            self.assertTrue(np.all(state.command == 0))
            self.assertEqual(episode.real_panorama_config.speed_rad_s, 0.8)
            trace = json.loads((episode.output_dir / "decision_000_panorama_capture.json").read_text())
            self.assertEqual(trace["status"], "PASS")
            self.assertEqual(trace["quarter_turns_completed"], 4)
            self.assertEqual(list(trace["captures"]), [*DIRECTION_ORDER, "forward_return"])
            for record in trace["captures"].values():
                self.assertGreaterEqual(record["capture_requested_monotonic_s"] -
                                        record["stop_returned_monotonic_s"], 0.5 - 1e-8)
            self.assertEqual(len(list(episode.output_dir.glob("*_depth_m.npy"))), 5)

    def test_threshold_uses_imu75_even_when_map_has_already_reached90(self):
        episode, state, tick = self.fixture(map_scale=1.3)
        while state.starts == 0:
            tick()
        state.yaw = state.initial_yaw + math.radians(72)
        result = tick()
        self.assertEqual(result.command[2], 0.8)
        self.assertGreater(math.degrees(wrap_to_pi(episode._map_pose().yaw -
                                                  episode._real_initial_map)), 90)
        self.assertEqual(len(state.stops), 1)
        state.yaw = state.initial_yaw + math.radians(76)
        result = tick()
        self.assertTrue(np.all(result.command == 0))
        self.assertEqual(len(state.stops), 2)
        self.assertEqual(len(state.captures), 1)

    def test_bad_slam_turn_stops_without_capture_or_next_quarter(self):
        episode, state, tick = self.fixture(map_scale=0.3)
        self.run_until_done(episode, tick)
        self.assertEqual(episode.state, EpisodeState.FAILED)
        self.assertIn("outside 70°–110°", episode.failure_reason)
        self.assertEqual(state.starts, 1)
        self.assertEqual(len(state.captures), 1)
        self.assertTrue(np.all(state.command == 0))

    def test_initial_capture_failure_never_starts_rotation(self):
        episode, state, tick = self.fixture(camera_failure=0)
        self.run_until_done(episode, tick)
        self.assertEqual(episode.failure_reason, "camera disconnected")
        self.assertEqual(state.starts, 0)
        self.assertTrue(np.all(state.command == 0))

    def test_camera_failure_stops_before_next_quarter_and_model(self):
        episode, state, tick = self.fixture(camera_failure=1)
        self.run_until_done(episode, tick)
        self.assertEqual(episode.failure_reason, "camera disconnected")
        self.assertEqual(state.starts, 1)
        self.assertFalse(hasattr(episode, "model_panorama"))
        self.assertTrue(np.all(state.command == 0))

    def test_lost_imu_or_slam_stops_active_rotation(self):
        for field in ("imu_live", "map_live"):
            with self.subTest(field=field):
                episode, state, tick = self.fixture()
                while state.starts == 0:
                    tick()
                setattr(state, field, False)
                result = tick()
                self.assertEqual(episode.state, EpisodeState.FAILED)
                self.assertTrue(np.all(result.command == 0))
                self.assertEqual(len(state.stops), 2)

    def test_stalled_imu_hits_time_limit_and_stops(self):
        episode, state, tick = self.fixture()
        while state.starts == 0:
            tick()
        initial = state.yaw
        for _ in range(220):
            state.yaw = initial
            tick()
            if episode.state == EpisodeState.FAILED:
                break
        self.assertIn("time limit", episode.failure_reason)
        self.assertTrue(np.all(state.command == 0))

    def test_simulation_still_uses_original_shared_class(self):
        # 仿真创建的原类不依赖真机IMU和主动停车回调。
        episode = LocalEndToEndEpisode(EpisodeConfig("sim", "go left", single_forward_panorama=True),
                                      LocalFollowerConfig(), camera=None, model=None, planner=None)
        self.assertFalse(hasattr(episode, "panorama_imu"))
        self.assertEqual(episode.rotation.speed_rad_s, 0.4)


if __name__ == "__main__":
    unittest.main()
