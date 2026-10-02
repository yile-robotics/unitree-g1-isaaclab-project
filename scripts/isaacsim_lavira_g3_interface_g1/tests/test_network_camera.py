"""离线验证真实网络收图接口：TCP分片、深度字节序、旧帧、断流和VLN工厂。

只监听本机回环地址，完全不访问机器人或调用DDS。
"""
import copy
import importlib
import json
from pathlib import Path
import socket
import struct
import sys
import tempfile
import threading
import time
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from camera_d435i.network_camera import NetworkCamera


def fixture(sequence=1):
    k = {"width": 2, "height": 2, "K": [[600, 0, 1], [0, 600, 1], [0, 0, 1]],
         "coeffs": [0] * 5, "distortion_model": "Inverse Brown Conrady"}
    meta = {"protocol": 1, "serial": "test-camera", "width": 2, "height": 2,
            "rgb_encoding": "RGB8", "depth_encoding": "Z16_BE", "depth_aligned_to": "color",
            "rgb_bytes": 12, "depth_bytes": 8, "depth_scale_m": 0.001,
            "sequence": sequence, "color_frame_number": sequence + 10,
            "depth_frame_number": sequence + 7, "server_frame_age_s": 0.01,
            # 有意模拟机器人1970年时钟；它不是笔记本的时间轴。
            "color_timestamp_ms": 3882746583.3669, "depth_timestamp_ms": 3882746583.3229,
            "color_timestamp_domain": "Global Time", "depth_timestamp_domain": "Global Time",
            "color_intrinsics": k, "aligned_depth_intrinsics": copy.deepcopy(k)}
    rgb = np.array([[[255, 0, 0], [0, 255, 0]], [[0, 0, 255], [10, 20, 30]]], dtype=np.uint8)
    depth = np.array([[1000, 0], [2000, 350]], dtype=">u2")
    return meta, rgb.tobytes(), depth.tobytes()


class Server:
    """模拟机器人发送格式，故意把TCP响应拆成小块，不假设recv一次收齐。"""
    def __init__(self, packets, *, truncate=False, delay=0.0):
        self.listener = socket.socket()
        self.listener.bind(("127.0.0.1", 0))
        self.listener.listen(1)
        self.listener.settimeout(2)
        self.config = {"host": "127.0.0.1", "port": self.listener.getsockname()[1],
                       "serial": "test-camera", "width": 2, "height": 2,
                       "timeout_s": 0.5, "max_round_trip_s": 0.3}
        self.error = None

        def send():
            try:
                with self.listener.accept()[0] as conn:
                    conn.settimeout(2)
                    for meta, rgb, depth in packets:
                        request = bytearray()
                        while len(request) < 4:
                            block = conn.recv(4 - len(request))
                            if not block:
                                return
                            request.extend(block)
                        if request != b"NEXT":
                            raise AssertionError(f"Unexpected request {request!r}")
                        time.sleep(delay)
                        header = json.dumps(meta).encode()
                        response = struct.pack("!I", len(header)) + header + rgb + depth
                        if truncate:
                            conn.sendall(response[:-3])
                            return
                        for offset in range(0, len(response), 37):
                            conn.sendall(response[offset:offset + 37])
            except (BrokenPipeError, ConnectionResetError):
                pass  # 客户端拒绝非法标定/旧帧后主动关闭是预期结果。
            except BaseException as error:
                self.error = error
            finally:
                self.listener.close()

        self.thread = threading.Thread(target=send, daemon=True)
        self.thread.start()

    def finish(self):
        self.thread.join(3)
        if self.thread.is_alive():
            raise AssertionError("Fake camera server did not finish")
        if self.error:
            raise self.error


class NetworkCameraTests(unittest.TestCase):
    def test_real_factory_contract_rgb_order_depth_units_and_timestamp(self):
        server = Server([fixture(1), fixture(3)])
        camera = None
        try:
            with tempfile.TemporaryDirectory() as folder:
                config = Path(folder) / "camera.json"
                config.write_text(json.dumps(server.config))
                # 使用正式runner的 MODULE:FUNCTION 调用约定，但不加载规划/DDS依赖。
                module_name, factory_name = "camera_d435i.network_camera:create_camera".split(":")
                camera = getattr(importlib.import_module(module_name), factory_name)(config)
                first = camera.capture_forward(12, 5.0)
                second = camera.capture_forward(13, 6.0)
            self.assertEqual(first.direction, "forward")
            self.assertEqual(first.sim_step, 12)
            self.assertEqual(first.rgb.dtype, np.uint8)
            np.testing.assert_array_equal(first.rgb[0, 0], [255, 0, 0])
            np.testing.assert_allclose(first.depth_m, [[1, 0], [2, 0.35]])
            self.assertEqual(first.depth_m.dtype, np.float32)
            self.assertTrue(5 <= first.timestamp < 5.3)
            self.assertEqual((first.frame_id, second.frame_id), (1, 3))
            first.rgb[:] = 0
            self.assertEqual(second.rgb[0, 0, 0], 255)
            self.assertAlmostEqual(camera.last_metadata["valid_depth_fraction"], 0.75)
        finally:
            if camera:
                camera.close()
                camera.close()
            server.finish()

    def test_reject_bad_frames_and_close_connection(self):
        bad_cases = {
            "stale": ("server_frame_age_s", 2.0),
            "serial": ("serial", "wrong-camera"),
            "payload size": ("depth_bytes", 1000000000),
            "unpaired": ("depth_timestamp_ms", 1.0),
            "time domains": ("depth_timestamp_domain", "Hardware Clock"),
            "scale": ("depth_scale_m", float("nan")),
        }
        for label, (key, value) in bad_cases.items():
            with self.subTest(label=label):
                packet = fixture()
                packet[0][key] = value
                self.assert_invalid(packet)
        for label in ("K", "distortion", "all zero depth"):
            with self.subTest(label=label):
                meta, rgb, depth = fixture()
                if label == "K":
                    meta["aligned_depth_intrinsics"]["K"][0][0] = 500
                elif label == "distortion":
                    meta["color_intrinsics"]["coeffs"][0] = 0.1
                else:
                    depth = bytes(8)
                self.assert_invalid((meta, rgb, depth))

    def assert_invalid(self, packet):
        server = Server([packet])
        camera = NetworkCamera(server.config)
        try:
            with self.assertRaises(ValueError):
                camera.capture_forward(0, 0)
            self.assertIsNone(camera._socket)
            with self.assertRaises(ConnectionError):
                camera.capture_forward(0, 0)
        finally:
            camera.close()
            server.finish()

    def test_repeated_frame_rejected(self):
        server = Server([fixture(), fixture()])
        camera = NetworkCamera(server.config)
        try:
            camera.capture_forward(0, 0)
            with self.assertRaisesRegex(ValueError, "Repeated"):
                camera.capture_forward(1, 1)
            self.assertIsNone(camera._socket)
        finally:
            camera.close()
            server.finish()

    def test_either_stream_freezing_is_rejected_despite_new_packet_sequence(self):
        for key in ("color_frame_number", "depth_frame_number"):
            with self.subTest(stream=key):
                first, second = fixture(1), fixture(2)
                second[0][key] = first[0][key]
                server = Server([first, second])
                camera = NetworkCamera(server.config)
                try:
                    camera.capture_forward(0, 0)
                    with self.assertRaisesRegex(ValueError, "Repeated/out-of-order.*camera frame"):
                        camera.capture_forward(1, 1)
                    self.assertIsNone(camera._socket)
                finally:
                    camera.close()
                    server.finish()

    def test_first_response_with_old_rgb_reports_actual_skew(self):
        meta, rgb, depth = fixture()
        meta["color_frame_number"] = 89
        meta["depth_frame_number"] = 1359
        meta["color_timestamp_ms"] = 3885406193.926892
        meta["depth_timestamp_ms"] = 3885497153.1228843
        server = Server([(meta, rgb, depth)])
        camera = NetworkCamera(server.config)
        try:
            with self.assertRaisesRegex(ValueError, r"RGB-depth=-90959.2ms.*RGB frame=89, depth frame=1359"):
                camera.capture_forward(0, 0)
        finally:
            camera.close()
            server.finish()

    def test_disconnect_mid_frame_rejected(self):
        server = Server([fixture()], truncate=True)
        camera = NetworkCamera(server.config)
        try:
            with self.assertRaises(ConnectionError):
                camera.capture_forward(0, 0)
            self.assertIsNone(camera._socket)
        finally:
            camera.close()
            server.finish()

    def test_slow_response_rejected_even_if_complete(self):
        server = Server([fixture()], delay=0.15)
        camera = NetworkCamera({**server.config, "max_round_trip_s": 0.05})
        try:
            with self.assertRaises(TimeoutError):
                camera.capture_forward(0, 0)
            self.assertIsNone(camera._socket)
        finally:
            camera.close()
            server.finish()

    def test_total_timeout_rejected(self):
        server = Server([fixture()], delay=0.15)
        camera = NetworkCamera({**server.config, "timeout_s": 0.05})
        try:
            with self.assertRaises(TimeoutError):
                camera.capture_forward(0, 0)
            self.assertIsNone(camera._socket)
        finally:
            camera.close()
            server.finish()


if __name__ == "__main__":
    unittest.main()
