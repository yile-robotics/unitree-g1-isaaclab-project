#!/usr/bin/env python3
"""笔记本端 RGB-D 接收器、VLN 相机工厂和独立预览入口；不发送运动命令。"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import socket
import struct
import sys
import time

import numpy as np

# 直接执行本文件时也能找到项目原有 ViewFrame；作为工厂导入时无需修改路径。
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from unified_vln.types import ViewFrame


class NetworkCamera:
    """每次 capture_forward 请求下一组新图，失败时关闭连接并向 VLN 抛出异常。

    网络传输：NEXT → 4字节大端JSON长度 → JSON → RGB8 → 大端Z16。
    RGB和深度来自同一 SDK frameset，深度已经对齐到彩色像素。
    帧新鲜度由单调时钟检查，不比较两台机器的epoch。
    """

    def __init__(self, config: dict):
        allowed = {"host", "port", "serial", "timeout_s", "max_round_trip_s",
                   "max_server_frame_age_s", "max_pair_delta_ms", "width", "height"}
        unknown = set(config) - allowed
        if unknown:
            raise ValueError(f"Unknown camera settings: {sorted(unknown)}")
        self.host = str(config.get("host", "192.168.123.164"))
        self.port = int(config.get("port", 8765))
        self.serial = str(config.get("serial", "344422072128"))
        self.width = int(config.get("width", 640))
        self.height = int(config.get("height", 480))
        if not 1 <= self.port <= 65535 or not 1 <= self.width <= 4096 or not 1 <= self.height <= 4096:
            raise ValueError("Invalid camera port/resolution")
        self.timeout = float(config.get("timeout_s", 3.0))
        self.max_round_trip = float(config.get("max_round_trip_s", 0.75))
        self.max_age = float(config.get("max_server_frame_age_s", 0.2))
        self.max_pair_delta = float(config.get("max_pair_delta_ms", 50.0))
        if any(not math.isfinite(v) or v <= 0 for v in
               (self.timeout, self.max_round_trip, self.max_age, self.max_pair_delta)):
            raise ValueError("Camera time limits must be finite and positive")
        self._socket: socket.socket | None = None
        self._sequence = 0
        self._color_frame = -1
        self._depth_frame = -1
        self.last_metadata: dict = {}
        self._socket = socket.create_connection((self.host, self.port), timeout=self.timeout)
        self._socket.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)

    def close(self) -> None:
        """可重复调用；Ctrl+C 与 VLN 资源清理均可使用。"""
        if self._socket is not None:
            self._socket.close()
            self._socket = None

    def _read_exact(self, size: int, deadline: float) -> bytes:
        parts = bytearray()
        while len(parts) < size:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("RGB-D response exceeded total timeout")
            assert self._socket is not None
            self._socket.settimeout(remaining)
            block = self._socket.recv(min(size - len(parts), 262144))
            if not block:
                raise ConnectionError("Robot camera stream disconnected")
            parts.extend(block)
        return bytes(parts)

    def _validate_header(self, meta: dict) -> tuple[np.ndarray, float]:
        expected = {"protocol": 1, "serial": self.serial, "width": self.width,
                    "height": self.height, "rgb_encoding": "RGB8", "depth_encoding": "Z16_BE",
                    "depth_aligned_to": "color", "rgb_bytes": self.width * self.height * 3,
                    "depth_bytes": self.width * self.height * 2}
        for key, value in expected.items():
            if meta.get(key) != value:
                raise ValueError(f"Camera {key}: expected {value!r}, got {meta.get(key)!r}")
        sequence = meta["sequence"]
        color_frame = meta["color_frame_number"]
        depth_frame = meta["depth_frame_number"]
        if not isinstance(sequence, int) or sequence <= self._sequence:
            raise ValueError("Repeated/out-of-order RGB-D sequence")
        if not isinstance(color_frame, int) or color_frame <= self._color_frame:
            raise ValueError(f"Repeated/out-of-order RGB camera frame: previous={self._color_frame}, current={color_frame}")
        if not isinstance(depth_frame, int) or depth_frame <= self._depth_frame:
            raise ValueError(f"Repeated/out-of-order depth camera frame: previous={self._depth_frame}, current={depth_frame}")
        age = float(meta["server_frame_age_s"])
        scale = float(meta["depth_scale_m"])
        if not math.isfinite(age) or age < 0 or age > self.max_age:
            raise ValueError(f"Stale camera frame: server age={age:.3f}s")
        if not math.isfinite(scale) or not 0 < scale < 1:
            raise ValueError("Invalid depth scale")
        # 两个时间域相同才可以相减；帧编号属于不同的流，不要求彼此相等。
        if meta["color_timestamp_domain"] != meta["depth_timestamp_domain"]:
            raise ValueError("RGB/depth timestamp domains differ")
        color_ts = float(meta["color_timestamp_ms"])
        depth_ts = float(meta["depth_timestamp_ms"])
        if not all(map(math.isfinite, (color_ts, depth_ts))):
            raise ValueError("Non-finite RGB/depth timestamp")
        if abs(color_ts - depth_ts) > self.max_pair_delta:
            raise ValueError(
                f"RGB/depth timestamps exceed allowed pairing difference: "
                f"RGB-depth={color_ts - depth_ts:+.1f}ms, limit={self.max_pair_delta:.1f}ms; "
                f"RGB frame={color_frame}, depth frame={depth_frame}; "
                f"domain={meta['color_timestamp_domain']}. "
                "Check whether BOTH streams keep updating; do not increase the limit to accept a frozen stream."
            )
        intrinsics = meta["color_intrinsics"]
        aligned = meta["aligned_depth_intrinsics"]
        K = np.asarray(intrinsics["K"], dtype=np.float64)
        if K.shape != (3, 3) or not np.isfinite(K).all() or K[0, 0] <= 0 or K[1, 1] <= 0:
            raise ValueError("Invalid camera K")
        if not np.array_equal(K[2], [0, 0, 1]) or K[0, 1] != 0 or K[1, 0] != 0:
            raise ValueError("Unsupported camera K")
        for item in (intrinsics, aligned):
            if (item["width"], item["height"]) != (self.width, self.height):
                raise ValueError("Calibration dimensions differ from image")
            coeffs = np.asarray(item["coeffs"], dtype=float)
            if coeffs.shape != (5,) or not np.isfinite(coeffs).all() or np.any(coeffs != 0):
                # 非零畸变需先做RGB-D校正，再交给VLN针孔投影。
                raise ValueError("Nonzero/invalid distortion requires RGB-D rectification")
        if not np.array_equal(K, np.asarray(aligned["K"], dtype=np.float64)):
            raise ValueError("Aligned depth K differs from RGB K")
        return K, scale

    def capture_forward(self, sim_step: int, timestamp: float) -> ViewFrame:
        """直接符合 run_g1_real.py 的工厂接口；深度零值保留为缺测，单位转换为米。

        timestamp 使用调用方的时间轴，标记笔记本收齐图像的时刻，不冒充曝光时间。
        SDK原始时间、接收单调时间和请求耗时放在 last_metadata，供后续位姿配对使用。
        """
        if self._socket is None:
            raise ConnectionError("Camera is closed; restart the camera backend")
        started = time.monotonic()
        deadline = started + self.timeout
        try:
            self._socket.settimeout(self.timeout)
            self._socket.sendall(b"NEXT")
            header_size = struct.unpack("!I", self._read_exact(4, deadline))[0]
            if not 1 <= header_size <= 65536:
                raise ValueError("Invalid camera JSON header length")
            meta = json.loads(self._read_exact(header_size, deadline))
            K, scale = self._validate_header(meta)
            rgb_bytes = self._read_exact(self.width * self.height * 3, deadline)
            depth_bytes = self._read_exact(self.width * self.height * 2, deadline)
            received = time.monotonic()
            round_trip = received - started
            if round_trip > self.max_round_trip:
                raise TimeoutError(f"RGB-D request/transfer too slow: {round_trip:.3f}s")
            rgb = np.frombuffer(rgb_bytes, dtype=np.uint8).reshape(self.height, self.width, 3).copy()
            depth_m = np.frombuffer(depth_bytes, dtype=">u2").reshape(self.height, self.width).astype(np.float32)
            depth_m *= scale
            if not np.any(depth_m > 0):
                raise ValueError("All depth pixels are invalid")
            frame = ViewFrame(direction="forward", frame_id=meta["sequence"], sim_step=int(sim_step),
                              timestamp=float(timestamp) + round_trip, rgb=rgb, depth_m=depth_m, K=K).validated()
            self._sequence = meta["sequence"]
            self._color_frame = meta["color_frame_number"]
            self._depth_frame = meta["depth_frame_number"]
            self.last_metadata = {**meta, "client_received_monotonic_s": received,
                                  "client_round_trip_s": round_trip,
                                  "pair_delta_ms": meta["color_timestamp_ms"] - meta["depth_timestamp_ms"],
                                  "valid_depth_fraction": float(np.mean(depth_m > 0))}
            return frame
        except BaseException:
            # 关闭不完整响应，不能在下一次调用中错把残余字节解成另一帧。
            # KeyboardInterrupt 同样关闭，正式 runner 负责自己的运动清理。
            self.close()
            raise


def create_camera(config_path: Path) -> NetworkCamera:
    """--camera-factory camera_d435i.network_camera:create_camera 的入口。"""
    return NetworkCamera(json.loads(Path(config_path).read_text(encoding="utf-8")))


def main() -> int:
    parser = argparse.ArgumentParser(description="只读取/预览机器人 RGB-D，不调用机器人运动服务")
    parser.add_argument("--config", type=Path, default=Path(__file__).with_name("network_camera.json"))
    parser.add_argument("--seconds", type=float, default=30.0, help="接收时间，默认30秒")
    parser.add_argument("--headless", action="store_true", help="只输出帧/深度/延迟，不开GUI")
    parser.add_argument("--output", type=Path, help="可选：保存最后一组和统计，目录必须不存在")
    args = parser.parse_args()
    if not math.isfinite(args.seconds) or args.seconds <= 0:
        parser.error("--seconds must be positive and finite")
    if args.output is not None and args.output.exists():
        parser.error("--output already exists; choose a new directory")
    # VLN工厂只依赖NumPy；独立预览/保存图像才加载已有的OpenCV。
    import cv2
    camera = None
    count = 0
    last_frame = None
    timings = []
    started = time.monotonic()
    last_print = started - 1.0
    last_received = None
    first_received = None
    try:
        camera = create_camera(args.config)
        while time.monotonic() - started < args.seconds:
            last_frame = camera.capture_forward(count, time.monotonic() - started)
            now = time.monotonic()
            first_received = now if first_received is None else first_received
            last_received = now
            count += 1
            timings.append(camera.last_metadata["client_round_trip_s"])
            if now - last_print >= 1:
                valid = last_frame.depth_m[last_frame.depth_m > 0]
                meta = camera.last_metadata
                print(f"[CAMERA] frame={last_frame.frame_id} "
                      f"RGB#{meta['color_frame_number']} depth#{meta['depth_frame_number']} "
                      f"RGB={last_frame.rgb.shape} "
                      f"depth median={np.median(valid):.3f}m valid={meta['valid_depth_fraction']:.1%} "
                      f"RGB-D Δt={meta['pair_delta_ms']:+.2f}ms "
                      f"request+transfer={meta['client_round_trip_s'] * 1000:.1f}ms", flush=True)
                last_print = now
            if not args.headless:
                # 颜色只用于显示；VLN接收的仍是原始RGB及米制深度。
                gray = np.clip(last_frame.depth_m / 5.0 * 255, 0, 255).astype(np.uint8)
                colored = cv2.applyColorMap(gray, cv2.COLORMAP_TURBO)
                colored[last_frame.depth_m <= 0] = 0
                cv2.imshow("G1 RGB | aligned depth (display 0..5 m)",
                           np.hstack((cv2.cvtColor(last_frame.rgb, cv2.COLOR_RGB2BGR), colored)))
                if cv2.waitKey(1) & 255 in (ord("q"), 27):
                    break
    except KeyboardInterrupt:
        print("[CAMERA] Ctrl+C: closing camera connection")
    except Exception as error:
        print(f"[CAMERA] failed: {error}", file=sys.stderr)
        return 1
    finally:
        if camera is not None:
            camera.close()
        if not args.headless:
            cv2.destroyAllWindows()
    if last_frame is not None and camera is not None:
        elapsed = max((last_received or started) - (first_received or started), 1e-9)
        report = {"received_frames": count, "receive_fps": (count - 1) / elapsed,
                  "request_transfer_ms_p50": float(np.percentile(timings, 50) * 1000),
                  "request_transfer_ms_p95": float(np.percentile(timings, 95) * 1000),
                  "last_frame": camera.last_metadata,
                  "timestamp_note": "ViewFrame timestamp is laptop receive time on caller timeline, not exposure time"}
        print(f"[CAMERA] received={count}, receive rate={report['receive_fps']:.2f}fps, "
              f"request+transfer p95={report['request_transfer_ms_p95']:.1f}ms")
        if args.output is not None:
            args.output.mkdir(parents=True, exist_ok=False)
            if not cv2.imwrite(str(args.output / "rgb.png"), cv2.cvtColor(last_frame.rgb, cv2.COLOR_RGB2BGR)):
                raise RuntimeError("Failed to save RGB")
            np.save(args.output / "depth_m.npy", last_frame.depth_m)
            (args.output / "stream_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
            print(f"[CAMERA] saved last RGB/depth/report: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
