"""四段旋转探针的RGB-D记录器：采集发生在停车后，不负责发送运动命令。"""
from dataclasses import replace
import json
import math
from pathlib import Path
import time

import cv2
import numpy as np

from camera_d435i.network_camera import create_camera


class RotationCapture:
    def __init__(self, config: Path, output: Path, camera_on_desk: bool, *, manual_follow: bool = False):
        self.output = output
        self.output.mkdir(parents=True, exist_ok=False)
        self.camera = None
        self.started = time.monotonic()
        self.monitor = self.slam = None
        self.manifest = {
            "format_version": 1, "status": "RUNNING",
            "capture_mode": ("rotation_manual_follow" if manual_follow else
                             "rotation_bench" if camera_on_desk else "robot_mounted_rotation"),
            "camera_moves_with_robot": None if manual_follow else not camera_on_desk,
            "robot_rotation_started": False, "navigation_executed": False,
            "camera_to_robot_extrinsics": None,
            "pose_timing": "SLAM与IMU记录于取图前后，未插值到曝光时间",
            "views": {}, "return_view": None,
        }
        self._save_manifest()
        try:
            self.camera = create_camera(config)
        except Exception as exc:
            self.finish("FAILED", error=str(exc))
            raise

    def _save_manifest(self):
        # 用原子替换保留上一次完整清单，避免中断导致半个JSON。
        temporary = self.output / "panorama.json.tmp"
        temporary.write_text(json.dumps(self.manifest, ensure_ascii=False, indent=2) + "\n")
        temporary.replace(self.output / "panorama.json")

    def preflight(self):
        frame = self.camera.capture_forward(0, time.monotonic() - self.started).validated()
        print(f"[CAPTURE] camera ready; frame={frame.frame_id} RGB={frame.rgb.shape}", flush=True)
        if self.manifest["capture_mode"] == "rotation_manual_follow":
            print("[CAPTURE] 手动跟随旋转：相机朝向未测量，先检查四张图的实际视角。", flush=True)
        elif not self.manifest["camera_moves_with_robot"]:
            print("[CAPTURE] 相机在桌上：方向名称仅表示机器人旋转阶段，不代表图像视角。", flush=True)

    def bind_monitors(self, monitor, slam):
        self.monitor, self.slam = monitor, slam

    def _pose_snapshot(self):
        imu = self.monitor.require_fresh()
        pose, received, error = self.slam.snapshot()
        now = time.monotonic()
        if pose is None or now - received > 1.0 or error:
            raise RuntimeError(f"采图时SLAM位姿缺失或过期：{error}")
        return {"observed_monotonic_s": now, "imu_yaw_deg": math.degrees(imu),
                "slam_received_monotonic_s": received, "slam_currentPose": dict(pose)}

    def capture(self, direction: str, quarter: int):
        before = self._pose_snapshot()
        frame = self.camera.capture_forward(quarter, time.monotonic() - self.started)
        after = self._pose_snapshot()
        # return图仍是前方ViewFrame，但文件名单独保留，不覆盖初始前方。
        frame = replace(frame, direction="forward" if quarter == 4 else direction).validated()
        rgb_name = f"current_{direction}.png"
        depth_name = f"{direction}_depth_m.npy"
        if not cv2.imwrite(str(self.output / rgb_name), cv2.cvtColor(frame.rgb, cv2.COLOR_RGB2BGR)):
            raise RuntimeError(f"保存{direction}图像失败")
        np.save(self.output / depth_name, frame.depth_m)
        record = {"frame_id": frame.frame_id, "sim_step": frame.sim_step,
                  "timestamp": frame.timestamp, "K": frame.K.tolist(),
                  "rgb_file": rgb_name, "depth_file": depth_name, "quarter": quarter,
                  "camera_metadata": dict(self.camera.last_metadata),
                  "pose_before_capture": before, "pose_after_capture": after}
        if quarter == 4:
            self.manifest["return_view"] = record
        else:
            self.manifest["views"][direction] = record
        self._save_manifest()
        print(f"[CAPTURE] 已保存 {direction}: frame={frame.frame_id}, "
              f"RGB={rgb_name}, depth={depth_name}", flush=True)

    def mark_rotation_started(self):
        self.manifest["robot_rotation_started"] = True
        self._save_manifest()

    def finish(self, status, **details):
        self.manifest.update(status=status, **details)
        self._save_manifest()

    def close(self):
        try:
            if self.camera is not None:
                self.camera.close()
        finally:
            if self.manifest["status"] == "RUNNING":
                self.finish("INTERRUPTED", error="采集流程未完成；可能为Ctrl+C中断")
