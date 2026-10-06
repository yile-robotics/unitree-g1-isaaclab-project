#!/usr/bin/env python3
"""桌面相机静态联调：手动四方向RGB-D → 保存 → 一次模型决策，不导入运控后端。"""
from __future__ import annotations

import argparse
from dataclasses import replace
from datetime import datetime
import json
import math
from pathlib import Path
import sys
import time
import uuid

import cv2
import numpy as np

from camera_d435i.network_camera import create_camera
from unified_vln.model_client import CombinedModelClient
from unified_vln.session_client import G3SessionClient
from unified_vln.types import DIRECTION_ORDER, PanoramaBundle, ViewFrame

ROOT = Path(__file__).resolve().parent
LABELS = dict(zip(DIRECTION_ORDER, ("前方", "左方", "后方", "右方")))


def write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def save_frame(output: Path, frame: ViewFrame, metadata: dict) -> dict:
    """RGB保存PNG，深度保留米制float32；相机曝光和接收信息另存到清单。"""
    frame.validated()
    if not cv2.imwrite(str(output / f"current_{frame.direction}.png"),
                       cv2.cvtColor(frame.rgb, cv2.COLOR_RGB2BGR)):
        raise RuntimeError("RGB保存失败")
    np.save(output / f"{frame.direction}_depth_m.npy", frame.depth_m)
    return {"frame_id": frame.frame_id, "sim_step": frame.sim_step,
            "timestamp": frame.timestamp, "K": frame.K.tolist(), "camera_metadata": metadata}


def capture_panorama(config: Path, output: Path, headless: bool = False) -> PanoramaBundle:
    camera = create_camera(config)
    views, records = {}, {}
    started = time.monotonic()
    manifest = {"format_version": 1, "capture_mode": "manual_desktop",
                "robot_motion_sent": False, "robot_pose": None,
                "camera_to_robot_extrinsics": None, "views": records}
    try:
        for index, direction in enumerate(DIRECTION_ORDER):
            print(f"[CAPTURE] {index + 1}/4 {LABELS[direction]} ({direction})："
                  "保持镜头位置，手动调整朝向并停稳。", flush=True)
            if headless:
                input("准备好后按 Enter 拍摄；Ctrl+C 退出：")
                frame = camera.capture_forward(index, time.monotonic() - started)
            else:
                while True:
                    frame = camera.capture_forward(index, time.monotonic() - started)
                    rgb = cv2.cvtColor(frame.rgb, cv2.COLOR_RGB2BGR)
                    cv2.putText(rgb, f"{index + 1}/4 {direction}: SPACE/ENTER capture, Q exit",
                                (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                    gray = np.clip(frame.depth_m / 5 * 255, 0, 255).astype(np.uint8)
                    depth = cv2.applyColorMap(gray, cv2.COLORMAP_TURBO)
                    depth[frame.depth_m <= 0] = 0
                    cv2.imshow("D415 static VLN: RGB | aligned depth 0..5m", np.hstack((rgb, depth)))
                    key = cv2.waitKey(1) & 255
                    if key in (27, ord("q")) or cv2.getWindowProperty(
                            "D415 static VLN: RGB | aligned depth 0..5m", cv2.WND_PROP_VISIBLE) < 1:
                        raise KeyboardInterrupt
                    if key in (10, 13, 32):
                        break
            frame = replace(frame, direction=direction).validated()
            views[direction] = frame
            records[direction] = save_frame(output, frame, dict(camera.last_metadata))
            # 每拍完一张就写入；中断后的部分清单不能被当作完整四方向数据读取。
            write_json(output / "panorama.json", manifest)
            print(f"[CAPTURE] 已保存 {direction}，frame={frame.frame_id}", flush=True)
    finally:
        camera.close()
        if not headless:
            cv2.destroyAllWindows()
    return PanoramaBundle(0, 0, time.monotonic() - started, views).validated()


def load_panorama(source: Path) -> PanoramaBundle:
    manifest = json.loads((source / "panorama.json").read_text(encoding="utf-8"))
    if manifest.get("capture_mode") == "rotation_bench":
        raise ValueError("相机在桌上的旋转测试仅验证时序，不能作为真实四方向全景上传模型")
    if manifest.get("capture_mode") == "rotation_manual_follow":
        raise ValueError("手动跟随旋转尚未验证相机朝向，请先检查图像，不能作为固定安装全景上传模型")
    if manifest.get("capture_mode") == "robot_mounted_rotation" and manifest.get("status") != "PASS":
        raise ValueError("机器人旋转采图流程未完成，不能上传模型")
    if manifest.get("format_version") != 1 or set(manifest.get("views", {})) != set(DIRECTION_ORDER):
        raise ValueError("需要本入口保存的完整四方向 panorama.json，不能用一张图替代四个方向")
    views = {}
    for direction in DIRECTION_ORDER:
        record = manifest["views"][direction]
        bgr = cv2.imread(str(source / f"current_{direction}.png"), cv2.IMREAD_COLOR)
        if bgr is None:
            raise ValueError(f"缺少或无法读取 {direction} RGB")
        depth = np.load(source / f"{direction}_depth_m.npy", allow_pickle=False)
        if not np.issubdtype(depth.dtype, np.floating) or not np.isfinite(depth).all() or np.any(depth < 0):
            raise ValueError(f"{direction} 深度必须是非负有限米制浮点数")
        if not np.any(depth > 0):
            raise ValueError(f"{direction} 没有非零深度")
        views[direction] = ViewFrame(direction, int(record["frame_id"]), int(record["sim_step"]),
                                     float(record["timestamp"]), cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB),
                                     depth, np.asarray(record["K"], dtype=np.float64)).validated()
    return PanoramaBundle(0, 0, max(f.timestamp for f in views.values()), views).validated()


def save_decision_image(bundle: PanoramaBundle, decision, output: Path) -> None:
    if decision is None or decision.bbox_2d is None:
        return
    frame = bundle.views[decision.direction]
    x1, y1, x2, y2 = decision.clipped_bbox(frame.rgb.shape[1], frame.rgb.shape[0])
    image = cv2.cvtColor(frame.rgb, cv2.COLOR_RGB2BGR)
    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.putText(image, f"{decision.action} / {decision.direction}", (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
    if not cv2.imwrite(str(output / "decision_bbox.png"), image):
        raise RuntimeError("决策预览保存失败")


def request_decision(bundle: PanoramaBundle, output: Path, instruction: str,
                     model_url: str, timeout: float, legacy: bool = False) -> dict:
    """复用正式协议；G3只做health/start/decision/end，不伪造任何运动执行报告。"""
    session_id = "camera_probe_" + uuid.uuid4().hex
    model = CombinedModelClient(model_url, timeout, send_instruction=legacy)
    request = model.make_request(bundle, session_id=session_id, instruction=instruction, decision_index=0)
    images = model.image_fields(bundle, request)
    write_json(output / "request_metadata.json", request.to_metadata())
    session = None if legacy else G3SessionClient.from_decision_url(model_url, timeout)
    active = False
    try:
        if session is not None:
            write_json(output / "g3_health.json", session.health_check())
            _, raw_start = session.start_session(session_id=session_id, instruction=instruction)
            active = True
            write_json(output / "g3_session_started.json", raw_start)
        print("[MODEL] 上传四张RGB图；深度和内参保留在本地。等待一次决策……", flush=True)
        decision, raw = model.decide(request, images)
        write_json(output / "response.json", raw)
        if session is not None:
            session.validate_decision_context(raw, decision_index=0)
        save_decision_image(bundle, decision, output)
        print(json.dumps(raw, ensure_ascii=False, indent=2), flush=True)
        return raw
    finally:
        if active:
            failed = sys.exc_info()[0] is not None
            try:
                _, raw_end = session.end_session(
                    status="FAILURE" if failed else "CANCELLED",
                    reason="static_probe_failed" if failed else "static_probe_no_robot_execution")
                write_json(output / "g3_session_ended.json", raw_end)
            except Exception as error:
                write_json(output / "g3_cleanup_error.json", {"error": str(error), "session_id": session_id})
                if not failed:
                    raise
                print(f"[MODEL] 会话清理失败：{error}", file=sys.stderr)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--camera-config", type=Path,
                        default=ROOT / "camera_d435i/network_camera_d415_local.json")
    parser.add_argument("--panorama-dir", type=Path, help="重放本入口已保存的完整四方向目录，不连接相机")
    parser.add_argument("--capture-only", action="store_true", help="只采集保存，不连接模型")
    parser.add_argument("--headless", action="store_true", help="不用窗口，终端Enter确认拍摄")
    parser.add_argument("--instruction", help="模型导航指令；上传时必填")
    parser.add_argument("--model-url", default="http://127.0.0.1:18765/v1/lavira/decision")
    parser.add_argument("--timeout-s", type=float, default=180.0)
    parser.add_argument("--legacy-model", action="store_true", help="仅旧版无Session服务使用；默认走G3会话")
    parser.add_argument("--output", type=Path, default=ROOT.parents[1] / "outputs/d415_static_vln" /
                        (datetime.now().strftime("run_%Y%m%d_%H%M%S_") + uuid.uuid4().hex[:8]))
    return parser


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not args.capture_only and (not args.instruction or not args.instruction.strip()):
        parser.error("上传模型时必须提供 --instruction")
    if not math.isfinite(args.timeout_s) or args.timeout_s <= 0:
        parser.error("--timeout-s 必须是正有限数")
    if args.capture_only and args.panorama_dir is not None:
        parser.error("--capture-only 与 --panorama-dir 不能同时使用")
    if args.output.exists():
        parser.error("--output 已存在，请换一个新目录")
    args.output.mkdir(parents=True)
    print(f"[STATIC PROBE] 手动采图/模型决策，不发送机器人命令。结果：{args.output}", flush=True)
    stage = "load_panorama" if args.panorama_dir else "capture"
    report = {"robot_motion_sent": False, "navigation_executed": False,
              "source_dir": str(args.panorama_dir.resolve()) if args.panorama_dir else None}
    try:
        bundle = (load_panorama(args.panorama_dir) if args.panorama_dir else
                  capture_panorama(args.camera_config, args.output, args.headless))
        if not args.capture_only:
            stage = "model"
            raw = request_decision(bundle, args.output, args.instruction, args.model_url,
                                   args.timeout_s, args.legacy_model)
            report["decision_action"] = raw.get("action")
        report.update(status="PASS", scope="capture_only" if args.capture_only else "static_model_decision")
        write_json(args.output / "probe_report.json", report)
        print(f"[STATIC PROBE] {report['scope']} 完成。结果：{args.output}", flush=True)
        return 0
    except (Exception, KeyboardInterrupt) as error:
        report.update(status="CANCELLED" if isinstance(error, KeyboardInterrupt) else "FAILED",
                      stage=stage, error=str(error) or "Ctrl+C / 用户退出")
        write_json(args.output / "probe_report.json", report)
        print(f"[STATIC PROBE] {stage}：{report['error']}。已保存的数据保留在 {args.output}", file=sys.stderr)
        return 130 if isinstance(error, KeyboardInterrupt) else 1


if __name__ == "__main__":
    raise SystemExit(main())
