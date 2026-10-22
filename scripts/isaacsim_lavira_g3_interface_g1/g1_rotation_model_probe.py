#!/usr/bin/env python3
"""真机四段旋转采图后请求一次 G3 决策；不调用 iPlanner 或执行模型动作。"""
from __future__ import annotations

import argparse
from datetime import datetime
import json
import math
from pathlib import Path
import subprocess
import sys
import uuid

from g1_camera_model_probe import load_panorama, request_decision, write_json
from unified_vln.session_client import G3SessionClient
from live_model_target import mark_display_motion

ROOT = Path(__file__).resolve().parent


def run_rotation(interface: str, camera_config: Path, output: Path) -> None:
    # 独立进程保留既有探针的 Ctrl+C 停车和资源清理路径。
    subprocess.run([
        sys.executable, str(ROOT / "g1_vln_rotation_probe.py"),
        "--network-interface", interface,
        "--mode", "imu75_panorama", "--quarters", "4",
        "--capture-panorama", "--camera-config", str(camera_config),
        "--output", str(output), "--execute",
    ], check=True)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--network-interface", required=True)
    parser.add_argument("--instruction", required=True)
    parser.add_argument("--camera-config", type=Path,
                        default=ROOT / "camera_d415/network_camera_d415_local.json")
    parser.add_argument("--model-url", default="http://127.0.0.1:18765/v1/lavira/decision")
    parser.add_argument("--timeout-s", type=float, default=180.0)
    parser.add_argument("--output", type=Path, default=ROOT.parents[1] / "outputs/d415_rotation_model" /
                        (datetime.now().strftime("run_%Y%m%d_%H%M%S_") + uuid.uuid4().hex[:8]))
    parser.add_argument("--execute", action="store_true", help="实际执行四段左转；模型决策不执行")
    args = parser.parse_args(argv)
    if not args.execute:
        parser.error("实际旋转需要 --execute")
    if not args.instruction.strip():
        parser.error("--instruction 不能为空")
    if not math.isfinite(args.timeout_s) or args.timeout_s <= 0:
        parser.error("--timeout-s 必须是正有限数")
    if not args.camera_config.is_file():
        parser.error("--camera-config 文件不存在")
    if args.output.exists():
        parser.error("--output 已存在，请使用新目录")
    args.output.mkdir(parents=True)
    panorama = args.output / "panorama"
    model_output = args.output / "model"
    report = {
        "scope": "robot_rotation_then_one_model_decision",
        "instruction": args.instruction, "model_url": args.model_url,
        "panorama_dir": str(panorama.resolve()),
        "model_output_dir": str(model_output.resolve()),
        "iplanner_called": False, "model_action_executed": False,
        "rotation_process_started": False, "rotation_capture_completed": False,
        "model_decision_started": False,
    }
    stage = "model_health"
    try:
        # 先检查现有服务协议；此时不初始化运控，不开始模型会话。
        session = G3SessionClient.from_decision_url(args.model_url, min(args.timeout_s, 10.0))
        write_json(args.output / "preflight_health.json", session.health_check())
        print(f"[FLOW] 指令：{args.instruction}", flush=True)
        print("[FLOW] G3健康检查通过；四段左转并停车采图，随后只请求一次决策。", flush=True)
        print(f"[FLOW] 结果：{args.output.resolve()}", flush=True)
        stage = "rotation_capture"
        # 先清除上一轮实时框和预览路径；显示通知本身不发送任何机器人命令。
        mark_display_motion(args.output)
        report["rotation_process_started"] = True
        run_rotation(args.network_interface, args.camera_config.resolve(), panorama.resolve())
        # 旧探针的 Ctrl+C 可能返回 0；退出码不能代替完整清单检查。
        manifest = json.loads((panorama / "panorama.json").read_text(encoding="utf-8"))
        if manifest.get("capture_mode") != "robot_mounted_rotation" or manifest.get("status") != "PASS":
            raise ValueError("四段固定安装全景未完整完成，不请求模型")
        bundle = load_panorama(panorama)
        report["rotation_capture_completed"] = True
        report["total_slam_turn_deg"] = manifest.get("total_slam_turn_deg")
        stage = "model_decision"
        model_output.mkdir()
        report["model_decision_started"] = True
        raw = request_decision(bundle, model_output, args.instruction, args.model_url, args.timeout_s)
        report.update(status="PASS", decision_action=raw.get("action"),
                      decision_direction=raw.get("direction"))
        print("[FLOW] 完成：四段旋转采图 + 一次模型决策；模型动作未执行。", flush=True)
        return 0
    except (Exception, KeyboardInterrupt) as error:
        cancelled = isinstance(error, KeyboardInterrupt)
        report.update(status="CANCELLED" if cancelled else "FAILED", stage=stage,
                      error="Ctrl+C" if cancelled else str(error))
        print(f"[FLOW] {stage}: {report['error']}；已保存的数据保留。", file=sys.stderr, flush=True)
        return 130 if cancelled else 1
    finally:
        write_json(args.output / "flow_report.json", report)


if __name__ == "__main__":
    raise SystemExit(main())
