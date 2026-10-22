"""保存真机模型目标的原图坐标可视化；不取图、不规划、不连接机器人。"""
from __future__ import annotations

import argparse
import html
import json
from pathlib import Path
import re

import cv2
import numpy as np

from .types import DIRECTION_ORDER, ViewFrame


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _write_index(output: Path) -> None:
    records = [json.loads(p.read_text(encoding="utf-8"))
               for p in output.glob("decision_*_target.json")]
    cards = []
    for item in sorted(records, key=lambda r: r["decision_index"]):
        def esc(value):
            return html.escape(str(value))
        projection = item["projection"] or {}
        title = (f"决策 {item['decision_index']}：{item['action']} / "
                 f"{item['selected_direction']} / 目标：{item['target']}")
        cards.append(
            f"<section><h2>{esc(title)}</h2>"
            f"<p>阶段判断：{esc(item['stage_progress'].get('current_stage'))}；"
            f"已完成：{esc(item['stage_progress'].get('stage_completed'))}/"
            f"{esc(item['stage_progress'].get('stage_total'))}</p>"
            f"<p>框 xyxy：{esc(item['bbox_xyxy_original'])}；"
            f"深度采样像素：{esc(projection.get('pixel_uv'))}；"
            f"采样深度：{esc(projection.get('depth_m'))} m；"
            f"有效样本：{esc(projection.get('valid_depth_count'))}</p>"
            f"<p>名义局部目标 [前方, 左方]："
            f"{esc(projection.get('goal_after_turn_xy_m'))} m；"
            f"是否使用前进兜底：{esc(projection.get('used_forward_fallback'))}；"
            f"客户端接受导航：{esc(item['navigation_accepted'])}</p>"
            f"<a href=\"{esc(item['image_file'])}\"><img src=\"{esc(item['image_file'])}\" "
            f"alt=\"{esc(title)}\"></a>"
            f"<p><a href=\"{esc(item['source_rgb_file'])}\">原图</a> · "
            f"<a href=\"{esc(item['metadata_file'])}\">目标坐标及来源 JSON</a></p></section>"
        )
    page = (
        "<!doctype html><html lang=\"zh-CN\"><meta charset=\"utf-8\">"
        "<meta name=\"viewport\" content=\"width=device-width,initial-scale=1\">"
        "<title>模型目标与深度采样记录</title>"
        "<style>body{font-family:sans-serif;max-width:1280px;margin:24px auto;padding:0 16px;"
        "background:#f5f5f5;color:#202020}section{background:white;padding:18px;margin:20px 0;"
        "border:1px solid #ddd}img{max-width:100%;height:auto}p{line-height:1.7;overflow-wrap:anywhere}</style>"
        "<h1>每次决策的模型目标</h1>"
        "<p>绿框：模型返回的框；黄十字/小框：本次投影实际使用的深度采样位置。"
        "这些标注保留模型原始选择，不代表目标识别正确。像素坐标以原图左上角为(0,0)，"
        "x向右、y向下；图片上方说明栏不计入坐标。局部目标仍是原有名义前/左坐标，"
        "不是标定后的地图坐标。无投影时不绘制深度采样标记。刷新页面查看新决策。</p>"
        + "".join(cards) + "</html>"
    )
    temporary = output / ".targets.tmp.html"
    temporary.write_text(page, encoding="utf-8")
    temporary.replace(output / "targets.html")


def save_decision_target(output: Path, decision_index: int, raw_response: dict,
                         frame: ViewFrame, *, projection: dict | None = None,
                         navigation_accepted: bool | None = None,
                         record_mode: str = "live_decision") -> dict:
    """在模型选择的源图上绘制框，保留原图和响应；辅助文件失败由调用方处理。"""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    direction = raw_response.get("direction")
    selected = direction in DIRECTION_ORDER
    if selected and frame.direction != direction:
        raise ValueError("目标框必须画在模型选择方向的原图，不能套到另一视图")
    height, width = frame.rgb.shape[:2]
    raw_bbox = raw_response.get("bbox_2d")
    bbox = None
    if selected and raw_bbox is not None:
        values = np.asarray(raw_bbox, dtype=float)
        if values.shape != (4,) or not np.isfinite(values).all():
            raise ValueError("Invalid model bbox")
        bbox = [min(max(int(round(v)), 0), limit - 1)
                for v, limit in zip(values, (width, height, width, height))]
        if bbox[0] > bbox[2] or bbox[1] > bbox[3]:
            raise ValueError("Invalid clipped model bbox")
    stem = f"decision_{decision_index:03d}"
    metadata = {
        "decision_index": int(decision_index),
        "observation_id": raw_response.get("observation_id"),
        "action": raw_response.get("action"),
        "target": raw_response.get("target"),
        "selected_direction": direction,
        "displayed_direction": frame.direction,
        "source_frame_id": int(frame.frame_id),
        "source_image_size_wh": [width, height],
        "preview_header_height_px": 116,
        "source_rgb_file": f"{stem}_{frame.direction}.png",
        "bbox_xyxy_original": raw_bbox,
        "bbox_xyxy_clipped": bbox,
        "projection": projection,
        "stage_progress": raw_response.get("stage_progress") or {},
        "control": raw_response.get("control"),
        "stop_phase": raw_response.get("stop_phase"),
        "navigation_accepted": navigation_accepted,
        "record_mode": record_mode,
        "image_file": f"{stem}_target.png",
        "metadata_file": f"{stem}_target.json",
        "coordinate_note": "bbox/pixel_uv use original RGB pixels; local goal uses unchanged nominal forward/left axes, not map coordinates",
    }
    # OpenCV渲染说明栏用ASCII；HTML/JSON保留完整中文目标和阶段，无新增字体依赖。
    header = 116
    canvas = np.zeros((height + header, width, 3), dtype=np.uint8)
    canvas[:] = (24, 24, 24)
    canvas[header:] = cv2.cvtColor(frame.rgb, cv2.COLOR_RGB2BGR)
    if bbox is not None:
        x1, y1, x2, y2 = bbox
        cv2.rectangle(canvas, (x1, y1 + header), (x2, y2 + header), (0, 255, 0), 3)
    if projection is not None and bbox is not None:
        u, v = projection["pixel_uv"]
        cv2.drawMarker(canvas, (int(u), int(v) + header), (0, 255, 255),
                       cv2.MARKER_CROSS, 20, 2)
        x0, y0, x1, y1 = projection["depth_window_xyxy_exclusive"]
        cv2.rectangle(canvas, (x0, y0 + header), (x1 - 1, y1 - 1 + header), (0, 255, 255), 1)
    info = projection or {}
    lines = [f"MODEL TARGET #{decision_index} | {raw_response.get('action')} | {direction}",
             f"bbox xyxy={bbox} | source frame={frame.frame_id} | {width}x{height}",
             f"depth pixel={info.get('pixel_uv')} | depth={info.get('depth_m')}m | valid={info.get('valid_depth_count')}",
             "Green: model bbox | Yellow: depth sample | Chinese target and stage: targets.html"]
    for row, line in enumerate(lines):
        cv2.putText(canvas, line, (10, 23 + row * 28), cv2.FONT_HERSHEY_SIMPLEX,
                    .58, (235, 235, 235), 1, cv2.LINE_AA)
    if not cv2.imwrite(str(output / metadata["image_file"]), canvas):
        raise RuntimeError("Failed to save model target image")
    _write_json(output / metadata["metadata_file"], metadata)
    _write_index(output)
    return metadata


def backfill_run(output: Path) -> list[dict]:
    """只用已保存的原图/响应/投影补齐可视化，不请求模型或机器人。"""
    output = Path(output)
    results = []
    for path in sorted(output.glob("decision_*_response.json")):
        match = re.fullmatch(r"decision_(\d+)_response\.json", path.name)
        if match is None:
            continue
        index = int(match[1])
        raw = json.loads(path.read_text(encoding="utf-8"))
        direction = raw.get("direction")
        direction = direction if direction in DIRECTION_ORDER else "forward"
        stem = f"decision_{index:03d}"
        bgr = cv2.imread(str(output / f"{stem}_{direction}.png"))
        if bgr is None:
            raise FileNotFoundError(f"Missing saved source image: {stem}_{direction}.png")
        trace_path = output / f"{stem}_panorama_capture.json"
        trace = json.loads(trace_path.read_text()) if trace_path.exists() else {}
        source = trace.get("captures", {}).get(direction, {})
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        frame = ViewFrame(direction, source.get("frame_id", index), source.get("sim_step", 0),
                          source.get("timestamp", 0.), rgb, np.zeros(rgb.shape[:2], np.float32), np.eye(3))
        projection_path = output / f"{stem}_projection.json"
        projection = json.loads(projection_path.read_text()) if projection_path.exists() else None
        results.append(save_decision_target(output, index, raw, frame, projection=projection,
                                           record_mode="reconstructed_from_saved_observation"))
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True, help="已保存的正式真机run目录；仅生成辅助文件")
    args = parser.parse_args()
    print(f"Saved {len(backfill_run(args.run))} decisions: {args.run / 'targets.html'}")
