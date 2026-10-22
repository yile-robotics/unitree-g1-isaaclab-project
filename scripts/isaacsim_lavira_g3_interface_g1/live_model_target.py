"""只读模型结果，将最近一次选择叠到实时RGB；不请求模型、不控制机器人。"""
from __future__ import annotations

from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
import time

DEFAULT_RESULTS = Path(__file__).resolve().parents[2] / "outputs/d415_rotation_model"
MOTION_EVENT = "display_motion.json"


def mark_display_motion(root=DEFAULT_RESULTS):
    """仅通知显示端清框，不发送运动命令；下一条决策写入后才能再次显示。"""
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    path = root / MOTION_EVENT
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps({"event": "motion_start", "robot_command_sent": False}) + "\n")
    temporary.replace(path)
    return path


def read_follow_preview(root=DEFAULT_RESULTS, since_ns=0):
    """只读单次或正式真机跟随器使用的路径；停止、未写完或0.5秒未更新时隐藏。"""
    root=Path(root)
    if root.is_file():root=root.parent
    try:
        paths=list(root.rglob('follow_live.json'))
        if not paths:return None
        path=max(paths,key=lambda p:p.stat().st_mtime_ns)
        if path.stat().st_mtime_ns<=since_ns:return None
        value=json.loads(path.read_text())
        if value.get('state')!='FOLLOWING' or not 0<=time.monotonic()-value['received_monotonic_s']<=.5:
            return None
        points=value['path_local_xyz_m']
        if len(points)<2 or any(len(p)!=3 or not all(math.isfinite(float(v)) for v in p) for p in points):
            return None
        for key in ('target_preview_xy_m','goal_preview_xy_m'):
            xy=value['target'][key]
            if len(xy)!=2 or not all(math.isfinite(float(v)) for v in xy):return None
        if value.get('status')!='PASS' or not (value.get('executed_by_probe') is True
                or value.get('execution_source')=='formal_real_episode'):return None
        return value
    except (OSError,ValueError,TypeError,KeyError):
        return None


@dataclass(frozen=True)
class ModelTarget:
    action: str
    direction: str
    name: str
    decision_id: str
    bbox: tuple[float, ...] | None
    width: int
    height: int
    source: str
    view_aligned: bool = False

    def box_for(self, shape):
        # 4:3到16:9可能改变裁剪范围，不能按宽高比例拉伸旧框。
        if self.action != "NAVIGATE" or (self.direction != "forward" and not self.view_aligned) or self.bbox is None:
            return None
        h, w = shape[:2]
        if (w, h) != (self.width, self.height):
            return None
        x1, y1, x2, y2 = self.bbox
        x1, x2 = max(0, min(w-1, round(x1))), max(0, min(w-1, round(x2)))
        y1, y2 = max(0, min(h-1, round(y1))), max(0, min(h-1, round(y2)))
        return (x1, y1, x2, y2) if x1 < x2 and y1 < y2 else None


class DecisionWatcher:
    """读取探针model/response.json或正式真机decision_NNN_response.json。

    按文件修改时间选择最近结果；不重新写结果。未写完的JSON稍后重试，
    下一条完整STOP/无框/无效决策清除旧框。旋转入口的display_motion.json清除运动前结果。
    可指定单次运行目录以隔离其他会话；since_ns用于跳过窗口打开前的历史响应。
    """
    def __init__(self, root=DEFAULT_RESULTS, interval=.25, since_ns=0):
        self.root = Path(root)
        self.interval = interval
        self.target = None
        self.error = None
        self.revision = 0
        self._signature = None
        self._next_poll = 0.
        self._cutoff_ns = since_ns
        self.hidden_by_motion = False

    def hide_until_next_decision(self):
        """外部遥控运动前可手动隐藏；这是本地显示操作，不控制机器人。"""
        self._cutoff_ns = max(self._cutoff_ns, time.time_ns())
        self._signature = None
        self.target = None
        self.error = None
        self.hidden_by_motion = True
        self.revision += 1
        self._next_poll = 0.

    def poll(self):
        now = time.monotonic()
        if now < self._next_poll:
            return self.target
        self._next_poll = now + self.interval
        event_root = self.root.parent if self.root.is_file() else self.root
        events = set(event_root.rglob(MOTION_EVENT))
        events.update((event_root.parent/MOTION_EVENT, event_root.parent.parent/MOTION_EVENT))
        for event in events:
            try:
                stamp = event.stat().st_mtime_ns
            except OSError:
                continue
            if stamp > self._cutoff_ns:
                self._cutoff_ns = stamp
                self._signature = None
                self.target = None
                self.error = None
                self.hidden_by_motion = True
                self.revision += 1
        candidates = []
        if self.root.is_file():
            paths = [self.root]
        elif self.root.is_dir():
            paths = list(self.root.rglob("response.json")) + list(self.root.rglob("decision_*_response.json"))
        else:
            return self.target
        for path in paths:
            try:
                stat = path.stat()
                candidates.append((stat.st_mtime_ns, str(path), stat.st_size, path))
            except OSError:
                continue
        if not candidates:
            return self.target
        stamp, name, size, path = max(candidates)
        if stamp <= self._cutoff_ns:
            return self.target
        alignment_path = path.with_name(path.name.replace("_response.json", "_overlay.json")) if path.name != "response.json" else None
        try:
            alignment_stamp = alignment_path.stat().st_mtime_ns if alignment_path else 0
        except OSError:
            alignment_stamp = 0
        signature = (name, stamp, size, alignment_stamp)
        if signature == self._signature:
            return self.target
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
            request_path = (path.with_name("request_metadata.json") if path.name == "response.json"
                            else path.with_name(path.name.replace("_response.json", "_request.json")))
            request = json.loads(request_path.read_text(encoding="utf-8")) if request_path.exists() else {}
            alignment = json.loads(alignment_path.read_text(encoding="utf-8")) if alignment_stamp else {}
        except (OSError, ValueError):
            return self.target  # 文件可能正在写入；不将半条响应当作新决策。
        self._signature = signature
        self.revision += 1
        self.hidden_by_motion = False
        self.target = None
        self.error = None
        try:
            action = str(raw.get("control") if raw.get("control") == "SAFE_STOP" else raw.get("action", "UNKNOWN")).upper()
            bbox = raw.get("bbox_2d") if action == "NAVIGATE" else None
            if bbox is not None:
                if len(bbox) != 4 or not all(math.isfinite(float(v)) for v in bbox):
                    raise ValueError("invalid model bbox")
                bbox = tuple(float(v) for v in bbox)
                if bbox[0] >= bbox[2] or bbox[1] >= bbox[3]:
                    raise ValueError("invalid model bbox order")
            self.target = ModelTarget(action, str(raw.get("direction", "")), str(raw.get("target") or ""),
                                      str(request.get("decision_index", raw.get("observation_id", "?"))),
                                      bbox, int(request.get("image_width", 0)), int(request.get("image_height", 0)), name,
                                      alignment.get("view_aligned") is True
                                      and alignment.get("direction") == raw.get("direction")
                                      and alignment.get("decision_index") == request.get("decision_index"))
        except (TypeError, ValueError, AttributeError) as error:
            self.error = str(error)
        return self.target

    def window_title(self, base, shape=None):
        if shape is not None:
            h, w = shape[:2]
            base += f" | 实时 {w}×{h}"
            if self.target is not None and self.target.bbox is not None and (w,h) != (self.target.width,self.target.height):
                return base + " | 等待新分辨率的模型决策（历史结果未显示）"
        if self.target is None:
            if self.hidden_by_motion:
                return base + " | 运动后旧框已清除，等待下一次决策"
            return base + (" | 模型结果无效，旧框已清除" if self.error else " | 等待模型决策")
        t = self.target
        return f"{base} | 决策 {t.decision_id} | {t.action}/{t.direction} | {t.name}"


def draw_model_target(bgr, target):
    """只改显示副本；停车期间显示最新框，不跟踪，不控制机器人。"""
    import cv2
    image = bgr.copy()
    h, w = image.shape[:2]
    prefix = f"LIVE RGB {w}x{h} | "
    if target is None:
        text = prefix + "Waiting for model decision"
        box = None
    else:
        box = target.box_for(image.shape)
        text = prefix + f"MODEL #{target.decision_id[:24]} {target.action} / {target.direction}"
    if box is not None:
        x1,y1,x2,y2 = box
        thickness = max(2, image.shape[1] // 400)
        cv2.rectangle(image, (x1,y1), (x2,y2), (0,255,0), thickness)
        text += " | stationary decision; not tracked"
    elif target is not None and target.bbox is not None:
        if (w,h) != (target.width,target.height):
            text = prefix + "Waiting for new model decision (old mode result skipped)"
        else:
            text += " | awaiting camera turn / visible box"
    scale = max(.45, image.shape[1]/1600)
    cv2.putText(image,text,(10,max(24,round(30*scale))),cv2.FONT_HERSHEY_SIMPLEX,scale,(0,255,0),2,cv2.LINE_AA)
    return image
