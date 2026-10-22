#!/usr/bin/env python3
"""D415/iPlanner 观察窗口。只有相机和 HTTP 客户端，不发送机器人命令。"""
from __future__ import annotations

import argparse
from dataclasses import dataclass, replace
from datetime import datetime
import json
import math
import os
from pathlib import Path
import queue
import threading
import time
import uuid

import cv2
import numpy as np

from camera_d415.network_camera import create_camera
from g1_camera_model_probe import load_panorama
from unified_vln.iplanner_client import IPlannerClient
from unified_vln.types import ViewFrame
from live_model_target import DEFAULT_RESULTS, DecisionWatcher, ModelTarget, draw_model_target, read_follow_preview

ROOT = Path(__file__).resolve().parent
HEADER, IMAGE_H, PANEL_W, PLOT_H = 32, 360, 640, 320
WINDOW = "D415 + iPlanner | PLAN ONLY - no robot commands"


def window_closed(name=WINDOW):
    """Qt关闭最后一个窗口后会删除guiReceiver；状态查询此时也可能抛异常。"""
    try:
        return cv2.getWindowProperty(name, cv2.WND_PROP_VISIBLE) < 1
    except cv2.error as error:
        if 'NULL guiReceiver' in str(error) or 'NULL window handler' in str(error):
            return True
        raise


def image_panel_geometry(shape):
    """保持原图比例；旧4:3图像在16:9面板两侧留黑。"""
    h, w = shape[:2]
    scale = min(PANEL_W / w, IMAGE_H / h)
    width, height = round(w * scale), round(h * scale)
    return (PANEL_W - width) // 2, (IMAGE_H - height) // 2, width, height


def panel_pixel_to_source(shape, x, y):
    """面板坐标含标题栏；黑边和深度面板不作为RGB目标。"""
    left, top, width, height = image_panel_geometry(shape)
    y -= HEADER
    if not (left <= x < left + width and top <= y < top + height):
        return None
    h, w = shape[:2]
    return int((x - left) * w / width), int((y - top) * h / height)


def goal_from_pixel(frame: ViewFrame, pixel, stand_off: float) -> dict:
    """局部预览坐标 X前/Y左，以RGB光心为原点；不是标定后的torso/map坐标。"""
    u, v = map(int, pixel)
    h, w = frame.depth_m.shape
    if not (0 <= u < w and 0 <= v < h):
        raise ValueError("Target pixel is outside the image")
    patch = frame.depth_m[max(0, v-3):min(h, v+4), max(0, u-3):min(w, u+4)]
    valid = patch[np.isfinite(patch) & (patch >= .1) & (patch <= 5.)]
    if not len(valid):
        raise ValueError("No valid 0.1..5m depth near target; select another pixel")
    z = float(np.percentile(valid, 30))
    K = np.asarray(frame.K)
    optical = np.array([(u-K[0, 2])*z/K[0, 0], (v-K[1, 2])*z/K[1, 1], z])
    target = np.array([z, -optical[0]])
    distance = float(np.linalg.norm(target))
    if distance <= stand_off:
        raise ValueError("Target is closer than the requested stand-off distance")
    goal = target * ((distance-stand_off)/distance)
    return {"pixel_uv": [u, v], "depth_m": z, "valid_depth_count": int(len(valid)),
            "target_camera_optical_xyz_m": optical.tolist(),
            "target_preview_xy_m": target.tolist(), "goal_preview_xy_m": goal.tolist(),
            "stand_off_m": stand_off,
            "frame": "RGB-origin nominal X-forward/Y-left; extrinsics NOT applied"}


def planner_frame(frame: ViewFrame) -> ViewFrame:
    # 原HTTP协议使用uint16的0.1mm单位，超过6.5535m会回绕；预览限定5m。
    d = np.asarray(frame.depth_m)
    return replace(frame, depth_m=np.where(np.isfinite(d) & (d >= .1) & (d <= 5), d, 0).astype(np.float32))


def model_goal_pixel(frame, target):
    """最近模型框底边中心作为预览目标；方向/分辨率未对齐时不生成规划目标。"""
    if target is None:
        return None
    box = target.box_for(frame.rgb.shape)
    if box is None:
        return None
    x1,y1,x2,y2 = box
    return (x1+x2)//2, y2


def model_target_record(target):
    if target is None:
        return None
    return {"decision_id":target.decision_id,"target":target.name,"action":target.action,
            "direction":target.direction,"bbox_2d":target.bbox,"response_file":target.source}


def plan_once(client, frame: ViewFrame, pixel, stand_off: float, version=0, model_target=None):
    target = goal_from_pixel(frame, pixel, stand_off)
    if not client.initialized and not client.reset(frame.K.tolist()):
        raise RuntimeError("iPlanner reset failed")
    started = time.monotonic()
    trajectory, fear = client.get_plan(planner_frame(frame), np.array(target["goal_preview_xy_m"]))
    path = np.asarray(trajectory, dtype=float)
    if path.ndim != 2 or path.shape[0] < 2 or path.shape[1] != 3 or not np.isfinite(path).all():
        raise RuntimeError("iPlanner did not return a finite N x 3 path")
    if fear is None or not math.isfinite(float(fear)):
        raise RuntimeError("iPlanner returned an invalid fear score")
    return {"status": "PASS", "source_frame_id": frame.frame_id, "target_version": version,
            "target_source":"model held bbox + current depth" if model_target else "manual pixel / replay",
            "model_decision":model_target_record(model_target),
            "target": target, "path_local_xyz_m": path.tolist(), "fear": float(fear),
            "request_duration_s": time.monotonic()-started, "received_monotonic_s": time.monotonic(),
            "robot_motion_sent": False, "extrinsics_applied": False,
            "coordinate_note": "Nominal camera-aligned local planner preview; not a verified robot/map path"}


@dataclass
class Job:
    frame: ViewFrame
    pixel: tuple[int, int]
    stand_off: float
    version: int
    model_target: ModelTarget | None = None


class PlanWorker:
    def __init__(self, url):
        self.client = IPlannerClient(url)
        self.jobs = queue.Queue(maxsize=1)
        self.result = None
        self.source_frame = None
        self.lock = threading.Lock()
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self.run, daemon=True)
        self.thread.start()

    def submit(self, job):
        try:
            self.jobs.get_nowait()
        except queue.Empty:
            pass
        self.jobs.put_nowait(job)

    def snapshot(self, version):
        with self.lock:
            if self.result is None or self.result["target_version"] != version:
                return None, None
            return self.result, self.source_frame

    def run(self):
        while not self.stop.is_set():
            try:
                job = self.jobs.get(timeout=.1)
            except queue.Empty:
                continue
            try:
                result = plan_once(self.client, job.frame, job.pixel, job.stand_off, job.version, job.model_target)
            except Exception as error:
                result = {"status": "FAILED", "error": str(error), "target_version": job.version,
                          "source_frame_id": job.frame.frame_id, "robot_motion_sent": False,
                          "received_monotonic_s": time.monotonic()}
            with self.lock:
                self.result, self.source_frame = result, job.frame

    def close(self):
        self.stop.set()
        self.thread.join(timeout=6)


class CameraFeed:
    def __init__(self, config):
        self.config = config
        self.camera = None
        self.lock = threading.Lock()
        self.frame = None
        self.error = None
        self.received = 0.
        self.generation = 0
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self.run, daemon=True)
        self.thread.start()

    def run(self):
        try:
            i = 0
            while not self.stop.is_set():
                try:
                    self.camera = create_camera(self.config)
                    first = True
                    while not self.stop.is_set():
                        frame = self.camera.capture_forward(i, time.monotonic())
                        with self.lock:
                            self.frame, self.received, self.error = frame, time.monotonic(), None
                            if first:
                                self.generation += 1
                                first = False
                        i += 1
                except Exception as error:
                    with self.lock:
                        self.error = str(error)
                finally:
                    if self.camera is not None:
                        self.camera.close()
                        self.camera = None
                self.stop.wait(.5)
        finally:
            if self.camera is not None:
                self.camera.close()

    def snapshot(self):
        with self.lock:
            return self.frame, self.received, self.error, self.generation

    def close(self):
        self.stop.set()
        self.thread.join(timeout=4)


def draw_plot(result):
    plot = np.full((PLOT_H, PANEL_W, 3), 25, np.uint8)
    origin = np.array([PANEL_W//2, PLOT_H-25])
    forward_limit = 3
    if result and result['status'] == 'PASS':
        forward_limit = max(2,min(8,math.ceil(max(
            result['target']['target_preview_xy_m'][0],
            max(p[0] for p in result['path_local_xyz_m']))+.5)))
    scale = (PLOT_H-65)/forward_limit
    side_limit = int((PANEL_W/2-15)/scale)

    def pixel(p):
        x, y = np.asarray(p)[:2]
        return tuple(np.clip(origin + np.array([-y, -x])*scale, -10000, 10000).astype(int))

    for y in range(-side_limit, side_limit+1):
        cv2.line(plot, pixel([0, y]), pixel([forward_limit, y]), (55, 55, 55), 1)
    for x in range(forward_limit+1):
        cv2.line(plot, pixel([x, -side_limit]), pixel([x, side_limit]), (55, 55, 55), 1)
        cv2.putText(plot, f"{x}m", pixel([x, side_limit]), cv2.FONT_HERSHEY_SIMPLEX, .35, (180, 180, 180), 1)
    cv2.putText(plot, "FORWARD ^    LEFT <    1 square = 1m", (140, 18),
                cv2.FONT_HERSHEY_SIMPLEX, .45, (220, 220, 220), 1)
    cv2.circle(plot, tuple(origin), 6, (255, 255, 255), -1)
    if result and result["status"] == "PASS":
        path = np.asarray([pixel(p) for p in result["path_local_xyz_m"]], np.int32)
        cv2.polylines(plot, [path], False, (255, 180, 40), 2, cv2.LINE_AA)
        cv2.circle(plot, pixel(result["target"]["target_preview_xy_m"]), 7, (0, 220, 255), 2)
        cv2.circle(plot, pixel(result["target"]["goal_preview_xy_m"]), 6, (0, 255, 0), -1)
        cv2.putText(plot, "blue: plan / yellow: target / green: stand-off goal", (75, PLOT_H-5),
                    cv2.FONT_HERSHEY_SIMPLEX, .4, (210, 210, 210), 1)
    return plot


def render(frame, pixel, result, *, mode, age, target_source, bbox=None, model_target=None, target_note=None):
    rgb = cv2.cvtColor(frame.rgb, cv2.COLOR_RGB2BGR)
    if model_target is not None:
        rgb = draw_model_target(rgb, model_target)
    if bbox is not None:
        x1,y1,x2,y2 = map(int,bbox)
        cv2.rectangle(rgb,(x1,y1),(x2,y2),(0,255,0),2)
    if pixel is not None:
        cv2.drawMarker(rgb, tuple(map(int, pixel)), (0, 230, 255), cv2.MARKER_CROSS, 24, 2)
    depth = cv2.applyColorMap((np.clip(frame.depth_m/5, 0, 1)*255).astype(np.uint8), cv2.COLORMAP_TURBO)
    depth[frame.depth_m <= 0] = 0
    top = []
    for title, image in [(f"RGB {mode} | {frame.rgb.shape[1]}x{frame.rgb.shape[0]} | frame {frame.frame_id}", rgb), ("ALIGNED DEPTH | display 0..5m", depth)]:
        header = np.full((HEADER, PANEL_W, 3), 35, np.uint8)
        cv2.putText(header, title, (10, 23), cv2.FONT_HERSHEY_SIMPLEX, .55, (255, 255, 255), 1)
        left, offset_y, width, height = image_panel_geometry(image.shape)
        panel = np.zeros((IMAGE_H, PANEL_W, 3), np.uint8)
        panel[offset_y:offset_y + height, left:left + width] = cv2.resize(image, (width, height))
        top.append(np.vstack([header, panel]))
    text = np.full((PLOT_H, PANEL_W, 3), 30, np.uint8)
    image_note = (f"Image age: {age:.2f}s" if mode == 'LIVE' else
                  f"CAMERA STALE: {age:.1f}s; planning paused" if mode == 'STALE' else 'Recorded image (not live)')
    lines = ["PLAN ONLY - NO ROBOT COMMANDS", "Axes: RGB-origin X forward / Y left (nominal)",
             "Camera extrinsics NOT applied; not a map path", f"{image_note} | target: {target_source}"]
    if result and result.get('executed_by_probe'):
        lines[0]='FOLLOWING IN PROBE - VIEWER IS READ ONLY'
        lines[1]='Path shown in current SLAM base XY; camera goal nominal'
        lines[2]='Actual shared follower path; camera extrinsics NOT applied'
    elif result and result.get('execution_source')=='formal_real_episode':
        lines[0]='FORMAL FOLLOWER - VIEWER IS READ ONLY'
        lines[1]='Original follower local path; nominal camera target'
        lines[2]='Local target extrinsics NOT applied; exploration map is separate'
    if result and result["status"] == "PASS":
        g = result["target"]["goal_preview_xy_m"]
        lines += [f"Path input frame: {result['source_frame_id']} (may precede live RGB)",
                  f"Goal: forward {g[0]:+.2f}m / left {g[1]:+.2f}m",
                  f"Plan age: {max(0,time.monotonic()-result['received_monotonic_s']):.2f}s",
                  f"Fear score: {result['fear']:.3f} (not a safety guarantee)",
                  f"Request: {result['request_duration_s']*1000:.0f}ms"]
    elif result:
        lines += ["PLAN FAILED: " + result["error"][:73]]
    else:
        lines += [target_note or "Click target near its bottom in the RGB panel.", "Waiting for target / planner..."]
    lines += ["Stationary decision only; no object tracking.", "H: hide old decision | S: save | C: manual clear | Q: quit"]
    for i, line in enumerate(lines):
        cv2.putText(text, line, (12, 22+i*24), cv2.FONT_HERSHEY_SIMPLEX, .44,
                    (0, 220, 255) if i == 0 else (230, 230, 230), 1, cv2.LINE_AA)
    return np.vstack([np.hstack(top), np.hstack([draw_plot(result), text])])


def render_waiting(error):
    image = np.full((HEADER+IMAGE_H+PLOT_H,2*PANEL_W,3),25,np.uint8)
    for i,line in enumerate(["Waiting for live D415 RGB-D; camera reconnecting...",
                             str(error or "Connecting to camera service")[:120],
                             "Planning paused. No robot commands. Q/ESC: quit"]):
        cv2.putText(image,line,(25,70+i*35),cv2.FONT_HERSHEY_SIMPLEX,.65,(0,220,255),1,cv2.LINE_AA)
    return image


def save_snapshot(output, image, result, source, *, announce=True):
    cv2.imwrite(str(output/"preview.png"), image)
    if result is not None:
        (output/"latest_plan.json").write_text(json.dumps(result, ensure_ascii=False, indent=2)+"\n")
    else:
        (output/"latest_plan.json").unlink(missing_ok=True)
    if source is not None:
        cv2.imwrite(str(output/"plan_input_rgb.png"), cv2.cvtColor(source.rgb, cv2.COLOR_RGB2BGR))
        np.save(output/"plan_input_depth_m.npy", source.depth_m)
        (output/"plan_input_K.json").write_text(json.dumps(source.K.tolist())+"\n")
    if announce:
        print(f"[VIEWER] Saved {output}", flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--camera-config", type=Path, default=ROOT/"camera_d415/network_camera_d415_local.json")
    parser.add_argument("--iplanner-url", default="http://127.0.0.1:8888")
    parser.add_argument("--replay-run", type=Path, help="使用一次已保存的全景和模型结果；不连接相机")
    parser.add_argument("--headless", action="store_true", help="仅重放：规划一次并保存预览")
    parser.add_argument("--stand-off-m", type=float, default=.5)
    parser.add_argument("--seconds", type=float, default=3600.)
    parser.add_argument("--replan-s", type=float, default=.5)
    parser.add_argument("--model-results", type=Path, default=DEFAULT_RESULTS,
                        help="监听模型决策，自动更新RGB目标框和规划目标")
    parser.add_argument("--target-source",choices=("model","manual"),default="model",
                        help="默认使用模型框；manual保留人工点击测试")
    parser.add_argument("--output", type=Path, default=ROOT.parents[1]/"outputs/iplanner_viewer"/
                        (datetime.now().strftime("run_%Y%m%d_%H%M%S_")+uuid.uuid4().hex[:8]))
    args = parser.parse_args(argv)
    if not all(math.isfinite(v) for v in (args.stand_off_m,args.seconds,args.replan_s)) or args.stand_off_m < 0 or min(args.seconds,args.replan_s) <= 0:
        parser.error("Invalid stand-off/time parameters")
    if args.headless and not args.replay_run:
        parser.error("--headless requires --replay-run")
    if args.output.exists():
        parser.error("--output must be a new directory")
    args.output.mkdir(parents=True)
    print(f"[VIEWER] PLAN ONLY. Output: {args.output}", flush=True)
    feed = worker = None
    image = result = source = None
    # 打开窗口时不把历史响应当作本轮的新决策；窗口保持运行即可接收之后每条结果。
    viewer_started_ns=time.time_ns()
    watcher = DecisionWatcher(args.model_results, since_ns=viewer_started_ns)
    display_title = None
    try:
        if args.replay_run:
            bundle = load_panorama(args.replay_run/"panorama")
            response = json.loads((args.replay_run/"model/response.json").read_text())
            if response.get("action") != "NAVIGATE" or response.get("direction") != "forward":
                raise ValueError("Replay planner test requires a NAVIGATE/forward decision")
            frame = bundle.views["forward"]
            x1,y1,x2,y2 = response["bbox_2d"]
            h,w = frame.depth_m.shape
            pixel = (int(np.clip((x1+x2)/2,0,w-1)),int(np.clip(y2,0,h-1)))
            result = plan_once(IPlannerClient(args.iplanner_url),frame,pixel,args.stand_off_m)
            result["source_run"] = str(args.replay_run.resolve())
            result["target_source"] = "saved model bbox bottom centre; not a fresh detection"
            source = frame
            image = render(frame,pixel,result,mode="RECORDED",age=0,target_source="saved model bbox",bbox=response['bbox_2d'])
            save_snapshot(args.output,image,result,source)
            print(json.dumps(result,ensure_ascii=False,indent=2),flush=True)
            if args.headless:
                return 0
        else:
            feed = CameraFeed(args.camera_config)
            worker = PlanWorker(args.iplanner_url)
            pixel = None
            frame = None
        selection = {"pixel":pixel, "version":0}
        # OpenCV wheel设置的qt/fonts目录可能不存在；只为本进程使用已安装的系统字体。
        font_dir = Path(os.environ.get('QT_QPA_FONTDIR', '/nonexistent'))
        system_fonts = Path('/usr/share/fonts/truetype/dejavu')
        if not font_dir.is_dir() and system_fonts.is_dir():
            os.environ['QT_QPA_FONTDIR'] = str(system_fonts)
        cv2.namedWindow(WINDOW,cv2.WINDOW_NORMAL)
        cv2.resizeWindow(WINDOW,2 * PANEL_W,HEADER + IMAGE_H + PLOT_H)

        def click(event,x,y,flags,param):
            if args.replay_run or frame is None or args.target_source != 'manual':
                return
            selected = panel_pixel_to_source(frame.rgb.shape, x, y)
            if event == cv2.EVENT_LBUTTONDOWN and selected is not None:
                selection["pixel"] = selected
                selection["version"] += 1
                print("[VIEWER] Manual target pixel:",selection["pixel"],flush=True)

        cv2.setMouseCallback(WINDOW,click)
        end = time.monotonic()+args.seconds
        next_plan = 0.
        last_saved_frame = None
        last_model_key = None
        last_generation = None
        degraded = False
        initial_preview_saved = False
        was_following = False
        while time.monotonic() < end:
            if feed:
                model_changed = False
                frame,received,error,generation = feed.snapshot()
                if frame is None:
                    cv2.imshow(WINDOW,render_waiting(error))
                    if cv2.waitKey(50)&255 in (27,ord('q')) or window_closed():
                        break
                    continue
                age = time.monotonic()-received
                model_target = watcher.poll()
                active_follow=read_follow_preview(args.model_results,viewer_started_ns)
                if generation != last_generation:
                    selection['version'] += 1
                    last_model_key = None
                    last_generation = generation
                if args.target_source == 'model':
                    model_key = (watcher.revision,generation,frame.rgb.shape[:2])
                    if model_key != last_model_key:
                        model_changed = True
                        selection['pixel'] = model_goal_pixel(frame,model_target)
                        selection['version'] += 1
                        last_model_key = model_key
                        next_plan = 0.
                        print('[VIEWER] Model target updated:',model_target_record(model_target),'pixel=',selection['pixel'],flush=True)
                        (args.output/'latest_target.json').write_text(json.dumps(model_target_record(model_target),ensure_ascii=False,indent=2))
                        (args.output/'latest_plan.json').unlink(missing_ok=True)
                healthy = not error and age <= 1.
                if not healthy and not degraded:
                    selection['version'] += 1
                degraded = not healthy
                pixel = selection["pixel"]
                result,source = worker.snapshot(selection["version"]) if healthy and pixel is not None else (None,None)
                if healthy and active_follow is not None:
                    result,source=active_follow,None
                    pixel=None
                if was_following and active_follow is None:
                    (args.output/'latest_plan.json').unlink(missing_ok=True)
                    model_changed=True
                was_following=active_follow is not None
                if healthy and active_follow is None and pixel is not None and time.monotonic() >= next_plan:
                    worker.submit(Job(frame,pixel,args.stand_off_m,selection["version"],
                                      model_target if args.target_source == 'model' else None))
                    next_plan = time.monotonic()+args.replan_s
                if not healthy:
                    (args.output/'latest_plan.json').unlink(missing_ok=True)
                image = render(frame,pixel if healthy else None,result,mode="LIVE" if healthy else "STALE",age=age,
                               target_source='actual single-follow probe' if active_follow else
                               "model decision" if args.target_source == 'model' else "manual click",
                               model_target=model_target if healthy and args.target_source == 'model' else None,
                               target_note=("Motion started; old target cleared. Waiting for next decision."
                                            if watcher.hidden_by_motion else "Model target pending / stopped / other view")
                               if args.target_source == 'model' else None)
                title = (watcher.window_title("D415 + iPlanner | 仅规划", frame.rgb.shape) if healthy else
                         "D415 + iPlanner | 相机重连中，规划暂停")
                if healthy and active_follow:
                    title=f"D415 真机单次跟随监视 | 实时 {frame.rgb.shape[1]}×{frame.rgb.shape[0]} | 只读"
                if title != display_title:
                    cv2.setWindowTitle(WINDOW, title)
                    display_title = title
                if not initial_preview_saved or model_changed:
                    cv2.imwrite(str(args.output/'preview.png'),image)
                if not initial_preview_saved:
                    print('[VIEWER] LIVE RGB-D ready. Target source:',args.target_source,flush=True)
                    initial_preview_saved = True
                result_key = (result['target_version'],result['source_frame_id']) if result else None
                if result and result["status"] == "PASS" and last_saved_frame != result_key:
                    save_snapshot(args.output,image,result,source,announce=last_saved_frame is None)
                    last_saved_frame = result_key
            cv2.imshow(WINDOW,image)
            key = cv2.waitKey(20)&255
            if key in (27,ord('q')) or window_closed():
                break
            if key == ord('s'):
                save_snapshot(args.output,image,result,source)
            if key == ord('h') and feed and args.target_source == 'model':
                watcher.hide_until_next_decision()
                selection['pixel'] = None
                selection['version'] += 1
                result = source = None
                (args.output/'latest_plan.json').unlink(missing_ok=True)
                (args.output/'latest_target.json').write_text('null\n')
            if key == ord('c') and feed and args.target_source == 'manual':
                selection["pixel"] = None
                selection["version"] += 1
                (args.output/'latest_plan.json').unlink(missing_ok=True)
        return 0
    except KeyboardInterrupt:
        print("[VIEWER] Closed",flush=True)
        return 130
    except Exception as error:
        (args.output/"error.json").write_text(json.dumps({"error":str(error),"robot_motion_sent":False})+"\n")
        print(f"[VIEWER] Failed: {error}",flush=True)
        return 1
    finally:
        if feed:
            feed.close()
        if worker:
            worker.close()
        if not args.headless:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    raise SystemExit(main())
