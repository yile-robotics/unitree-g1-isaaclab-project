#!/usr/bin/env python3
"""D415/iPlanner 观察窗口。只有相机和 HTTP 客户端，不发送机器人命令。"""
from __future__ import annotations

import argparse
from dataclasses import dataclass, replace
from datetime import datetime
import json
import math
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

ROOT = Path(__file__).resolve().parent
HEADER, IMAGE_H, PANEL_W, PLOT_H = 32, 480, 640, 320
WINDOW = "D415 + iPlanner | PLAN ONLY - no robot commands"


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


def plan_once(client, frame: ViewFrame, pixel, stand_off: float, version=0):
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
                result = plan_once(self.client, job.frame, job.pixel, job.stand_off, job.version)
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
        self.camera = create_camera(config)
        self.lock = threading.Lock()
        self.frame = None
        self.error = None
        self.received = 0.
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self.run, daemon=True)
        self.thread.start()

    def run(self):
        try:
            i = 0
            while not self.stop.is_set():
                frame = self.camera.capture_forward(i, time.monotonic())
                with self.lock:
                    self.frame, self.received = frame, time.monotonic()
                i += 1
        except Exception as error:
            with self.lock:
                self.error = str(error)
        finally:
            self.camera.close()

    def snapshot(self):
        with self.lock:
            return self.frame, self.received, self.error

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


def render(frame, pixel, result, *, mode, age, target_source, bbox=None):
    rgb = cv2.cvtColor(frame.rgb, cv2.COLOR_RGB2BGR)
    if bbox is not None:
        x1,y1,x2,y2 = map(int,bbox)
        cv2.rectangle(rgb,(x1,y1),(x2,y2),(0,255,0),2)
    if pixel is not None:
        cv2.drawMarker(rgb, tuple(map(int, pixel)), (0, 230, 255), cv2.MARKER_CROSS, 24, 2)
    depth = cv2.applyColorMap((np.clip(frame.depth_m/5, 0, 1)*255).astype(np.uint8), cv2.COLORMAP_TURBO)
    depth[frame.depth_m <= 0] = 0
    top = []
    for title, image in [(f"RGB {mode} | frame {frame.frame_id}", rgb), ("ALIGNED DEPTH | display 0..5m", depth)]:
        header = np.full((HEADER, PANEL_W, 3), 35, np.uint8)
        cv2.putText(header, title, (10, 23), cv2.FONT_HERSHEY_SIMPLEX, .55, (255, 255, 255), 1)
        top.append(np.vstack([header, cv2.resize(image, (PANEL_W, IMAGE_H))]))
    text = np.full((PLOT_H, PANEL_W, 3), 30, np.uint8)
    image_note = f"Image age: {age:.2f}s" if mode == 'LIVE' else 'Recorded image (not live)'
    lines = ["PLAN ONLY - NO ROBOT COMMANDS", "Axes: RGB-origin X forward / Y left (nominal)",
             "Camera extrinsics NOT applied; not a map path", f"{image_note} | target: {target_source}"]
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
        lines += ["Click target near its bottom in the RGB panel.", "Waiting for target / planner..."]
    lines += ["Fixed pixel selection; no object tracking.", "S: save | C: clear target | Q/ESC: quit"]
    for i, line in enumerate(lines):
        cv2.putText(text, line, (12, 22+i*24), cv2.FONT_HERSHEY_SIMPLEX, .44,
                    (0, 220, 255) if i == 0 else (230, 230, 230), 1, cv2.LINE_AA)
    return np.vstack([np.hstack(top), np.hstack([draw_plot(result), text])])


def save_snapshot(output, image, result, source, *, announce=True):
    cv2.imwrite(str(output/"preview.png"), image)
    if result is not None:
        (output/"latest_plan.json").write_text(json.dumps(result, ensure_ascii=False, indent=2)+"\n")
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
        cv2.namedWindow(WINDOW,cv2.WINDOW_NORMAL)
        cv2.resizeWindow(WINDOW,1280,832)

        def click(event,x,y,flags,param):
            if args.replay_run or frame is None:
                return
            if event == cv2.EVENT_LBUTTONDOWN and 0 <= x < PANEL_W and HEADER <= y < HEADER+IMAGE_H:
                selection["pixel"] = (int(x*frame.rgb.shape[1]/PANEL_W),int((y-HEADER)*frame.rgb.shape[0]/IMAGE_H))
                selection["version"] += 1
                print("[VIEWER] Manual target pixel:",selection["pixel"],flush=True)

        cv2.setMouseCallback(WINDOW,click)
        end = time.monotonic()+args.seconds
        next_plan = 0.
        last_saved_frame = None
        initial_preview_saved = False
        while time.monotonic() < end:
            if feed:
                frame,received,error = feed.snapshot()
                if error:
                    raise RuntimeError("Camera: "+error)
                if frame is None:
                    time.sleep(.02)
                    continue
                age = time.monotonic()-received
                if age > 1:
                    raise RuntimeError("No fresh camera frame for 1 second")
                pixel = selection["pixel"]
                result,source = worker.snapshot(selection["version"])
                if pixel is not None and time.monotonic() >= next_plan:
                    worker.submit(Job(frame,pixel,args.stand_off_m,selection["version"]))
                    next_plan = time.monotonic()+args.replan_s
                image = render(frame,pixel,result,mode="LIVE",age=age,target_source="manual click" if pixel else "none")
                if not initial_preview_saved:
                    cv2.imwrite(str(args.output/'preview.png'),image)
                    print('[VIEWER] LIVE RGB-D ready. Click a target near its bottom in the RGB panel.',flush=True)
                    initial_preview_saved = True
                if result and result["status"] == "PASS" and last_saved_frame != result["source_frame_id"]:
                    save_snapshot(args.output,image,result,source,announce=last_saved_frame is None)
                    last_saved_frame = result["source_frame_id"]
            cv2.imshow(WINDOW,image)
            key = cv2.waitKey(20)&255
            if key in (27,ord('q')) or cv2.getWindowProperty(WINDOW,cv2.WND_PROP_VISIBLE)<1:
                break
            if key == ord('s'):
                save_snapshot(args.output,image,result,source)
            if key == ord('c') and feed:
                selection["pixel"] = None
                selection["version"] += 1
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
