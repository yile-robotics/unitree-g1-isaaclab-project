"""真机外围的数据失效停车；不改变Episode、规划器或跟随算法。"""
import math
import threading
import time

from .odometry import wrap_to_pi


class RealInputGuard:
    def __init__(self, backend, camera, pose_source, lowstate):
        self.backend, self.camera = backend, camera
        self.pose_source, self.lowstate = pose_source, lowstate
        self.lock = threading.Lock()
        self.done = threading.Event()
        self.command = (0., 0., 0.)
        self.error = None
        self.shutdown_requested = threading.Event()
        self.shutdown_stop_lock = threading.Lock()
        self.shutdown_reason = None
        self.shutdown_times = {}
        self.shutdown_stop_done = False
        self.shutdown_stop_error = None
        self.previous_pose = pose_source.get_pose()
        self.thread = threading.Thread(target=self._watch, name='real-input-guard', daemon=True)
        self.thread.start()

    def __getattr__(self, name):
        return getattr(self.backend, name)

    def check(self):
        if self.lowstate.get_yaw() is None:
            raise RuntimeError('lowstate missing or stale')
        pose = self.pose_source.get_pose()
        previous = self.previous_pose
        if (math.hypot(pose.x-previous.x, pose.y-previous.y) > .25
                or abs(wrap_to_pi(pose.yaw-previous.yaw)) > math.radians(30)):
            raise RuntimeError('SLAM pose jumped')
        self.previous_pose = pose
        if self.camera.input_error is not None:
            raise RuntimeError(self.camera.input_error)
        with self.lock:
            walking = self.command[0] != 0. or self.command[1] != 0.
        # 转圈采全景时有意不连续取图；行走期间必须持续取得新图。
        if walking:
            received = self.camera.last_metadata.get('client_received_monotonic_s')
            if received is None or not 0 <= time.monotonic()-received <= 1.:
                raise RuntimeError('camera input stale while following')

    def _watch(self):
        while not self.done.is_set():
            if self.shutdown_requested.wait(.02):
                try:
                    self._perform_shutdown_stop()
                except Exception as error:
                    print(f'[REAL INPUT GUARD] shutdown stop request failed: {error}', flush=True)
                return
            if self.done.is_set():
                return
            try:
                self.check()
            except Exception as error:
                with self.lock:
                    self.error = str(error)
                    self.command = (0., 0., 0.)
                    self.backend.set_velocity(0., 0., 0.)
                print(f'[REAL INPUT GUARD] {self.error}; requesting stop', flush=True)
                self.backend.stop()
                return

    def set_velocity(self, vx, vy, wz):
        with self.lock:
            if self.shutdown_requested.is_set() and any((vx, vy, wz)):
                raise RuntimeError('Manual shutdown latched; nonzero command rejected.')
            if self.error is not None and any((vx, vy, wz)):
                raise RuntimeError('Real input guard latched: '+self.error)
            self.command = (vx, vy, wz)
            self.backend.set_velocity(vx, vy, wz)

    def stop(self):
        if self.shutdown_requested.is_set():
            self._perform_shutdown_stop()
            return
        with self.lock:
            self.command = (0., 0., 0.)
            self.backend.set_velocity(0., 0., 0.)
        self.backend.stop()

    def request_shutdown_stop(self, reason='user_interrupt'):
        # 信号处理器可能打断正在持有self.lock的主线程，此处不能取该锁或同步等待SDK。
        if self.shutdown_reason is None:
            self.shutdown_reason = reason
            self.shutdown_times['signal_requested_monotonic_s'] = time.monotonic()
        if callable(getattr(type(self.backend), 'request_shutdown_stop', None)):
            self.backend.request_shutdown_stop(reason)
        self.shutdown_requested.set()

    def _perform_shutdown_stop(self):
        # 主线程收尾与独立保护线程竞争时只执行一次该次关停的StopMove。
        with self.shutdown_stop_lock:
            if self.shutdown_stop_done:
                if self.shutdown_stop_error is not None:
                    raise RuntimeError(self.shutdown_stop_error)
                return
            try:
                with self.lock:
                    self.command = (0., 0., 0.)
                    try:
                        self.backend.set_velocity(0., 0., 0.)
                        self.shutdown_times['zero_target_monotonic_s'] = time.monotonic()
                    except Exception as error:
                        self.shutdown_stop_error = f'zero target failed: {error}'
                self.shutdown_times['stop_call_started_monotonic_s'] = time.monotonic()
                self.backend.stop()
                if self.shutdown_stop_error is not None:
                    raise RuntimeError(self.shutdown_stop_error)
            except Exception as error:
                self.shutdown_stop_error = str(error)
                raise
            finally:
                self.shutdown_times['stop_call_returned_monotonic_s'] = time.monotonic()
                self.shutdown_stop_done = True

    def close(self):
        self.done.set()
        self.backend.close()
        self.thread.join(timeout=1.)

    def guard_report(self):
        with self.lock:
            return {'latched_error': self.error, 'last_requested_command': list(self.command),
                    'manual_shutdown_reason':self.shutdown_reason,
                    'manual_shutdown_times':dict(self.shutdown_times),
                    'manual_shutdown_error':self.shutdown_stop_error,
                    'note': 'software stop requests; not confirmation of physical stopping'}
