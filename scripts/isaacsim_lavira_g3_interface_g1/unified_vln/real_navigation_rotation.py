"""真机选定方向转向：沿用全景的IMU75°停车、停稳、SLAM70..110°核验。"""
import math

from .rotation import RotationCommand, TimedFixedSpeedRotation


class RealNavigationRotation(TimedFixedSpeedRotation):
    def __init__(self, speed_rad_s, duration_scale, *, yaw_reader, imu_reader, clock,
                 on_stop, stop_deg=75., settle_s=.5, quarter_timeout_s=10.):
        super().__init__(speed_rad_s, duration_scale)
        self.yaw_reader, self.imu_reader = yaw_reader, imu_reader
        self.clock, self.on_stop = clock, on_stop
        self.stop_deg, self.settle_s = float(stop_deg), float(settle_s)
        self.quarter_timeout_s = float(quarter_timeout_s)
        self.status, self.phase = 'IDLE', 'idle'
        self.initial_yaw_rad = None
        self.failure_reason = None
        self.quarter_records = []

    def start(self, direction):
        super().start(direction)
        self.status, self.phase = ('RUNNING','ready_to_turn') if self.active else ('SKIPPED','idle')
        self.initial_yaw_rad = None
        self.failure_reason = None
        self.quarter_records = []
        self.quarters_required = round(abs(self.target_yaw_rad)/(math.pi/2))
        self.sign = 1 if self.target_yaw_rad > 0 else -1
        self.started_at = None
        self.slam_accumulated_rad = 0.

    @staticmethod
    def _read(reader, label):
        value = reader()
        if value is None or not math.isfinite(float(value)):
            raise RuntimeError(f'Real navigation rotation lost fresh {label}.')
        return float(value)

    def _command(self, wz=0., done=False):
        return RotationCommand(0.,0.,wz,done,self.elapsed_s,self.duration_s)

    def update(self, dt):
        if not math.isfinite(dt) or dt < 0:
            raise ValueError('Rotation dt must be finite and non-negative.')
        if self.status == 'FAILED':
            raise RuntimeError(self.failure_reason)
        if not self.active:
            return self._command(done=True)
        try:
            imu = self._read(self.imu_reader,'rt/lowstate IMU')
            yaw = self._read(self.yaw_reader,'SLAM yaw')
            now = self.clock()
            if self.started_at is None:
                self.started_at, self.initial_yaw_rad = now, yaw
            self.elapsed_s = now-self.started_at
            if self.phase == 'ready_to_turn':
                self.initial_quarter_imu = self.previous_imu = imu
                self.initial_quarter_yaw, self.quarter_started_at = yaw, now
                self.quarter_imu_rad = 0.
                self.phase, self.status = 'turning','RUNNING'
                return self._command(self.sign*self.speed_rad_s)

            self.quarter_imu_rad += self._wrap_to_pi(imu-self.previous_imu)*self.sign
            self.previous_imu = imu
            self.accumulated_yaw_rad = sum(r['imu_deg'] for r in self.quarter_records)*math.pi/180+self.quarter_imu_rad
            if self.phase == 'turning':
                if self.quarter_imu_rad <= math.radians(-20):
                    raise RuntimeError('Real navigation rotation IMU shows wrong direction; planning blocked.')
                if now-self.quarter_started_at >= self.quarter_timeout_s:
                    self.feedback_timed_out = True
                    raise RuntimeError(
                        f'Real navigation rotation timed out: {self.direction}, '
                        f'quarter={len(self.quarter_records)+1}/{self.quarters_required}, '
                        f'IMU={math.degrees(self.quarter_imu_rad):.1f}deg; planning and translation blocked.')
                if self.quarter_imu_rad+1e-9 >= math.radians(self.stop_deg):
                    self.phase, self.status = 'settling','SETTLING'
                    self.on_stop()  # 先停车，返回后开始额外的停稳等待。
                    self.settle_after = self.clock()+self.settle_s
                    return self._command()
                return self._command(self.sign*self.speed_rad_s)

            if self.phase == 'settling':
                if now < self.settle_after:
                    return self._command()
                slam_deg = math.degrees(self._wrap_to_pi(yaw-self.initial_quarter_yaw))*self.sign
                if not 70. <= slam_deg <= 110.:
                    raise RuntimeError(f'Real navigation rotation SLAM quarter {slam_deg:+.1f}deg outside 70..110; planning blocked.')
                self.slam_accumulated_rad += math.radians(slam_deg)
                self.quarter_records.append({'imu_deg':math.degrees(self.quarter_imu_rad),
                                             'slam_deg':slam_deg, 'sign':self.sign})
                if len(self.quarter_records) < self.quarters_required:
                    self.phase = 'ready_to_turn'
                    return self._command()
                self.active = False
                self.phase, self.status = 'complete','ANGLE_REACHED'
                return self._command(done=True)
            raise RuntimeError(f'Unexpected real navigation rotation phase: {self.phase}')
        except Exception as error:
            self.active = False
            self.status, self.failure_reason = 'FAILED', str(error)
            raise

    def verify_settled(self, direction, tolerance_deg=20.):
        if direction == 'forward':
            return None
        if (self.direction != direction or self.status not in ('ANGLE_REACHED','SETTLED_VERIFIED')
                or self.initial_yaw_rad is None):
            raise RuntimeError('Real navigation rotation not completed; planning blocked.')
        try:
            actual_yaw = self._read(self.yaw_reader,'SLAM yaw')
        except Exception as error:
            self.status, self.failure_reason = 'FAILED', str(error)
            raise
        error = self._wrap_to_pi(actual_yaw-self.initial_yaw_rad-self.target_yaw_rad)
        if abs(error) > math.radians(tolerance_deg):
            self.status = 'FAILED'
            self.failure_reason = f'Real navigation rotation settled error {math.degrees(error):+.1f}deg; planning blocked.'
            raise RuntimeError(self.failure_reason)
        self.status = 'SETTLED_VERIFIED'
        return {'settled_yaw_rad':actual_yaw, 'settled_error_deg':math.degrees(error),
                'settled_tolerance_deg':tolerance_deg}

    def snapshot(self):
        return {'status':self.status, 'phase':self.phase, 'direction':self.direction,
                'target_yaw_deg':math.degrees(self.target_yaw_rad),
                'measured_imu_yaw_deg':math.degrees(self.accumulated_yaw_rad)*getattr(self,'sign',1),
                'measured_slam_yaw_deg':math.degrees(getattr(self,'slam_accumulated_rad',0.))*getattr(self,'sign',1),
                'initial_yaw_rad':self.initial_yaw_rad, 'quarter_turns':self.quarter_records,
                'speed_rad_s':self.speed_rad_s, 'imu_stop_deg':self.stop_deg,
                'settle_s':self.settle_s, 'elapsed_s':self.elapsed_s,
                'quarter_timeout_s':self.quarter_timeout_s, 'failure_reason':self.failure_reason,
                'feedback_timed_out':self.feedback_timed_out}
