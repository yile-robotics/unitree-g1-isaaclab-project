from __future__ import annotations

"""记录 WaistYaw 实验 CSV，并计算摔倒、接触和足部滑动指标。"""

import csv
import math
from pathlib import Path

import torch


def _roll_pitch_from_wxyz(quaternion: torch.Tensor) -> tuple[float, float]:
    """把 IsaacLab 的 ``(w,x,y,z)`` 四元数转换成 roll 和 pitch。"""

    # yaw 不参与当前摔倒判断，所以这里不额外计算 yaw。
    w, x, y, z = (float(value) for value in quaternion)
    roll = math.atan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
    # 浮点误差可能让 asin 的输入略微超出 [-1,1]，先夹紧避免数学域错误。
    pitch_sine = min(max(2.0 * (w * y - z * x), -1.0), 1.0)
    return roll, math.asin(pitch_sine)


class ExperimentMetrics:
    """按 policy 控制频率记录单个仿真环境的稳定性指标。"""

    # 固定列顺序便于人工阅读，也便于后续脚本稳定地解析 CSV。
    FIELDNAMES = (
        "time_s",
        "step",
        "stage_index",
        "mode",
        "control_state",
        "arm_sdk_takeover_weight",
        "user_blend_weight",
        "waist_actual_rad",
        "waist_velocity_rad_s",
        "waist_policy_baseline_q_des_rad",
        "waist_user_requested_rad",
        "waist_arm_sdk_reference_rad",
        "waist_final_q_des_rad",
        "waist_tracking_error_rad",
        "base_roll_rad",
        "base_pitch_rad",
        "base_height_m",
        "base_vx_m_s",
        "base_vy_m_s",
        "base_wz_rad_s",
        "left_foot_contact",
        "right_foot_contact",
        "left_foot_slip_m_s",
        "right_foot_slip_m_s",
        "fallen",
        "fall_reason",
    )

    def __init__(
        self,
        raw_env,
        action_term,
        csv_path: Path,
        *,
        contact_force_threshold_n: float,
        fall_height_m: float,
        fall_tilt_rad: float,
    ):
        # 当前实验强制 num_envs=1，因此下面读取数据时环境索引固定使用 0。
        self.raw_env = raw_env
        self.robot = raw_env.scene["robot"]
        self.action_term = action_term
        self.contact_force_threshold_n = float(contact_force_threshold_n)
        self.fall_height_m = float(fall_height_m)
        self.fall_tilt_rad = float(fall_tilt_rad)
        self.csv_path = Path(csv_path)
        self.csv_path.parent.mkdir(parents=True, exist_ok=True)
        # 每次实验目录都是新的，这里直接创建一份新的 CSV 并先写表头。
        self._file = self.csv_path.open("w", newline="", encoding="utf-8")
        self._writer = csv.DictWriter(self._file, fieldnames=self.FIELDNAMES)
        self._writer.writeheader()
        self.rows = 0
        self.fallen = False
        self.fall_reason = ""
        self.max_abs_roll_rad = 0.0
        self.max_abs_pitch_rad = 0.0
        self.min_base_height_m = float("inf")
        self.max_abs_tracking_error_rad = 0.0
        self.max_foot_slip_m_s = 0.0

        # 足部指标属于辅助诊断：找不到接触传感器时不终止主实验，而是将对应
        # 数据记为不可用。ankle_roll body 在当前 G1 资产中代表左右脚末端。
        self.contact_sensor = None
        self.contact_foot_ids: list[int] = []
        self.robot_foot_ids: list[int] = []
        self.foot_names: list[str] = []
        try:
            self.contact_sensor = raw_env.scene["contact_forces"]
            contact_ids, contact_names = self.contact_sensor.find_bodies(
                ".*ankle_roll.*", preserve_order=True
            )
            robot_ids, robot_names = self.robot.find_bodies(
                ".*ankle_roll.*", preserve_order=True
            )
            if len(contact_ids) == len(robot_ids) == 2:
                self.contact_foot_ids = list(contact_ids)
                self.robot_foot_ids = list(robot_ids)
                self.foot_names = list(robot_names or contact_names)
        except Exception as exc:
            print(f"[WAIST-YAW WARN] Foot contact/slip metrics unavailable: {exc}")

    def close(self) -> None:
        """刷新缓冲并关闭 CSV；允许重复调用。"""

        if not self._file.closed:
            self._file.flush()
            self._file.close()

    def summary(self) -> dict:
        """返回整次实验累计的极值和最终控制状态。"""

        return {
            "rows": self.rows,
            "fallen": self.fallen,
            "fall_reason": self.fall_reason,
            "max_abs_roll_rad": self.max_abs_roll_rad,
            "max_abs_pitch_rad": self.max_abs_pitch_rad,
            "min_base_height_m": (
                self.min_base_height_m if self.rows else None
            ),
            "max_abs_tracking_error_rad": self.max_abs_tracking_error_rad,
            "max_foot_slip_m_s": self.max_foot_slip_m_s,
            "final_control_state": self.action_term.control_state,
            "final_arm_sdk_takeover_weight": self.action_term.weight,
        }

    def _foot_metrics(self) -> dict[str, float | bool]:
        """读取左右脚接触状态，以及接触时脚底的水平滑动速度。"""

        result: dict[str, float | bool] = {
            "left_foot_contact": False,
            "right_foot_contact": False,
            "left_foot_slip_m_s": float("nan"),
            "right_foot_slip_m_s": float("nan"),
        }
        if self.contact_sensor is None or len(self.contact_foot_ids) != 2:
            return result
        # net_forces_w 和 body_lin_vel_w 都在世界坐标系中。接触力取三维模长；
        # 滑动只关心水平面 x/y 速度，不把脚的竖直运动算作打滑。
        forces = self.contact_sensor.data.net_forces_w[0, self.contact_foot_ids]
        velocities = self.robot.data.body_lin_vel_w[0, self.robot_foot_ids]
        for index, name in enumerate(self.foot_names):
            side = "left" if "left" in name.lower() else "right"
            contact = float(torch.linalg.norm(forces[index]).item()) >= self.contact_force_threshold_n
            slip = float(torch.linalg.norm(velocities[index, :2]).item()) if contact else 0.0
            result[f"{side}_foot_contact"] = contact
            result[f"{side}_foot_slip_m_s"] = slip
        return result

    def record(self, *, step: int, time_s: float, stage_index: int) -> dict:
        """采集一个控制周期的数据、更新累计极值，并写入一行 CSV。"""

        data = self.robot.data
        yaw_id = self.action_term.yaw_joint_id
        actual = float(data.joint_pos[0, yaw_id].item())
        velocity = float(data.joint_vel[0, yaw_id].item())
        baseline = float(self.action_term.baseline_yaw_target[0].item())
        requested = float(self.action_term.requested_user_yaw_target[0].item())
        smoothed = float(self.action_term.smoothed_user_yaw_target[0].item())
        final = float(self.action_term.final_yaw_target[0].item())
        roll, pitch = _roll_pitch_from_wxyz(data.root_quat_w[0])
        height = float(data.root_pos_w[0, 2].item())

        # 摔倒判断采用“高度过低或机身倾角过大”。第一次满足时保存原因，后续
        # 即使机器人弹起，整次实验的 fallen 汇总仍保持 True。
        reasons: list[str] = []
        if height < self.fall_height_m:
            reasons.append(f"height<{self.fall_height_m:.3f}")
        if abs(roll) > self.fall_tilt_rad:
            reasons.append(f"abs(roll)>{self.fall_tilt_rad:.3f}")
        if abs(pitch) > self.fall_tilt_rad:
            reasons.append(f"abs(pitch)>{self.fall_tilt_rad:.3f}")
        fallen = bool(reasons)
        if fallen and not self.fallen:
            self.fallen = True
            self.fall_reason = ";".join(reasons)

        foot_metrics = self._foot_metrics()
        finite_slips = [
            float(foot_metrics[name])
            for name in ("left_foot_slip_m_s", "right_foot_slip_m_s")
            if math.isfinite(float(foot_metrics[name]))
        ]
        self.max_abs_roll_rad = max(self.max_abs_roll_rad, abs(roll))
        self.max_abs_pitch_rad = max(self.max_abs_pitch_rad, abs(pitch))
        self.min_base_height_m = min(self.min_base_height_m, height)
        # 跟踪误差比较“最终下发 q_des”和“实际 WaistYaw”，而不是用户原始目标。
        self.max_abs_tracking_error_rad = max(
            self.max_abs_tracking_error_rad, abs(final - actual)
        )
        if finite_slips:
            self.max_foot_slip_m_s = max(
                self.max_foot_slip_m_s, max(finite_slips)
            )

        # 同时保存用户请求、限速参考和最终 q_des，便于判断误差来自限速、权重
        # 混合还是执行器实际跟踪。
        row = {
            "time_s": float(time_s),
            "step": int(step),
            "stage_index": int(stage_index),
            "mode": self.action_term.mode,
            "control_state": self.action_term.control_state,
            "arm_sdk_takeover_weight": self.action_term.weight,
            "user_blend_weight": self.action_term.blend_weight,
            "waist_actual_rad": actual,
            "waist_velocity_rad_s": velocity,
            "waist_policy_baseline_q_des_rad": baseline,
            "waist_user_requested_rad": requested,
            "waist_arm_sdk_reference_rad": smoothed,
            "waist_final_q_des_rad": final,
            "waist_tracking_error_rad": final - actual,
            "base_roll_rad": roll,
            "base_pitch_rad": pitch,
            "base_height_m": height,
            "base_vx_m_s": float(data.root_lin_vel_w[0, 0].item()),
            "base_vy_m_s": float(data.root_lin_vel_w[0, 1].item()),
            "base_wz_rad_s": float(data.root_ang_vel_w[0, 2].item()),
            **foot_metrics,
            "fallen": fallen,
            "fall_reason": ";".join(reasons),
        }
        self._writer.writerow(row)
        self.rows += 1
        # 每100行主动刷新一次，程序意外退出时也能尽量保留最近的数据。
        if self.rows % 100 == 0:
            self._file.flush()
        return row
