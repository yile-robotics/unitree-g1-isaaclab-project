from __future__ import annotations

"""G1 policy、路径执行和 LaViRA 导航的主循环。

本模块是真正“跑起来”的地方：
- run_stand：加载 RSL-RL checkpoint，只测试原生 stand env。
- run_locomotion：加载 locomotion ONNX，只测试速度命令跟踪。
- run_switch：同时加载 stand/locomotion ONNX，支持键盘切换、path follower、
  四视图抓取、一次性决策验证和有限多轮 LaViRA episode。

为了和 IsaacLab 训练环境对齐，仿真步进仍然走 env.step(actions)，动作仍然交给
IsaacLab action manager/actuator stack 处理。这里主要负责 policy inference、
键盘事件、命令写入、LaViRA episode 调度和状态打印。
"""

import select
import sys
import termios
import time
import tty

import numpy as np
import torch
from isaaclab.devices import Se2Keyboard, Se2KeyboardCfg
from rsl_rl.runners import OnPolicyRunner

from .config import STAND_ACTION_JOINT_NAMES, STAND_OBS_JOINT_NAMES
from .control import (
    PolicySwitchState,
    StandObservationHistory,
    SwitchCommandController,
    clamp_command,
    get_policy_obs,
    print_native_stack_diagnostics,
    resolve_joint_ids,
    set_velocity_command,
    tensor_first,
)
from .frame_bundle import FourViewCameraRig
from .lavira_episode import LaViRABoundedEpisodeController
from .lavira_offline import NavigationDecisionOfflineProbe
from .path import (
    PathVisualizer,
    WaypointPathFollower,
    parse_path_waypoints,
    prepare_fmm_path_for_execution,
)


def make_four_view_camera_rig(raw_env, args_cli) -> FourViewCameraRig | None:
    """按命令行开关构造 env_0 的四相机读取器；不开启时保持零开销。"""
    if not getattr(args_cli, "four_rgbd_cameras", False):
        return None
    return FourViewCameraRig(raw_env, args_cli, env_index=0)


def update_camera_debug_after_step(
    camera_rig: FourViewCameraRig | None,
    completed_step: int,
    step_dt: float,
) -> None:
    """在 env.step 完成后触发可选的一次性四视图调试快照。"""
    if camera_rig is not None:
        camera_rig.maybe_save_debug_snapshot(completed_step, step_dt)


def report_camera_debug_status(camera_rig: FourViewCameraRig | None) -> None:
    if camera_rig is not None:
        camera_rig.report_debug_status()


def update_lavira_decision_probe_after_step(
    probe: NavigationDecisionOfflineProbe,
    camera_rig: FourViewCameraRig | None,
    completed_step: int,
    step_dt: float,
) -> None:
    probe.maybe_run(
        camera_rig,
        completed_step=completed_step,
        step_dt=step_dt,
    )


def start_lavira_fmm_path_execution(
    probe: NavigationDecisionOfflineProbe,
    path_follower: WaypointPathFollower,
    path_visualizer: PathVisualizer,
    command_controller: SwitchCommandController,
    switch_state: PolicySwitchState,
    args_cli,
) -> bool:
    """把本轮内存 FMMPlan 安全交给原有 follower/locomotion 控制链。"""
    if probe.fmm_plan is None:
        print(
            "[WARN] FMM execution was requested but this decision produced no valid "
            "FMM plan. Robot remains stopped."
        )
        command_controller.zero()
        return False

    current_pose = path_follower.current_robot_pose()
    execution_max_path_m = float(
        getattr(
            probe.fmm_plan,
            "execution_max_path_m",
            args_cli.fmm_execute_max_path_m,
        )
    )
    prepared = prepare_fmm_path_for_execution(
        probe.fmm_plan,
        current_pose,
        max_start_drift_m=float(args_cli.fmm_execute_start_tolerance_m),
        max_path_length_m=execution_max_path_m,
    )
    final = prepared.waypoints[-1]
    final_distance = float(
        np.hypot(final.x - current_pose.x, final.y - current_pose.y)
    )
    if final_distance <= float(args_cli.goal_tolerance):
        print(
            "[LAVIRA] FMM safe target is already within goal tolerance: "
            f"distance={final_distance:.3f}m. Robot remains in stand."
        )
        command_controller.zero()
        return False

    execution_source = str(
        getattr(
            probe.fmm_plan,
            "execution_source",
            f"lavira_fmm_bundle_{probe.fmm_plan.bundle_id}",
        )
    )
    path_follower.replace_waypoints(
        prepared.waypoints,
        source=execution_source,
        cross_track_abort_m=float(args_cli.fmm_execute_cross_track_abort_m),
        tilt_abort_rad=float(args_cli.fmm_execute_tilt_abort_rad),
        velocity_limits=(
            float(args_cli.fmm_execute_max_vx),
            float(args_cli.fmm_execute_max_vy),
            float(args_cli.fmm_execute_max_wz),
        ),
        lookahead_distance_m=float(args_cli.fmm_execute_lookahead_m),
    )
    path_visualizer.draw_static_path(path_follower.waypoints)
    command_controller.zero()
    path_follower.start()
    switch_state.request_locomotion()
    print(
        "[LAVIRA] Accepted navigation path for locomotion: "
        f"source={execution_source} "
        f"bundle={probe.fmm_plan.bundle_id} "
        f"length={prepared.path_length_m:.3f}m "
        f"waypoints={len(prepared.waypoints)} "
        f"start_drift={prepared.start_drift_m:.3f}m "
        f"speed_caps=({args_cli.fmm_execute_max_vx:.2f},"
        f"{args_cli.fmm_execute_max_vy:.2f},"
        f"{args_cli.fmm_execute_max_wz:.2f})."
    )
    return True


def hot_swap_lavira_fmm_path_execution(
    fmm_plan,
    execution_max_path_m: float,
    path_follower: WaypointPathFollower,
    path_visualizer: PathVisualizer,
    args_cli,
) -> bool:
    """Replace an active FMM path without zeroing commands or switching policy."""

    current_pose = path_follower.current_robot_pose()
    prepared = prepare_fmm_path_for_execution(
        fmm_plan,
        current_pose,
        max_start_drift_m=float(args_cli.fmm_execute_start_tolerance_m),
        max_path_length_m=float(execution_max_path_m),
    )
    final = prepared.waypoints[-1]
    final_distance = float(
        np.hypot(final.x - current_pose.x, final.y - current_pose.y)
    )
    if final_distance <= float(args_cli.goal_tolerance):
        # The episode controller owns NAVIGATE/BACKTRACK/STOP completion.  Keep
        # the existing short remainder so its normal threshold logic can finish.
        return True

    source = (
        f"lavira_online_replan_bundle_{int(fmm_plan.bundle_id):06d}"
    )
    path_follower.replace_waypoints_while_active(
        prepared.waypoints,
        source=source,
        cross_track_abort_m=float(args_cli.fmm_execute_cross_track_abort_m),
        tilt_abort_rad=float(args_cli.fmm_execute_tilt_abort_rad),
        velocity_limits=(
            float(args_cli.fmm_execute_max_vx),
            float(args_cli.fmm_execute_max_vy),
            float(args_cli.fmm_execute_max_wz),
        ),
        lookahead_distance_m=float(args_cli.fmm_execute_lookahead_m),
    )
    path_visualizer.draw_static_path(path_follower.waypoints)
    print(
        "[LAVIRA ONLINE] Hot-swapped FMM path: "
        f"bundle={fmm_plan.bundle_id} "
        f"length={prepared.path_length_m:.3f}m "
        f"waypoints={len(prepared.waypoints)} "
        f"start_drift={prepared.start_drift_m:.3f}m."
    )
    return True


def get_keyboard_command(keyboard, raw_env, num_envs: int) -> torch.Tensor:
    """读取 IsaacLab Se2Keyboard 的连续命令，并扩展到所有 env。"""
    command = keyboard.advance().to(device=raw_env.device, dtype=torch.float32)
    return command.reshape(1, 3).repeat(num_envs, 1)


class TerminalVelocityKeyboard:
    """终端非阻塞键盘输入。

    Isaac Sim 窗口没有焦点时，Omniverse keyboard event 收不到按键；
    这个类从终端 stdin 读取单字符，作为备用控制入口。
    """

    def __init__(self, enabled: bool, device: str, num_envs: int):
        self._enabled = enabled and sys.stdin.isatty()
        self._device = device
        self._num_envs = num_envs
        self._old_settings = None
        self._command = torch.zeros((num_envs, 3), device=device, dtype=torch.float32)
        self._active = False
        self._last_key = None

    def __enter__(self):
        """进入 raw/cbreak 终端模式，让按键不需要回车即可读取。"""
        if self._enabled:
            self._old_settings = termios.tcgetattr(sys.stdin)
            tty.setcbreak(sys.stdin.fileno())
            print("[INFO] Terminal keyboard enabled: G=path, W/S=vx step, A/D=vy step, Q/E=wz step, X/L=zero.")
        return self

    def __exit__(self, exc_type, exc, tb):
        """退出时恢复终端设置，否则 shell 会留在奇怪的输入模式。"""
        if self._enabled and self._old_settings is not None:
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self._old_settings)

    def advance(self) -> tuple[torch.Tensor, bool]:
        """轮询一次键盘；返回值保留给早期连续命令接口，目前主要通过 consume_key 使用。"""
        key = self._poll()
        self._last_key = key.lower() if key is not None else None
        if key is None:
            return self._command, self._active

        key = key.lower()
        if key in ("1", "2", "3", "g", "w", "s", "a", "d", "q", "e", "x", "l", " "):
            return self._command, self._active
        else:
            return self._command, self._active

    def consume_key(self) -> str | None:
        """取出最近一次按键，并清空缓存，避免一个键被重复消费。"""
        key = self._last_key
        self._last_key = None
        return key

    def _poll(self) -> str | None:
        """非阻塞读取 stdin 中最后一个字符。"""
        if not self._enabled:
            return None
        key = None
        while select.select([sys.stdin], [], [], 0.0)[0]:
            key = sys.stdin.read(1)
        return key


class OmniverseKeyEvents:
    """Isaac Sim 窗口内的离散键盘事件。

    switch 模式需要“按一下增加 0.05m/s”这种离散事件，所以不用连续 Se2Keyboard，
    而是订阅 carb.input 的 KEY_PRESS/KEY_RELEASE。
    """

    def __init__(self, enabled: bool):
        self._enabled = enabled
        self._input = None
        self._keyboard = None
        self._subscription = None
        self._carb_input = None
        self._events: list[str] = []
        self._pressed: set[str] = set()

    def __enter__(self):
        if not self._enabled:
            return self
        try:
            import carb.input
            import omni.appwindow

            self._carb_input = carb.input
            appwindow = omni.appwindow.get_default_app_window()
            if appwindow is None:
                return self
            self._input = carb.input.acquire_input_interface()
            self._keyboard = appwindow.get_keyboard()
            self._subscription = self._input.subscribe_to_keyboard_events(self._keyboard, self._on_keyboard_event)
            print("[INFO] Isaac Sim discrete keyboard enabled: G=path, W/S/A/D/Q/E increments, X/L zero.")
        except Exception as exc:
            print(f"[WARN] Could not subscribe to Isaac Sim keyboard events: {exc}")
        return self

    def __exit__(self, exc_type, exc, tb):
        if self._input is not None and self._keyboard is not None and self._subscription is not None:
            try:
                self._input.unsubscribe_to_keyboard_events(self._keyboard, self._subscription)
            except Exception as exc:
                print(f"[WARN] Could not unsubscribe Isaac Sim keyboard events: {exc}")

    def poll(self) -> str | None:
        """返回最近一次按键事件；如果一帧内多个事件，只取最后一个。"""
        if not self._events:
            return None
        key = self._events[-1]
        self._events.clear()
        return key

    def _on_keyboard_event(self, event, *args, **kwargs) -> bool:
        """carb.input 回调：只在按下瞬间记录事件，长按不会无限重复触发。"""
        key = self._normalize_key(event.input.name)
        if key is None:
            return True
        if event.type == self._carb_input.KeyboardEventType.KEY_PRESS:
            if key not in self._pressed:
                self._events.append(key)
            self._pressed.add(key)
        elif event.type == self._carb_input.KeyboardEventType.KEY_RELEASE:
            self._pressed.discard(key)
        return True

    def _normalize_key(self, name: str) -> str | None:
        """把 Isaac Sim 不同键名格式归一成脚本内部使用的小写字符。"""
        mapping = {
            "1": "1",
            "KEY_1": "1",
            "NUMPAD_1": "1",
            "2": "2",
            "KEY_2": "2",
            "NUMPAD_2": "2",
            "3": "3",
            "KEY_3": "3",
            "NUMPAD_3": "3",
            "W": "w",
            "S": "s",
            "A": "a",
            "D": "d",
            "Q": "q",
            "E": "e",
            "G": "g",
            "X": "x",
            "L": "l",
            "SPACE": " ",
        }
        return mapping.get(name)



def add_wasd_keyboard_bindings(keyboard) -> None:
    """给 IsaacLab Se2Keyboard 补 W/S/A/D/Q/E 映射。

    IsaacLab 默认可能用方向键或其它键位；这里强制加上和当前脚本一致的 WASDQE。
    """
    keyboard._INPUT_KEY_MAPPING.update(
        {
            "W": np.asarray([1.0, 0.0, 0.0]) * keyboard.v_x_sensitivity,
            "S": np.asarray([-1.0, 0.0, 0.0]) * keyboard.v_x_sensitivity,
            "A": np.asarray([0.0, 1.0, 0.0]) * keyboard.v_y_sensitivity,
            "D": np.asarray([0.0, -1.0, 0.0]) * keyboard.v_y_sensitivity,
            "Q": np.asarray([0.0, 0.0, 1.0]) * keyboard.omega_z_sensitivity,
            "E": np.asarray([0.0, 0.0, -1.0]) * keyboard.omega_z_sensitivity,
        }
    )


def run_stand(env, agent_cfg, args_cli, simulation_app) -> None:
    """只运行 stand checkpoint 的原生 IsaacLab probe。

    这个模式主要用来确认：stand 任务本身、RSL-RL checkpoint、机器人 USD/actuator
    配置是否能在 IsaacLab 原生栈里稳定站立。
    """
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(str(args_cli.checkpoint))
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    raw_env = env.unwrapped
    step_dt = float(raw_env.step_dt)
    camera_rig = make_four_view_camera_rig(raw_env, args_cli)
    obs = env.get_observations()
    if isinstance(obs, tuple):
        obs = obs[0]

    step = 0
    print(f"[INFO] Running native IsaacLab stand probe, step_dt={step_dt:.4f}s.")
    try:
        while simulation_app.is_running() and (args_cli.max_steps < 0 or step < args_cli.max_steps):
            start_time = time.time()
            with torch.inference_mode():
                actions = policy(obs)
                obs, _, _, _ = env.step(actions)

            update_camera_debug_after_step(camera_rig, step + 1, step_dt)
            print_state(env, step, args_cli=args_cli)
            step += 1
            sleep_to_real_time(step_dt, start_time, args_cli)
    finally:
        report_camera_debug_status(camera_rig)


def run_locomotion(env, env_cfg, args_cli, simulation_app) -> None:
    """只运行 locomotion ONNX，测试速度命令跟踪。

    这个模式不加载 stand policy，也不做 policy 切换。它适合单独排查 locomotion
    policy、ONNX 输入维度、命令范围和键盘速度控制。
    """
    import onnxruntime as ort

    providers = ["CPUExecutionProvider"]
    session = ort.InferenceSession(str(args_cli.locomotion_onnx), providers=providers)
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    input_shape = session.get_inputs()[0].shape
    output_shape = session.get_outputs()[0].shape

    raw_env = env.unwrapped
    step_dt = float(raw_env.step_dt)
    camera_rig = make_four_view_camera_rig(raw_env, args_cli)
    command = torch.tensor([[args_cli.vx, args_cli.vy, args_cli.wz]], device=raw_env.device, dtype=torch.float32)
    command = command.repeat(raw_env.num_envs, 1)
    command = clamp_command(command, env_cfg)
    keyboard = None
    if args_cli.keyboard and not args_cli.headless:
        keyboard_cfg = Se2KeyboardCfg(
            v_x_sensitivity=args_cli.keyboard_vx,
            v_y_sensitivity=args_cli.keyboard_vy,
            omega_z_sensitivity=args_cli.keyboard_wz,
            sim_device=raw_env.device,
        )
        keyboard = Se2Keyboard(keyboard_cfg)
        add_wasd_keyboard_bindings(keyboard)
        print(keyboard)
        print("[INFO] Extra keyboard bindings: W/S=vx, A/D=vy, Q/E=wz, L=zero.")
    elif args_cli.keyboard and args_cli.headless:
        print("[INFO] Headless mode: keyboard disabled, using --vx/--vy/--wz.")

    set_velocity_command(raw_env, command)
    obs = env.get_observations()
    obs_tensor = get_policy_obs(obs)
    expected_obs_dim = int(input_shape[-1])
    if obs_tensor.shape[-1] != expected_obs_dim:
        raise RuntimeError(f"ONNX expects obs dim {expected_obs_dim}, but env produced {obs_tensor.shape[-1]}.")
    print(
        "[INFO] Running native IsaacLab locomotion probe, "
        f"step_dt={step_dt:.4f}s, onnx_input={input_shape}, onnx_output={output_shape}."
    )

    step = 0
    terminal_keyboard = TerminalVelocityKeyboard(args_cli.keyboard, raw_env.device, raw_env.num_envs)
    try:
        with terminal_keyboard:
            while simulation_app.is_running() and (args_cli.max_steps < 0 or step < args_cli.max_steps):
                start_time = time.time()
                if keyboard is not None:
                    command = get_keyboard_command(keyboard, raw_env, raw_env.num_envs)
                terminal_keyboard.advance()
                key = terminal_keyboard.consume_key()
                if key == "w":
                    command[:, 0] = args_cli.keyboard_vx
                elif key == "s":
                    command[:, 0] = -args_cli.keyboard_vx
                elif key == "a":
                    command[:, 1] = args_cli.keyboard_vy
                elif key == "d":
                    command[:, 1] = -args_cli.keyboard_vy
                elif key == "q":
                    command[:, 2] = args_cli.keyboard_wz
                elif key == "e":
                    command[:, 2] = -args_cli.keyboard_wz
                elif key in ("x", "l", " "):
                    command.zero_()
                command = clamp_command(command, env_cfg)
                set_velocity_command(raw_env, command)

                with torch.inference_mode():
                    obs_np = get_policy_obs(obs).detach().cpu().numpy().astype(np.float32)
                    actions_np = session.run([output_name], {input_name: obs_np})[0].astype(np.float32)
                    actions = torch.from_numpy(actions_np).to(device=raw_env.device)
                    obs, _, _, _ = env.step(actions)

                update_camera_debug_after_step(camera_rig, step + 1, step_dt)
                print_state(env, step, command, args_cli=args_cli)
                step += 1
                sleep_to_real_time(step_dt, start_time, args_cli)
    finally:
        report_camera_debug_status(camera_rig)


def run_switch(env, env_cfg, args_cli, simulation_app) -> None:
    """运行 stand/locomotion 双 policy 切换和 path follower。

    这是当前最完整的部署测试模式：
    - locomotion ONNX 负责速度命令跟踪。
    - stand ONNX 负责低速/停止后的下半身稳定站立。
    - PolicySwitchState 负责平滑动作切换。
    - WaypointPathFollower 负责把路径转换成速度命令。
    """
    import onnxruntime as ort

    if env.unwrapped.num_envs != 1:
        # 当前状态机、键盘事件和 path follower 都按单机器人写的；多环境会让打印和控制含义混乱。
        raise RuntimeError("--mode switch currently expects --num_envs 1.")

    # 两个 ONNX 分开加载。locomotion 输入来自 env observation；
    # stand 输入由 StandObservationHistory 手工拼接。
    providers = ["CPUExecutionProvider"]
    locomotion_session = ort.InferenceSession(str(args_cli.locomotion_onnx), providers=providers)
    stand_session = ort.InferenceSession(str(args_cli.stand_onnx), providers=providers)
    locomotion_input = locomotion_session.get_inputs()[0].name
    locomotion_output = locomotion_session.get_outputs()[0].name
    stand_input = stand_session.get_inputs()[0].name
    stand_output = stand_session.get_outputs()[0].name
    locomotion_input_dim = int(locomotion_session.get_inputs()[0].shape[-1])
    stand_input_dim = int(stand_session.get_inputs()[0].shape[-1])

    raw_env = env.unwrapped
    step_dt = float(raw_env.step_dt)
    camera_rig = make_four_view_camera_rig(raw_env, args_cli)
    lavira_decision_probe = NavigationDecisionOfflineProbe(args_cli)
    lavira_history_probe = LaViRABoundedEpisodeController(args_cli)
    robot = raw_env.scene["robot"]

    # 通过关节名解析 id，而不是写死整数，避免 USD/IsaacLab joint order 微调后悄悄错位。
    stand_action_joint_ids = resolve_joint_ids(robot, STAND_ACTION_JOINT_NAMES)
    stand_obs_joint_ids = resolve_joint_ids(robot, STAND_OBS_JOINT_NAMES)
    switch_state = PolicySwitchState(raw_env, stand_action_joint_ids, args_cli)
    stand_history = StandObservationHistory(raw_env, stand_obs_joint_ids)
    last_stand_action = torch.zeros(
        (raw_env.num_envs, len(stand_action_joint_ids)), device=raw_env.device, dtype=torch.float32
    )

    command_controller = SwitchCommandController(raw_env, env_cfg, args_cli)
    command_controller.set_initial(args_cli.vx, args_cli.vy, args_cli.wz)
    if torch.linalg.norm(command_controller.requested).item() > 1.0e-5:
        switch_state.request_locomotion()
    initial_waypoints = (
        []
        if bool(getattr(args_cli, "lavira_execute_fmm_path", False))
        else parse_path_waypoints(args_cli.path_waypoints)
    )
    path_follower = WaypointPathFollower(raw_env, env_cfg, initial_waypoints, args_cli)
    path_visualizer = PathVisualizer(args_cli.show_path)
    path_visualizer.draw_static_path(path_follower.waypoints)
    if args_cli.start_path_on_enter:
        path_follower.start()
        switch_state.request_locomotion()

    if args_cli.keyboard and args_cli.headless:
        print("[INFO] Headless mode: keyboard disabled, using --vx/--vy/--wz.")

    set_velocity_command(raw_env, torch.zeros_like(command_controller.filtered))
    obs = env.get_observations()
    obs_tensor = get_policy_obs(obs)
    # 启动时先检查 ONNX 输入维度，防止跑到一半才发现 policy 和 env 不匹配。
    if obs_tensor.shape[-1] != locomotion_input_dim:
        raise RuntimeError(f"Locomotion ONNX expects obs dim {locomotion_input_dim}, got {obs_tensor.shape[-1]}.")
    stand_obs = stand_history.reset(last_stand_action)
    if stand_obs.shape[-1] != stand_input_dim:
        raise RuntimeError(f"Stand ONNX expects obs dim {stand_input_dim}, got {stand_obs.shape[-1]}.")

    print(
        "[INFO] Running native IsaacLab switch probe, "
        f"step_dt={step_dt:.4f}s, locomotion_obs={locomotion_input_dim}, stand_obs={stand_input_dim}."
    )
    print(
        "[INFO] Switch keys: 1=stand, 2=locomotion, 3=force stand, "
        "G=start path, W/S/A/D/Q/E=manual increment, X/L/Space=zero+stand."
    )

    step = 0
    fmm_execution_handled = False
    terminal_keyboard = TerminalVelocityKeyboard(args_cli.keyboard, raw_env.device, raw_env.num_envs)
    omni_keys = OmniverseKeyEvents(args_cli.keyboard and not args_cli.headless)
    try:
        with terminal_keyboard, omni_keys:
            while simulation_app.is_running() and (args_cli.max_steps < 0 or step < args_cli.max_steps):
                start_time = time.time()

                # 1. 读取键盘事件，并把它翻译成“切换模式/开始路径/手动改速度”等高层命令。
                terminal_keyboard.advance()
                key = omni_keys.poll() or terminal_keyboard.consume_key()
                if key == "1":
                    path_follower.stop("stand requested")
                    command_controller.zero()
                    switch_state.request_stand()
                elif key == "2":
                    switch_state.request_locomotion()
                elif key == "3":
                    path_follower.stop("force stand requested")
                    command_controller.zero()
                    command_controller.filtered.zero_()
                    switch_state.force_stand()
                elif key == "g":
                    path_follower.start()
                    switch_state.request_locomotion()
                elif key in ("x", "l", " "):
                    path_follower.stop("zero command")
                    command_controller.zero()
                    switch_state.request_stand()
                elif key in ("w", "s", "a", "d", "q", "e"):
                    path_follower.stop("manual command")
                    command_controller.apply_key(key)
                    switch_state.request_locomotion()

                # 2. path follower 根据当前机器人位姿更新 requested velocity。
                path_follower.update(command_controller, switch_state)
                path_visualizer.draw_tracking(path_follower.last_pose, path_follower.last_target, path_follower.enabled)

                # 3. requested velocity 经过 ramp/filter 后写入 IsaacLab command manager。
                command_for_env = command_controller.update_filtered(step_dt, switch_state.should_zero_command())
                switch_state.update_waiting_for_stand(command_for_env, step_dt)
                set_velocity_command(raw_env, command_for_env)

                with torch.inference_mode():
                    # 4. locomotion policy 用原生 env observation 推理 29 维动作。
                    locomotion_obs_np = get_policy_obs(obs).detach().cpu().numpy().astype(np.float32)
                    locomotion_action_np = locomotion_session.run(
                        [locomotion_output], {locomotion_input: locomotion_obs_np}
                    )[0].astype(np.float32)
                    locomotion_action = torch.from_numpy(locomotion_action_np).to(device=raw_env.device)

                    # 5. stand policy 用手工历史观测推理 12 维腿部动作。
                    stand_obs = stand_history.append(last_stand_action)
                    stand_obs_np = stand_obs.detach().cpu().numpy().astype(np.float32)
                    stand_action_np = stand_session.run([stand_output], {stand_input: stand_obs_np})[0].astype(np.float32)
                    stand_action = torch.from_numpy(stand_action_np).to(device=raw_env.device)
                    last_stand_action = stand_action

                    # 6. 状态机选择 stand/locomotion 或二者混合后的最终动作，再 step 环境。
                    actions = switch_state.action(stand_action, locomotion_action, step_dt)
                    obs, _, _, _ = env.step(actions)

                update_camera_debug_after_step(camera_rig, step + 1, step_dt)
                update_lavira_decision_probe_after_step(
                    lavira_decision_probe,
                    camera_rig,
                    step + 1,
                    step_dt,
                )
                if (
                    bool(getattr(args_cli, "lavira_execute_fmm_path", False))
                    and not bool(getattr(args_cli, "lavira_history_probe", False))
                    and lavira_decision_probe.completed
                    and not fmm_execution_handled
                ):
                    # One-shot only: whether accepted or rejected, never retry a
                    # stale plan automatically on later simulation steps.
                    fmm_execution_handled = True
                    try:
                        start_lavira_fmm_path_execution(
                            lavira_decision_probe,
                            path_follower,
                            path_visualizer,
                            command_controller,
                            switch_state,
                            args_cli,
                        )
                    except Exception as exc:
                        path_follower.stop("FMM execution validation failed")
                        command_controller.zero()
                        switch_state.request_stand()
                        print(
                            "[WARN] Rejected FMM path execution: "
                            f"{exc}. Robot remains stopped."
                        )
                lavira_history_probe.update_after_step(
                    camera_rig,
                    completed_step=step + 1,
                    step_dt=step_dt,
                    path_follower=path_follower,
                    path_visualizer=path_visualizer,
                    command_controller=command_controller,
                    switch_state=switch_state,
                    start_path=lambda probe: start_lavira_fmm_path_execution(
                        probe,
                        path_follower,
                        path_visualizer,
                        command_controller,
                        switch_state,
                        args_cli,
                    ),
                    hot_swap_path=lambda plan, max_path_m: (
                        hot_swap_lavira_fmm_path_execution(
                            plan,
                            max_path_m,
                            path_follower,
                            path_visualizer,
                            args_cli,
                        )
                    ),
                    applied_velocity_command=(
                        command_for_env[0].detach().cpu().numpy()
                    ),
                )
                print_state(
                    env,
                    step,
                    command_for_env,
                    switch_state.active_mode,
                    switch_state.transition_mode,
                    command_controller.requested,
                    path_follower,
                    args_cli=args_cli,
                )
                step += 1
                sleep_to_real_time(step_dt, start_time, args_cli)
    finally:
        report_camera_debug_status(camera_rig)
        lavira_decision_probe.report_status()
        lavira_history_probe.report_status()


def print_state(
    env,
    step: int,
    command: torch.Tensor | None = None,
    mode: str | None = None,
    transition: str | None = None,
    requested_command: torch.Tensor | None = None,
    path_follower: WaypointPathFollower | None = None,
    args_cli=None,
) -> None:
    """按固定频率打印机器人状态、命令、模式和路径跟踪信息。"""
    if step % args_cli.print_every != 0:
        return
    robot = env.unwrapped.scene["robot"]
    root_pos = tensor_first(robot.data.root_pos_w)
    root_lin_vel = tensor_first(robot.data.root_lin_vel_w)
    root_ang_vel = tensor_first(robot.data.root_ang_vel_w)
    prefix = (
        f"cmd=({float(command[0, 0]):+.2f},{float(command[0, 1]):+.2f},{float(command[0, 2]):+.2f}) "
        if command is not None
        else ""
    )
    requested_prefix = (
        f"req=({float(requested_command[0, 0]):+.2f},{float(requested_command[0, 1]):+.2f},{float(requested_command[0, 2]):+.2f}) "
        if requested_command is not None
        else ""
    )
    mode_prefix = f"mode={mode}/{transition} " if mode is not None and transition is not None else ""
    path_prefix = ""
    if path_follower is not None and path_follower.enabled and path_follower.last_target.valid:
        # path=on 时额外打印当前路径段、lookahead target 和 cross-track error。
        target = path_follower.last_target
        path_prefix = (
            f"path=on seg={target.segment_index} "
            f"tgt=({target.x:.2f},{target.y:.2f},{target.yaw:.2f}) "
            f"cte={target.cross_track_error:.2f} "
        )
    print(
        "[STATE] "
        f"step={step:05d} "
        f"{mode_prefix}"
        f"{path_prefix}"
        f"{requested_prefix}"
        f"{prefix}"
        f"z={float(root_pos[2]):.3f} "
        f"vx={float(root_lin_vel[0]):+.3f} "
        f"vy={float(root_lin_vel[1]):+.3f} "
        f"wz={float(root_ang_vel[2]):+.3f}"
    )


def sleep_to_real_time(step_dt: float, start_time: float, args_cli) -> None:
    """可选实时限速：调试时可以让仿真大致按真实时间播放。"""
    sleep_time = step_dt - (time.time() - start_time)
    if args_cli.real_time and sleep_time > 0:
        time.sleep(sleep_time)
