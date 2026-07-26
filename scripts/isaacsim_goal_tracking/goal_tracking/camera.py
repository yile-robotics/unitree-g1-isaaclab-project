from __future__ import annotations

"""G1 四方向 RGB-D 相机的配置、固定挂载和 GUI 调试入口。

正式的 VLN 感知入口是 :func:`configure_four_rgbd_cameras`。它在 IsaacLab
环境创建前向 ``env_cfg.scene`` 注册四个可读取数组的 ``CameraCfg``，分别对应
机器人身体坐标系下的 forward / left / behind / right 四个方向。

早期的单 USD Camera 调试入口 :func:`attach_head_camera` 也保留在本模块中，
但主程序不导入、不调用它，也不再暴露对应 CLI 参数。

这里创建的是可读取 RGB-D 数组的传感器 prim，不包含相机外壳 mesh。
"""

import math


# 四方向名称和传感器 key 是整个 Isaac/LaViRA 接口的稳定契约。模型适配器以后
# 可以决定发送顺序，但相机层永远不通过 list 下标猜测方向。
FOUR_VIEW_DIRECTIONS = ("forward", "left", "behind", "right")
FOUR_VIEW_PARENT_BODY_NAME = "torso_link"
FOUR_VIEW_SENSOR_NAMES = {
    "forward": "camera_forward",
    "left": "camera_left",
    "behind": "camera_behind",
    "right": "camera_right",
}
FOUR_VIEW_CAMERA_PRIM_NAMES = {
    direction: f"lavira_camera_{direction}" for direction in FOUR_VIEW_DIRECTIONS
}
FOUR_VIEW_YAW_DEG = {
    # Calibrated in the open bedroom against the G1 body and its actual
    # locomotion directions: forward=+X and left=+Y in torso_link.
    "forward": 0.0,
    "left": 90.0,
    "behind": 180.0,
    "right": -90.0,
}


def configure_four_rgbd_cameras(env_cfg, args_cli) -> None:
    """在 ``gym.make`` 前给 G1 注册四个正式 IsaacLab RGB-D Camera sensor。

    相机直接挂到 ``torso_link``，不依赖可能在 URDF 转 USD 时被合并的
    ``head_link`` / ``d435_link``。四个光心位于头顶篮子四个外侧面：篮子中心
    为 ``(0, 0, camera_rig_height)``，水平半径为 ``camera_rig_radius``。

    ``CameraCfg.OffsetCfg(convention="world")`` 在这里表示相机局部坐标采用
    +X optical-forward / +Y optical-left / +Z optical-up。当前 G1 的导航前方经
    GUI 和实际行走校准为 torso_link +X；``camera_down_tilt_deg`` 为正时光轴向下。
    """
    if not getattr(args_cli, "four_rgbd_cameras", False):
        return

    try:
        import isaaclab.sim as sim_utils
        from isaaclab.sensors import CameraCfg
    except Exception as exc:
        raise RuntimeError(f"Could not import IsaacLab RGB-D camera APIs: {exc}") from exc

    width = int(args_cli.rgbd_camera_width)
    height = int(args_cli.rgbd_camera_height)
    rig_height = float(args_cli.camera_rig_height)
    rig_radius = float(args_cli.camera_rig_radius)
    down_tilt_deg = float(args_cli.camera_down_tilt_deg)
    hfov_deg = float(args_cli.rgbd_camera_hfov_deg)
    update_period = float(args_cli.rgbd_camera_update_period)
    near = float(args_cli.rgbd_camera_near)
    far = float(args_cli.rgbd_camera_far)

    if width <= 0 or height <= 0:
        raise ValueError(f"RGB-D camera resolution must be positive, got {width}x{height}.")
    if rig_radius < 0.0:
        raise ValueError(f"--camera_rig_radius must be non-negative, got {rig_radius}.")
    if not 1.0 < hfov_deg < 179.0:
        raise ValueError(f"--rgbd_camera_hfov_deg must be in (1, 179), got {hfov_deg}.")
    if not -89.0 < down_tilt_deg < 89.0:
        raise ValueError(
            f"--camera_down_tilt_deg must be in (-89, 89), got {down_tilt_deg}."
        )
    if abs(update_period) > 1.0e-9:
        raise ValueError(
            "The first four-view implementation requires --rgbd_camera_update_period 0. "
            "A non-zero period can pair an older RGB-D frame with the robot's current pose."
        )
    if near <= 0.0 or far <= near:
        raise ValueError(f"Invalid RGB-D clipping range ({near}, {far}).")

    # Camera buffers and robot pose are copied after one RL env.step.  The last
    # physics sub-step must therefore also be a render step; otherwise a fresh
    # base pose could be paired with an older image.  The native G1 tasks use
    # render_interval == decimation and satisfy this condition.
    decimation = int(getattr(env_cfg, "decimation", 0))
    render_interval = int(getattr(env_cfg.sim, "render_interval", 0))
    if (
        decimation <= 0
        or render_interval <= 0
        or render_interval > decimation
        or decimation % render_interval != 0
    ):
        raise ValueError(
            "Synchronized four-view capture requires the final physics sub-step "
            "of every env.step to be rendered, but got "
            f"decimation={decimation}, render_interval={render_interval}."
        )

    # Isaac/UsdCamera specifies focal length and aperture rather than HFOV.
    # This conversion makes the default exactly match LaViRA's Habitat config
    # (640x480, HFOV=79 degrees) while retaining a calibration override.
    focal_length = 18.0
    horizontal_aperture = 2.0 * focal_length * math.tan(
        math.radians(hfov_deg) * 0.5
    )

    local_poses = get_four_view_local_poses(
        rig_height=rig_height,
        rig_radius=rig_radius,
        down_tilt_deg=down_tilt_deg,
    )

    for direction in FOUR_VIEW_DIRECTIONS:
        sensor_name = FOUR_VIEW_SENSOR_NAMES[direction]
        if getattr(env_cfg.scene, sensor_name, None) is not None:
            raise ValueError(f"Scene already contains an entity named {sensor_name!r}.")

        camera_cfg = CameraCfg(
            prim_path=(
                f"{{ENV_REGEX_NS}}/Robot/{FOUR_VIEW_PARENT_BODY_NAME}/"
                f"{FOUR_VIEW_CAMERA_PRIM_NAMES[direction]}"
            ),
            update_period=update_period,
            height=height,
            width=width,
            data_types=["rgb", "distance_to_image_plane"],
            spawn=sim_utils.PinholeCameraCfg(
                focal_length=focal_length,
                focus_distance=2.0,
                horizontal_aperture=horizontal_aperture,
                clipping_range=(near, far),
            ),
            offset=CameraCfg.OffsetCfg(
                pos=local_poses[direction][0],
                rot=local_poses[direction][1],
                convention="world",
            ),
            depth_clipping_behavior="none",
        )
        setattr(env_cfg.scene, sensor_name, camera_cfg)

    print(
        "[INFO] Configured four IsaacLab RGB-D cameras: "
        f"directions={FOUR_VIEW_DIRECTIONS}, resolution={width}x{height}, "
        f"hfov={hfov_deg:.1f}deg, "
        f"rig_height={rig_height:.3f}m, rig_radius={rig_radius:.3f}m, "
        f"down_tilt={down_tilt_deg:.1f}deg, clipping=({near:.2f},{far:.2f})m."
    )


def get_four_view_local_poses(
    rig_height: float,
    rig_radius: float,
    down_tilt_deg: float,
) -> dict[
    str,
    tuple[
        tuple[float, float, float],
        tuple[float, float, float, float],
    ],
]:
    """返回四台相机相对 torso_link 的 world-convention 固定安装位姿。

    这里按机器人实际外观和运动方向标定：forward=+X，left=+Y，
    behind=-X，right=-Y。
    """
    positions = {
        "forward": (rig_radius, 0.0, rig_height),
        "left": (0.0, rig_radius, rig_height),
        "behind": (-rig_radius, 0.0, rig_height),
        "right": (0.0, -rig_radius, rig_height),
    }
    return {
        direction: (
            tuple(float(value) for value in positions[direction]),
            _quat_wxyz_from_yaw_down_tilt_deg(
                FOUR_VIEW_YAW_DEG[direction], down_tilt_deg
            ),
        )
        for direction in FOUR_VIEW_DIRECTIONS
    }


def set_forward_rgbd_camera_viewport(args_cli) -> None:
    """把 GUI viewport 切到 env_0 的正式 forward RGB-D Camera prim。"""
    if (
        not getattr(args_cli, "four_rgbd_cameras", False)
        or not getattr(args_cli, "four_rgbd_set_viewport", False)
        or getattr(args_cli, "headless", False)
    ):
        return

    try:
        import omni.usd
    except Exception as exc:
        print(f"[WARN] Could not inspect RGB-D camera prims for viewport: {exc}")
        return

    stage = omni.usd.get_context().get_stage()
    if stage is None:
        print("[WARN] Could not switch viewport to forward RGB-D camera: no USD stage.")
        return

    prim_name = FOUR_VIEW_CAMERA_PRIM_NAMES["forward"]
    candidates = [
        str(prim.GetPath())
        for prim in stage.Traverse()
        if prim.GetName() == prim_name
    ]
    if not candidates:
        print(f"[WARN] Could not find forward RGB-D camera prim named {prim_name!r}.")
        return

    candidates.sort(key=lambda path: (0 if "/env_0/" in path else 1, path))
    _set_active_viewport_camera(candidates[0])


def draw_four_rgbd_camera_debug_points(args_cli) -> None:
    """在 env_0 四个真实相机光心处创建临时彩色球体。"""
    if not getattr(args_cli, "four_rgbd_debug_points", False):
        return
    if not getattr(args_cli, "four_rgbd_cameras", False):
        print("[WARN] --four_rgbd_debug_points requires --four_rgbd_cameras.")
        return

    try:
        import omni.usd
    except Exception as exc:
        print(f"[WARN] Could not draw four-view camera debug points: {exc}")
        return

    stage = omni.usd.get_context().get_stage()
    if stage is None:
        print("[WARN] Could not draw four-view camera debug points: no USD stage.")
        return

    colors = {
        "forward": (1.0, 0.05, 0.05),
        "left": (0.05, 1.0, 0.05),
        "behind": (0.05, 0.25, 1.0),
        "right": (1.0, 0.85, 0.05),
    }
    camera_paths_by_name: dict[str, list[str]] = {
        prim_name: [] for prim_name in FOUR_VIEW_CAMERA_PRIM_NAMES.values()
    }
    for prim in stage.Traverse():
        if prim.GetName() in camera_paths_by_name:
            camera_paths_by_name[prim.GetName()].append(str(prim.GetPath()))

    created: list[str] = []
    for direction in FOUR_VIEW_DIRECTIONS:
        prim_name = FOUR_VIEW_CAMERA_PRIM_NAMES[direction]
        candidates = camera_paths_by_name[prim_name]
        candidates.sort(key=lambda path: (0 if "/env_0/" in path else 1, path))
        if not candidates:
            print(f"[WARN] Camera debug point skipped: could not find {prim_name!r}.")
            continue

        camera_path = candidates[0]
        marker_path = f"{camera_path}/DebugOpticalCenter"
        if stage.GetPrimAtPath(marker_path):
            stage.RemovePrim(marker_path)
        _sphere(
            stage,
            marker_path,
            (0.0, 0.0, 0.0),
            radius=0.025,
            color=colors[direction],
        )
        created.append(f"{direction}={marker_path}")

    if created:
        print(
            "[INFO] Four-view camera debug points created "
            "(forward=red, left=green, behind=blue, right=yellow):"
        )
        for item in created:
            print(f"  {item}")


def _quat_wxyz_from_yaw_down_tilt_deg(
    yaw_deg: float, down_tilt_deg: float
) -> tuple[float, float, float, float]:
    """返回 body/world-convention camera frame 相对父节点的 wxyz 四元数。

    组合顺序是 ``Rz(yaw) @ Ry(down_tilt)``。因此相机的 +X forward 轴会变为：

    ``[cos(yaw) cos(tilt), sin(yaw) cos(tilt), -sin(tilt)]``。

    这个显式定义避免直接猜 USD/OpenGL camera 的 pitch 正负号；CameraCfg 会再把
    world convention 转换成 USD Camera 的 -Z forward convention。
    """
    yaw = math.radians(float(yaw_deg))
    tilt = math.radians(float(down_tilt_deg))
    cy, sy = math.cos(yaw * 0.5), math.sin(yaw * 0.5)
    ct, st = math.cos(tilt * 0.5), math.sin(tilt * 0.5)

    # Hamilton product q_yaw(z) * q_tilt(y).
    return (
        cy * ct,
        -sy * st,
        cy * st,
        sy * ct,
    )


def attach_head_camera(raw_env, args_cli) -> str | None:
    """保留的旧单目 USD Camera 挂载入口；当前主程序不会调用它。"""
    del raw_env
    if not getattr(args_cli, "head_camera", False):
        return None

    try:
        import omni.usd
        from pxr import Gf, Sdf, UsdGeom
    except Exception as exc:
        print(f"[WARN] Head camera disabled: could not import USD/Omniverse APIs: {exc}")
        return None

    stage = omni.usd.get_context().get_stage()
    if stage is None:
        print("[WARN] Head camera disabled: no USD stage.")
        return None

    parent_path = getattr(args_cli, "head_camera_parent", None)
    if parent_path is None:
        parent_path = _find_camera_parent_path(stage)
    if parent_path is None:
        print(
            "[WARN] Head camera disabled: could not find a d435_link/head_link "
            "prim. Set args_cli.head_camera_parent."
        )
        _print_head_candidates(stage)
        return None

    parent_prim = stage.GetPrimAtPath(parent_path)
    if not parent_prim or not parent_prim.IsValid():
        print(f"[WARN] Head camera disabled: parent prim does not exist: {parent_path}")
        return None

    camera_name = str(getattr(args_cli, "head_camera_name", "head_camera"))
    camera_pos = tuple(getattr(args_cli, "head_camera_pos", (0.0, 0.0, 0.0)))
    camera_rpy = tuple(getattr(args_cli, "head_camera_rpy", (90.0, 0.0, -90.0)))
    focal_length = float(getattr(args_cli, "head_camera_focal_length", 18.0))
    horizontal_aperture = float(
        getattr(args_cli, "head_camera_horizontal_aperture", 24.742)
    )
    vertical_aperture = float(
        getattr(args_cli, "head_camera_vertical_aperture", 13.819)
    )
    focus_distance = float(getattr(args_cli, "head_camera_focus_distance", 2.0))

    mount_path = f"{parent_path.rstrip('/')}/{camera_name}_mount"
    camera_path = f"{mount_path}/{camera_name}"
    if stage.GetPrimAtPath(mount_path):
        stage.RemovePrim(mount_path)

    mount = UsdGeom.Xform.Define(stage, Sdf.Path(mount_path))
    mount_xform = UsdGeom.Xformable(mount.GetPrim())
    mount_xform.ClearXformOpOrder()
    mount_xform.AddTranslateOp().Set(Gf.Vec3d(*[float(v) for v in camera_pos]))
    mount_xform.AddOrientOp().Set(Gf.Quatf(1.0, Gf.Vec3f(0.0, 0.0, 0.0)))

    camera = UsdGeom.Camera.Define(stage, Sdf.Path(camera_path))
    camera_xform = UsdGeom.Xformable(camera.GetPrim())
    camera_xform.ClearXformOpOrder()
    camera_xform.AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, 0.0))
    camera_xform.AddOrientOp().Set(
        _quat_from_rpy_deg(*[float(v) for v in camera_rpy])
    )
    camera.CreateFocalLengthAttr(focal_length)
    camera.CreateHorizontalApertureAttr(horizontal_aperture)
    camera.CreateVerticalApertureAttr(vertical_aperture)
    camera.CreateFocusDistanceAttr(focus_distance)
    camera.CreateClippingRangeAttr(Gf.Vec2f(0.01, 100.0))
    _draw_head_camera_debug_visualization(
        stage,
        mount_path,
        focal_length=focal_length,
        horizontal_aperture=horizontal_aperture,
        vertical_aperture=vertical_aperture,
        enabled=bool(getattr(args_cli, "head_camera_debug_vis", False)),
    )

    print(f"[INFO] Legacy head camera parent: {parent_path}")
    print(f"[INFO] Legacy head camera attached: {camera_path}")
    if (
        getattr(args_cli, "head_camera_set_viewport", True)
        and not getattr(args_cli, "headless", False)
    ):
        _set_active_viewport_camera(camera_path)
    return camera_path


def _draw_head_camera_debug_visualization(
    stage,
    mount_path: str,
    *,
    focal_length: float,
    horizontal_aperture: float,
    vertical_aperture: float,
    enabled: bool,
) -> None:
    if not enabled:
        return
    try:
        from pxr import Sdf, UsdGeom

        root_path = f"{mount_path}/DebugViz"
        if stage.GetPrimAtPath(root_path):
            stage.RemovePrim(root_path)
        UsdGeom.Xform.Define(stage, Sdf.Path(root_path))

        z = -0.35
        half_width = abs(z) * horizontal_aperture / (2.0 * focal_length)
        half_height = abs(z) * vertical_aperture / (2.0 * focal_length)
        origin = (0.0, 0.0, 0.0)
        corners = [
            (-half_width, -half_height, z),
            (half_width, -half_height, z),
            (half_width, half_height, z),
            (-half_width, half_height, z),
        ]

        _sphere(
            stage,
            f"{root_path}/OpticalCenter",
            origin,
            radius=0.018,
            color=(0.0, 0.8, 1.0),
        )
        _curve(
            stage,
            f"{root_path}/OpticalAxis",
            [origin, (0.0, 0.0, z)],
            width=0.01,
            color=(1.0, 0.05, 0.05),
        )
        _curve(
            stage,
            f"{root_path}/UpAxis",
            [origin, (0.0, half_height * 0.75, 0.0)],
            width=0.008,
            color=(0.0, 0.9, 0.2),
        )
        _curve(
            stage,
            f"{root_path}/FovFrame",
            [*corners, corners[0]],
            width=0.006,
            color=(1.0, 0.8, 0.0),
        )
        for index, corner in enumerate(corners):
            _curve(
                stage,
                f"{root_path}/FovRay_{index}",
                [origin, corner],
                width=0.004,
                color=(1.0, 0.8, 0.0),
            )
        print(f"[INFO] Legacy head camera debug visualization: {root_path}")
    except Exception as exc:
        print(f"[WARN] Could not draw head camera debug visualization: {exc}")


def _sphere(
    stage,
    path: str,
    position: tuple[float, float, float],
    radius: float,
    color: tuple[float, float, float],
) -> None:
    from pxr import Gf, UsdGeom

    sphere = UsdGeom.Sphere.Define(stage, path)
    sphere.CreateRadiusAttr(radius)
    UsdGeom.XformCommonAPI(sphere).SetTranslate(Gf.Vec3d(*position))
    sphere.CreateDisplayColorAttr([Gf.Vec3f(*color)])


def _curve(
    stage,
    path: str,
    points: list[tuple[float, float, float]],
    width: float,
    color: tuple[float, float, float],
) -> None:
    from pxr import Gf, UsdGeom

    curve = UsdGeom.BasisCurves.Define(stage, path)
    curve.CreateTypeAttr("linear")
    curve.CreateCurveVertexCountsAttr([len(points)])
    curve.CreatePointsAttr([Gf.Vec3f(*point) for point in points])
    curve.CreateWidthsAttr([width])
    curve.CreateDisplayColorAttr([Gf.Vec3f(*color)])


def _find_camera_parent_path(stage) -> str | None:
    """优先返回 env_0 机器人中的 d435_link，其次返回 head_link。"""
    preferred_names = ("d435_link", "head_link")
    candidates = [
        str(prim.GetPath())
        for prim in stage.Traverse()
        if prim.GetName() in preferred_names
    ]
    if not candidates:
        return None

    def score(path: str) -> tuple[int, int, int, str]:
        env0_bonus = 0 if "/env_0/" in path or "/envs/env_0/" in path else 1
        robot_bonus = 0 if "robot" in path.lower() or "g1" in path.lower() else 1
        link_bonus = 0 if path.rsplit("/", 1)[-1] == "d435_link" else 1
        return (env0_bonus, robot_bonus, link_bonus, path)

    return sorted(candidates, key=score)[0]


def _print_head_candidates(stage) -> None:
    candidates = [
        str(prim.GetPath())
        for prim in stage.Traverse()
        if "head" in prim.GetName().lower() or "camera" in prim.GetName().lower()
    ]
    if candidates:
        print("[INFO] Head/camera candidate prims:")
        for path in candidates[:30]:
            print(f"  {path}")


def _quat_from_rpy_deg(roll_deg: float, pitch_deg: float, yaw_deg: float):
    """把 intrinsic XYZ roll/pitch/yaw（度）转换为 USD/Gf 四元数。"""
    from pxr import Gf

    roll = math.radians(roll_deg)
    pitch = math.radians(pitch_deg)
    yaw = math.radians(yaw_deg)
    cr, sr = math.cos(roll * 0.5), math.sin(roll * 0.5)
    cp, sp = math.cos(pitch * 0.5), math.sin(pitch * 0.5)
    cy, sy = math.cos(yaw * 0.5), math.sin(yaw * 0.5)
    qw = cr * cp * cy + sr * sp * sy
    qx = sr * cp * cy - cr * sp * sy
    qy = cr * sp * cy + sr * cp * sy
    qz = cr * cp * sy - sr * sp * cy
    return Gf.Quatf(float(qw), Gf.Vec3f(float(qx), float(qy), float(qz)))


def _set_active_viewport_camera(camera_path: str) -> None:
    """把当前 Isaac Sim 活跃视口切换到指定相机。"""
    try:
        # viewport utility 只在带 GUI 的 Isaac Sim 中可用；headless 模式下调用方会跳过这里。
        import omni.kit.viewport.utility
        from pxr import Sdf

        viewport = omni.kit.viewport.utility.get_active_viewport()
        if viewport is None:
            print("[WARN] Could not switch viewport camera: no active viewport.")
            return

        # viewport.camera_path 需要 Sdf.Path 类型，而不是普通字符串。
        viewport.camera_path = Sdf.Path(camera_path)
        print(f"[INFO] Active viewport camera: {camera_path}")
    except Exception as exc:
        # 视口切换失败不影响相机 prim 本身创建，所以这里只警告，不向外抛异常。
        print(f"[WARN] Could not switch viewport camera: {exc}")
