"""Export a Unitree RL Lab-style deploy.yaml for ManagerBasedRLEnv policies."""

from __future__ import annotations

import os
import re
from collections.abc import Sequence

import numpy as np
import yaml

from isaaclab.assets import Articulation
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.utils import class_to_dict


G1_29DOF_SDK_JOINT_NAMES = [
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
]

G1_29DOF_FALLBACK_STIFFNESS = {
    ".*_hip_yaw_joint": 100.0,
    ".*_hip_roll_joint": 100.0,
    ".*_hip_pitch_joint": 100.0,
    ".*_knee_joint": 150.0,
    ".*_ankle_pitch_joint": 28.5,
    ".*_ankle_roll_joint": 28.5,
    "waist_yaw_joint": 300.0,
    "waist_roll_joint": 300.0,
    "waist_pitch_joint": 300.0,
    ".*_shoulder_.*_joint": 200.0,
    ".*_elbow_joint": 200.0,
    ".*_wrist_.*_joint": 200.0,
}

G1_29DOF_FALLBACK_DAMPING = {
    ".*_hip_yaw_joint": 2.0,
    ".*_hip_roll_joint": 2.0,
    ".*_hip_pitch_joint": 2.0,
    ".*_knee_joint": 4.0,
    ".*_ankle_pitch_joint": 1.8,
    ".*_ankle_roll_joint": 1.8,
    "waist_yaw_joint": 8.0,
    "waist_roll_joint": 8.0,
    "waist_pitch_joint": 8.0,
    ".*_shoulder_.*_joint": 8.0,
    ".*_elbow_joint": 8.0,
    ".*_wrist_.*_joint": 8.0,
}


def _to_list(value):
    if value is None:
        return None
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, range):
        return list(value)
    if isinstance(value, Sequence) and not isinstance(value, str):
        return list(value)
    return value


def _format_value(value):
    value = _to_list(value)
    if isinstance(value, float):
        return float(f"{value:.6g}")
    if isinstance(value, list):
        return [_format_value(item) for item in value]
    if isinstance(value, dict):
        return {key: _format_value(item) for key, item in value.items()}
    return value


def _term_scale_list(term_cfg, action_term):
    scale = getattr(term_cfg, "scale", None)
    if isinstance(scale, float):
        return [scale for _ in range(action_term.action_dim)]
    if hasattr(action_term, "_scale"):
        return _to_list(action_term._scale[0])
    return _to_list(scale)


def _joint_ids(action_term):
    ids = getattr(action_term, "_joint_ids", None)
    if ids == slice(None):
        return None
    return _to_list(ids)


def _deploy_action_name(action_name: str, action_term) -> str:
    class_name = action_term.__class__.__name__
    if "JointPositionAction" in class_name:
        return "JointPositionAction"
    if "JointVelocityAction" in class_name:
        return "JointVelocityAction"
    return action_name


def _clean_dict(data: dict, keys: list[str]) -> dict:
    for key in keys:
        data.pop(key, None)
    return data


def _pattern_value(table: dict[str, float], joint_name: str, default_value: float = 0.0):
    for pattern, value in table.items():
        if re.fullmatch(pattern, joint_name):
            return value
    return default_value


def _sdk_index_by_name():
    return {name: idx for idx, name in enumerate(G1_29DOF_SDK_JOINT_NAMES)}


def _sdk_order_values(asset: Articulation, values, fallback_table: dict[str, float], default_value: float = 0.0):
    asset_values = dict(zip(asset.data.joint_names, values))
    ordered = []
    for joint_name in G1_29DOF_SDK_JOINT_NAMES:
        ordered.append(asset_values.get(joint_name, _pattern_value(fallback_table, joint_name, default_value)))
    return ordered


def _asset_scale_to_sdk_order(asset: Articulation, values, default_value: float):
    asset_values = dict(zip(asset.data.joint_names, values))
    return [asset_values.get(joint_name, default_value) for joint_name in G1_29DOF_SDK_JOINT_NAMES]


def _expanded_joint_position_action(term_dict, action_term, action_scale, default_joint_pos, asset: Articulation):
    action_joint_ids = _joint_ids(action_term)
    if action_joint_ids is None:
        return term_dict

    sdk_index = _sdk_index_by_name()
    full_scale = [0.0 for _ in range(len(G1_29DOF_SDK_JOINT_NAMES))]
    full_offset = list(default_joint_pos)
    action_offset = _to_list(action_term._offset[0]) if hasattr(action_term, "_offset") else None
    policy_sdk_joint_ids = []
    for action_idx, joint_id in enumerate(action_joint_ids):
        joint_name = asset.data.joint_names[joint_id]
        sdk_joint_id = sdk_index[joint_name]
        policy_sdk_joint_ids.append(sdk_joint_id)
        full_scale[sdk_joint_id] = action_scale[action_idx]
        if action_offset is not None:
            full_offset[sdk_joint_id] = action_offset[action_idx]

    term_dict["joint_names"] = [".*"]
    term_dict["scale"] = full_scale
    term_dict["offset"] = full_offset
    term_dict["joint_ids"] = None
    term_dict["policy_joint_ids"] = policy_sdk_joint_ids
    term_dict["policy_action_dim"] = len(action_joint_ids)
    return term_dict


def export_deploy_cfg(env: ManagerBasedRLEnv, log_dir: str, expand_actions_to_all_joints: bool = True):
    """Save deploy.yaml beside env.yaml and agent.yaml.

    The output matches Unitree RL Lab's deploy config format where possible.
    For local lock-waist G1 assets that do not define ``joint_sdk_names``, the
    standard G1 29DOF SDK order is used.
    """

    asset: Articulation = env.scene["robot"]
    joint_sdk_names = G1_29DOF_SDK_JOINT_NAMES
    joint_ids_map = list(range(len(joint_sdk_names)))

    cfg = {}
    cfg["joint_ids_map"] = _to_list(joint_ids_map)
    cfg["step_dt"] = env.cfg.sim.dt * env.cfg.decimation

    asset_stiffness = asset.data.default_joint_stiffness[0].detach().cpu().numpy().tolist()
    cfg["stiffness"] = _sdk_order_values(asset, asset_stiffness, G1_29DOF_FALLBACK_STIFFNESS)

    asset_damping = asset.data.default_joint_damping[0].detach().cpu().numpy().tolist()
    cfg["damping"] = _sdk_order_values(asset, asset_damping, G1_29DOF_FALLBACK_DAMPING)

    asset_default_joint_pos = asset.data.default_joint_pos[0].detach().cpu().numpy().tolist()
    default_joint_pos = _sdk_order_values(asset, asset_default_joint_pos, {}, 0.0)
    cfg["default_joint_pos"] = default_joint_pos

    cfg["commands"] = {}
    if hasattr(env.cfg.commands, "base_velocity"):
        ranges_cfg = (
            env.cfg.commands.base_velocity.limit_ranges
            if hasattr(env.cfg.commands.base_velocity, "limit_ranges")
            else env.cfg.commands.base_velocity.ranges
        )
        ranges = ranges_cfg.to_dict()
        for name in ["lin_vel_x", "lin_vel_y", "ang_vel_z"]:
            if name in ranges and ranges[name] is not None:
                ranges[name] = list(ranges[name])
        cfg["commands"]["base_velocity"] = {"ranges": ranges}

    cfg["actions"] = {}
    for action_name, action_term in zip(env.action_manager.active_terms, env.action_manager._terms.values()):
        term_cfg = action_term.cfg.copy()
        action_scale = _term_scale_list(term_cfg, action_term)
        term_cfg.scale = action_scale

        if getattr(term_cfg, "clip", None) is not None and hasattr(action_term, "_clip"):
            term_cfg.clip = _to_list(action_term._clip[0])

        if _deploy_action_name(action_name, action_term) in ["JointPositionAction", "JointVelocityAction"]:
            if getattr(term_cfg, "use_default_offset", False):
                term_cfg.offset = _to_list(action_term._offset[0])
            else:
                term_cfg.offset = [0.0 for _ in range(action_term.action_dim)]

        deploy_name = _deploy_action_name(action_name, action_term)
        term_dict = term_cfg.to_dict()
        _clean_dict(term_dict, ["class_type", "asset_name", "debug_vis", "preserve_order", "use_default_offset"])
        term_dict["joint_ids"] = _joint_ids(action_term)
        if expand_actions_to_all_joints and deploy_name == "JointPositionAction":
            term_dict = _expanded_joint_position_action(
                term_dict=term_dict,
                action_term=action_term,
                action_scale=action_scale,
                default_joint_pos=default_joint_pos,
                asset=asset,
            )
        cfg["actions"][deploy_name] = term_dict

    cfg["observations"] = {}
    for obs_name, obs_cfg in zip(
        env.observation_manager.active_terms["policy"],
        env.observation_manager._group_obs_term_cfgs["policy"],
    ):
        obs_dims = tuple(obs_cfg.func(env, **obs_cfg.params).shape)
        term_cfg = obs_cfg.copy()
        if term_cfg.scale is not None:
            scale = _to_list(term_cfg.scale)
            if isinstance(scale, float):
                term_cfg.scale = [scale for _ in range(obs_dims[1])]
            else:
                term_cfg.scale = scale
        else:
            term_cfg.scale = [1.0 for _ in range(obs_dims[1])]

        if expand_actions_to_all_joints and obs_name in ["joint_pos_rel", "joint_vel_rel", "last_action"]:
            if obs_name == "joint_pos_rel":
                term_cfg.scale = _asset_scale_to_sdk_order(asset, term_cfg.scale, 1.0)
            elif obs_name == "joint_vel_rel":
                term_cfg.scale = _asset_scale_to_sdk_order(asset, term_cfg.scale, 0.05)
            else:
                term_cfg.scale = [1.0 for _ in range(len(joint_sdk_names))]

        if term_cfg.clip is not None:
            term_cfg.clip = list(term_cfg.clip)
        if term_cfg.history_length == 0:
            term_cfg.history_length = 1

        term_dict = term_cfg.to_dict()
        _clean_dict(term_dict, ["func", "modifiers", "noise", "flatten_history_dim"])
        cfg["observations"][obs_name] = term_dict

    filename = os.path.join(log_dir, "params", "deploy.yaml")
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    if not isinstance(cfg, dict):
        cfg = class_to_dict(cfg)
    with open(filename, "w", encoding="utf-8") as file:
        yaml.dump(_format_value(cfg), file, default_flow_style=None, sort_keys=False)
