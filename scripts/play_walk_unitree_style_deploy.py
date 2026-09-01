"""Export a deploy-compatible 29DOF ONNX wrapper for the 12DOF Unitree-style policy."""

from pathlib import Path
import argparse
import copy
import os
import sys
import time

PROJECT_DIR = Path(__file__).resolve().parents[1]
PROJECTS_DIR = PROJECT_DIR.parent
ISAACLAB_DIR = PROJECTS_DIR / "IsaacLab"
RSL_RL_SCRIPT_DIR = ISAACLAB_DIR / "scripts" / "reinforcement_learning" / "rsl_rl"

sys.path.insert(0, str(PROJECT_DIR / "source" / "unitree_g1_isaaclab"))
sys.path.insert(0, str(PROJECTS_DIR / "unitree_rl_lab" / "source" / "unitree_rl_lab"))
sys.path.insert(0, str(RSL_RL_SCRIPT_DIR))

import unitree_g1_isaaclab.tasks.walk_flat_unitree_style  # noqa: E402,F401

from isaaclab.app import AppLauncher  # noqa: E402

import cli_args  # noqa: E402

parser = argparse.ArgumentParser(description="Play a checkpoint and export deploy-compatible ONNX.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during play.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video in steps.")
parser.add_argument("--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O.")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--agent", type=str, default="rsl_rl_cfg_entry_point", help="Name of the RL agent configuration entry point."
)
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment.")
parser.add_argument("--real-time", action="store_true", default=False, help="Run in real-time, if possible.")
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

if args_cli.video:
    args_cli.enable_cameras = True

sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402
from rsl_rl.runners import DistillationRunner, OnPolicyRunner  # noqa: E402

import isaaclab_tasks  # noqa: E402,F401
from isaaclab.envs import (  # noqa: E402
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.assets import retrieve_file_path  # noqa: E402
from isaaclab.utils.dict import print_dict  # noqa: E402
from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, RslRlVecEnvWrapper, export_policy_as_jit  # noqa: E402
from isaaclab_tasks.utils import get_checkpoint_path  # noqa: E402
from isaaclab_tasks.utils.hydra import hydra_task_config  # noqa: E402

from unitree_g1_isaaclab.utils.export_deploy_cfg import G1_29DOF_SDK_JOINT_NAMES, export_deploy_cfg  # noqa: E402


class DeployCompatPolicy(torch.nn.Module):
    """Adapt 29DOF deploy observations/actions to a 12DOF trained policy."""

    def __init__(self, policy, normalizer, obs_indices, action_joint_ids, deploy_obs_dim, full_action_dim):
        super().__init__()
        if getattr(policy, "is_recurrent", False):
            raise ValueError("DeployCompatPolicy currently supports feed-forward policies only.")
        if hasattr(policy, "actor"):
            self.actor = copy.deepcopy(policy.actor)
        elif hasattr(policy, "student"):
            self.actor = copy.deepcopy(policy.student)
        else:
            raise ValueError("Policy does not have an actor/student module.")
        self.normalizer = copy.deepcopy(normalizer) if normalizer is not None else torch.nn.Identity()
        self.deploy_obs_dim = deploy_obs_dim
        self.full_action_dim = full_action_dim
        self.register_buffer("obs_indices", torch.as_tensor(obs_indices, dtype=torch.long))
        projection = torch.zeros(len(action_joint_ids), full_action_dim)
        for action_idx, joint_id in enumerate(action_joint_ids):
            projection[action_idx, joint_id] = 1.0
        self.register_buffer("action_projection", projection)

    def forward(self, obs):
        policy_obs = obs.index_select(1, self.obs_indices)
        raw_action_12 = self.actor(self.normalizer(policy_obs))
        return raw_action_12.matmul(self.action_projection.to(dtype=raw_action_12.dtype))


def _as_list(ids):
    if ids == slice(None):
        return None
    if hasattr(ids, "detach"):
        ids = ids.detach().cpu().numpy()
    return list(ids)


def _sdk_index_by_name():
    return {name: idx for idx, name in enumerate(G1_29DOF_SDK_JOINT_NAMES)}


def _policy_joint_sdk_ids(env):
    sdk_index = _sdk_index_by_name()
    return [sdk_index[name] for name in env.unwrapped.scene["robot"].data.joint_names if name in sdk_index]


def _deploy_obs_indices(env, policy_action_sdk_ids, full_action_dim):
    raw_env = env.unwrapped
    policy_joint_sdk_ids = _policy_joint_sdk_ids(env)
    obs_names = raw_env.observation_manager.active_terms["policy"]
    obs_cfgs = raw_env.observation_manager._group_obs_term_cfgs["policy"]
    deploy_cursor = 0
    indices = []
    for obs_name, obs_cfg in zip(obs_names, obs_cfgs):
        obs_dim = int(obs_cfg.func(raw_env, **obs_cfg.params).shape[1])
        history = obs_cfg.history_length if obs_cfg.history_length != 0 else 1
        deploy_dim = full_action_dim if obs_name in ["joint_pos_rel", "joint_vel_rel", "last_action"] else obs_dim
        for hist_idx in range(history):
            block_start = deploy_cursor + hist_idx * deploy_dim
            if obs_name == "last_action":
                indices.extend(block_start + joint_id for joint_id in policy_action_sdk_ids)
            elif obs_name in ["joint_pos_rel", "joint_vel_rel"]:
                indices.extend(block_start + joint_id for joint_id in policy_joint_sdk_ids)
            else:
                indices.extend(range(block_start, block_start + obs_dim))
        deploy_cursor += deploy_dim * history
    return indices, deploy_cursor


def _action_sdk_joint_ids(env):
    raw_env = env.unwrapped
    sdk_index = _sdk_index_by_name()
    for action_term in raw_env.action_manager._terms.values():
        ids = _as_list(getattr(action_term, "_joint_ids", None))
        if ids is not None:
            return [sdk_index[raw_env.scene["robot"].data.joint_names[joint_id]] for joint_id in ids]
    raise ValueError("Expected a joint action term with explicit 12DOF joint ids.")


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    """Play with RSL-RL agent and export deploy-compatible policy.onnx."""

    task_name = args_cli.task.split(":")[-1]
    train_task_name = task_name.replace("-Play", "")

    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    log_root_path = os.path.abspath(os.path.join("logs", "rsl_rl", agent_cfg.experiment_name))
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    if args_cli.checkpoint:
        resume_path = retrieve_file_path(args_cli.checkpoint)
    else:
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
    log_dir = os.path.dirname(resume_path)
    env_cfg.log_dir = log_dir

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    action_joint_ids = _action_sdk_joint_ids(env)
    full_action_dim = len(G1_29DOF_SDK_JOINT_NAMES)
    obs_indices, deploy_obs_dim = _deploy_obs_indices(env, action_joint_ids, full_action_dim)
    export_deploy_cfg(env.unwrapped, log_dir)
    print(f"[INFO] Exported deploy config to: {os.path.join(log_dir, 'params', 'deploy.yaml')}")

    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "play"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during play.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    print(f"[INFO]: Loading model checkpoint from: {resume_path}")
    if agent_cfg.class_name == "OnPolicyRunner":
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    elif agent_cfg.class_name == "DistillationRunner":
        runner = DistillationRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    else:
        raise ValueError(f"Unsupported runner class: {agent_cfg.class_name}")
    runner.load(resume_path)
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    try:
        policy_nn = runner.alg.policy
    except AttributeError:
        policy_nn = runner.alg.actor_critic

    if hasattr(policy_nn, "actor_obs_normalizer"):
        normalizer = policy_nn.actor_obs_normalizer
    elif hasattr(policy_nn, "student_obs_normalizer"):
        normalizer = policy_nn.student_obs_normalizer
    else:
        normalizer = None

    export_model_dir = os.path.join(log_dir, "exported")
    export_policy_as_jit(policy_nn, normalizer=normalizer, path=export_model_dir, filename="policy_12dof.pt")

    deploy_policy = DeployCompatPolicy(
        policy=policy_nn,
        normalizer=normalizer,
        obs_indices=obs_indices,
        action_joint_ids=action_joint_ids,
        deploy_obs_dim=deploy_obs_dim,
        full_action_dim=full_action_dim,
    )
    deploy_policy.to("cpu")
    deploy_policy.eval()
    os.makedirs(export_model_dir, exist_ok=True)
    torch.onnx.export(
        deploy_policy,
        torch.zeros(1, deploy_obs_dim),
        os.path.join(export_model_dir, "policy.onnx"),
        export_params=True,
        opset_version=18,
        input_names=["obs"],
        output_names=["actions"],
        dynamic_axes={},
    )
    print(f"[INFO] Exported deploy-compatible 29DOF ONNX to: {os.path.join(export_model_dir, 'policy.onnx')}")
    print(f"[INFO] Wrapped {len(action_joint_ids)} policy actions into {full_action_dim} deploy actions.")

    dt = env.unwrapped.step_dt
    obs = env.get_observations()
    timestep = 0
    while simulation_app.is_running():
        start_time = time.time()
        with torch.inference_mode():
            actions = policy(obs)
            obs, _, _, _ = env.step(actions)
        if args_cli.video:
            timestep += 1
            if timestep == args_cli.video_length:
                break
        sleep_time = dt - (time.time() - start_time)
        if args_cli.real_time and sleep_time > 0:
            time.sleep(sleep_time)

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
