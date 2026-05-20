"""Register local Unitree G1 tasks, then run Isaac Lab's RSL-RL trainer."""

from pathlib import Path
import runpy
import sys

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR / "source" / "unitree_g1_isaaclab"))
import unitree_g1_isaaclab.tasks  # noqa: E402,F401

ISAACLAB_DIR = Path(__file__).resolve().parents[2] / "IsaacLab"
RSL_RL_SCRIPT_DIR = ISAACLAB_DIR / "scripts" / "reinforcement_learning" / "rsl_rl"

sys.path.insert(0, str(RSL_RL_SCRIPT_DIR))
runpy.run_path(str(RSL_RL_SCRIPT_DIR / "train.py"), run_name="__main__")
