"""Register the robust Unitree-style G1 task, then run Isaac Lab's RSL-RL trainer."""

from pathlib import Path
import runpy
import sys

PROJECT_DIR = Path(__file__).resolve().parents[1]
PROJECTS_DIR = PROJECT_DIR.parent

sys.path.insert(0, str(PROJECT_DIR / "source" / "unitree_g1_isaaclab"))
sys.path.insert(0, str(PROJECTS_DIR / "unitree_rl_lab" / "source" / "unitree_rl_lab"))

import unitree_g1_isaaclab.tasks.walk_flat_unitree_style  # noqa: E402,F401

ISAACLAB_DIR = PROJECTS_DIR / "IsaacLab"
RSL_RL_SCRIPT_DIR = ISAACLAB_DIR / "scripts" / "reinforcement_learning" / "rsl_rl"

sys.path.insert(0, str(RSL_RL_SCRIPT_DIR))
runpy.run_path(str(RSL_RL_SCRIPT_DIR / "train.py"), run_name="__main__")
