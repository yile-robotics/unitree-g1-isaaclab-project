from __future__ import annotations

"""Thin data adapter around Uni-LaViRA G1's original iPlanner client."""

import importlib.util
from pathlib import Path
import sys
from types import ModuleType
from typing import Sequence

import numpy as np

from .types import ViewFrame


def _load_uni_iplanner_client_class():
    """Load the original client file without importing Uni's heavy robot package."""

    projects_dir = Path(__file__).resolve().parents[4]
    uni_g1_dir = (
        projects_dir / "uni-lavira-code" / "real-world-code" / "unitree_g1"
    )
    config_path = uni_g1_dir / "config.py"
    client_path = uni_g1_dir / "robot" / "iplanner_client.py"
    if not config_path.is_file() or not client_path.is_file():
        raise FileNotFoundError(
            "Uni-LaViRA G1 iPlanner client source is unavailable under "
            f"{uni_g1_dir}"
        )

    config_spec = importlib.util.spec_from_file_location(
        "_uni_lavira_g1_config", config_path
    )
    client_spec = importlib.util.spec_from_file_location(
        "_uni_lavira_g1_iplanner_client", client_path
    )
    if config_spec is None or config_spec.loader is None:
        raise ImportError(f"Cannot load Uni-LaViRA config from {config_path}")
    if client_spec is None or client_spec.loader is None:
        raise ImportError(f"Cannot load Uni-LaViRA iPlanner client from {client_path}")

    config_module = importlib.util.module_from_spec(config_spec)
    config_spec.loader.exec_module(config_module)
    client_module = importlib.util.module_from_spec(client_spec)

    # Uni's original file imports ``Config`` as the top-level module ``config``.
    # Supply exactly that module only while executing the source file, then put
    # back any unrelated module that the host application had already imported.
    previous_config: ModuleType | None = sys.modules.get("config")
    previous_client: ModuleType | None = sys.modules.get(client_spec.name)
    sys.modules["config"] = config_module
    sys.modules[client_spec.name] = client_module
    try:
        client_spec.loader.exec_module(client_module)
    except Exception:
        if previous_client is None:
            sys.modules.pop(client_spec.name, None)
        else:
            sys.modules[client_spec.name] = previous_client
        raise
    finally:
        if previous_config is None:
            sys.modules.pop("config", None)
        else:
            sys.modules["config"] = previous_config
    return client_module.IPlannerRemoteClient


_UniIPlannerRemoteClient = _load_uni_iplanner_client_class()


class IPlannerClient:
    """Adapt RGB/metres ``ViewFrame`` data to Uni's original BGR/mm client."""

    def __init__(self, server_url: str, timeout_s: float = 5.0):
        # Uni hard-codes a five-second HTTP timeout.  Keep the existing argument
        # for runner CLI compatibility, but require the Uni value when supplied.
        if float(timeout_s) != 5.0:
            raise ValueError(
                "Exact Uni-LaViRA iPlanner mode requires timeout_s=5.0."
            )
        self._client = _UniIPlannerRemoteClient(server_url)

    @property
    def initialized(self) -> bool:
        return bool(self._client.initialized)

    def reset(self, intrinsic: Sequence[Sequence[float]]) -> bool:
        return bool(self._client.reset(intrinsic=intrinsic))

    @staticmethod
    def depth_metres_to_uni_millimetres(depth_m: np.ndarray) -> np.ndarray:
        """Quantize floating metres to the uint16 millimetres Uni expects."""

        return (np.asarray(depth_m) * 1000.0).astype(np.uint16)

    def get_plan(
        self,
        front_frame: ViewFrame,
        goal_local_xy: np.ndarray,
    ) -> tuple[np.ndarray | None, float | None]:
        import cv2

        rgb_bgr = cv2.cvtColor(
            np.ascontiguousarray(front_frame.rgb), cv2.COLOR_RGB2BGR
        )
        depth_mm = self.depth_metres_to_uni_millimetres(front_frame.depth_m)
        return self._client.get_plan(
            rgb_bgr,
            depth_mm,
            np.asarray(goal_local_xy).reshape(2),
        )
