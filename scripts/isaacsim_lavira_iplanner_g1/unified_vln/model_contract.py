from __future__ import annotations

"""Read-only import of the already working schema-v2 model contract.

The new navigation stack deliberately reuses these exact dataclasses instead of
forking their validation rules.  The legacy directory remains untouched.
"""

from pathlib import Path
import sys


_SCRIPTS_DIR = Path(__file__).resolve().parents[2]
_LEGACY_DIR = _SCRIPTS_DIR / "isaacsim_goal_tracking"
if str(_LEGACY_DIR) not in sys.path:
    sys.path.insert(0, str(_LEGACY_DIR))

from goal_tracking.lavira_protocol import (  # noqa: E402,F401
    LAVIRA_ACTIONS,
    LAVIRA_DIRECTIONS,
    LAVIRA_MAX_HISTORY_IMAGE_WAYPOINTS,
    LAVIRA_REQUEST_TYPE,
    LAVIRA_RESPONSE_TYPE,
    LAVIRA_SCHEMA_VERSION,
    NavigationDecisionRequest,
    NavigationDecisionResponse,
    NavigationHistoryEntry,
)

__all__ = [
    "LAVIRA_ACTIONS",
    "LAVIRA_DIRECTIONS",
    "LAVIRA_MAX_HISTORY_IMAGE_WAYPOINTS",
    "LAVIRA_REQUEST_TYPE",
    "LAVIRA_RESPONSE_TYPE",
    "LAVIRA_SCHEMA_VERSION",
    "NavigationDecisionRequest",
    "NavigationDecisionResponse",
    "NavigationHistoryEntry",
]
