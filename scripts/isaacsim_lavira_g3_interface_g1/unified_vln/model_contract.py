from __future__ import annotations

"""只读复用已经运行稳定的 schema-v2 模型通信协议。

新的导航代码直接导入旧目录中的数据类和校验规则，而不是复制一份再维护。
这样客户端与已经部署的模型服务端始终使用完全相同的字段定义，旧目录本身不变。
"""

from pathlib import Path
import sys


# 找到 scripts 目录，再把旧协议包的父目录加入模块搜索路径。
# 这是运行时修改 ``sys.path``，编辑器的静态分析器可能仍会显示黄色波浪线。
_SCRIPTS_DIR = Path(__file__).resolve().parents[2]
_LEGACY_DIR = _SCRIPTS_DIR / "isaacsim_goal_tracking"
if str(_LEGACY_DIR) not in sys.path:
    sys.path.insert(0, str(_LEGACY_DIR))

# 由于必须先设置搜索路径，导入不在文件最顶部；noqa 用于忽略相应代码风格告警。
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

# 明确声明这个兼容层允许其他模块导入的协议常量和数据类型。
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
