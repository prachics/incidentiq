"""Diagnostic and remediation tools.

Importing this package registers every tool, so `registry` is populated as a
side effect of import. That is why the imports below look unused.
"""

from incidentiq.tools import actions, diagnostics  # noqa: F401  (registration)
from incidentiq.tools.actions import execute_action
from incidentiq.tools.base import (
    FailureInjector,
    FailureMode,
    Tool,
    ToolError,
    ToolResult,
    ToolStatus,
    execute_tool,
    registry,
)

__all__ = [
    "Tool", "ToolError", "ToolResult", "ToolStatus", "FailureMode",
    "FailureInjector", "execute_tool", "execute_action", "registry",
]
