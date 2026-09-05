"""Tool definitions, the registry, and failure injection.

Every tool declares a Pydantic argument model. That model does three jobs at
once: it validates what the LLM sends, it generates the JSON Schema the LLM is
shown, and it documents the tool. Keeping them as one artifact means the schema
the model sees and the validation the code performs cannot drift apart - a class
of bug that is otherwise very hard to notice, because the symptom is the model
"being bad at tool calling".

Failure injection is a feature here, not a test fixture
-------------------------------------------------------
`FAILURE_INJECTION_ENABLED=true` makes tools fail at a configured rate, in three
distinct ways, because they demand different handling:

  timeout    - no result at all. Retry is appropriate.
  malformed  - a result that does not match the schema. Retry may help; the
               agent must not treat the garbage as data.
  partial    - a *valid* result that is quietly incomplete. This is the
               dangerous one: nothing raises, and an agent that does not notice
               the truncation reasons confidently from missing evidence.

Injection is seeded, so a given investigation fails the same way on every run.
Without that, an eval scenario "with injected tool failures" would score
differently each time and the number would be meaningless.
"""

from __future__ import annotations

import hashlib
import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import StrEnum
from typing import Any, ClassVar

import psycopg
from pydantic import BaseModel, ValidationError

from incidentiq.config import Settings, get_settings

log = logging.getLogger(__name__)


class ToolStatus(StrEnum):
    SUCCESS = "success"
    FAILED = "failed"
    TIMEOUT = "timeout"
    MALFORMED = "malformed"
    PARTIAL = "partial"


class FailureMode(StrEnum):
    TIMEOUT = "timeout"
    MALFORMED = "malformed"
    PARTIAL = "partial"


class ToolError(Exception):
    """Raised by a tool when it cannot produce a result."""

    def __init__(self, message: str, status: ToolStatus = ToolStatus.FAILED,
                 injected: bool = False):
        super().__init__(message)
        self.status = status
        self.injected = injected


@dataclass
class ToolResult:
    """What one tool invocation produced."""
    tool_name: str
    status: ToolStatus
    arguments: dict[str, Any]
    data: dict[str, Any] | None = None
    error: str | None = None
    latency_ms: int = 0
    attempt: int = 1
    injected_failure: bool = False
    truncated: bool = False   # set on PARTIAL so the agent can see the gap

    @property
    def ok(self) -> bool:
        return self.status in (ToolStatus.SUCCESS, ToolStatus.PARTIAL)

    def for_prompt(self) -> str:
        """How this result is rendered into the agent's context."""
        import json

        if self.status == ToolStatus.SUCCESS:
            return f"{self.tool_name} -> {json.dumps(self.data, default=str)}"
        if self.status == ToolStatus.PARTIAL:
            return (
                f"{self.tool_name} -> PARTIAL RESULT (incomplete data, some records "
                f"were not returned): {json.dumps(self.data, default=str)}"
            )
        return f"{self.tool_name} -> FAILED ({self.status.value}): {self.error}"


class Tool(ABC):
    """One callable capability."""

    name: ClassVar[str]
    description: ClassVar[str]
    args_model: ClassVar[type[BaseModel]]
    requires_approval: ClassVar[bool] = False

    @abstractmethod
    def run(self, conn: psycopg.Connection, args: BaseModel) -> dict[str, Any]:
        """Execute. Raise ToolError on failure. Return a JSON-serialisable dict."""

    @classmethod
    def schema(cls) -> dict[str, Any]:
        """JSON Schema in the shape the LLM APIs expect."""
        return {
            "name": cls.name,
            "description": cls.description,
            "input_schema": cls.args_model.model_json_schema(),
        }

    def validate_args(self, raw: dict[str, Any]) -> BaseModel:
        try:
            return self.args_model.model_validate(raw)
        except ValidationError as exc:
            # Surface the validation error back to the model: it is usually
            # recoverable, and a specific message ("service is required") gets a
            # correct retry far more often than a generic failure.
            raise ToolError(
                f"invalid arguments for {self.name}: {exc.errors()}",
                status=ToolStatus.MALFORMED,
            ) from exc


class FailureInjector:
    """Decides, deterministically, whether a given call should fail.

    Keyed on (investigation_id, tool_name, attempt) hashed with the configured
    seed, so the same investigation fails identically on every run - and a retry
    is a *different* key, so retries can succeed. A purely random injector would
    make retry behaviour untestable.
    """

    def __init__(self, settings: Settings | None = None):
        s = settings or get_settings()
        self.enabled = s.failure_injection_enabled
        self.rate = s.failure_injection_rate
        self.seed = s.failure_injection_seed

    def _roll(self, investigation_id: str, tool_name: str, attempt: int) -> float:
        key = f"{self.seed}:{investigation_id}:{tool_name}:{attempt}".encode()
        digest = hashlib.sha256(key).digest()
        return int.from_bytes(digest[:8], "big") / float(1 << 64)

    def decide(self, investigation_id: str, tool_name: str,
               attempt: int) -> FailureMode | None:
        if not self.enabled or self.rate <= 0:
            return None
        roll = self._roll(investigation_id, tool_name, attempt)
        if roll >= self.rate:
            return None
        # Which failure, chosen from a second, independent draw.
        which = self._roll(f"{investigation_id}#mode", tool_name, attempt)
        if which < 0.45:
            return FailureMode.TIMEOUT
        if which < 0.75:
            return FailureMode.MALFORMED
        return FailureMode.PARTIAL


class ToolRegistry:
    """Name -> tool, with the read/write split made explicit."""

    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> Tool:
        if tool.name in self._tools:
            raise ValueError(f"tool {tool.name!r} is already registered")
        self._tools[tool.name] = tool
        return tool

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    def all(self) -> list[Tool]:
        return list(self._tools.values())

    def read_only(self) -> list[Tool]:
        return [t for t in self._tools.values() if not t.requires_approval]

    def write(self) -> list[Tool]:
        return [t for t in self._tools.values() if t.requires_approval]

    def schemas(self, *, read_only: bool = False) -> list[dict]:
        tools = self.read_only() if read_only else self.all()
        return [t.schema() for t in tools]


registry = ToolRegistry()


def execute_tool(
    conn: psycopg.Connection,
    tool: Tool,
    raw_args: dict[str, Any],
    *,
    investigation_id: str,
    attempt: int = 1,
    injector: FailureInjector | None = None,
) -> ToolResult:
    """Run one tool once, with failure injection and timing. No retries here -
    retry policy belongs to the caller, which is what makes it testable."""
    injector = injector or FailureInjector()
    started = time.perf_counter()

    def elapsed() -> int:
        return int((time.perf_counter() - started) * 1000)

    injected = injector.decide(investigation_id, tool.name, attempt)
    if injected is FailureMode.TIMEOUT:
        return ToolResult(
            tool_name=tool.name, status=ToolStatus.TIMEOUT, arguments=raw_args,
            error="upstream did not respond within 30000ms",
            latency_ms=30000, attempt=attempt, injected_failure=True,
        )
    if injected is FailureMode.MALFORMED:
        return ToolResult(
            tool_name=tool.name, status=ToolStatus.MALFORMED, arguments=raw_args,
            error="upstream returned a response that could not be parsed",
            latency_ms=elapsed(), attempt=attempt, injected_failure=True,
        )

    try:
        args = tool.validate_args(raw_args)
        data = tool.run(conn, args)
    except ToolError as exc:
        return ToolResult(
            tool_name=tool.name, status=exc.status, arguments=raw_args,
            error=str(exc), latency_ms=elapsed(), attempt=attempt,
        )
    except Exception as exc:  # noqa: BLE001 - a tool bug must not kill the graph
        log.exception("tool %s raised", tool.name)
        return ToolResult(
            tool_name=tool.name, status=ToolStatus.FAILED, arguments=raw_args,
            error=f"{type(exc).__name__}: {exc}", latency_ms=elapsed(), attempt=attempt,
        )

    if injected is FailureMode.PARTIAL:
        data = _truncate(data)
        return ToolResult(
            tool_name=tool.name, status=ToolStatus.PARTIAL, arguments=raw_args,
            data=data, latency_ms=elapsed(), attempt=attempt,
            injected_failure=True, truncated=True,
        )

    return ToolResult(
        tool_name=tool.name, status=ToolStatus.SUCCESS, arguments=raw_args,
        data=data, latency_ms=elapsed(), attempt=attempt,
    )


def _truncate(data: dict[str, Any]) -> dict[str, Any]:
    """Halve the longest list in the result and flag it.

    The flag matters. A partial result that does not announce itself is
    indistinguishable from a complete one, and the agent would reason
    confidently from evidence it does not have. Real systems signal this too -
    a truncation marker, a `has_more` field - so the agent must learn to look.
    """
    out = dict(data)
    lists = [(k, v) for k, v in out.items() if isinstance(v, list) and len(v) > 1]
    if lists:
        key, value = max(lists, key=lambda kv: len(kv[1]))
        out[key] = value[: len(value) // 2]
        out["_truncated"] = True
        out["_truncated_field"] = key
        out["_note"] = (
            f"Result is incomplete: {key} was truncated. Conclusions drawn from "
            "this data may be based on partial evidence."
        )
    return out
