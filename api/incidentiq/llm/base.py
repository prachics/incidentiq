"""The LLM interface the agent programs against.

Three implementations: Anthropic (paid, best reasoning), Ollama (local, free),
and a deterministic stub. The stub is not a second-class citizen - it is what
makes the graph testable. Every structural property of the agent (does the loop
terminate, does state survive a crash, does a rejected approval feed back into
the scratchpad) can be verified without an LLM in the loop at all, which means
those tests are fast, free, and never flaky.

Design notes
------------
The interface is deliberately narrow: `complete` for text and `complete_tool`
for structured tool selection. Anything richer would leak provider-specific
concepts into the graph.

Tool calling is normalised across providers. Anthropic returns `tool_use`
content blocks natively; Ollama's models vary in how reliably they emit tool
calls, so the Ollama provider falls back to parsing JSON out of the text. That
difference is confined here rather than being visible to the agent.

Token accounting lives on the response because cost per investigation is a
reported metric, and a number nobody collects is a number nobody reports.
"""

from __future__ import annotations

import json
import logging
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

log = logging.getLogger(__name__)


@dataclass
class ToolCall:
    """A tool the model wants invoked. Provider-neutral."""
    name: str
    arguments: dict[str, Any]
    call_id: str = ""


@dataclass
class LLMResponse:
    text: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    input_tokens: int = 0
    output_tokens: int = 0
    model: str = ""
    stop_reason: str = ""

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens


class LLMProvider(ABC):
    """Text in, text or tool calls out."""

    @property
    @abstractmethod
    def model_name(self) -> str:
        ...

    @abstractmethod
    def complete(
        self,
        system: str,
        messages: list[dict[str, str]],
        *,
        max_tokens: int = 2048,
        temperature: float | None = None,
    ) -> LLMResponse:
        """Free-text completion."""

    def complete_json(
        self,
        system: str,
        messages: list[dict[str, str]],
        *,
        max_tokens: int = 2048,
        schema: dict[str, Any] | None = None,
    ) -> LLMResponse:
        """Completion that must return JSON.

        Separate from `complete` because some providers can *enforce* it at
        decode time rather than merely asking for it in the prompt. Measured on
        this project: a local 14B model returned valid JSON reliably at small
        context and degraded badly as the prompt grew, until five of seven
        planning calls were unparseable. Constrained decoding removed that
        failure mode entirely. The default implementation just delegates, so
        providers that cannot constrain still work.

        When `schema` is given, providers that support it constrain the output
        to that exact shape. This matters more than it sounds: JSON mode alone
        guarantees syntax, not structure, and a model that returns valid JSON
        with invented field names fails *silently* - every field reads back as
        None and the node produces an empty result having reasoned correctly.
        """
        return self.complete(system, messages, max_tokens=max_tokens)

    @abstractmethod
    def complete_tool(
        self,
        system: str,
        messages: list[dict[str, str]],
        tools: list[dict[str, Any]],
        *,
        max_tokens: int = 2048,
    ) -> LLMResponse:
        """Completion where the model may request tool calls.

        `tools` is a list of {name, description, input_schema} - JSON Schema in
        the shape the Anthropic API expects. Other providers adapt from it.
        """


def extract_json(text: str) -> dict | None:
    """Best-effort JSON extraction from a text response.

    Needed because smaller local models frequently wrap JSON in prose or a
    fenced code block despite instructions not to. Being tolerant here is the
    difference between a local model being usable and not.
    """
    if not text:
        return None

    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    candidates = [fenced.group(1)] if fenced else []

    # Balanced-brace scan for the first complete object in the text.
    depth, start = 0, None
    for i, ch in enumerate(text):
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0 and start is not None:
                candidates.append(text[start:i + 1])
                start = None

    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            continue
    return None
