"""Concrete LLM providers: Anthropic, Ollama, and a deterministic stub."""

from __future__ import annotations

import json
import logging
from functools import lru_cache
from typing import Any

from incidentiq.config import Settings, get_settings
from incidentiq.llm.base import LLMProvider, LLMResponse, ToolCall, extract_json

log = logging.getLogger(__name__)


class AnthropicProvider(LLMProvider):
    """Claude via the Anthropic API. Pay-per-token."""

    def __init__(self, api_key: str, model: str):
        import anthropic

        self._client = anthropic.Anthropic(api_key=api_key)
        self._model = model

    @property
    def model_name(self) -> str:
        return self._model

    def _call(self, system, messages, tools, max_tokens) -> LLMResponse:
        kwargs: dict[str, Any] = {
            "model": self._model,
            "max_tokens": max_tokens,
            "system": system,
            "messages": messages,
        }
        if tools:
            kwargs["tools"] = tools

        resp = self._client.messages.create(**kwargs)

        text_parts, calls = [], []
        for block in resp.content:
            if block.type == "text":
                text_parts.append(block.text)
            elif block.type == "tool_use":
                calls.append(ToolCall(name=block.name, arguments=dict(block.input),
                                      call_id=block.id))
        return LLMResponse(
            text="\n".join(text_parts),
            tool_calls=calls,
            input_tokens=resp.usage.input_tokens,
            output_tokens=resp.usage.output_tokens,
            model=self._model,
            stop_reason=resp.stop_reason or "",
        )

    def complete(self, system, messages, *, max_tokens=2048, temperature=None):
        return self._call(system, messages, None, max_tokens)

    def complete_tool(self, system, messages, tools, *, max_tokens=2048):
        return self._call(system, messages, tools, max_tokens)


class OllamaProvider(LLMProvider):
    """A local model served by Ollama. Free, offline, slower, weaker.

    Ollama exposes an OpenAI-shaped tool-calling API, but small models are
    inconsistent about using it - they often describe the call in prose instead
    of emitting it. So this provider tries the structured path first and falls
    back to parsing JSON out of the text. Without that fallback, local runs fail
    for reasons that have nothing to do with reasoning quality.
    """

    def __init__(self, base_url: str, model: str, timeout: float = 180.0):
        import httpx

        self._client = httpx.Client(base_url=base_url.rstrip("/"), timeout=timeout)
        self._model = model

    @property
    def model_name(self) -> str:
        return self._model

    @staticmethod
    def _to_ollama_tools(tools: list[dict]) -> list[dict]:
        return [
            {
                "type": "function",
                "function": {
                    "name": t["name"],
                    "description": t["description"],
                    "parameters": t["input_schema"],
                },
            }
            for t in tools
        ]

    def _post(self, system, messages, tools, max_tokens, json_mode: bool = False,
              schema: dict | None = None) -> dict:
        payload: dict[str, Any] = {
            "model": self._model,
            "messages": [{"role": "system", "content": system}, *messages],
            "stream": False,
            "options": {"num_predict": max_tokens, "temperature": 0.1},
        }
        if schema is not None:
            # A full JSON Schema constrains the *shape*, not just the syntax.
            # Ollama compiles it to a grammar and restricts the sampler to
            # tokens that keep the output conformant.
            payload["format"] = schema
        elif json_mode:
            payload["format"] = "json"
        if tools:
            payload["tools"] = self._to_ollama_tools(tools)
        resp = self._client.post("/api/chat", json=payload)
        resp.raise_for_status()
        return resp.json()

    def _parse(self, data: dict, expect_tools: bool) -> LLMResponse:
        message = data.get("message", {})
        text = message.get("content", "") or ""
        calls = [
            ToolCall(name=c["function"]["name"],
                     arguments=c["function"].get("arguments", {}) or {})
            for c in message.get("tool_calls", []) or []
        ]

        # Fallback: the model described a call in prose instead of emitting one.
        if expect_tools and not calls:
            parsed = extract_json(text)
            if parsed and "name" in parsed:
                calls.append(ToolCall(
                    name=parsed["name"],
                    arguments=parsed.get("arguments") or parsed.get("input") or {},
                ))
                log.debug("recovered a tool call from prose: %s", parsed["name"])

        return LLMResponse(
            text=text,
            tool_calls=calls,
            input_tokens=data.get("prompt_eval_count", 0),
            output_tokens=data.get("eval_count", 0),
            model=self._model,
            stop_reason=data.get("done_reason", ""),
        )

    def complete(self, system, messages, *, max_tokens=2048, temperature=None):
        return self._parse(self._post(system, messages, None, max_tokens), False)

    def complete_json(self, system, messages, *, max_tokens=2048, schema=None):
        return self._parse(
            self._post(system, messages, None, max_tokens, json_mode=True, schema=schema),
            False,
        )

    def complete_tool(self, system, messages, tools, *, max_tokens=2048):
        return self._parse(self._post(system, messages, tools, max_tokens), True)


class StubProvider(LLMProvider):
    """Deterministic, scripted, offline. Not a mock - a test double with rules.

    Structural properties of the graph - does the loop terminate, does state
    survive a crash, does a rejection re-enter planning - have nothing to do
    with model quality. Verifying them against a real model would make those
    tests slow, costly, and non-deterministic, so they would be run rarely and
    trusted less.

    Behaviour is driven by simple rules over the prompt so a scripted run still
    exercises the real control flow: it calls tools, reflects, and proposes,
    rather than short-circuiting to a canned answer.
    """

    def __init__(self, script: list[LLMResponse] | None = None):
        self._script = list(script or [])
        self.calls: list[dict] = []   # inspectable by tests

    @property
    def model_name(self) -> str:
        return "stub"

    def _record(self, system, messages, tools):
        self.calls.append({
            "system": system,
            "messages": messages,
            "tools": [t["name"] for t in (tools or [])],
        })

    def _next_scripted(self) -> LLMResponse | None:
        return self._script.pop(0) if self._script else None

    def complete(self, system, messages, *, max_tokens=2048, temperature=None):
        self._record(system, messages, None)
        scripted = self._next_scripted()
        if scripted:
            return scripted
        last = messages[-1]["content"] if messages else ""
        return LLMResponse(
            text=json.dumps({"summary": "stub summary", "echo": last[:80]}),
            input_tokens=len(str(messages)) // 4,
            output_tokens=16,
            model="stub",
            stop_reason="end_turn",
        )

    def complete_tool(self, system, messages, tools, *, max_tokens=2048):
        self._record(system, messages, tools)
        scripted = self._next_scripted()
        if scripted:
            return scripted
        # Default rule: call the first offered read-only tool once, then stop.
        names = [t["name"] for t in (tools or [])]
        if names and not any("call the tool" in str(m) for m in messages):
            return LLMResponse(
                tool_calls=[ToolCall(name=names[0], arguments={})],
                input_tokens=32, output_tokens=8, model="stub", stop_reason="tool_use",
            )
        return LLMResponse(text="done", input_tokens=32, output_tokens=4,
                           model="stub", stop_reason="end_turn")


@lru_cache(maxsize=3)
def _build(provider: str, model: str, key_or_url: str) -> LLMProvider:
    if provider == "anthropic":
        return AnthropicProvider(key_or_url, model)
    if provider == "ollama":
        return OllamaProvider(key_or_url, model)
    if provider == "stub":
        return StubProvider()
    raise ValueError(f"unknown llm provider {provider!r}")


def get_llm(settings: Settings | None = None) -> LLMProvider:
    s = settings or get_settings()
    s.require_llm_credentials()
    if s.llm_provider == "anthropic":
        return _build("anthropic", s.anthropic_model, s.anthropic_api_key)
    if s.llm_provider == "ollama":
        return _build("ollama", s.ollama_model, s.ollama_base_url)
    return StubProvider()
