"""Response schemas for each reasoning node.

These exist because "return exactly this JSON shape" in a prompt is a request,
not a constraint. Measured on this project: with Ollama's `format: "json"` the
local model reliably returned *syntactically* valid JSON, and then, as the
prompt grew past ~3000 tokens, began returning valid JSON with an entirely
invented schema - `{"incident_updates": ...}` where `root_cause`,
`evidence_citations` and `remediation_tool` were expected. Parsing succeeded and
every field read back as None, so the failure was silent: the agent produced an
empty proposal having actually reasoned its way to the correct answer.

Passing the JSON Schema to the sampler makes the wrong shape unrepresentable
rather than merely discouraged. Ollama accepts a schema in `format`; the
Anthropic path uses the same models to validate rather than constrain, since
Claude follows shape instructions reliably.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

# Nullable types are avoided throughout. Two reasons, both practical:
# llama.cpp's grammar compiler handles `["string", "null"]` unevenly, and a
# field must be REQUIRED for the grammar to force the model to emit it - which
# means it needs a representable "absent" value. Empty string and empty object
# serve that purpose, and the nodes convert them back to None.


class IntakeOut(BaseModel):
    service: str = Field(default="", description="Exact service name, or empty if not named")
    related_services: list[str] = Field(default_factory=list)
    error_signature: str = ""
    time_window: str = "6h"
    severity: str = ""
    symptoms: list[str] = Field(default_factory=list)
    confidence: float = 0.0


class PlanOut(BaseModel):
    tool_name: str = Field(description="One of the available tool names")
    arguments: dict[str, Any] = Field(default_factory=dict)
    rationale: str = ""


class ReflectOut(BaseModel):
    sufficient: bool
    reasoning: str = ""
    missing: list[str] = Field(default_factory=list)


class ProposeOut(BaseModel):
    root_cause: str = Field(default="", description="What is broken and why")
    confidence: float = 0.0
    evidence_citations: list[str] = Field(default_factory=list)
    remediation: str = ""
    remediation_tool: str = Field(default="", description="Tool name, or empty for none")
    remediation_arguments: dict[str, Any] = Field(default_factory=dict)
    abstained: bool = False
    abstention_reason: str = ""


class SummarizeOut(BaseModel):
    summary: str = ""
    uncertainties: list[str] = Field(default_factory=list)


def schema_of(model: type[BaseModel]) -> dict[str, Any]:
    """JSON Schema for constrained decoding.

    `$defs`/`$ref` are inlined and unsupported keywords dropped, because
    llama.cpp's grammar compiler - which is what Ollama uses under the hood -
    rejects schemas it cannot convert and falls back to unconstrained output
    silently, which would reintroduce exactly the bug this prevents.
    """
    raw = model.model_json_schema()
    defs = raw.pop("$defs", {})

    def inline(node: Any) -> Any:
        if isinstance(node, dict):
            if "$ref" in node:
                name = node["$ref"].rsplit("/", 1)[-1]
                return inline(defs.get(name, {}))
            if "anyOf" in node:
                # Optional fields render as anyOf[T, null]; keep the concrete
                # branch, which the grammar compiler handles cleanly.
                branches = [b for b in node["anyOf"]
                            if not (isinstance(b, dict) and b.get("type") == "null")]
                merged = inline(branches[0]) if branches else {"type": "string"}
                for key in ("description", "default", "title"):
                    if key in node and key != "default":
                        merged.setdefault(key, node[key])
                return merged
            return {k: inline(v) for k, v in node.items()
                    if k not in ("$defs", "additionalProperties", "default")}
        if isinstance(node, list):
            return [inline(v) for v in node]
        return node

    schema = inline(raw)
    # Mark every property required. Pydantic omits fields that have defaults,
    # which leaves the grammar free to skip them - and it does: constrained to
    # the right field *names*, the model still dropped `root_cause` entirely
    # and the proposal came back empty. Required makes emitting them
    # unavoidable; the empty-string sentinels above give it a way to say
    # "nothing here" without violating the schema.
    if "properties" in schema:
        schema["required"] = list(schema["properties"])
    return schema
