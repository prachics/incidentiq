"""MCP server: the read-only diagnostic tools, for any MCP client.

Write tools are neither listed nor callable, and every call runs on a read-only
database session (docs/DECISIONS.md #22). Failure injection is off: it exists to
exercise the agent's recovery in evaluations.

    python -m incidentiq.mcp_server          # stdio, which is what MCP clients launch
"""

from __future__ import annotations

import json
from typing import Any

import anyio
import psycopg
from mcp import types
from mcp.server.lowlevel import Server
from mcp.server.stdio import stdio_server

from incidentiq.config import Settings, get_settings
from incidentiq.rag.embeddings import get_embedding_provider
from incidentiq.telemetry import setup_tracing
from incidentiq.tools import FailureInjector, Tool, ToolStatus, execute_tool, registry

SERVER_NAME = "incidentiq"
# The failure injector keys its decisions on an investigation id; MCP calls have none.
_CALLER = "mcp"


def exposed_tools() -> dict[str, Tool]:
    """The tools an MCP client may see and call: the read-only ones."""
    return {tool.name: tool for tool in registry.read_only()}


def _listing(tool: Tool) -> types.Tool:
    return types.Tool(
        name=tool.name,
        description=tool.description,
        input_schema=tool.args_model.model_json_schema(),
        annotations=types.ToolAnnotations(read_only_hint=True, destructive_hint=False,
                                          idempotent_hint=True, open_world_hint=False),
    )


def _error(message: str) -> types.CallToolResult:
    return types.CallToolResult(content=[types.TextContent(type="text", text=message)],
                                is_error=True)


def _connect(settings: Settings) -> psycopg.Connection:
    from pgvector.psycopg import register_vector

    # default_transaction_read_only applies to autocommit statements too, unlike
    # psycopg's Connection.read_only, which only covers transactions it opens.
    conn = psycopg.connect(settings.database_url, autocommit=True, connect_timeout=3,
                           options="-c default_transaction_read_only=on")
    register_vector(conn)
    return conn


def run_tool(name: str, arguments: dict[str, Any] | None,
             settings: Settings | None = None) -> types.CallToolResult:
    """One MCP tool call, synchronously: what the async handler runs in a thread."""
    settings = settings or get_settings()
    tool = exposed_tools().get(name)
    if tool is None:
        if registry.get(name) is not None:
            return _error(f"{name} changes infrastructure; it runs only after a person "
                          "approves it in IncidentIQ and is not available over MCP")
        return _error(f"unknown tool {name!r}")

    injector = FailureInjector(settings.model_copy(update={"failure_injection_enabled": False}))
    try:
        conn = _connect(settings)
    except psycopg.OperationalError as exc:
        return _error(f"database unavailable: {exc}")
    with conn:
        result = execute_tool(conn, tool, arguments or {}, investigation_id=_CALLER,
                              injector=injector,
                              embeddings=lambda: get_embedding_provider(settings))
    if result.status is not ToolStatus.SUCCESS:
        return _error(f"{name} failed ({result.status.value}): {result.error}")
    return types.CallToolResult(
        content=[types.TextContent(type="text", text=json.dumps(result.data, default=str))],
        structured_content=json.loads(json.dumps(result.data, default=str)),
    )


async def _list_tools(_ctx: Any, _params: Any) -> types.ListToolsResult:
    return types.ListToolsResult(tools=[_listing(t) for t in exposed_tools().values()])


async def _call_tool(_ctx: Any, params: types.CallToolRequestParams) -> types.CallToolResult:
    # The tools are synchronous psycopg code; a thread keeps them off the event loop.
    return await anyio.to_thread.run_sync(run_tool, params.name, params.arguments)


def build_server() -> Server:
    return Server(SERVER_NAME, version="0.1.0",
                  instructions="Read-only diagnostics over IncidentIQ's services: logs, "
                               "metrics, deploys, dependencies and past incidents.",
                  on_list_tools=_list_tools, on_call_tool=_call_tool)


async def _serve_stdio() -> None:
    server = build_server()
    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream, server.create_initialization_options())


def main() -> None:
    settings = get_settings()
    setup_tracing("incidentiq-mcp", settings.otel_exporter_otlp_endpoint,
                  headers=settings.otel_exporter_otlp_headers)
    anyio.run(_serve_stdio)


if __name__ == "__main__":
    main()
