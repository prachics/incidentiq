"""MCP server: what a client can see and call, and what it cannot.

The property that matters most is negative: no write tool is reachable over MCP,
whatever the client sends. The tests check it at both guards - the tool list and
the read-only database session - and that a write tool added later is excluded
without anyone remembering to exclude it.
"""

import os
import sys
from pathlib import Path
from typing import Any, ClassVar

import psycopg
import pytest
from pydantic import BaseModel

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "api"))

from conftest import database_missing  # noqa: E402
from incidentiq import mcp_server  # noqa: E402
from incidentiq.config import get_settings  # noqa: E402
from incidentiq.tools import Tool, registry  # noqa: E402
from incidentiq.tools.base import ToolRegistry  # noqa: E402

WRITE_TOOLS = {"restart_service", "scale_service", "rollback_deploy"}
READ_TOOLS = {"get_service_logs", "get_metrics", "get_recent_deploys",
              "get_service_dependencies", "search_similar_incidents"}


@pytest.fixture(scope="module")
def seeded():
    try:
        with psycopg.connect(get_settings().database_url, connect_timeout=3) as conn:
            if conn.execute("SELECT count(*) FROM services").fetchone()[0] == 0:
                database_missing("no seed data")
    except psycopg.OperationalError:
        database_missing("Postgres not reachable")


class _Args(BaseModel):
    service: str = "checkout-service"


class _Recorder(Tool):
    """A tool that records whether it ran, for proving a call never got there."""
    name: ClassVar[str] = "recorder"
    description: ClassVar[str] = "test tool"
    args_model: ClassVar[type[BaseModel]] = _Args
    calls: ClassVar[list[Any]] = []

    def run(self, conn, args):
        self.calls.append(args)
        return {"ran": True}


@pytest.fixture
def scratch_registry(monkeypatch):
    """The real tools plus whatever a test registers, without touching the real registry."""
    copy = ToolRegistry()
    for tool in registry.all():
        copy.register(tool)
    monkeypatch.setattr(mcp_server, "registry", copy)
    _Recorder.calls = []
    return copy


# ── What is exposed ─────────────────────────────────────────
def test_exactly_the_read_only_tools_are_exposed():
    assert set(mcp_server.exposed_tools()) == READ_TOOLS
    assert {t.name for t in registry.write()} == WRITE_TOOLS


def test_a_write_tool_added_later_is_not_exposed(scratch_registry):
    class NewWrite(_Recorder):
        name = "drain_node"
        requires_approval = True

    scratch_registry.register(NewWrite())
    assert "drain_node" not in mcp_server.exposed_tools()


async def test_the_listing_carries_each_tools_own_schema_and_marks_it_read_only():
    listed = (await mcp_server._list_tools(None, None)).tools
    assert {t.name for t in listed} == READ_TOOLS
    for entry in listed:
        tool = registry.get(entry.name)
        assert entry.input_schema == tool.args_model.model_json_schema()
        assert entry.description == tool.description
        assert entry.annotations.read_only_hint is True
        assert entry.annotations.destructive_hint is False


# ── What cannot be called ───────────────────────────────────
@pytest.mark.parametrize("name", sorted(WRITE_TOOLS))
def test_calling_a_write_tool_is_refused_before_it_runs(name, monkeypatch):
    ran = []
    monkeypatch.setattr(registry.get(name), "run", lambda *a, **k: ran.append(a))
    monkeypatch.setattr(mcp_server, "execute_tool", lambda *a, **k: ran.append(a))
    monkeypatch.setattr(mcp_server, "_connect", lambda *a, **k: ran.append(a))

    result = mcp_server.run_tool(name, {"service": "checkout-service"})

    assert result.is_error
    assert "approves" in result.content[0].text
    assert ran == []


def test_an_unknown_tool_is_an_error():
    result = mcp_server.run_tool("drop_database", {})
    assert result.is_error
    assert "unknown tool" in result.content[0].text


def test_a_diagnostic_tool_that_writes_is_refused_by_the_database(seeded, scratch_registry):
    class Sneaky(_Recorder):
        name = "sneaky_read"

        def run(self, conn, args):
            conn.execute("UPDATE services SET replica_count = replica_count WHERE name = %s",
                         (args.service,))
            return {"ran": True}

    scratch_registry.register(Sneaky())
    result = mcp_server.run_tool("sneaky_read", {})
    assert result.is_error
    assert "read-only transaction" in result.content[0].text


# ── Calls that work ─────────────────────────────────────────
def test_a_diagnostic_call_returns_the_tools_data(seeded):
    result = mcp_server.run_tool("get_service_dependencies", {"service": "checkout-service"})
    assert not result.is_error
    assert result.structured_content["service"] == "checkout-service"
    assert "depends_on" in result.structured_content


def test_failure_injection_is_off_even_when_settings_enable_it(seeded, monkeypatch):
    on = get_settings().model_copy(update={"failure_injection_enabled": True,
                                           "failure_injection_rate": 1.0})
    for _ in range(3):
        result = mcp_server.run_tool("get_recent_deploys", {"service": "checkout-service"}, on)
        assert not result.is_error


def test_a_wrong_service_name_comes_back_with_the_suggestion(seeded):
    result = mcp_server.run_tool("get_metrics", {"service": "checkout-servic"})
    assert result.is_error
    assert "Did you mean: checkout-service" in result.content[0].text


def test_invalid_arguments_are_an_error_not_a_crash(seeded):
    result = mcp_server.run_tool("get_service_logs", {"service": "checkout-service",
                                                      "time_window": "1y"})
    assert result.is_error
    assert "time_window" in result.content[0].text


def test_no_database_is_an_error_not_a_crash():
    unreachable = get_settings().model_copy(
        update={"database_url": "postgresql://nobody@127.0.0.1:1/none"})
    result = mcp_server.run_tool("get_metrics", {"service": "checkout-service"}, unreachable)
    assert result.is_error
    assert "database unavailable" in result.content[0].text


# ── Over the protocol ───────────────────────────────────────
async def test_a_client_lists_and_calls_tools_in_process(seeded):
    from mcp import Client

    async with Client(mcp_server.build_server()) as client:
        names = {t.name for t in (await client.list_tools()).tools}
        assert names == READ_TOOLS
        ok = await client.call_tool("get_recent_deploys",
                                    {"service": "checkout-service", "time_window": "7d"})
        assert not ok.is_error
        assert ok.structured_content["service"] == "checkout-service"
        refused = await client.call_tool("rollback_deploy", {"service": "checkout-service"})
        assert refused.is_error


async def test_the_stdio_server_answers_from_another_directory(seeded, tmp_path):
    """What an MCP client actually launches, from its own working directory, with only
    DATABASE_URL. The search loads the embedding model on first use; anything printed
    to stdout besides protocol messages, at startup or then, would break the session."""
    from mcp import Client, StdioServerParameters

    params = StdioServerParameters(
        command=sys.executable, args=["-m", "incidentiq.mcp_server"], cwd=str(tmp_path),
        env={"PATH": os.environ["PATH"], "PYTHONPATH": str(ROOT / "api"),
             "DATABASE_URL": get_settings().database_url},
    )
    async with Client(params) as client:
        assert {t.name for t in (await client.list_tools()).tools} == READ_TOOLS
        found = await client.call_tool("search_similar_incidents",
                                       {"query": "connection pool exhausted", "limit": 3})
        assert not found.is_error
        assert found.structured_content["result_count"] == 3
        assert all(r["doc_id"] for r in found.structured_content["results"])
