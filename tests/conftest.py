"""Shared test setup.

The test tracer provider is installed here, when pytest loads this file and
before any test module imports the app. OpenTelemetry allows one global
provider per process; if the app's import got there first (an
OTEL_EXPORTER_OTLP_ENDPOINT in the environment or .env), spans would go to that
exporter and the tests would see none.
"""

import os
import sys
from pathlib import Path

import pytest
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "api"))

from incidentiq import telemetry  # noqa: E402

SPANS = InMemorySpanExporter()
telemetry.setup_tracing("incidentiq-test", exporter=SPANS)


def database_missing(reason: str) -> None:
    """Skip when the database is not set up, or fail when INCIDENTIQ_REQUIRE_DB=1
    says it must exist, so a CI run cannot pass on skipped database tests."""
    if os.environ.get("INCIDENTIQ_REQUIRE_DB"):
        pytest.fail(f"required database setup is missing: {reason}")
    pytest.skip(reason)
