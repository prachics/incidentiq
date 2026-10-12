"""Shared test helpers."""

import os

import pytest


def database_missing(reason: str) -> None:
    """Skip when the database is not set up, or fail when INCIDENTIQ_REQUIRE_DB=1
    says it must exist, so a CI run cannot pass on skipped database tests."""
    if os.environ.get("INCIDENTIQ_REQUIRE_DB"):
        pytest.fail(f"required database setup is missing: {reason}")
    pytest.skip(reason)
