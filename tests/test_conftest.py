"""database_missing: a skip by default, a failure when CI says the database must exist."""

import pytest
from conftest import database_missing


def test_without_the_flag_a_missing_database_skips(monkeypatch):
    monkeypatch.delenv("INCIDENTIQ_REQUIRE_DB", raising=False)
    with pytest.raises(pytest.skip.Exception):
        database_missing("no database")


def test_with_the_flag_a_missing_database_fails(monkeypatch):
    monkeypatch.setenv("INCIDENTIQ_REQUIRE_DB", "1")
    with pytest.raises(pytest.fail.Exception, match="no database"):
        database_missing("no database")
