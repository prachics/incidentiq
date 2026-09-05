#!/usr/bin/env python3
"""Apply SQL migrations in order, exactly once each.

Why a 30-line script instead of Alembic: the schema here is the teaching
surface of the project. Hand-written SQL keeps the pgvector index definitions,
the generated tsvector column, and the append-only audit trigger visible and
reviewable. Alembic's autogenerate would hide all three behind Python DSL calls
it does not model well anyway. See docs/DECISIONS.md #3.

Usage:
    python scripts/migrate.py            # apply pending migrations
    python scripts/migrate.py --status   # show what is applied
    python scripts/migrate.py --reset    # DROP the schema, then re-apply
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

import psycopg

REPO_ROOT = Path(__file__).resolve().parent.parent
MIGRATIONS_DIR = REPO_ROOT / "db" / "migrations"

# Bootstrap table: tracks which migrations have run.
LEDGER_DDL = """
CREATE TABLE IF NOT EXISTS schema_migrations (
    filename    TEXT PRIMARY KEY,
    checksum    TEXT        NOT NULL,
    applied_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);
"""


def _database_url() -> str:
    sys.path.insert(0, str(REPO_ROOT / "api"))
    from incidentiq.config import get_settings

    return get_settings().database_url


def _migration_files() -> list[Path]:
    return sorted(MIGRATIONS_DIR.glob("*.sql"))


def _checksum(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


def status(conn: psycopg.Connection) -> None:
    conn.execute(LEDGER_DDL)
    applied = {
        row[0]: row[1]
        for row in conn.execute("SELECT filename, checksum FROM schema_migrations").fetchall()
    }
    for path in _migration_files():
        current = _checksum(path)
        if path.name not in applied:
            mark, note = "pending", ""
        elif applied[path.name] != current:
            mark, note = "MODIFIED", "  <- file changed since it was applied"
        else:
            mark, note = "applied", ""
        print(f"  [{mark:>8}] {path.name}{note}")


def migrate(conn: psycopg.Connection) -> int:
    conn.execute(LEDGER_DDL)
    applied = {
        row[0] for row in conn.execute("SELECT filename FROM schema_migrations").fetchall()
    }
    count = 0
    for path in _migration_files():
        if path.name in applied:
            continue
        print(f"  applying {path.name} ...", end=" ", flush=True)
        # Each migration runs in its own transaction: a failure half-way
        # through leaves the ledger and the schema consistent with each other.
        with conn.transaction():
            conn.execute(path.read_text())
            conn.execute(
                "INSERT INTO schema_migrations (filename, checksum) VALUES (%s, %s)",
                (path.name, _checksum(path)),
            )
        print("ok")
        count += 1
    return count


def reset(conn: psycopg.Connection) -> None:
    print("  dropping schema public ...", end=" ", flush=True)
    with conn.transaction():
        conn.execute("DROP SCHEMA public CASCADE")
        conn.execute("CREATE SCHEMA public")
    print("ok")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--status", action="store_true", help="show migration status and exit")
    parser.add_argument("--reset", action="store_true", help="drop everything, then re-apply")
    args = parser.parse_args()

    url = _database_url()
    # Redact the password when echoing the target.
    shown = url.split("@")[-1] if "@" in url else url
    print(f"database: {shown}")

    try:
        conn = psycopg.connect(url, autocommit=True)
    except psycopg.OperationalError as exc:
        print(f"\ncannot reach Postgres: {exc}", file=sys.stderr)
        print("is the container up?  docker compose up -d postgres", file=sys.stderr)
        return 1

    with conn:
        if args.status:
            status(conn)
            return 0
        if args.reset:
            reset(conn)
        applied = migrate(conn)
        print(f"\n{applied} migration(s) applied." if applied else "\nalready up to date.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
