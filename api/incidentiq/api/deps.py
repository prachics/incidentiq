"""Shared request dependencies: a database connection and the worker."""

from __future__ import annotations

import threading
from collections.abc import Iterator
from typing import Annotated

import psycopg
from fastapi import Depends, Request
from psycopg.rows import dict_row

from incidentiq.api.worker import Worker
from incidentiq.config import get_settings


def db() -> Iterator[psycopg.Connection]:
    with psycopg.connect(get_settings().database_url, autocommit=True,
                         connect_timeout=3, row_factory=dict_row) as conn:
        yield conn


_worker_lock = threading.Lock()


def worker(request: Request) -> Worker:
    """The app's single worker. Created in the lifespan hook; the locked fallback
    covers clients that skip the lifespan, without ever making two."""
    current = getattr(request.app.state, "worker", None)
    if current is None:
        with _worker_lock:
            current = getattr(request.app.state, "worker", None)
            if current is None:
                current = request.app.state.worker = Worker()
    return current


DB = Annotated[psycopg.Connection, Depends(db)]
Jobs = Annotated[Worker, Depends(worker)]
