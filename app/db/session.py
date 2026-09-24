"""DB 연결."""
from __future__ import annotations

from contextlib import contextmanager

import psycopg

from app.config import settings


@contextmanager
def connect():
    with psycopg.connect(host=settings.pg_host, port=settings.pg_port, dbname=settings.pg_db,
                          user=settings.pg_user, password=settings.pg_password, connect_timeout=5) as conn:
        yield conn
