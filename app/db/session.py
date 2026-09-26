"""DB 연결 풀.

요청마다 새 연결을 만들면 동시 사용자가 생기는 순간 연결 수립 비용이
지연으로 드러난다(PostgreSQL 은 연결당 프로세스를 띄운다). PoC 라도 여러
사람이 동시에 시연을 눌러 보므로 풀을 쓴다.

풀은 지연 생성한다. DB 가 내려가 있어도 프로세스는 떠야 하고, 그 상태를
`/ready` 가 보고해야 한다. 기동 시점에 연결을 요구하면 두 요구가 충돌한다.
"""
from __future__ import annotations

import atexit
import logging
import threading
from contextlib import contextmanager

from psycopg_pool import ConnectionPool

from app.config import settings

log = logging.getLogger(__name__)

_pool: ConnectionPool | None = None
_lock = threading.Lock()


def pool() -> ConnectionPool:
    global _pool
    if _pool is None:
        with _lock:
            if _pool is None:
                _pool = ConnectionPool(
                    conninfo=settings.dsn,
                    kwargs=settings.pg_kwargs,
                    min_size=settings.pg_pool_min,
                    max_size=settings.pg_pool_max,
                    timeout=float(settings.pg_connect_timeout),
                    max_idle=300.0,
                    # 유휴 중에 끊긴 연결을 요청에 물려 주지 않는다.
                    check=ConnectionPool.check_connection,
                    open=False,
                    name="specrag",
                )
                _pool.open(wait=False)
                log.info(
                    "DB 연결 풀 생성", extra={"min": settings.pg_pool_min,
                                              "max": settings.pg_pool_max}
                )
    return _pool


@contextmanager
def connect():
    """풀에서 연결을 빌린다. 블록을 정상 종료하면 커밋, 예외면 롤백된다."""
    with pool().connection() as conn:
        yield conn


def close_pool() -> None:
    global _pool
    with _lock:
        if _pool is not None:
            _pool.close()
            _pool = None


atexit.register(close_pool)
