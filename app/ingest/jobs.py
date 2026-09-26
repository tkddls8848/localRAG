"""색인 작업 큐.

787페이지 PDF 6건을 임베딩하는 데 노트북에서 수십 분이 걸린다. HTTP 요청
안에서 처리하면 프록시 타임아웃에 걸리고, 브라우저 탭을 닫으면 진행 상황을
알 수 없다. 그래서 업로드는 작업을 등록하고 즉시 응답하며, 진행 상황은
`GET /jobs/{id}` 로 확인한다.

큐를 Redis·Celery 로 만들지 않은 이유는 구성요소를 늘리기 때문이다
(architecture.md 1.1). 상태는 DB(`ingest_jobs`)에 있으므로 프로세스가 죽어도
무엇이 돌다 말았는지 남는다. 여러 인스턴스로 확장할 때 워커만 분리하면 된다.
"""
from __future__ import annotations

import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Callable

from app.config import settings
from app.db.session import connect
from app.observability import metrics, new_request_id, principal_id, request_id

log = logging.getLogger(__name__)

# 작업 함수는 진행 보고 콜백을 받는다. (stage, done, total)
JobFn = Callable[[Callable[[str, int, int], None]], dict]

_executor: ThreadPoolExecutor | None = None
_lock = threading.Lock()


def executor() -> ThreadPoolExecutor:
    global _executor
    if _executor is None:
        with _lock:
            if _executor is None:
                _executor = ThreadPoolExecutor(
                    max_workers=max(1, settings.ingest_workers),
                    thread_name_prefix="ingest",
                )
    return _executor


def create(kind: str, source: str, principal: str | None = None) -> int:
    with connect() as conn:
        row = conn.execute(
            """INSERT INTO ingest_jobs (kind, source, principal, status)
               VALUES (%s, %s, %s, 'queued') RETURNING id""",
            (kind, source, principal),
        ).fetchone()
        conn.commit()
    return row[0]


def _update(job_id: int, **fields) -> None:
    if not fields:
        return
    import json

    sets, values = [], []
    for key, value in fields.items():
        sets.append(f"{key} = %s")
        values.append(json.dumps(value, ensure_ascii=False, default=str)
                      if key == "stats" else value)
    values.append(job_id)
    with connect() as conn:
        conn.execute(f"UPDATE ingest_jobs SET {', '.join(sets)} WHERE id = %s", values)
        conn.commit()


def submit(job_id: int, fn: JobFn, principal: str | None = None) -> None:
    """작업을 백그라운드로 보낸다. 예외는 작업 행에 기록된다."""
    rid = request_id.get()

    def run() -> None:
        # 백그라운드 스레드는 요청 컨텍스트를 물려받지 않는다. 로그를 잇기
        # 위해 요청 ID 를 직접 심는다.
        request_id.set(rid if rid != "-" else new_request_id())
        principal_id.set(principal or "-")
        _update(job_id, status="running", started_at=_now())
        last_report = 0.0

        def progress(stage: str, done: int, total: int) -> None:
            nonlocal last_report
            # 배치마다 UPDATE 를 치면 DB 왕복이 색인보다 비싸진다.
            if time.monotonic() - last_report < 2.0 and done < total:
                return
            last_report = time.monotonic()
            _update(job_id, stats={"stage": stage, "done": done, "total": total})

        try:
            stats = fn(progress) or {}
            _update(job_id, status="done", finished_at=_now(), stats=stats)
            metrics.inc("ingest_jobs_total", status="done")
        except Exception as exc:
            log.exception("색인 작업 실패", extra={"job_id": job_id})
            _update(job_id, status="failed", finished_at=_now(),
                    detail=str(exc)[:2000])
            metrics.inc("ingest_jobs_total", status="failed")

    executor().submit(run)


def _now():
    from datetime import datetime, timezone

    return datetime.now(tz=timezone.utc)


def get(job_id: int) -> dict | None:
    with connect() as conn:
        row = conn.execute(
            """SELECT id, kind, source, status, detail, stats, principal,
                      created_at, started_at, finished_at
               FROM ingest_jobs WHERE id = %s""",
            (job_id,),
        ).fetchone()
    return _row(row) if row else None


def recent(limit: int = 20) -> list[dict]:
    with connect() as conn:
        rows = conn.execute(
            """SELECT id, kind, source, status, detail, stats, principal,
                      created_at, started_at, finished_at
               FROM ingest_jobs ORDER BY created_at DESC LIMIT %s""",
            (limit,),
        ).fetchall()
    return [_row(r) for r in rows]


def _row(r) -> dict:
    return {
        "id": r[0], "kind": r[1], "source": r[2], "status": r[3], "detail": r[4],
        "stats": r[5], "principal": r[6],
        "created_at": r[7].isoformat() if r[7] else None,
        "started_at": r[8].isoformat() if r[8] else None,
        "finished_at": r[9].isoformat() if r[9] else None,
    }
