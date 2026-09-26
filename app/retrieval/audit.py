"""질의 로그.

"무엇을 못 찾았는지"를 모으는 것이 품질 개선의 가장 빠른 길이다
(architecture.md 8.4). PoC 기간 동안 실무자가 던진 질문과 그 결과가 남아
있으면, 다음 개선 대상을 추측이 아니라 목록으로 고를 수 있다.

로그 실패가 답변을 망치면 안 된다. 그래서 모든 쓰기는 세이브포인트 안에서
하고, 실패하면 경고만 남기고 넘어간다.
"""
from __future__ import annotations

import json
import logging

log = logging.getLogger(__name__)


def log_query(
    conn,
    *,
    question: str,
    request_id: str | None = None,
    principal: str | None = None,
    products: list[str] | None = None,
    top_k: int | None = None,
    used_rag: bool = True,
    answered: bool = False,
    grounded: bool | None = None,
    unsupported: list[str] | None = None,
    chunk_ids: list[int] | None = None,
    answer: str | None = None,
    model: str | None = None,
    latency_ms: dict | None = None,
) -> int | None:
    if conn is None:
        return None
    try:
        with conn.transaction():
            row = conn.execute(
                """INSERT INTO query_log
                     (request_id, principal, question, products, top_k, used_rag,
                      answered, grounded, unsupported, chunk_ids, answer, model, latency_ms)
                   VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                   RETURNING id""",
                (
                    request_id, principal, question, products or [], top_k, used_rag,
                    answered, grounded, json.dumps(unsupported or [], ensure_ascii=False),
                    chunk_ids or [], answer, model,
                    json.dumps(latency_ms or {}, ensure_ascii=False),
                ),
            ).fetchone()
        return row[0] if row else None
    except Exception:
        log.warning("질의 로그 기록 실패", exc_info=True)
        return None


def set_feedback(conn, query_id: int, rating: int, comment: str | None = None) -> bool:
    """사용자 평가를 남긴다. 틀린 답을 모으는 것이 이 기능의 목적이다."""
    if rating not in (-1, 1):
        raise ValueError("rating 은 -1 또는 1 이어야 한다")
    row = conn.execute(
        "UPDATE query_log SET rating = %s, comment = %s WHERE id = %s RETURNING id",
        (rating, comment, query_id),
    ).fetchone()
    conn.commit()
    return bool(row)


def recent(conn, limit: int = 50, only_failed: bool = False) -> list[dict]:
    """최근 질의. only_failed 는 근거를 못 찾았거나 평가가 나쁜 것만 본다."""
    where = ""
    if only_failed:
        where = "WHERE answered = false OR rating = -1 OR grounded = false"
    rows = conn.execute(
        f"""SELECT id, asked_at, principal, question, products, answered, grounded,
                   unsupported, rating, comment, model, latency_ms, left(answer, 300)
            FROM query_log {where}
            ORDER BY asked_at DESC LIMIT %s""",
        (limit,),
    ).fetchall()
    return [
        {
            "id": r[0], "asked_at": r[1].isoformat(), "principal": r[2],
            "question": r[3], "products": list(r[4] or []), "answered": r[5],
            "grounded": r[6], "unsupported": r[7], "rating": r[8], "comment": r[9],
            "model": r[10], "latency_ms": r[11], "answer": r[12],
        }
        for r in rows
    ]


def stats(conn, days: int = 7) -> dict:
    """PoC 보고에 쓸 요약. 응답 시간은 품질과 분리해서 본다."""
    row = conn.execute(
        """SELECT count(*),
                  count(*) FILTER (WHERE answered),
                  count(*) FILTER (WHERE grounded IS false),
                  count(*) FILTER (WHERE rating = 1),
                  count(*) FILTER (WHERE rating = -1),
                  percentile_disc(0.5) WITHIN GROUP (
                      ORDER BY (latency_ms->>'total')::float),
                  percentile_disc(0.95) WITHIN GROUP (
                      ORDER BY (latency_ms->>'total')::float)
           FROM query_log
           WHERE asked_at > now() - make_interval(days => %s)""",
        (days,),
    ).fetchone()
    total = row[0] or 0
    return {
        "days": days,
        "queries": total,
        "answered": row[1],
        "answer_rate": round(row[1] / total, 3) if total else None,
        "ungrounded": row[2],
        "thumbs_up": row[3],
        "thumbs_down": row[4],
        "latency_p50_ms": row[5],
        "latency_p95_ms": row[6],
    }
