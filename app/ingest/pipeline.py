"""색인 파이프라인: 문서 → 파싱 → 청킹 → 임베딩 → 적재.

지켜야 할 성질이 두 가지다.

  1. **실패를 숨기지 않는다.** 색인된 줄 알았는데 빠져 있는 문서가 PoC 에서
     가장 위험하다(architecture.md 3.1). 상태와 원인을 문서 행에 남긴다.
  2. **재색인이 기존 색인을 깨뜨리지 않는다.** 새 임베딩이 전부 준비된 뒤에
     한 트랜잭션에서 청크를 교체한다. 중간에 실패하면 이전 청크가 그대로
     검색된다. 모델이 죽은 동안 검색이 빈손이 되는 것보다 낫다.
"""
from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

from app.config import settings
from app.db.session import connect
from app.ingest.chunker import Chunk, chunk_document
from app.ingest.documents import parse_document
from app.observability import metrics, timed
from app.providers.base import EmbeddingProvider, get_embedding_provider

log = logging.getLogger(__name__)

Progress = Callable[[str, int, int], None]


@dataclass
class IngestResult:
    path: str
    status: str            # indexed | skipped | failed
    products: list[str] | None = None
    chunks: int = 0
    detail: str = ""
    document_id: int | None = None
    warnings: list[str] = field(default_factory=list)
    tokens: int = 0
    elapsed_ms: float = 0.0


def _embed_all(
    provider: EmbeddingProvider,
    chunks: list[Chunk],
    on_progress: Progress | None = None,
) -> list[list[float]]:
    size = settings.embed_batch_size
    if size < 1:
        raise ValueError("EMBED_BATCH_SIZE must be positive")
    vectors: list[list[float]] = []
    for i in range(0, len(chunks), size):
        batch = [c.content for c in chunks[i : i + size]]
        result = provider.embed(batch)
        if len(result) != len(batch) or any(
            len(v) != settings.embedding_dim or not all(math.isfinite(x) for x in v)
            or not any(v) for v in result
        ):
            raise ValueError("Invalid embedding count, dimension or values")
        vectors.extend(result)
        done = min(i + size, len(chunks))
        log.info("임베딩 %d/%d", done, len(chunks))
        if on_progress:
            on_progress("embed", done, len(chunks))
    return vectors


def _source_stat(path: Path) -> tuple[int | None, datetime | None]:
    try:
        info = path.stat()
    except OSError:
        return None, None
    return info.st_size, datetime.fromtimestamp(info.st_mtime, tz=timezone.utc)


def ingest_document(
    path: str | Path,
    provider: EmbeddingProvider | None = None,
    force: bool = False,
    source_name: str | None = None,
    acl_groups: list[str] | None = None,
    on_progress: Progress | None = None,
    supersede: bool = False,
) -> IngestResult:
    """supersede=True 면 같은 경로의 이전 판을 지운다.

    문서는 사내에서 계속 개정된다. 내용이 바뀌면 sha256 이 달라지므로 새 행이
    생기는데, 이전 판을 남겨 두면 옛 스펙이 영원히 검색된다. 파일 경로가 곧
    문서 정체성인 디렉터리 동기화에서는 이전 판을 걷어내야 한다. 업로드는
    같은 파일명이 다른 문서일 수 있으므로 기본값을 False 로 둔다.
    """
    path = Path(path)
    provider = provider or get_embedding_provider()

    with timed("ingest_document_ms") as clock:
        result = _ingest(path, provider, force, source_name, acl_groups,
                         on_progress, supersede)
    result.elapsed_ms = round(clock["ms"], 1)
    metrics.inc("ingest_documents_total", status=result.status)
    metrics.inc("ingest_chunks_total", result.chunks)
    return result


def _ingest(path, provider, force, source_name, acl_groups, on_progress,
            supersede=False) -> IngestResult:
    if on_progress:
        on_progress("parse", 0, 1)
    doc = parse_document(path)
    byte_size, mtime = _source_stat(path)

    meta = {
        "format": path.suffix.lower(),
        "embedding_provider": settings.embedding_provider,
        "embedding_model": settings.embedding_model,
        "chunk_target_chars": settings.chunk_target_chars,
        "warnings": doc.warnings,
        # 사람이 한 번 봐야 하는 문서. 스캔 PDF 가 조용히 섞여 들어오는 것을 막는다.
        "needs_review": bool(doc.likely_scanned),
    }

    with connect() as conn, conn.cursor() as cur:
        # 같은 파일의 중복 색인을 막는다. 사내 파일서버에는 같은 PDF 가
        # 여러 경로에 복사돼 있고, 중복 청크는 검색 결과를 오염시킨다.
        cur.execute("SELECT id, status FROM documents WHERE sha256 = %s", (doc.sha256,))
        row = cur.fetchone()
        if row and row[1] == "indexed" and not force:
            return IngestResult(str(path), "skipped", doc.products, document_id=row[0],
                                detail="이미 색인됨 (--force 로 재색인)",
                                warnings=doc.warnings)
        cur.execute(
            """INSERT INTO documents
                 (source_path, title, products, sha256, page_count, status, meta,
                  acl_groups, source_kind, byte_size, source_mtime, updated_at)
               VALUES (%s, %s, %s, %s, %s, 'indexing', %s, %s, %s, %s, %s, now())
               ON CONFLICT (sha256) DO UPDATE SET
                   source_path = EXCLUDED.source_path,
                   status      = 'indexing',
                   meta        = EXCLUDED.meta,
                   acl_groups  = EXCLUDED.acl_groups,
                   source_kind = EXCLUDED.source_kind,
                   byte_size   = EXCLUDED.byte_size,
                   source_mtime= EXCLUDED.source_mtime,
                   updated_at  = now()
               RETURNING id""",
            (source_name or str(path), doc.title, doc.products, doc.sha256,
             doc.page_count, json.dumps(meta, ensure_ascii=False),
             acl_groups or [], path.suffix.lower().lstrip("."), byte_size, mtime),
        )
        doc_id = cur.fetchone()[0]
        if supersede:
            # 같은 경로의 이전 판을 걷어낸다. 남겨 두면 옛 스펙이 계속 검색된다.
            cur.execute(
                "DELETE FROM documents WHERE source_path = %s AND id <> %s",
                (source_name or str(path), doc_id),
            )
            if cur.rowcount:
                log.info("이전 판 %d건 제거", cur.rowcount,
                         extra={"source_path": source_name or str(path)})
        conn.commit()

        try:
            chunks = chunk_document(doc)
            if not chunks:
                raise ValueError("청크가 하나도 생성되지 않았다. 파싱 실패로 간주한다.")

            vectors = _embed_all(provider, chunks, on_progress)
            if len(vectors) != len(chunks):
                raise ValueError(
                    f"임베딩 개수 불일치: 청크 {len(chunks)}, 벡터 {len(vectors)}"
                )

            # 여기서부터가 교체 구간이다. 한 트랜잭션 안에서 끝낸다.
            cur.execute("DELETE FROM chunks WHERE document_id=%s", (doc_id,))
            cur.executemany(
                """INSERT INTO chunks
                     (document_id, ordinal, kind, content, section_path,
                      page_from, page_to, products, embedding, content_hash, token_count)
                   VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)""",
                [
                    (doc_id, c.ordinal, c.kind, c.content, c.section_path,
                     c.page_from, c.page_to, c.products, str(v),
                     c.content_hash or None, c.token_count or None)
                    for c, v in zip(chunks, vectors)
                ],
            )
            cur.execute(
                "UPDATE documents SET status='indexed', error=NULL, ingested_at=now(),"
                " updated_at=now() WHERE id=%s",
                (doc_id,),
            )
            conn.commit()
            return IngestResult(
                str(path), "indexed", doc.products, len(chunks),
                document_id=doc_id, warnings=doc.warnings,
                tokens=sum(c.token_count for c in chunks),
            )

        except Exception as exc:
            conn.rollback()
            # 실패를 조용히 넘기지 않는다. 이전에 색인돼 있었다면 그 상태를
            # 유지한다. 재색인 실패로 검색이 빈손이 되는 것이 더 나쁘다.
            keep = "indexed" if row and row[1] == "indexed" else "failed"
            cur.execute(
                "UPDATE documents SET status=%s, error=%s, updated_at=now() WHERE id=%s",
                (keep, str(exc)[:2000], doc_id),
            )
            conn.commit()
            log.error("색인 실패", extra={"path": str(path), "document_id": doc_id},
                      exc_info=True)
            return IngestResult(str(path), "failed", doc.products,
                                detail=str(exc), document_id=doc_id,
                                warnings=doc.warnings)


def delete_document(document_id: int) -> bool:
    """문서와 청크를 함께 지운다(ON DELETE CASCADE).

    잘못 색인된 문서를 지울 수 없으면 오염된 검색 결과를 되돌릴 방법이 없다.
    """
    with connect() as conn:
        row = conn.execute(
            "DELETE FROM documents WHERE id = %s RETURNING title", (document_id,)
        ).fetchone()
        conn.commit()
    if row:
        log.info("문서 삭제", extra={"document_id": document_id, "title": row[0]})
    return bool(row)


def document_rows(conn) -> list[dict]:
    rows = conn.execute(
        """SELECT d.id, d.title, d.products, d.page_count, d.status, d.error,
                  count(c.id), coalesce(sum(c.token_count), 0), d.acl_groups,
                  d.source_kind, d.byte_size, d.ingested_at, d.meta
           FROM documents d LEFT JOIN chunks c ON c.document_id = d.id
           GROUP BY d.id ORDER BY d.title"""
    ).fetchall()
    return [
        {
            "id": r[0], "title": r[1], "products": list(r[2] or []), "pages": r[3],
            "status": r[4], "error": r[5], "chunks": r[6], "tokens": int(r[7] or 0),
            "acl_groups": list(r[8] or []), "kind": r[9], "bytes": r[10],
            "ingested_at": r[11].isoformat() if r[11] else None,
            "warnings": (r[12] or {}).get("warnings") or [],
            "needs_review": bool((r[12] or {}).get("needs_review")),
        }
        for r in rows
    ]


# 기존 PDF 호출자와 호환
ingest_pdf = ingest_document
