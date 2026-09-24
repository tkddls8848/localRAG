"""색인 파이프라인: PDF → 파싱 → 청킹 → 임베딩 → 적재."""
from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass
from pathlib import Path

from app.config import settings
from app.db.session import connect
from app.ingest.chunker import Chunk, chunk_document
from app.ingest.documents import parse_document
from app.providers.base import EmbeddingProvider, get_embedding_provider

log = logging.getLogger(__name__)


@dataclass
class IngestResult:
    path: str
    status: str            # indexed | skipped | failed
    products: list[str] | None = None
    chunks: int = 0
    detail: str = ""


def _embed_all(provider: EmbeddingProvider, chunks: list[Chunk]) -> list[list[float]]:
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
        log.info("임베딩 %d/%d", min(i + size, len(chunks)), len(chunks))
    return vectors


def ingest_document(
    path: str | Path,
    provider: EmbeddingProvider | None = None,
    force: bool = False,
    source_name: str | None = None,
) -> IngestResult:
    path = Path(path)
    provider = provider or get_embedding_provider()

    doc = parse_document(path)

    with connect() as conn, conn.cursor() as cur:
        # 같은 파일의 중복 색인을 막는다. 사내 파일서버에는 같은 PDF 가
        # 여러 경로에 복사돼 있고, 중복 청크는 검색 결과를 오염시킨다.
        cur.execute("SELECT id, status FROM documents WHERE sha256 = %s", (doc.sha256,))
        row = cur.fetchone()
        if row and row[1] == "indexed" and not force:
            return IngestResult(str(path), "skipped", doc.products,
                                detail="이미 색인됨 (--force 로 재색인)")
        cur.execute(
            """INSERT INTO documents
                 (source_path, title, products, sha256, page_count, status, meta)
               VALUES (%s, %s, %s, %s, %s, 'pending', %s)
               ON CONFLICT (sha256) DO UPDATE SET source_path=EXCLUDED.source_path
               RETURNING id""",
            (source_name or str(path), doc.title, doc.products, doc.sha256, doc.page_count,
             json.dumps({"format": path.suffix.lower(), "embedding_provider": settings.embedding_provider, "embedding_model": settings.embedding_model})),
        )
        doc_id = cur.fetchone()[0]
        conn.commit()

        try:
            chunks = chunk_document(doc)
            if not chunks:
                raise ValueError("청크가 하나도 생성되지 않았다. 파싱 실패로 간주한다.")

            vectors = _embed_all(provider, chunks)
            if len(vectors) != len(chunks):
                raise ValueError(
                    f"임베딩 개수 불일치: 청크 {len(chunks)}, 벡터 {len(vectors)}"
                )

            cur.execute("DELETE FROM chunks WHERE document_id=%s", (doc_id,))
            cur.executemany(
                """INSERT INTO chunks
                     (document_id, ordinal, kind, content, section_path,
                      page_from, page_to, products, embedding)
                   VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)""",
                [
                    (doc_id, c.ordinal, c.kind, c.content, c.section_path,
                     c.page_from, c.page_to, c.products, str(v))
                    for c, v in zip(chunks, vectors)
                ],
            )
            cur.execute("UPDATE documents SET status='indexed', error=NULL, ingested_at=now() WHERE id=%s", (doc_id,))
            conn.commit()
            return IngestResult(str(path), "indexed", doc.products, len(chunks))

        except Exception as exc:
            conn.rollback()
            # 실패를 조용히 넘기지 않는다. 색인된 줄 알았는데 빠져 있는 문서가
            # 가장 위험하다(docs/architecture.md 3.1).
            cur.execute(
                "UPDATE documents SET status=%s, error=%s WHERE id=%s",
                ("indexed" if row and row[1] == "indexed" else "failed", str(exc)[:2000], doc_id),
            )
            conn.commit()
            return IngestResult(str(path), "failed", doc.products, detail=str(exc))


# 기존 PDF 호출자와 호환
ingest_pdf = ingest_document
