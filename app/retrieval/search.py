"""하이브리드 검색: 밀집(벡터) + 희소(전문검색)를 RRF 로 결합.

벡터 단독으로 가지 않는 이유는 이 도메인에 있다. 제품 스펙 문서는 모델명·
파트번호·규격 코드로 가득한데, 벡터 검색은 SR650 V4 와 SR650a V4 처럼
한 글자 다른 식별자를 잘 구분하지 못한다(decisions.md D5).
"""
from __future__ import annotations

from dataclasses import dataclass

from app.config import settings
from app.ingest.parser import _MODEL_RE

RRF_K = 60  # 표준값. 상위 순위에 과도한 가중이 실리지 않게 한다.


@dataclass
class Hit:
    chunk_id: int
    content: str
    section_path: str
    page_from: int
    page_to: int
    products: list[str]
    doc_title: str
    source_path: str
    score: float = 0.0

    @property
    def citation(self) -> str:
        if not self.page_from:
            return f"{self.doc_title} · {self.section_path}"
        page = (
            f"p{self.page_from}"
            if self.page_from == self.page_to
            else f"p{self.page_from}-{self.page_to}"
        )
        if self.source_path.lower().endswith(".pptx"):
            page = page.replace("p", "슬라이드 ")
        return f"{self.doc_title} {page}"


_COLUMNS = """
    c.id, c.content, c.section_path, c.page_from, c.page_to, c.products,
    d.title, d.source_path
"""


def detect_products(query: str) -> list[str]:
    """질문에서 모델명을 뽑는다.

    "SR650a V4 최대 메모리" 처럼 모델을 특정한 질문에 다른 모델 행이 섞여
    들어오는 것을 막는 2차 방어선이다(1차는 색인 시 청크 태깅).
    """
    return list(
        dict.fromkeys(
            f"ThinkSystem {m.group(1)} {m.group(2)}" for m in _MODEL_RE.finditer(query)
        )
    )


def _rows_to_hits(rows) -> list[Hit]:
    return [
        Hit(
            chunk_id=r[0], content=r[1], section_path=r[2] or "",
            page_from=r[3], page_to=r[4], products=list(r[5] or []),
            doc_title=r[6], source_path=r[7],
        )
        for r in rows
    ]


def _dense(cur, vector: list[float], products: list[str], limit: int) -> list[Hit]:
    cur.execute(
        f"""SELECT {_COLUMNS}
            FROM chunks c JOIN documents d ON d.id = c.document_id
            WHERE d.status = 'indexed' AND (%(products)s::text[] IS NULL OR c.products && %(products)s::text[])
            ORDER BY c.embedding <=> %(vec)s::vector
            LIMIT %(limit)s""",
        {"vec": str(vector), "products": products or None, "limit": limit},
    )
    return _rows_to_hits(cur.fetchall())


def _sparse(cur, query: str, products: list[str], limit: int) -> list[Hit]:
    """전문검색. 모델명·파트번호 같은 정확한 토큰에 강하다.

    'simple' 설정에는 한국어 형태소 분석기가 없다(decisions.md D5 미해결).
    영문·숫자 식별자에는 충분하고, 한국어는 어절 단위 일치까지만 잡는다.
    """
    cur.execute(
        f"""SELECT {_COLUMNS}
            FROM chunks c
            JOIN documents d ON d.id = c.document_id,
                 websearch_to_tsquery('simple', %(q)s) AS q
            WHERE d.status = 'indexed' AND to_tsvector('simple', c.content) @@ q
              AND (%(products)s::text[] IS NULL OR c.products && %(products)s::text[])
            ORDER BY ts_rank(to_tsvector('simple', c.content), q) DESC
            LIMIT %(limit)s""",
        {"q": query, "products": products or None, "limit": limit},
    )
    return _rows_to_hits(cur.fetchall())


def fuse(rankings: list[list[Hit]], top_k: int) -> list[Hit]:
    """Reciprocal Rank Fusion.

    점수 체계가 다른 검색 결과(코사인 거리 vs ts_rank)를 정규화 없이 합친다.
    가중치 튜닝이 필요 없어 PoC 단계에 적합하다.
    """
    scores: dict[int, float] = {}
    best: dict[int, Hit] = {}
    for ranking in rankings:
        for rank, hit in enumerate(ranking, start=1):
            scores[hit.chunk_id] = scores.get(hit.chunk_id, 0.0) + 1.0 / (RRF_K + rank)
            best.setdefault(hit.chunk_id, hit)

    ordered = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)[:top_k]
    out = []
    for chunk_id, score in ordered:
        hit = best[chunk_id]
        hit.score = score
        out.append(hit)
    return out


def _trigram(cur, query: str, products: list[str], limit: int) -> list[Hit]:
    """한국어 어절의 부분 일치를 pg_trgm의 단어 유사도로 보완한다."""
    cur.execute(
        f"""SELECT {_COLUMNS}
            FROM chunks c JOIN documents d ON d.id = c.document_id
            WHERE d.status = 'indexed' AND %(q)s <%% c.content
              AND (%(products)s::text[] IS NULL OR c.products && %(products)s::text[])
            ORDER BY word_similarity(%(q)s, c.content) DESC, c.id
            LIMIT %(limit)s""",
        {"q": query, "products": products or None, "limit": limit},
    )
    return _rows_to_hits(cur.fetchall())


def search(
    conn,
    query: str,
    embedding: list[float],
    top_k: int | None = None,
    products: list[str] | None = None,
) -> list[Hit]:
    top_k = top_k or settings.search_top_k
    n = settings.search_candidates
    products = products if products is not None else detect_products(query)

    with conn.cursor() as cur:
        dense = _dense(cur, embedding, products, n)
        sparse = _sparse(cur, query, products, n)
        trigram = _trigram(cur, query, products, n)

    return fuse([dense, sparse, trigram], top_k)
