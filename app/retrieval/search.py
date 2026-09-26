"""하이브리드 검색: 밀집(벡터) + 희소(전문검색) + 정확 일치 + 유사도를 RRF 로 결합.

벡터 단독으로 가지 않는 이유는 이 도메인에 있다. 제품 스펙 문서는 모델명·
파트번호·규격 코드로 가득한데, 벡터 검색은 SR650 V4 와 SR650a V4 처럼
한 글자 다른 식별자를 잘 구분하지 못한다(decisions.md D5).

경로는 네 개다.

| 경로 | 근거 | 강점 |
|---|---|---|
| dense   | pgvector 코사인 | 표현이 달라도 의미로 찾음 |
| sparse  | tsvector OR + ts_rank_cd | 여러 어휘가 겹치는 문단 |
| exact   | 식별자 AND | 파트번호·피처코드 한 건 조회 |
| trigram | pg_trgm word_similarity | 표기 흔들림·부분 일치 보완 |

`exact` 경로를 따로 둔 이유: OR 결합만 쓰면 "CF3G" 를 물었을 때 CF 가 들어간
수백 행이 같은 취급을 받는다. 식별자가 있는 질문은 그 식별자를 모두 포함한
청크가 정답일 확률이 압도적이다.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field

from app.config import settings
from app.observability import metrics, timed
from app.retrieval.query import AnalyzedQuery, analyze

log = logging.getLogger(__name__)

RRF_K = settings.rrf_k  # 표준값 60. 상위 순위에 과도한 가중이 실리지 않게 한다.

__all__ = [
    "Hit", "Filters", "search", "fuse", "RRF_K", "ALL_PATHS", "dedupe", "diversify",
]

_TOKEN = re.compile(r"[A-Za-z0-9]+|[가-힣]+")


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
    kind: str = ""
    document_id: int = 0
    ordinal: int = 0
    content_hash: str | None = None
    # 어느 경로가 이 청크를 올렸는지. 검색 실패를 진단할 때 이것부터 본다.
    paths: dict[str, int] = field(default_factory=dict)
    # 앞뒤 표 행. 인용에는 넣지 않고 컨텍스트로만 쓴다.
    neighbors: list[str] = field(default_factory=list)

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

    @property
    def context_block(self) -> str:
        """LLM 에 넘길 본문. 앞뒤 행이 있으면 덧붙인다."""
        if not self.neighbors:
            return self.content
        return self.content + "\n(인접 행) " + " / ".join(self.neighbors)


@dataclass(frozen=True)
class Filters:
    """검색 범위. None 은 '제한 없음' 이고 빈 리스트와 뜻이 다르다."""

    products: list[str] | None = None
    groups: list[str] | None = None        # 주체의 소속 그룹. None 이면 전체 열람
    document_ids: list[int] | None = None

    def params(self) -> dict:
        return {
            "products": self.products or None,
            "groups": self.groups if self.groups else None,
            "doc_ids": self.document_ids or None,
        }


_COLUMNS = """
    c.id, c.content, c.section_path, c.page_from, c.page_to, c.products,
    d.title, d.source_path, c.kind, c.document_id, c.ordinal, c.content_hash
"""

# 문서 상태, 제품, 열람 그룹, 문서 지정을 한 곳에서 건다.
# acl_groups 가 빈 배열이면 전사 공개 문서로 본다(migrations/002).
_FILTER = """
    d.status = 'indexed'
    AND (%(products)s::text[] IS NULL OR c.products && %(products)s::text[])
    AND (%(groups)s::text[] IS NULL
         OR cardinality(d.acl_groups) = 0
         OR d.acl_groups && %(groups)s::text[])
    AND (%(doc_ids)s::bigint[] IS NULL OR d.id = ANY(%(doc_ids)s::bigint[]))
"""


def _rows_to_hits(rows) -> list[Hit]:
    return [
        Hit(
            chunk_id=r[0], content=r[1], section_path=r[2] or "",
            page_from=r[3] or 0, page_to=r[4] or 0, products=list(r[5] or []),
            doc_title=r[6], source_path=r[7], kind=r[8] or "",
            document_id=r[9] or 0, ordinal=r[10] or 0, content_hash=r[11],
        )
        for r in rows
    ]


_ef_search_supported: bool | None = None


def _tune_dense(cur) -> None:
    """HNSW 탐색 폭을 넓힌다.

    기본값(ef_search=40)은 필터와 겹치면 후보를 너무 일찍 잘라낸다. 트랜잭션
    범위로만 올려 다른 질의에 영향을 주지 않는다. pgvector 판이 낮아 설정이
    없으면 한 번만 확인하고 넘어간다.
    """
    global _ef_search_supported
    if _ef_search_supported is False:
        return
    try:
        # SET 은 파라미터 바인딩을 받지 않는다. 정수로 확정해 문자열에 넣는다.
        # 실패하면 트랜잭션이 중단되므로 세이브포인트 안에서 시도한다.
        with cur.connection.transaction():
            cur.execute(f"SET LOCAL hnsw.ef_search = {int(settings.hnsw_ef_search)}")
        _ef_search_supported = True
    except Exception:
        _ef_search_supported = False
        log.info("hnsw.ef_search 를 지원하지 않는 pgvector 다. 기본값으로 진행한다.")


def _dense(cur, vector: list[float], filters: Filters, limit: int) -> list[Hit]:
    _tune_dense(cur)
    params = filters.params() | {"vec": str(vector), "limit": limit}
    cur.execute(
        f"""SELECT {_COLUMNS}
            FROM chunks c JOIN documents d ON d.id = c.document_id
            WHERE {_FILTER} AND c.embedding IS NOT NULL
            ORDER BY c.embedding <=> %(vec)s::vector
            LIMIT %(limit)s""",
        params,
    )
    return _rows_to_hits(cur.fetchall())


def _sparse(cur, aq: AnalyzedQuery, filters: Filters, limit: int) -> list[Hit]:
    """전문검색. 확장된 어휘를 OR 로 묶고 겹침 정도로 순위를 낸다.

    AND(`websearch_to_tsquery` 기본값)로 묶으면 질문이 길어질수록 결과가 0 이
    된다. 한국어 질문을 용어집으로 영문화한 뒤 OR 로 넣는 것이 이 문서군에서
    희소 검색을 살리는 유일한 방법이다.
    """
    query = aq.sparse_query(bool(filters.products))
    if not query:
        return []
    params = filters.params() | {"q": query, "limit": limit}
    # 정규화 플래그 1 = 문서 길이의 로그로 나눈다. 없으면 같은 어휘를 여러 번
    # 쓴 긴 산문이 한 줄짜리 스펙 행을 항상 이긴다. 이 문서군에서는 대개
    # 한 줄짜리 스펙 행이 정답이다.
    cur.execute(
        f"""SELECT {_COLUMNS}
            FROM chunks c
            JOIN documents d ON d.id = c.document_id,
                 websearch_to_tsquery('simple', %(q)s) AS q
            WHERE {_FILTER} AND c.content_tsv @@ q
            ORDER BY ts_rank_cd(c.content_tsv, q, 1) DESC, c.id
            LIMIT %(limit)s""",
        params,
    )
    return _rows_to_hits(cur.fetchall())


def _exact(cur, aq: AnalyzedQuery, filters: Filters, limit: int) -> list[Hit]:
    """식별자를 모두 포함하는 청크. 파트번호·피처코드 조회의 정답 경로다.

    모델명만 있는 질문에서는 건너뛴다. 모델명은 그 문서의 모든 청크 머리에
    붙어 있어서 AND 검색이 문서 전체를 같은 취급하기 때문이다.
    """
    if not aq.exact_identifiers:
        return []
    params = filters.params() | {"q": aq.websearch_exact, "limit": limit}
    cur.execute(
        f"""SELECT {_COLUMNS}
            FROM chunks c
            JOIN documents d ON d.id = c.document_id,
                 websearch_to_tsquery('simple', %(q)s) AS q
            WHERE {_FILTER} AND c.content_tsv @@ q
            ORDER BY ts_rank_cd(c.content_tsv, q) DESC, c.id
            LIMIT %(limit)s""",
        params,
    )
    return _rows_to_hits(cur.fetchall())


def _trigram(cur, aq: AnalyzedQuery, filters: Filters, limit: int) -> list[Hit]:
    """표기 흔들림을 pg_trgm 단어 유사도로 보완한다."""
    params = filters.params() | {"q": aq.trigram_text, "limit": limit}
    cur.execute(
        f"""SELECT {_COLUMNS}
            FROM chunks c JOIN documents d ON d.id = c.document_id
            WHERE {_FILTER} AND %(q)s <%% c.content
            ORDER BY word_similarity(%(q)s, c.content) DESC, c.id
            LIMIT %(limit)s""",
        params,
    )
    return _rows_to_hits(cur.fetchall())


def fuse(
    rankings: dict[str, list[Hit]] | list[list[Hit]],
    top_k: int,
    weights: dict[str, float] | None = None,
) -> list[Hit]:
    """가중 Reciprocal Rank Fusion.

    점수 체계가 다른 결과(코사인 거리 vs ts_rank vs 유사도)를 정규화 없이
    합친다. 가중치는 경로 신뢰도만 반영하며, 기본값에서 벗어날 근거는
    평가 결과여야 한다.
    """
    if isinstance(rankings, list):
        rankings = {f"path{i}": r for i, r in enumerate(rankings)}
    weights = weights or {}

    scores: dict[int, float] = {}
    best: dict[int, Hit] = {}
    for name, ranking in rankings.items():
        weight = weights.get(name, 1.0)
        for rank, hit in enumerate(ranking, start=1):
            scores[hit.chunk_id] = scores.get(hit.chunk_id, 0.0) + weight / (RRF_K + rank)
            kept = best.setdefault(hit.chunk_id, hit)
            kept.paths[name] = rank

    ordered = sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))
    out: list[Hit] = []
    for chunk_id, score in ordered[:top_k]:
        hit = best[chunk_id]
        hit.score = score
        out.append(hit)
    return out


def _tokens(text: str) -> set[str]:
    return {t.lower() for t in _TOKEN.findall(text)}


def dedupe(hits: list[Hit]) -> list[Hit]:
    """같은 본문을 여러 건 돌려주지 않는다.

    사내 문서는 같은 표·머리말이 문서 여러 곳에 복사돼 있다. 중복 청크는
    컨텍스트 자리만 차지하고 LLM 에는 새 정보를 주지 않는다.
    """
    seen: set[str] = set()
    out: list[Hit] = []
    for hit in hits:
        key = hit.content_hash or hit.content.strip()
        if key in seen:
            continue
        seen.add(key)
        out.append(hit)
    return out


def _body_tokens(hit: Hit) -> set[str]:
    """유사도 비교용 토큰. 청크 머리의 제품·섹션 줄은 제외한다.

    모든 청크가 같은 머리를 달고 있어서(청킹 규칙), 머리를 포함하면 문서 안의
    모든 청크가 서로 비슷하다고 판정된다. 실제로 이 때문에 초기 구현이
    "가장 안 비슷한 것" = 무관한 청크를 골라 올렸다.
    """
    lines = hit.content.splitlines()
    body = "\n".join(lines[1:]) if len(lines) > 1 else hit.content
    return _tokens(body)


def diversify(
    hits: list[Hit],
    top_k: int,
    threshold: float | None = None,
    section_cap: int | None = None,
) -> list[Hit]:
    """한 표가 상위를 독점하는 것을 막는다.

    두 가지를 본다.

      - **근접 중복** — 본문이 거의 같은 청크는 한 건만 남긴다
      - **섹션 상한** — 같은 섹션에서 올릴 건수를 제한한다. 이 문서군에서는
        섹션 하나가 대개 표 하나다

    섹션 상한이 실측에서 결정적이었다. "SR650 V4 최대 드라이브 베이"를 물으면
    `Drive bay field upgrades` 표의 비슷한 행 여덟 건이 상위 10개를 채우고,
    정답인 `Standard specifications` 한 줄이 밀려났다. 행마다 파트번호가 달라
    근접 중복 판정으로는 걸러지지 않는다.

    관련도 순서는 흔들지 않는다. 순위대로 훑으며 초과분만 뒤로 미루고, 자리가
    남으면 원래 순서대로 채운다. 후보를 버리지 않으므로 검색 품질을 떨어뜨릴 수 없다.

    고전적 MMR(관련도와 다양성의 가중합)을 쓰지 않은 이유도 실측 때문이다.
    RRF 점수 차이가 작아 다양성 항이 관련도를 압도했고, "가장 안 비슷한 것"을
    고르다 보니 질문과 무관한 청크가 1위로 올라왔다.
    """
    threshold = settings.near_dup_similarity if threshold is None else threshold
    cap = settings.section_cap if section_cap is None else section_cap
    if len(hits) <= 1 or (threshold >= 1.0 and cap <= 0):
        return hits[:top_k]

    token_sets = {h.chunk_id: _body_tokens(h) for h in hits}
    selected: list[Hit] = []
    deferred: list[Hit] = []
    per_section: dict[tuple[int, str], int] = {}

    for hit in hits:
        if len(selected) >= top_k:
            deferred.append(hit)
            continue
        key = (hit.document_id, hit.section_path)
        if cap > 0 and per_section.get(key, 0) >= cap:
            deferred.append(hit)
            continue
        near = False
        if threshold < 1.0:
            a = token_sets[hit.chunk_id]
            for chosen in selected:
                b = token_sets[chosen.chunk_id]
                union = len(a | b)
                if union and len(a & b) / union >= threshold:
                    near = True
                    break
        if near:
            deferred.append(hit)
            continue
        selected.append(hit)
        per_section[key] = per_section.get(key, 0) + 1

    for hit in deferred:
        if len(selected) >= top_k:
            break
        selected.append(hit)
    return selected[:top_k]


def attach_neighbors(cur, hits: list[Hit], window: int | None = None) -> None:
    """표 행 청크에 앞뒤 행을 붙인다.

    표 행은 대개 자기완결적이지만, 그룹 구분 행 아래 여러 행이 이어지는 표에서
    "그 다음 행" 을 물으면 단독 행만으로는 답할 수 없다. 인용은 원래 청크로
    유지하고 본문만 넓힌다. 출처가 흐려지면 안 된다.
    """
    window = settings.neighbor_window if window is None else window
    targets = [h for h in hits if h.kind == "table_row" and h.document_id]
    if window < 1 or not targets:
        return

    wanted: list[tuple[int, int]] = []
    for hit in targets:
        for delta in range(-window, window + 1):
            if delta:
                wanted.append((hit.document_id, hit.ordinal + delta))
    if not wanted:
        return

    cur.execute(
        """SELECT document_id, ordinal, content
           FROM chunks
           WHERE (document_id, ordinal) IN (
               SELECT d, o FROM unnest(%(docs)s::bigint[], %(ords)s::int[]) AS t(d, o)
           ) AND kind = 'table_row'""",
        {"docs": [d for d, _ in wanted], "ords": [o for _, o in wanted]},
    )
    lookup = {(r[0], r[1]): r[2] for r in cur.fetchall()}
    for hit in targets:
        for delta in range(-window, window + 1):
            if delta == 0:
                continue
            body = lookup.get((hit.document_id, hit.ordinal + delta))
            if body:
                # 청크 머리의 제품·섹션 줄은 이미 본문에 있으므로 마지막 줄만 쓴다.
                hit.neighbors.append(body.strip().splitlines()[-1][:200])


ALL_PATHS = ("dense", "sparse", "exact", "trigram")

_RUNNERS = {
    "dense":   lambda cur, aq, vec, f, n: _dense(cur, vec, f, n),
    "sparse":  lambda cur, aq, vec, f, n: _sparse(cur, aq, f, n),
    "exact":   lambda cur, aq, vec, f, n: _exact(cur, aq, f, n),
    "trigram": lambda cur, aq, vec, f, n: _trigram(cur, aq, f, n),
}


def _paths(conn, aq: AnalyzedQuery, embedding: list[float], filters: Filters,
           candidates: int, enabled: tuple[str, ...] = ALL_PATHS) -> dict[str, list[Hit]]:
    """경로를 골라 실행한다.

    경로를 끌 수 있어야 평가에서 각 경로의 기여를 따로 잴 수 있다. "하이브리드가
    낫다"는 주장은 단독 경로 점수와 나란히 놓아야 근거가 된다.
    """
    out: dict[str, list[Hit]] = {}
    with conn.cursor() as cur:
        for name in enabled:
            runner = _RUNNERS.get(name)
            if runner is None:
                raise ValueError(f"알 수 없는 검색 경로: {name!r}")
            with timed("rag_search_path_ms", path=name):
                out[name] = runner(cur, aq, embedding, filters, candidates)
    return out


def _weights() -> dict[str, float]:
    return {
        "dense": settings.weight_dense,
        "sparse": settings.weight_sparse,
        # 식별자 전량 일치는 이 도메인에서 가장 믿을 수 있는 신호다.
        "exact": max(settings.weight_sparse, settings.weight_dense) * 1.5,
        "trigram": settings.weight_trigram,
    }


def _search_one(conn, aq, embedding, filters, top_k, candidates,
                enabled=ALL_PATHS) -> list[Hit]:
    rankings = _paths(conn, aq, embedding, filters, candidates, enabled)
    for name, ranking in rankings.items():
        metrics.inc("rag_search_path_hits_total", len(ranking), path=name)
    fused = dedupe(fuse(rankings, candidates, _weights()))
    return fused[: max(top_k, 1)]


def _interleave(groups: list[list[Hit]], top_k: int) -> list[Hit]:
    """제품별 결과를 라운드로빈으로 섞는다.

    비교 질문에서 한 제품이 상위를 독점하면 LLM 이 비교할 재료가 없다.
    """
    out: list[Hit] = []
    seen: set[int] = set()
    for row in range(max((len(g) for g in groups), default=0)):
        for group in groups:
            if row < len(group) and group[row].chunk_id not in seen:
                seen.add(group[row].chunk_id)
                out.append(group[row])
                if len(out) >= top_k:
                    return out
    return out


def search(
    conn,
    query: str,
    embedding: list[float],
    top_k: int | None = None,
    products: list[str] | None = None,
    groups: list[str] | None = None,
    document_ids: list[int] | None = None,
    aq: AnalyzedQuery | None = None,
    candidates: int | None = None,
    neighbors: bool = True,
    paths: tuple[str, ...] | None = None,
) -> list[Hit]:
    top_k = top_k or settings.search_top_k
    candidates = candidates or settings.search_candidates
    aq = aq or analyze(query)
    products = products if products is not None else aq.products
    enabled = tuple(paths) if paths else ALL_PATHS

    # 비교 질문은 제품별로 따로 검색한 뒤 섞는다. 한 번에 뽑으면 문서 분량이
    # 많은 모델이 상위를 다 먹는다.
    if len(products) > 1:
        per_product = max(3, (top_k // len(products)) + 1)
        groups_of_hits = [
            _search_one(
                conn, aq, embedding,
                Filters([product], groups, document_ids),
                per_product, candidates, enabled,
            )
            for product in products
        ]
        hits = _interleave(groups_of_hits, top_k)
    else:
        filters = Filters(products or None, groups, document_ids)
        # 근접 중복 정리는 리랭킹 뒤에 한다(retrieval.answer.retrieve). 후보
        # 단계에서 줄이면 재정렬이 볼 수 있는 범위가 좁아진다.
        hits = _search_one(
            conn, aq, embedding, filters, top_k, candidates, enabled
        )

    if neighbors and hits:
        with conn.cursor() as cur:
            attach_neighbors(cur, hits)
    return hits
