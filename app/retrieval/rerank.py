"""리랭킹.

RRF 는 "여러 경로가 함께 올린 것"을 위로 보내지만, 질문의 뜻은 보지 않는다.
검색 후보 30개 안에 정답이 들어 있어도 8위에 있으면 LLM 이 놓친다. 그래서
후보를 질문에 맞춰 한 번 더 정렬한다.

제공자를 둘 준비한다.

  - `rules` (기본) — 규칙 기반 특징 점수. 지연이 0 이고 순위 근거를 설명할 수
    있다. 사내 도입 검토에서 "왜 이게 1위인가"에 답할 수 있어야 한다
  - `llm` — 로컬 생성 모델로 한 번에 재정렬. 품질은 더 좋지만 질문마다
    호출이 한 번 더 붙는다. 평가로 이득을 확인하고 켠다

교차 인코더(`bge-reranker-v2-m3`)를 쓰지 않은 이유는 Ollama 가 리랭킹
엔드포인트를 제공하지 않아 런타임을 하나 더 띄워야 하기 때문이다.
구성요소를 늘리지 않는다는 원칙(architecture.md 1.1)이 우선이다.
"""
from __future__ import annotations

import logging
import re
from typing import Protocol

from app.config import settings
from app.observability import metrics, timed
from app.retrieval.query import AnalyzedQuery
from app.retrieval.search import Hit

log = logging.getLogger(__name__)

_TOKEN = re.compile(r"[A-Za-z0-9]+|[가-힣]+")
_NUMBER = re.compile(r"\d")


def _keys_of(content: str) -> str:
    """표 행 청크의 항목명 부분만 모아 돌려준다.

    청크 본문은 "항목: 값" 여러 줄이다. 질문의 어휘가 항목명 자리에 있는 것과
    값 자리에 있는 것은 전혀 다른 신호다. 앞부분만 떼어 비교한다.
    """
    keys = []
    for line in content.splitlines():
        head, sep, _ = line.partition(":")
        if sep and len(head) <= 80:
            keys.append(head)
    return " ".join(keys).lower()


class Reranker(Protocol):
    @property
    def name(self) -> str: ...

    def rerank(self, aq: AnalyzedQuery, hits: list[Hit], top_k: int) -> list[Hit]: ...


class NoopReranker:
    @property
    def name(self) -> str:
        return "none"

    def rerank(self, aq: AnalyzedQuery, hits: list[Hit], top_k: int) -> list[Hit]:
        return hits[:top_k]


class RulesReranker:
    """규칙 기반 특징 점수.

    가중치는 이 문서군의 실패 사례에서 나왔다.
      - 식별자가 본문에 다 있으면 거의 항상 정답이다 -> 가장 큰 가중
      - 제품 태그가 질문의 모델과 정확히 맞아야 한다 -> 모델 혼동 차단
      - 스펙 질문의 답은 표 행에, 절차 질문의 답은 산문에 있다
      - 섹션 경로에 질문 어휘가 있으면 본문에만 있는 경우보다 강한 신호다
      - **묻는 항목이 값이 아니라 항목명 자리에 있어야 한다.** 실측에서 가장
        효과가 컸다. "Memory maximum: Up to 8TB"(항목명이 memory maximum)와
        "Feature: Memory"(값이 Memory)는 같은 질문에 전혀 다른 답이다
    """

    W_IDENT = 3.0
    W_PRODUCT = 1.5
    W_TERMS = 1.5
    W_KEY = 1.6
    W_SECTION = 1.0
    W_KIND = 0.8
    W_AGREE = 0.6
    W_RETRIEVAL = 2.0

    @property
    def name(self) -> str:
        return "rules"

    @staticmethod
    def discriminating(aq: AnalyzedQuery, hits: list[Hit]) -> tuple[list[str], list[str]]:
        """후보 중 어디에도 없는 검색어를 떼어낸다.

        커버리지 비율의 분모에 그런 어휘가 끼면 **모든** 후보의 점수가 똑같이
        희석되고, 그만큼 검색 순위(W_RETRIEVAL)의 영향이 커져 순서가 뒤집힌다.

        실측: 코퍼스에 한국어 문서가 섞이면서 질의 확장이 한국어 어간을 함께
        넣게 됐다. 영문 문서 질의에서는 그 어간이 한 건도 맞지 않는데, 분모만
        키워 정답(`Weight: Maximum weight: 42 kg`)이 2위로 밀렸다. 어디에도
        없는 어휘는 변별 정보가 0 이므로 분모에서 빼는 것이 맞다.
        """
        corpus = " ".join(h.content.lower() for h in hits)
        idents = [i.lower() for i in aq.identifiers]
        terms = [t.lower() for t in aq.terms if t.lower() not in idents]
        return (
            [i for i in idents if i in corpus] or idents,
            [t for t in terms if t in corpus] or terms,
        )

    def features(
        self,
        aq: AnalyzedQuery,
        hit: Hit,
        idents: list[str] | None = None,
        terms: list[str] | None = None,
    ) -> dict[str, float]:
        content_low = hit.content.lower()
        section_low = hit.section_path.lower()

        if idents is None:
            idents = [i.lower() for i in aq.identifiers]
        ident_cover = (
            sum(1 for i in idents if i in content_low) / len(idents) if idents else 0.0
        )

        if terms is None:
            terms = [t.lower() for t in aq.terms if t.lower() not in idents]
        term_cover = (
            sum(1 for t in terms if t in content_low) / len(terms) if terms else 0.0
        )
        section_cover = (
            sum(1 for t in terms if t in section_low) / len(terms) if terms else 0.0
        )
        keys = _keys_of(hit.content)
        key_cover = (
            sum(1 for t in terms if t in keys) / len(terms) if terms and keys else 0.0
        )

        product_match = 0.0
        if aq.products:
            wanted = set(aq.products)
            have = set(hit.products)
            if wanted & have:
                # 질문이 한 모델만 가리키는데 청크가 그 모델만 달고 있으면 최상.
                product_match = 1.0 if have <= wanted else 0.6

        kind_match = 0.0
        if aq.intent == "spec" and hit.kind == "table_row":
            kind_match = 1.0
        elif aq.intent == "procedure" and hit.kind == "prose":
            kind_match = 1.0
        elif aq.intent == "spec" and _NUMBER.search(hit.content):
            kind_match = 0.4

        return {
            "ident": ident_cover,
            "product": product_match,
            "terms": term_cover,
            "key": key_cover,
            "section": section_cover,
            "kind": kind_match,
            # 여러 경로가 함께 올린 청크는 한 경로만 올린 것보다 믿을 만하다.
            "agree": min(len(hit.paths), 3) / 3.0,
        }

    def rerank(self, aq: AnalyzedQuery, hits: list[Hit], top_k: int) -> list[Hit]:
        if not hits:
            return []
        top_score = max(h.score for h in hits) or 1.0
        idents, terms = self.discriminating(aq, hits)
        scored: list[tuple[float, int, Hit]] = []
        for index, hit in enumerate(hits):
            f = self.features(aq, hit, idents, terms)
            value = (
                self.W_IDENT * f["ident"]
                + self.W_PRODUCT * f["product"]
                + self.W_TERMS * f["terms"]
                + self.W_KEY * f["key"]
                + self.W_SECTION * f["section"]
                + self.W_KIND * f["kind"]
                + self.W_AGREE * f["agree"]
                + self.W_RETRIEVAL * (hit.score / top_score)
            )
            # 동점일 때 검색 순위를 유지한다(index 가 작을수록 앞).
            scored.append((-value, index, hit))
        scored.sort()
        out = []
        for value, _, hit in scored[:top_k]:
            hit.score = -value
            out.append(hit)
        return out


_ORDER_RE = re.compile(r"\d+")

_SYSTEM = """당신은 검색 결과를 질문 적합도 순으로 정렬한다.

규칙:
1. 질문에 직접 답하는 발췌를 앞에 둔다.
2. 질문이 특정 제품 모델을 가리키면 다른 모델의 발췌는 뒤로 보낸다.
3. 숫자·파트번호를 물었다면 그 값이 실제로 적힌 발췌를 앞에 둔다.
4. 출력은 발췌 번호를 쉼표로 구분한 목록 하나만. 설명하지 않는다.
   예: 3,1,7,2
5. 모든 번호를 빠짐없이 한 번씩 포함한다."""


class LLMReranker:
    """로컬 생성 모델로 한 번에 재정렬(listwise).

    점수를 하나씩 물으면 후보 수만큼 호출이 늘어 노트북에서 못 쓴다. 번호
    순서만 받아 파싱하고, 형식이 깨지면 원래 순서를 유지한다. 리랭커가
    검색을 망치는 것이 가장 나쁜 결과다.
    """

    def __init__(self, llm=None) -> None:
        self._llm = llm

    @property
    def name(self) -> str:
        return "llm"

    def _provider(self):
        if self._llm is None:
            from app.providers.base import get_llm_provider

            self._llm = get_llm_provider()
        return self._llm

    def rerank(self, aq: AnalyzedQuery, hits: list[Hit], top_k: int) -> list[Hit]:
        if len(hits) <= 1:
            return hits[:top_k]
        excerpts = []
        for i, hit in enumerate(hits, start=1):
            body = " ".join(hit.content.split())[:300]
            excerpts.append(f"[{i}] ({hit.citation}) {body}")
        user = "질문: " + aq.raw + "\n\n" + "\n".join(excerpts)

        try:
            with timed("rag_rerank_ms", provider="llm"):
                raw = self._provider().generate(_SYSTEM, user)
        except Exception:
            log.warning("LLM 리랭킹 실패. 검색 순위를 유지한다.", exc_info=True)
            metrics.inc("rag_rerank_failed_total", provider="llm")
            return hits[:top_k]

        order: list[int] = []
        for match in _ORDER_RE.finditer(raw):
            index = int(match.group(0))
            if 1 <= index <= len(hits) and index not in order:
                order.append(index)
        if not order:
            metrics.inc("rag_rerank_failed_total", provider="llm")
            return hits[:top_k]
        # 모델이 빠뜨린 번호는 원래 순서 그대로 뒤에 붙인다.
        order += [i for i in range(1, len(hits) + 1) if i not in order]
        return [hits[i - 1] for i in order][:top_k]


def get_reranker(name: str | None = None, llm=None) -> Reranker:
    name = (name or settings.rerank_provider).lower()
    if name in {"none", "off", ""}:
        return NoopReranker()
    if name == "rules":
        return RulesReranker()
    if name == "llm":
        return LLMReranker(llm)
    raise ValueError(f"알 수 없는 리랭커: {name!r}")
