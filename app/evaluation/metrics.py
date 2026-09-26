"""채점.

검색이 실패하면 생성은 무조건 실패한다. 그래서 Recall 을 먼저 본다.
답변이 나쁠 때 프롬프트부터 손대면 대개 시간을 낭비한다.

지표를 이렇게 나눈 이유:

| 지표 | 무엇을 판단하는가 |
|---|---|
| Recall@k | 정답이 상위 k 안에 들어왔는가. **생성 품질의 천장** |
| MRR / nDCG@k | 몇 번째로 맞혔는가. 순서가 나쁘면 LLM 이 놓친다 |
| Precision@k | 가져온 것 중 쓸모 있는 비율. 낮으면 컨텍스트가 소음이다 |
| 출처 정확도 | 답변이 제시한 출처가 실제 정답 위치인가 |
| 답변 정확도 | 기대 문자열이 답변에 있는가 |
| 정직도 | 답이 없는 질문에 "찾지 못했다"고 답했는가 |
| 지연 | 품질과 분리해서 기록한다. 느린 것을 나쁜 것으로 오진하지 않기 위해 |
"""
from __future__ import annotations

import math
import statistics
from dataclasses import dataclass, field
from typing import Protocol, Sequence

DEFAULT_KS = (1, 3, 5, 10)


class HitLike(Protocol):
    doc_title: str
    page_from: int
    page_to: int


class AnswerLike(Protocol):
    document: str
    pages: list[int]


def hit_matches(hit: HitLike, answers: Sequence[AnswerLike]) -> bool:
    """청크가 정답 위치를 덮는가.

    청크는 페이지를 걸칠 수 있으므로 범위 겹침으로 판정한다.
    """
    for a in answers:
        if hit.doc_title != a.document:
            continue
        if any(hit.page_from <= p <= hit.page_to for p in a.pages):
            return True
    return False


def matched_ranks(hits: Sequence[HitLike], answers: Sequence[AnswerLike]) -> list[int]:
    """정답 위치를 덮은 모든 순위(1-based)."""
    return [i for i, hit in enumerate(hits, start=1) if hit_matches(hit, answers)]


def matched_rank(hits: Sequence[HitLike], answers: Sequence[AnswerLike]) -> int | None:
    """정답을 처음 맞힌 순위(1-based). 못 찾으면 None."""
    ranks = matched_ranks(hits, answers)
    return ranks[0] if ranks else None


@dataclass
class QuestionScore:
    id: str
    rank: int | None                  # 정답 문단의 첫 순위
    tags: list[str] = field(default_factory=list)
    unanswerable: bool = False
    answered_not_found: bool | None = None   # 정직도: "찾지 못했다"고 답했는가
    expect_hit: bool | None = None            # 기대 문자열이 답변에 있는가
    ranks: list[int] = field(default_factory=list)   # 정답을 덮은 모든 순위
    retrieved: int = 0
    # 답변이 제시한 출처가 실제 정답 위치였는가. 출처가 틀리면 사용자는
    # 원문을 확인할 수 없고, 그 답변은 검증 불가능하다.
    citation_ok: bool | None = None
    grounded: bool | None = None
    latency_ms: dict[str, float] = field(default_factory=dict)

    def recall_at(self, k: int) -> bool:
        return self.rank is not None and self.rank <= k

    def precision_at(self, k: int) -> float:
        if not self.retrieved:
            return 0.0
        return sum(1 for r in self.ranks if r <= k) / min(k, self.retrieved)

    def ndcg_at(self, k: int) -> float:
        """이진 적합도 nDCG.

        이상적인 순서는 '찾아낸 정답들이 맨 위에 붙어 있는 경우'로 잡는다.
        전체 정답 개수를 모르는 상태에서 계산 가능한 정의이며, 재현율이 아니라
        **순서 품질**만 본다. 재현율은 Recall 로 따로 본다.
        """
        found = [r for r in self.ranks if r <= k]
        if not found:
            return 0.0
        dcg = sum(1.0 / math.log2(r + 1) for r in found)
        ideal = sum(1.0 / math.log2(i + 1) for i in range(1, len(found) + 1))
        return dcg / ideal if ideal else 0.0


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = min(len(ordered) - 1, int(round(q * (len(ordered) - 1))))
    return round(ordered[index], 1)


@dataclass
class Report:
    scores: list[QuestionScore]
    ks: tuple[int, ...] = DEFAULT_KS
    # 무검색 대비군은 검색 단계가 없다. Recall 을 0 으로 표시하면
    # "검색에 실패했다"로 오독되므로 아예 해당 없음으로 다룬다.
    retrieval: bool = True
    label: str = ""
    mode: str = "hybrid+rerank"

    @property
    def answerable(self) -> list[QuestionScore]:
        return [s for s in self.scores if not s.unanswerable]

    @property
    def traps(self) -> list[QuestionScore]:
        return [s for s in self.scores if s.unanswerable]

    def recall(self, k: int) -> float | None:
        rows = self.answerable
        if not self.retrieval or not rows:
            return None
        return sum(s.recall_at(k) for s in rows) / len(rows)

    def precision(self, k: int) -> float | None:
        rows = self.answerable
        if not self.retrieval or not rows:
            return None
        return _mean([s.precision_at(k) for s in rows])

    def ndcg(self, k: int) -> float | None:
        rows = self.answerable
        if not self.retrieval or not rows:
            return None
        return _mean([s.ndcg_at(k) for s in rows])

    def mrr(self) -> float | None:
        rows = self.answerable
        if not self.retrieval or not rows:
            return None
        return sum(1.0 / s.rank if s.rank else 0.0 for s in rows) / len(rows)

    def answer_accuracy(self) -> float | None:
        rows = [s for s in self.answerable if s.expect_hit is not None]
        return sum(s.expect_hit for s in rows) / len(rows) if rows else None

    def citation_precision(self) -> float | None:
        rows = [s for s in self.answerable if s.citation_ok is not None]
        return sum(s.citation_ok for s in rows) / len(rows) if rows else None

    def grounded_rate(self) -> float | None:
        rows = [s for s in self.scores if s.grounded is not None]
        return sum(s.grounded for s in rows) / len(rows) if rows else None

    def honesty(self) -> float | None:
        """답이 없는 질문에 '찾지 못했다'고 답한 비율."""
        rows = [s for s in self.traps if s.answered_not_found is not None]
        return sum(s.answered_not_found for s in rows) / len(rows) if rows else None

    def latency(self, stage: str) -> dict[str, float | None]:
        values = [s.latency_ms[stage] for s in self.scores if stage in s.latency_ms]
        return {
            "p50": _percentile(values, 0.5),
            "p95": _percentile(values, 0.95),
            "mean": round(statistics.mean(values), 1) if values else None,
        }

    def by_tag(self, k: int) -> dict[str, float]:
        if not self.retrieval:
            return {}
        buckets: dict[str, list[QuestionScore]] = {}
        for s in self.answerable:
            for t in s.tags:
                buckets.setdefault(t, []).append(s)
        return {
            t: sum(x.recall_at(k) for x in rows) / len(rows)
            for t, rows in sorted(buckets.items())
        }

    def failures(self) -> list[str]:
        """정답을 못 찾은 질문 id. 다음 개선 대상 목록이다."""
        return [s.id for s in self.answerable if s.rank is None]

    def to_dict(self) -> dict:
        top = max(self.ks)
        return {
            "label": self.label,
            "mode": self.mode,
            "questions": len(self.scores),
            "answerable": len(self.answerable),
            "unanswerable": len(self.traps),
            "recall": {f"@{k}": self.recall(k) for k in self.ks},
            "precision": {f"@{k}": self.precision(k) for k in self.ks},
            "ndcg": {f"@{k}": self.ndcg(k) for k in self.ks},
            "mrr": self.mrr(),
            "answer_accuracy": self.answer_accuracy(),
            "citation_precision": self.citation_precision(),
            "grounded_rate": self.grounded_rate(),
            "honesty": self.honesty(),
            "latency_ms": {
                stage: self.latency(stage)
                for stage in ("embed", "search", "rerank", "generate", "total")
            },
            "by_tag": {f"@{top}": self.by_tag(top)},
            "retrieval_failures": self.failures(),
            "per_question": [
                {"id": s.id, "rank": s.rank, "ranks": s.ranks, "tags": s.tags,
                 "unanswerable": s.unanswerable, "expect_hit": s.expect_hit,
                 "citation_ok": s.citation_ok, "grounded": s.grounded,
                 "answered_not_found": s.answered_not_found,
                 "latency_ms": s.latency_ms}
                for s in self.scores
            ],
        }


def _pct(v: float | None) -> str:
    return "  -  " if v is None else f"{v:6.3f}"


def _ms(v: float | None) -> str:
    return "   -   " if v is None else f"{v:7.0f}"


def format_report(report: Report, title: str = "") -> str:
    lines = []
    if title:
        lines.append(title)
    lines.append(
        f"  질문 {len(report.scores)}개 "
        f"(정답 있음 {len(report.answerable)} / 함정 {len(report.traps)})"
    )
    if report.retrieval:
        for k in report.ks:
            lines.append(
                f"  @{k:<3} Recall {_pct(report.recall(k))}"
                f"  Precision {_pct(report.precision(k))}"
                f"  nDCG {_pct(report.ndcg(k))}"
            )
        lines.append(f"  MRR             {_pct(report.mrr())}")
    else:
        lines.append("  Recall/MRR       해당 없음 (검색 단계가 없음)")
    if report.answer_accuracy() is not None:
        lines.append(f"  답변 정확도      {_pct(report.answer_accuracy())}")
    if report.citation_precision() is not None:
        lines.append(f"  출처 정확도      {_pct(report.citation_precision())}")
    if report.grounded_rate() is not None:
        lines.append(f"  근거 일치율      {_pct(report.grounded_rate())}")
    if report.honesty() is not None:
        lines.append(f"  정직도           {_pct(report.honesty())}")

    stages = [(s, report.latency(s)) for s in
              ("embed", "search", "rerank", "generate", "total")]
    stages = [(s, v) for s, v in stages if v["p50"] is not None]
    if stages:
        lines.append("  지연(ms)         p50 /    p95")
        for stage, value in stages:
            lines.append(f"    {stage:<13} {_ms(value['p50'])} / {_ms(value['p95'])}")

    tags = report.by_tag(max(report.ks))
    if tags:
        lines.append(f"  태그별 Recall@{max(report.ks)}:")
        for t, v in tags.items():
            lines.append(f"    {t:<22} {_pct(v)}")
    if report.retrieval and report.failures():
        lines.append("  검색 실패 질문   " + ", ".join(report.failures()[:12]))
    return "\n".join(lines)


# 비교에서 이 폭 안의 변화는 잡음으로 본다. 질문 30~50개 규모에서 질문 하나가
# 2~3%p 를 움직이므로, 그보다 작은 차이를 개선이라 부르면 자기 기만이다.
NOISE = 0.02


def compare(before: dict, after: dict, ks: tuple[int, ...] = DEFAULT_KS) -> str:
    """두 평가 결과의 차이를 표로 만든다. 무엇을 바꿨을 때 실제로 나아졌는지 본다."""
    rows: list[tuple[str, float | None, float | None]] = []
    for k in ks:
        rows.append((f"Recall@{k}",
                     (before.get("recall") or {}).get(f"@{k}"),
                     (after.get("recall") or {}).get(f"@{k}")))
    rows.append((f"nDCG@{max(ks)}",
                 (before.get("ndcg") or {}).get(f"@{max(ks)}"),
                 (after.get("ndcg") or {}).get(f"@{max(ks)}")))
    for key, name in (("mrr", "MRR"), ("answer_accuracy", "답변 정확도"),
                      ("citation_precision", "출처 정확도"), ("honesty", "정직도")):
        rows.append((name, before.get(key), after.get(key)))

    lines = [f"  {'지표':<16}{'이전':>8}{'이후':>8}{'차이':>9}  판정"]
    for name, old, new in rows:
        if old is None and new is None:
            continue
        if old is None or new is None:
            lines.append(f"  {name:<16}{_pct(old):>8}{_pct(new):>8}{'':>9}  비교 불가")
            continue
        delta = new - old
        verdict = "개선" if delta > NOISE else ("악화" if delta < -NOISE else "차이 없음")
        lines.append(
            f"  {name:<16}{old:>8.3f}{new:>8.3f}{delta:>+9.3f}  {verdict}"
        )
    return "\n".join(lines)
