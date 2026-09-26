"""평가 실행.

    python -m app.evaluation.run                                  # 검색만 (빠름)
    python -m app.evaluation.run --modes dense,sparse,hybrid,hybrid+rerank
    python -m app.evaluation.run --answers                         # 답변 생성까지 채점
    python -m app.evaluation.run --answers --baseline              # 무검색 대비군과 비교
    python -m app.evaluation.run --compare data/eval/results/A.json
    python -m app.evaluation.run --report data/eval/report.md

검색만 돌리는 모드가 기본이다. Recall 이 생성 품질의 천장이므로 여기부터
보는 것이 맞고, 자동 채점이라 초 단위로 끝난다.

`--modes` 로 경로를 하나씩 끄고 켜면 "하이브리드가 낫다"를 주장이 아니라
숫자로 보일 수 있다. 사내 도입 검토에서 결국 요구되는 것이 이 표다.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

from app.config import settings
from app.db.session import connect
from app.evaluation.dataset import DatasetError, Question, load_questions
from app.evaluation.metrics import (
    DEFAULT_KS, QuestionScore, Report, compare, format_report, hit_matches,
    matched_ranks,
)
from app.observability import configure_logging
from app.providers.base import get_embedding_provider, get_llm_provider
from app.retrieval.answer import NOT_FOUND, answer_question, retrieve

DEFAULT_SET = Path("data/eval/questions.yaml")
RESULTS_DIR = Path("data/eval/results")

# 모드 -> (검색 경로, 리랭커)
MODES: dict[str, tuple[tuple[str, ...] | None, str]] = {
    "dense": (("dense",), "none"),
    "sparse": (("sparse",), "none"),
    "exact": (("exact",), "none"),
    "trigram": (("trigram",), "none"),
    "hybrid": (None, "none"),
    "hybrid+rerank": (None, "rules"),
    "hybrid+llm": (None, "llm"),
}
DEFAULT_MODE = "hybrid+rerank"


def _said_not_found(text: str) -> bool:
    return NOT_FOUND.rstrip(".") in text


def _citation_precision(sources, answers) -> float | None:
    """제시한 출처 중 실제 정답 위치를 덮는 비율.

    출처가 틀리면 사용자는 원문을 확인할 수 없고, 그 답변은 검증 불가능하다.
    답변 문장이 맞았는지와 별개로 따로 재야 한다.
    """
    if not sources:
        return None
    return sum(1 for s in sources if hit_matches(s, answers)) / len(sources)


def evaluate_mode(
    questions: list[Question],
    mode: str,
    with_answers: bool,
    baseline: bool,
    top_k: int,
    embedder,
    llm,
) -> tuple[Report, Report | None]:
    paths, reranker = MODES[mode]
    scores: list[QuestionScore] = []
    base_scores: list[QuestionScore] = []

    with connect() as conn:
        for q in questions:
            hits, aq, timings = retrieve(
                conn, q.question, embedder, top_k=top_k,
                reranker=reranker, paths=paths,
            )
            ranks = [] if q.unanswerable else matched_ranks(hits, q.answers)
            score = QuestionScore(
                id=q.id,
                rank=ranks[0] if ranks else None,
                tags=q.tags,
                unanswerable=q.unanswerable,
                ranks=ranks,
                retrieved=len(hits),
                latency_ms=dict(timings),
            )

            if with_answers:
                res = answer_question(
                    conn, q.question, embedder, llm, top_k=top_k,
                    reranker=reranker, log_query=False,
                )
                score.latency_ms = dict(res.latency_ms)
                score.grounded = res.grounded
                if q.unanswerable:
                    score.answered_not_found = _said_not_found(res.answer)
                else:
                    score.citation_ok = _citation_precision(res.sources, q.answers)
                    if q.expect:
                        score.expect_hit = q.expect.lower() in res.answer.lower()

                if baseline:
                    b = answer_question(
                        conn, q.question, embedder, llm, top_k=top_k,
                        use_rag=False, log_query=False,
                    )
                    bs = QuestionScore(
                        id=q.id, rank=None, tags=q.tags, unanswerable=q.unanswerable,
                        latency_ms=dict(b.latency_ms),
                    )
                    if q.unanswerable:
                        bs.answered_not_found = _said_not_found(b.answer)
                    elif q.expect:
                        bs.expect_hit = q.expect.lower() in b.answer.lower()
                    base_scores.append(bs)

            scores.append(score)
            print(f"  [{mode}] {q.id} rank={score.rank}", file=sys.stderr)

    return (
        Report(scores, mode=mode),
        Report(base_scores, retrieval=False, mode="no-retrieval") if baseline else None,
    )


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="검색·답변 품질 평가")
    ap.add_argument("--questions", type=Path, default=DEFAULT_SET)
    ap.add_argument("--top-k", type=int, default=max(DEFAULT_KS))
    ap.add_argument("--answers", action="store_true", help="답변 생성까지 채점")
    ap.add_argument("--baseline", action="store_true",
                    help="무검색(use_rag=false) 대비군과 비교. --answers 필요")
    ap.add_argument("--modes", default=DEFAULT_MODE,
                    help="쉼표로 구분. " + " | ".join(MODES))
    ap.add_argument("--label", default="", help="결과 파일에 붙일 이름")
    ap.add_argument("--compare", type=Path,
                    help="이전 결과 JSON 과 비교한다(같은 모드끼리)")
    ap.add_argument("--report", type=Path, help="마크다운 보고서를 이 경로에 쓴다")
    args = ap.parse_args(argv)

    configure_logging()
    if args.baseline and not args.answers:
        ap.error("--baseline 은 --answers 와 함께 써야 한다.")

    modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    unknown = [m for m in modes if m not in MODES]
    if unknown:
        ap.error(f"알 수 없는 모드: {unknown}. 가능: {', '.join(MODES)}")

    try:
        questions = load_questions(args.questions)
    except DatasetError as exc:
        print(exc, file=sys.stderr)
        return 2

    embedder = get_embedding_provider()
    llm = get_llm_provider() if (args.answers or args.baseline) else None
    if args.answers and len(modes) > 1:
        print(f"주의: 모드 {len(modes)}개 × 질문 {len(questions)}개만큼 생성 호출이 "
              f"일어난다. 시간이 걸린다.", file=sys.stderr)

    reports: dict[str, Report] = {}
    base: Report | None = None
    for mode in modes:
        report, mode_base = evaluate_mode(
            questions, mode, args.answers, args.baseline, args.top_k, embedder, llm
        )
        report.label = args.label
        reports[mode] = report
        base = base or mode_base

    print()
    for mode, report in reports.items():
        print(format_report(report, f"=== {mode} ==="))
        print()
    if base:
        print(format_report(base, "=== 무검색 대비군 (동일 모델) ==="))
        primary = reports[modes[-1]]
        a, b = primary.answer_accuracy(), base.answer_accuracy()
        if a is not None and b is not None:
            print(f"\n답변 정확도: 무검색 {b:.3f} -> RAG {a:.3f}  (차이 {a - b:+.3f})")

    payload = {
        "label": args.label,
        "ran_at": datetime.now().isoformat(timespec="seconds"),
        "questions_file": str(args.questions),
        "top_k": args.top_k,
        "embedding_model": settings.embedding_model,
        "llm_model": settings.llm_model if args.answers else None,
        "reranker_default": settings.rerank_provider,
        "modes": {mode: report.to_dict() for mode, report in reports.items()},
        "baseline": base.to_dict() if base else None,
        # 같은 코퍼스·같은 설정에서 재현할 수 있어야 비교가 의미를 갖는다.
        "settings": {
            "chunk_target_chars": settings.chunk_target_chars,
            "search_candidates": settings.search_candidates,
            "rerank_candidates": settings.rerank_candidates,
            "weights": {"dense": settings.weight_dense, "sparse": settings.weight_sparse,
                        "trigram": settings.weight_trigram},
            "query_expansion": settings.query_expansion,
            "grounding_mode": settings.grounding_mode,
        },
    }

    if args.compare:
        try:
            previous = json.loads(args.compare.read_text(encoding="utf-8"))
        except Exception as exc:
            print(f"비교 대상을 읽을 수 없다: {exc}", file=sys.stderr)
            return 2
        print("\n=== 변경 전후 비교 ===")
        for mode, report in reports.items():
            before = (previous.get("modes") or {}).get(mode)
            if not before:
                print(f"  {mode}: 이전 결과에 없음")
                continue
            print(f"  [{mode}]  ({args.compare.name} -> 현재)")
            print(compare(before, report.to_dict()))

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    name = f"{stamp}{'-' + args.label if args.label else ''}.json"
    out = RESULTS_DIR / name
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n결과 저장: {out}")

    if args.report:
        from app.evaluation.report import write_report

        write_report(payload, args.report)
        print(f"보고서 저장: {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
