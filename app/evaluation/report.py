"""평가 결과 → 마크다운 보고서.

PoC 의 결론은 "돌아간다"가 아니라 "쓸 만한가"여야 한다(architecture.md 7).
그 판단을 하는 사람은 대개 코드를 읽지 않으므로, 측정 결과를 그대로 붙여
쓸 수 있는 문서로 내보낸다.

이 보고서가 담는 것은 숫자와 실패 목록뿐이다. 결론은 사람이 쓴다.
"""
from __future__ import annotations

import json
from pathlib import Path


def _num(value, digits: int = 3) -> str:
    if value is None:
        return "–"
    return f"{value:.{digits}f}" if isinstance(value, float) else str(value)


def _ms(value) -> str:
    return "–" if value is None else f"{value:,.0f}"


def build_report(payload: dict) -> str:
    modes: dict = payload.get("modes") or {}
    if not modes:
        return "# 평가 보고서\n\n측정 결과가 없다.\n"

    ks = sorted(
        int(k.lstrip("@"))
        for k in (next(iter(modes.values())).get("recall") or {})
    )
    top = max(ks) if ks else 10
    lines: list[str] = []

    lines.append("# 로컬 RAG 품질 측정 보고서")
    lines.append("")
    lines.append(f"- 측정 시각: {payload.get('ran_at', '-')}")
    lines.append(f"- 정답표: `{payload.get('questions_file', '-')}`")
    lines.append(f"- 임베딩 모델: `{payload.get('embedding_model', '-')}`")
    lines.append(f"- 생성 모델: `{payload.get('llm_model') or '(생성 미측정)'}`")
    lines.append(f"- top_k: {payload.get('top_k', '-')}")
    first = next(iter(modes.values()))
    lines.append(
        f"- 질문 {first.get('questions')}개 "
        f"(정답 있음 {first.get('answerable')} / 함정 {first.get('unanswerable')})"
    )
    lines.append("")

    lines.append("## 1. 검색 경로별 비교")
    lines.append("")
    lines.append(
        "하이브리드 검색이 필요한지는 주장이 아니라 이 표로 판단한다. "
        "`dense` 는 벡터 단독, `sparse` 는 전문검색 단독, `exact` 는 식별자 "
        "정확 일치, `hybrid` 는 RRF 결합, `+rerank` 는 재정렬까지다."
    )
    lines.append("")
    header = ["모드"] + [f"Recall@{k}" for k in ks] + [f"nDCG@{top}", "MRR"]
    lines.append("| " + " | ".join(header) + " |")
    lines.append("|" + "---|" * len(header))
    for mode, data in modes.items():
        row = [f"`{mode}`"]
        row += [_num((data.get("recall") or {}).get(f"@{k}")) for k in ks]
        row.append(_num((data.get("ndcg") or {}).get(f"@{top}")))
        row.append(_num(data.get("mrr")))
        lines.append("| " + " | ".join(row) + " |")
    lines.append("")

    answered = {m: d for m, d in modes.items() if d.get("answer_accuracy") is not None}
    if answered:
        lines.append("## 2. 답변 품질")
        lines.append("")
        lines.append("| 모드 | 답변 정확도 | 출처 정확도 | 근거 일치율 | 정직도 |")
        lines.append("|---|---|---|---|---|")
        for mode, data in answered.items():
            lines.append(
                f"| `{mode}` | {_num(data.get('answer_accuracy'))} "
                f"| {_num(data.get('citation_precision'))} "
                f"| {_num(data.get('grounded_rate'))} "
                f"| {_num(data.get('honesty'))} |"
            )
        baseline = payload.get("baseline")
        if baseline:
            lines.append(
                f"| `무검색 대비군` | {_num(baseline.get('answer_accuracy'))} | – | – "
                f"| {_num(baseline.get('honesty'))} |"
            )
            lines.append("")
            lines.append(
                "대비군은 같은 모델에 검색만 끄고 같은 질문을 물은 결과다. "
                "RAG 의 효과는 이 차이로만 말할 수 있다."
            )
        lines.append("")

    lines.append("## 3. 응답 시간")
    lines.append("")
    lines.append("품질과 분리해서 기록한다. 느린 것을 나쁜 것으로 오진하지 않기 위해서다.")
    lines.append("")
    lines.append("| 모드 | 단계 | p50(ms) | p95(ms) |")
    lines.append("|---|---|---|---|")
    for mode, data in modes.items():
        for stage, value in (data.get("latency_ms") or {}).items():
            if value.get("p50") is None:
                continue
            lines.append(
                f"| `{mode}` | {stage} | {_ms(value.get('p50'))} | {_ms(value.get('p95'))} |"
            )
    lines.append("")

    tag_rows = []
    for mode, data in modes.items():
        tags = (data.get("by_tag") or {}).get(f"@{top}") or {}
        for tag, value in tags.items():
            tag_rows.append((mode, tag, value))
    if tag_rows:
        lines.append(f"## 4. 질문 유형별 Recall@{top}")
        lines.append("")
        lines.append("어떤 유형에서 깨지는지 보면 다음에 무엇을 고칠지 정할 수 있다.")
        lines.append("")
        lines.append("| 모드 | 태그 | Recall |")
        lines.append("|---|---|---|")
        for mode, tag, value in tag_rows:
            lines.append(f"| `{mode}` | {tag} | {_num(value)} |")
        lines.append("")

    lines.append("## 5. 검색 실패 질문")
    lines.append("")
    lines.append(
        "정답 문단을 상위 후보에 올리지 못한 질문이다. **여기부터 고친다.** "
        "검색이 실패하면 생성은 무조건 실패한다."
    )
    lines.append("")
    for mode, data in modes.items():
        failures = data.get("retrieval_failures") or []
        if failures:
            lines.append(f"- `{mode}`: {', '.join(failures)}")
        else:
            lines.append(f"- `{mode}`: 없음")
    lines.append("")

    lines.append("## 6. 해석 시 주의")
    lines.append("")
    answerable = first.get("answerable") or 0
    step = round(1 / answerable, 3) if answerable else None
    lines.append(
        f"- **표본이 작다.** 정답 있는 질문 {answerable}개이므로 질문 하나가 "
        f"{step if step else '?'} 만큼 지표를 움직인다. 2%p 미만의 차이를 "
        f"개선이라고 부르면 안 된다."
    )
    lines.append(
        "- **정답표를 보고 튜닝하면 점수가 부풀려진다.** 용어집 어휘나 가중치를 "
        "이 정답표의 실패 사례에서 고쳤다면, 본 적 없는 질문에서는 이보다 낮게 "
        "나온다. 정직한 검증은 `query_log` 에 쌓인 새 질문으로 다시 재는 것이다."
    )
    lines.append(
        "- **출처 정확도는 엄격한 정의다.** 답변이 인용한 발췌 **각각**이 정답 "
        "위치를 덮는 비율이므로, 여러 발췌를 함께 인용하면 낮게 나온다. 답이 "
        "맞았는지와 별개로 읽어야 한다."
    )
    lines.append(
        "- **응답 시간은 품질과 분리해서 본다.** 첫 질문이 느린 것은 대개 모델 "
        "로딩이다(`OLLAMA_KEEP_ALIVE`)."
    )
    lines.append("")

    lines.append("## 7. 재현 정보")
    lines.append("")
    lines.append("```json")
    lines.append(json.dumps(payload.get("settings") or {}, ensure_ascii=False, indent=2))
    lines.append("```")
    lines.append("")
    return "\n".join(lines)


def write_report(payload: dict, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(build_report(payload), encoding="utf-8")
    return path
