"""여러 평가 결과를 하나의 보고서로 합친다.

    python tools/build_report.py data/eval/results/*.json -o docs/eval-report.md

왜 필요한가. 검색만 재는 실행(경로별 ablation)은 몇 초면 끝나지만 답변까지
재는 실행은 모드마다 질문 수만큼 생성 호출이 붙는다. 그래서 실무에서는
"경로별 비교는 전 모드로, 답변 품질은 한 모드로" 나눠 돌리게 된다. 보고서는
그 둘을 한 장에 놓아야 읽을 수 있다.

뒤에 오는 파일이 같은 모드를 덮어쓴다. 메타데이터(모델·정답표)는 가장 마지막
파일 것을 쓰되, 생성 모델명은 실제로 답변을 측정한 파일에서 가져온다.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.evaluation.report import write_report  # noqa: E402


def merge(paths: list[Path]) -> dict:
    merged: dict = {"modes": {}}
    for path in paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        modes = payload.pop("modes", {}) or {}
        baseline = payload.pop("baseline", None)
        merged.update({k: v for k, v in payload.items() if v is not None})
        merged["modes"].update(modes)
        if baseline:
            merged["baseline"] = baseline
        # 생성 모델명은 답변을 실제로 측정한 실행에서만 의미가 있다.
        if any(m.get("answer_accuracy") is not None for m in modes.values()):
            merged["llm_model"] = payload.get("llm_model") or merged.get("llm_model")
    if not merged["modes"]:
        raise SystemExit("합칠 측정 결과가 없다.")
    # 경로 단독 → 결합 → 재정렬 순으로 읽히게 정렬한다.
    order = ["dense", "sparse", "exact", "trigram", "hybrid", "hybrid+rerank", "hybrid+llm"]
    merged["modes"] = dict(
        sorted(merged["modes"].items(),
               key=lambda kv: (order.index(kv[0]) if kv[0] in order else 99, kv[0]))
    )
    return merged


def main() -> int:
    ap = argparse.ArgumentParser(description="평가 결과 병합 보고서")
    ap.add_argument("results", nargs="+", type=Path)
    ap.add_argument("-o", "--out", type=Path, default=Path("docs/eval-report.md"))
    args = ap.parse_args()

    payload = merge(args.results)
    write_report(payload, args.out)
    print(f"모드 {len(payload['modes'])}개를 합쳐 보고서를 만들었다: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
