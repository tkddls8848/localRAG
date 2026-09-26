"""실행 중인 로컬 RAG 를 실제 요청으로 점검한다.

배포 직후 "무엇이 깨졌는지"를 1분 안에 판단하기 위한 도구다. 자동 테스트가
검증하지 못하는 것 — 진짜 모델, 진짜 색인, 진짜 HTTP — 만 본다.

    python tools/check_local.py
    python tools/check_local.py --question "SR650 V4 최대 메모리 용량은?" --key $API_KEY
"""
from __future__ import annotations

import argparse
import json
import sys

import httpx

FAIL = "[!!]"
OK = "[OK]"


def _headers(key: str) -> dict:
    return {"authorization": f"Bearer {key}"} if key else {}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--question", default="SR630 V4 최대 메모리 용량은?")
    parser.add_argument("--expect", default="8TB",
                        help="답변에 반드시 있어야 하는 문자열")
    parser.add_argument("--key", default="", help="API 키(인증을 켰다면 필요)")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    failures: list[str] = []
    headers = _headers(args.key)

    with httpx.Client(base_url=args.url, timeout=300, headers=headers) as client:
        # 1) 준비 상태
        ready = client.get("/ready")
        checks = ready.json().get("checks", {})
        if ready.status_code != 200:
            failures.append(f"준비되지 않음: {checks}")
        print(f"{OK if ready.status_code == 200 else FAIL} /ready  {checks}")

        # 2) 색인 상태 — 실패한 문서가 남아 있으면 안 된다
        documents = client.get("/documents").json()["documents"]
        broken = [d["title"] for d in documents if d["status"] != "indexed"]
        review = [d["title"] for d in documents if d.get("needs_review")]
        chunks = sum(d["chunks"] for d in documents)
        print(f"{FAIL if broken else OK} /documents  문서 {len(documents)}건 "
              f"청크 {chunks:,}개" + (f"  색인 실패: {broken}" if broken else ""))
        if broken:
            failures.append(f"색인되지 않은 문서: {broken}")
        if review:
            print(f"     검토 필요(스캔 의심): {review}")

        # 3) 검색만 — 생성 품질과 분리해서 본다
        diagnosis = client.post("/search", json={"question": args.question}).json()
        if not diagnosis.get("hits"):
            failures.append("검색 후보가 0건이다")
        print(f"{FAIL if not diagnosis.get('hits') else OK} /search  "
              f"의도 {diagnosis.get('intent')}  어휘 {diagnosis.get('terms')}  "
              f"후보 {len(diagnosis.get('hits', []))}건")

        # 4) 실제 답변과 근거
        answer = client.post("/ask", json={"question": args.question, "top_k": 5}).json()
        if args.verbose:
            print(json.dumps(answer, ensure_ascii=False, indent=2))
        if not answer.get("sources"):
            failures.append("답변에 근거가 없다")
        elif args.expect and args.expect.lower() not in answer["answer"].lower():
            failures.append(f"답변에 기대 문자열 {args.expect!r} 이 없다: {answer['answer'][:120]}")
        if answer.get("grounded") is False:
            failures.append(f"발췌에서 확인되지 않은 값: {answer.get('unsupported')}")
        print(f"{FAIL if failures and '답변' in failures[-1] else OK} /ask  "
              f"{answer['answer'][:80]}")
        print(f"     출처 {len(answer.get('sources', []))}건  "
              f"근거검증 {answer.get('grounded')}  지연 {answer.get('latency_ms')}")

        # 5) 없는 것을 지어내지 않는가 — 가장 중요한 검사다
        unknown = client.post("/ask", json={"question": "SR999 V9 스펙은?"}).json()
        honest = not unknown.get("sources") and "찾지 못했" in unknown["answer"]
        if not honest:
            failures.append(f"없는 모델에 답을 지어냈다: {unknown['answer'][:120]}")
        print(f"{OK if honest else FAIL} 정직도  미등록 모델 질문 → "
              f"{unknown['answer'][:60]}")

    print()
    if failures:
        for item in failures:
            print(f"{FAIL} {item}", file=sys.stderr)
        return 1
    print(f"{OK} 준비 상태 · 색인 · 검색 · 답변 · 출처 · 정직도 모두 확인 완료")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
