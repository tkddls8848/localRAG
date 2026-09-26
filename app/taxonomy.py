"""제품 모델 식별 규칙.

색인 단계(청크에 모델 태그를 붙일 때)와 검색 단계(질문에서 모델을 감지할 때)가
**같은 규칙**을 써야 한다. 두 곳에 규칙을 따로 두면 조용히 어긋나고, 그 결과는
이 프로젝트가 막으려는 바로 그 증상 — SR650a V4 질문에 SR650i V4 행이 섞이는
것 — 으로 나타난다. 그래서 규칙을 한 파일에 둔다.

사내 도입 시 제품 체계가 다르면 이 파일만 바꾸면 된다.
"""
from __future__ import annotations

import re

# 예: "SR650a V4", "SR630 V4", "ST250 V3"
MODEL_RE = re.compile(r"\b([A-Z]{2}\d{2,4}[a-z]?)\s+(V\d+)\b")

PREFIX = "ThinkSystem"


def extract_products(text: str) -> list[str]:
    """텍스트에서 제품명을 뽑는다(중복 제거, 등장 순서 유지)."""
    return list(
        dict.fromkeys(
            f"{PREFIX} {m.group(1)} {m.group(2)}" for m in MODEL_RE.finditer(text)
        )
    )


def match_products(text: str, candidates: list[str]) -> list[str]:
    """본문이 특정 모델만 가리키면 그 모델로 좁힌다.

    문서가 SR650a/SR650i 를 함께 다루더라도, 개별 표 행은 대개 한쪽 이야기다.
    두 모델 다 나오거나 하나도 없으면 좁히지 않는다. 잘못 좁히면 검색에서
    영영 걸리지 않는 청크가 생긴다.
    """
    hits = [c for c in candidates if c.split()[1] in text]
    return hits if len(hits) == 1 else candidates
