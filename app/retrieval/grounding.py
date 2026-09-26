"""근거 검증.

인용 번호가 붙어 있어도 그 안에 없는 수치를 쓸 수 있다. 실제로 가장 잘
일어나는 실패는 "출처는 맞는데 숫자가 다르다"다. 8TB 를 4TB 로 적은 답변은
출처가 붙어 있어서 오히려 더 위험하다.

그래서 답변에 등장한 **수치와 식별자**가 인용한 발췌 안에 실제로 있는지
문자열로 확인한다. 의미 검증이 아니라 표기 검증이다. 값을 그대로 옮기라는
규칙(answer.py 규칙 4)을 지켰는지만 본다.

검사 대상을 수치·식별자로 좁힌 이유는 오탐을 줄이기 위해서다. 문장 표현은
바꿔 쓰는 것이 정상이지만 스펙 값은 바꿔 쓸 이유가 없다.
"""
from __future__ import annotations

import re

# 인용 표시는 본문이 아니다. 검사 전에 지운다.
_CITATION = re.compile(r"\[\d+\]")

_UNIT = (
    "TB|GB|MB|PB|KB|Gbps|Gb/s|MT/s|MHz|GHz|W|V|A|U|mm|cm|kg|lb|inch|nm|rpm|BTU"
)
# 수치 + 단위. 문서군의 스펙 값은 거의 전부 이 모양이다.
_MEASURE = re.compile(r"\b\d+(?:[.,]\d+)?\s?(?:" + _UNIT + r")\b", re.IGNORECASE)
# 영문자와 숫자가 섞인 토큰. 파트번호·피처코드·규격코드.
_CODE = re.compile(r"\b(?=[A-Za-z0-9./\-]*\d)(?=[A-Za-z0-9./\-]*[A-Za-z])[A-Za-z0-9][A-Za-z0-9./\-]{2,}\b")


def _normalize(text: str) -> str:
    """공백·쉼표·대소문자 차이를 없앤다.

    문서는 "Up to 8TB", 답변은 "8 TB" 로 쓸 수 있다. 이것을 불일치로 보면
    검증이 소음만 만든다.
    """
    return re.sub(r"[\s,]+", "", text).lower()


def claims(answer: str) -> list[str]:
    """답변에서 검증할 수치·식별자를 뽑는다."""
    body = _CITATION.sub(" ", answer)
    found = [m.group(0) for m in _MEASURE.finditer(body)]
    found += [m.group(0) for m in _CODE.finditer(body)]
    out: list[str] = []
    for item in found:
        item = item.strip()
        if _normalize(item) and item not in out:
            out.append(item)
    return out


def check(answer: str, evidence: list[str]) -> list[str]:
    """발췌에서 찾을 수 없는 수치·식별자 목록을 돌려준다. 빈 목록이면 통과."""
    haystack = _normalize("\n".join(evidence))
    if not haystack:
        return claims(answer)
    return [c for c in claims(answer) if _normalize(c) not in haystack]
