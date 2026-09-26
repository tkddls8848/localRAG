"""테스트용 결정적 임베딩.

모델 런타임 없이 색인·검색 경로를 통째로 돌려보기 위한 구현이다.
해시 기반 bag-of-words 라 의미 유사도는 빈약하지만, 같은 토큰을 공유하는
문서가 가까워지므로 SQL 과 파이프라인의 정합성을 확인하기에는 충분하다.

운영에 쓰면 안 된다. EMBEDDING_PROVIDER=fake 로만 켜진다.
"""
from __future__ import annotations

import hashlib
import math
import re
from typing import Sequence

from app.config import settings

_TOKEN = re.compile(r"[A-Za-z0-9]+|[가-힣]+")


class FakeEmbeddingProvider:
    def __init__(self, dimension: int | None = None) -> None:
        self._dimension = dimension or settings.embedding_dim

    @property
    def dimension(self) -> int:
        return self._dimension

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        return [self._one(t) for t in texts]

    def _one(self, text: str) -> list[float]:
        vec = [0.0] * self._dimension
        for tok in _TOKEN.findall(text.lower()):
            h = hashlib.blake2b(tok.encode(), digest_size=8).digest()
            idx = int.from_bytes(h[:4], "big") % self._dimension
            sign = 1.0 if h[4] & 1 else -1.0
            vec[idx] += sign
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        return [v / norm for v in vec]


class EchoLLMProvider:
    """검색 결과만 확인하고 싶을 때 쓰는 더미 생성기."""

    @property
    def name(self) -> str:
        return "fake/echo"

    def generate(self, system: str, user: str) -> str:
        blocks = len(re.findall(r"^\[\d+\] 출처:", user, re.MULTILINE))
        return f"(테스트 생성기) 발췌 {blocks}건을 받았다." + (" [1]" if blocks else "")

    def generate_stream(self, system: str, user: str):
        """스트리밍 경로도 제공자를 모른 채 시험할 수 있어야 한다."""
        text = self.generate(system, user)
        for i in range(0, len(text), 8):
            yield text[i : i + 8]
