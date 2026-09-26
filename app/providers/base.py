"""제공자 추상화.

인터페이스를 의도적으로 좁게 유지한다. 특정 제공자에만 있는 기능을 여기 넣는
순간 추상화가 무너진다(decisions.md D3).

교체 가능성은 두 번째 어댑터를 붙여 보기 전까지 증명되지 않는다. 그래서
`fake` 를 테스트 전용이 아니라 정식 두 번째 구현으로 유지한다. 색인·검색·
API 경로 전체가 제공자를 모른 채 동작하는지 매번 확인하는 장치다.
"""
from __future__ import annotations

from typing import Iterator, Protocol, Sequence, runtime_checkable


@runtime_checkable
class LLMProvider(Protocol):
    """답변 생성기. 임베딩과 달리 교체해도 재색인이 필요 없다."""

    @property
    def name(self) -> str: ...

    def generate(self, system: str, user: str) -> str: ...

    def generate_stream(self, system: str, user: str) -> Iterator[str]:
        """토큰 단위 생성.

        선택 기능이다. 지원하지 않는 제공자는 `generate` 결과를 한 번에
        내보내면 된다. 스트리밍을 필수로 만들면 어댑터 진입 장벽이 올라간다.
        """
        ...


@runtime_checkable
class EmbeddingProvider(Protocol):
    """임베딩 생성기.

    주의: 이 구현을 바꾸면 벡터 공간이 달라져 전체 재색인이 필요하다.
    """

    @property
    def dimension(self) -> int: ...

    def embed(self, texts: Sequence[str]) -> list[list[float]]: ...


def get_embedding_provider(name: str | None = None) -> EmbeddingProvider:
    from app.config import settings

    name = (name or settings.embedding_provider).lower()
    if name == "ollama":
        from app.providers.ollama import OllamaEmbeddingProvider

        return OllamaEmbeddingProvider()
    if name == "fake":
        from app.providers.fake import FakeEmbeddingProvider

        return FakeEmbeddingProvider()
    raise ValueError(f"알 수 없는 임베딩 제공자: {name!r}")


def get_llm_provider(name: str | None = None) -> LLMProvider:
    from app.config import settings

    name = (name or settings.llm_provider).lower()
    if name == "ollama":
        from app.providers.ollama import OllamaLLMProvider

        return OllamaLLMProvider()
    if name == "fake":
        from app.providers.fake import EchoLLMProvider

        return EchoLLMProvider()
    raise ValueError(f"알 수 없는 LLM 제공자: {name!r}")
