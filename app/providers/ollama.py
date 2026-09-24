"""Ollama 제공자 (임베딩 + 생성)."""
from __future__ import annotations

from typing import Sequence

import httpx

from app.config import settings


class OllamaEmbeddingProvider:
    def __init__(self, base_url: str | None = None, model: str | None = None) -> None:
        self._base_url = (base_url or settings.ollama_base_url).rstrip("/")
        self._model = model or settings.embedding_model
        self._dimension = settings.embedding_dim

    @property
    def dimension(self) -> int:
        return self._dimension

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []
        resp = httpx.post(
            f"{self._base_url}/api/embed",
            json={"model": self._model, "input": list(texts)},
            timeout=180.0,
        )
        resp.raise_for_status()
        vectors = resp.json()["embeddings"]

        if vectors and len(vectors[0]) != self._dimension:
            raise ValueError(
                f"임베딩 차원 불일치: 설정 {self._dimension}, 실제 {len(vectors[0])}. "
                f"EMBEDDING_DIM 을 맞추고 재색인하라."
            )
        return vectors


class OllamaLLMProvider:
    def __init__(self, base_url: str | None = None, model: str | None = None) -> None:
        self._base_url = (base_url or settings.ollama_base_url).rstrip("/")
        self._model = model or settings.llm_model

    @property
    def name(self) -> str:
        return f"ollama/{self._model}"

    def generate(self, system: str, user: str) -> str:
        resp = httpx.post(
            f"{self._base_url}/api/chat",
            json={
                "model": self._model,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                "stream": False,
                "think": False,
                "options": {"temperature": 0.0, "num_ctx": settings.llm_context_size,
                            "num_predict": 1024},
            },
            timeout=300.0,
        )
        resp.raise_for_status()
        return resp.json()["message"]["content"].strip()
