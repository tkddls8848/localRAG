"""Ollama 제공자 (임베딩 + 생성).

PoC 에서 가장 자주 넘어지는 구성요소가 모델 런타임이다. 모델을 메모리에
올리는 동안 타임아웃이 나거나, 다른 프로세스가 GPU 를 물고 있거나,
`ollama pull` 이 덜 끝난 상태로 호출된다. 그때마다 색인 전체가 실패로
끝나면 787페이지를 처음부터 다시 돌려야 한다.

그래서 세 가지를 넣었다.

  - **연결 재사용** — 요청마다 TCP/TLS 를 새로 세우지 않는다
  - **재시도** — 일시적 실패(연결·타임아웃·5xx)는 지수 백오프로 다시 시도
  - **차단기** — 연속 실패가 임계를 넘으면 잠시 즉시 실패시킨다. 죽은
    런타임에 요청을 쌓아 대기열만 늘리는 것을 막는다
"""
from __future__ import annotations

import json
import logging
import threading
import time
from typing import Iterator, Sequence

import httpx

from app.config import settings
from app.observability import metrics

log = logging.getLogger(__name__)


class ModelUnavailable(RuntimeError):
    """모델 런타임에 도달할 수 없거나 연속 실패로 차단된 상태."""


class _Breaker:
    """연속 실패가 임계를 넘으면 쿨다운 동안 즉시 실패시킨다."""

    def __init__(self, threshold: int, cooldown: float) -> None:
        self.threshold = threshold
        self.cooldown = cooldown
        self._failures = 0
        self._open_until = 0.0
        self._lock = threading.Lock()

    def guard(self) -> None:
        with self._lock:
            if time.monotonic() < self._open_until:
                remaining = int(self._open_until - time.monotonic()) + 1
                raise ModelUnavailable(
                    f"모델 런타임이 연속 실패로 차단됐다. {remaining}초 후 재시도한다."
                )

    def record(self, ok: bool) -> None:
        with self._lock:
            if ok:
                self._failures = 0
                return
            self._failures += 1
            if self._failures >= self.threshold:
                self._open_until = time.monotonic() + self.cooldown
                self._failures = 0
                metrics.inc("model_breaker_opened_total")
                log.error("모델 런타임 차단기 작동", extra={"cooldown_s": self.cooldown})

    @property
    def state(self) -> str:
        return "open" if time.monotonic() < self._open_until else "closed"


_breaker = _Breaker(settings.breaker_threshold, settings.breaker_cooldown_s)
_client_lock = threading.Lock()
_client: httpx.Client | None = None


def _http() -> httpx.Client:
    global _client
    if _client is None:
        with _client_lock:
            if _client is None:
                _client = httpx.Client(
                    base_url=settings.ollama_base_url.rstrip("/"),
                    timeout=httpx.Timeout(settings.model_timeout_s, connect=10.0),
                    limits=httpx.Limits(max_keepalive_connections=8, max_connections=16),
                )
    return _client


_RETRYABLE = (httpx.ConnectError, httpx.ReadTimeout, httpx.WriteTimeout,
              httpx.PoolTimeout, httpx.RemoteProtocolError)


def _post(path: str, payload: dict, timeout: float | None = None) -> dict:
    _breaker.guard()
    attempts = max(1, settings.model_retries)
    last: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            response = _http().post(path, json=payload, timeout=timeout)
            if response.status_code >= 500:
                raise httpx.HTTPStatusError(
                    f"{response.status_code} {response.text[:200]}",
                    request=response.request, response=response,
                )
            response.raise_for_status()
            _breaker.record(True)
            return response.json()
        except _RETRYABLE as exc:
            last = exc
        except httpx.HTTPStatusError as exc:
            # 4xx 는 요청이 잘못된 것이므로 재시도해도 같다.
            if exc.response is not None and exc.response.status_code < 500:
                _breaker.record(True)
                raise
            last = exc
        metrics.inc("model_retry_total", path=path.strip("/"))
        if attempt < attempts:
            delay = settings.model_retry_backoff_s * (2 ** (attempt - 1))
            log.warning("모델 호출 재시도 %d/%d (%.1fs 후)", attempt, attempts, delay)
            time.sleep(delay)
    _breaker.record(False)
    raise ModelUnavailable(f"{path} 호출이 {attempts}회 모두 실패했다: {last}") from last


def installed_models() -> set[str]:
    response = _http().get("/api/tags", timeout=5.0)
    response.raise_for_status()
    return {m["name"] for m in response.json().get("models", [])}


def breaker_state() -> str:
    return _breaker.state


class OllamaEmbeddingProvider:
    def __init__(self, base_url: str | None = None, model: str | None = None) -> None:
        self._base_url = (base_url or settings.ollama_base_url).rstrip("/")
        self._model = model or settings.embedding_model
        self._dimension = settings.embedding_dim

    @property
    def dimension(self) -> int:
        return self._dimension

    @property
    def name(self) -> str:
        return f"ollama/{self._model}"

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []
        data = _post(
            "/api/embed",
            {
                "model": self._model,
                "input": list(texts),
                "keep_alive": settings.ollama_keep_alive,
            },
        )
        vectors = data["embeddings"]
        if vectors and len(vectors[0]) != self._dimension:
            raise ValueError(
                f"임베딩 차원 불일치: 설정 {self._dimension}, 실제 {len(vectors[0])}. "
                f"EMBEDDING_DIM 을 맞추고 재색인하라."
            )
        metrics.inc("model_embed_texts_total", len(vectors))
        return vectors


class OllamaLLMProvider:
    def __init__(self, base_url: str | None = None, model: str | None = None) -> None:
        self._base_url = (base_url or settings.ollama_base_url).rstrip("/")
        self._model = model or settings.llm_model

    @property
    def name(self) -> str:
        return f"ollama/{self._model}"

    def _payload(self, system: str, user: str, stream: bool) -> dict:
        return {
            "model": self._model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "stream": stream,
            "think": False,
            "keep_alive": settings.ollama_keep_alive,
            "options": {
                "temperature": settings.llm_temperature,
                "num_ctx": settings.llm_context_size,
                "num_predict": settings.llm_max_tokens,
            },
        }

    def generate(self, system: str, user: str) -> str:
        data = _post(
            "/api/chat",
            self._payload(system, user, stream=False),
            timeout=settings.model_generate_timeout_s,
        )
        metrics.inc("model_generate_total", model=self._model)
        return data["message"]["content"].strip()

    def generate_stream(self, system: str, user: str) -> Iterator[str]:
        """토큰 단위 스트리밍.

        스트리밍에는 재시도를 걸지 않는다. 일부를 이미 화면에 보낸 뒤 다시
        시작하면 답변이 두 번 겹쳐 나온다. 실패는 그대로 올린다.
        """
        _breaker.guard()
        payload = self._payload(system, user, stream=True)
        try:
            with _http().stream(
                "POST", "/api/chat", json=payload,
                timeout=settings.model_generate_timeout_s,
            ) as response:
                response.raise_for_status()
                for line in response.iter_lines():
                    if not line:
                        continue
                    chunk = json.loads(line)
                    piece = (chunk.get("message") or {}).get("content", "")
                    if piece:
                        yield piece
                    if chunk.get("done"):
                        break
            _breaker.record(True)
            metrics.inc("model_generate_total", model=self._model)
        except Exception:
            _breaker.record(False)
            raise
