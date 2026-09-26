"""환경변수 기반 설정. 코드에 값을 하드코딩하지 않는다(decisions.md D6).

PoC 를 사내에서 실제로 돌리는 순간 설정 항목이 늘어난다. 늘어난 값을 코드 곳곳에
흘리지 않기 위해 환경변수는 이 파일에서만 읽고, 기동 시 한 번 `validate()` 로
검증한다. 잘못된 설정으로 절반만 동작하는 서버가 뜨는 것이 가장 나쁜 경우다.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


def _int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    try:
        return int(raw.strip())
    except ValueError as exc:
        raise ConfigError(f"{name} 은 정수여야 한다: {raw!r}") from exc


def _float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    try:
        return float(raw.strip())
    except ValueError as exc:
        raise ConfigError(f"{name} 은 실수여야 한다: {raw!r}") from exc


def _bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _list(name: str, default: str = "") -> list[str]:
    raw = os.getenv(name, default)
    return [p.strip() for p in raw.split(",") if p.strip()]


class ConfigError(ValueError):
    """설정이 잘못되어 기동을 중단해야 하는 경우."""


@dataclass(frozen=True)
class Settings:
    # --- 실행 환경 ---
    app_env: str = os.getenv("APP_ENV", "local")          # local | poc | prod
    log_level: str = os.getenv("LOG_LEVEL", "INFO").upper()
    log_format: str = os.getenv("LOG_FORMAT", "text").lower()   # text | json

    # --- 저장소 ---
    pg_host: str = os.getenv("POSTGRES_HOST", "localhost")
    pg_port: int = _int("POSTGRES_PORT", 5432)
    pg_db: str = os.getenv("POSTGRES_DB", "specrag")
    pg_user: str = os.getenv("POSTGRES_USER", "specrag")
    pg_password: str = os.getenv("POSTGRES_PASSWORD", "change-me")
    pg_pool_min: int = _int("POSTGRES_POOL_MIN", 1)
    pg_pool_max: int = _int("POSTGRES_POOL_MAX", 8)
    pg_connect_timeout: int = _int("POSTGRES_CONNECT_TIMEOUT", 5)
    pg_statement_timeout_ms: int = _int("POSTGRES_STATEMENT_TIMEOUT_MS", 30_000)

    # --- 모델 런타임 ---
    embedding_provider: str = os.getenv("EMBEDDING_PROVIDER", "ollama")
    llm_provider: str = os.getenv("LLM_PROVIDER", "ollama")
    llm_model: str = os.getenv("LLM_MODEL", "qwen3:8b")
    llm_context_size: int = _int("LLM_CONTEXT_SIZE", 8192)
    llm_max_tokens: int = _int("LLM_MAX_TOKENS", 1024)
    llm_temperature: float = _float("LLM_TEMPERATURE", 0.0)
    ollama_base_url: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    # 모델을 메모리에 얼마나 붙잡아 둘지. 짧으면 질문마다 로딩 시간이 붙어
    # 품질 문제로 오해받는다.
    ollama_keep_alive: str = os.getenv("OLLAMA_KEEP_ALIVE", "30m")
    embedding_model: str = os.getenv("EMBEDDING_MODEL", "bge-m3")
    embedding_dim: int = _int("EMBEDDING_DIM", 1024)

    # 모델 런타임은 PoC 에서 가장 자주 넘어지는 구성요소다. 재시도와 차단기를
    # 넣어 두면 일시적 장애가 "색인 전체 실패"로 번지지 않는다.
    model_timeout_s: float = _float("MODEL_TIMEOUT_S", 180.0)
    model_generate_timeout_s: float = _float("MODEL_GENERATE_TIMEOUT_S", 300.0)
    model_retries: int = _int("MODEL_RETRIES", 3)
    model_retry_backoff_s: float = _float("MODEL_RETRY_BACKOFF_S", 1.0)
    breaker_threshold: int = _int("MODEL_BREAKER_THRESHOLD", 5)
    breaker_cooldown_s: float = _float("MODEL_BREAKER_COOLDOWN_S", 30.0)

    # --- 색인 ---
    chunk_target_chars: int = _int("CHUNK_TARGET_CHARS", 1200)
    chunk_overlap_chars: int = _int("CHUNK_OVERLAP_CHARS", 200)
    embed_batch_size: int = _int("EMBED_BATCH_SIZE", 16)
    doc_root: str = os.getenv("DOC_ROOT", "data/pdfs")
    ingest_workers: int = _int("INGEST_WORKERS", 2)
    max_upload_mb: int = _int("MAX_UPLOAD_MB", 100)
    # 페이지당 추출 문자가 이보다 적으면 스캔 PDF 로 의심한다. 조용히 빈
    # 색인을 만드는 대신 문서 상태에 남긴다(architecture.md 3.1).
    min_chars_per_page: int = _int("MIN_CHARS_PER_PAGE", 120)

    # --- 검색 ---
    search_top_k: int = _int("SEARCH_TOP_K", 10)
    search_candidates: int = _int("SEARCH_CANDIDATES", 50)
    rrf_k: int = _int("RRF_K", 60)
    weight_dense: float = _float("WEIGHT_DENSE", 1.0)
    weight_sparse: float = _float("WEIGHT_SPARSE", 1.0)
    weight_trigram: float = _float("WEIGHT_TRIGRAM", 0.6)
    hnsw_ef_search: int = _int("HNSW_EF_SEARCH", 100)
    # 표 행 청크는 앞뒤 행을 함께 보여줘야 뜻이 통하는 경우가 있다.
    neighbor_window: int = _int("SEARCH_NEIGHBOR_WINDOW", 1)
    # 거의 같은 표 행이 상위를 독점하는 것만 막는다. 1.0 이면 끈다.
    near_dup_similarity: float = _float("NEAR_DUP_SIMILARITY", 0.9)
    # 같은 섹션(= 대개 같은 표)에서 상위에 올릴 최대 건수. 0 이면 끈다.
    section_cap: int = _int("SEARCH_SECTION_CAP", 3)
    query_expansion: bool = _bool("QUERY_EXPANSION", True)
    glossary_path: str = os.getenv("GLOSSARY_PATH", "app/retrieval/glossary.yaml")

    # --- 리랭킹 ---
    rerank_provider: str = os.getenv("RERANK_PROVIDER", "rules")   # none|rules|llm
    rerank_candidates: int = _int("RERANK_CANDIDATES", 30)

    # --- 답변 ---
    # off: 검사하지 않음 / warn: 근거 없는 수치를 응답에 표시 / strict: 답변 거부
    grounding_mode: str = os.getenv("GROUNDING_MODE", "warn").lower()

    # --- 접근 제어 ---
    # "키:역할:그룹1|그룹2, 키:역할:그룹" 형식. 비어 있으면 인증을 끈다.
    api_keys: str = os.getenv("API_KEYS", "")
    rate_limit_per_min: int = _int("RATE_LIMIT_PER_MIN", 60)
    cors_origins: list[str] = field(default_factory=lambda: _list("CORS_ORIGINS"))

    # --- 운영 ---
    query_log_enabled: bool = _bool("QUERY_LOG_ENABLED", True)
    metrics_enabled: bool = _bool("METRICS_ENABLED", True)

    @property
    def dsn(self) -> str:
        return (
            f"host={self.pg_host} port={self.pg_port} dbname={self.pg_db} "
            f"user={self.pg_user} password={self.pg_password} "
            f"connect_timeout={self.pg_connect_timeout}"
        )

    @property
    def pg_kwargs(self) -> dict:
        """연결 시 함께 넘길 옵션.

        statement_timeout 을 DSN 문자열에 넣으면 공백 이스케이프 규칙 때문에
        libpq 구현별로 다르게 해석된다. 별도 인자로 넘겨 모호함을 없앤다.
        """
        return {"options": f"-c statement_timeout={self.pg_statement_timeout_ms}"}

    @property
    def auth_enabled(self) -> bool:
        return bool(self.api_keys.strip())

    def validate(self) -> list[str]:
        """치명적 오류는 예외로, 운영상 위험은 경고 목록으로 돌려준다."""
        if self.chunk_target_chars <= 0 or not 0 <= self.chunk_overlap_chars < self.chunk_target_chars:
            raise ConfigError("0 <= CHUNK_OVERLAP_CHARS < CHUNK_TARGET_CHARS 이어야 한다")
        if self.embed_batch_size < 1:
            raise ConfigError("EMBED_BATCH_SIZE 는 1 이상이어야 한다")
        if self.embedding_dim < 1:
            raise ConfigError("EMBEDDING_DIM 은 1 이상이어야 한다")
        if self.search_candidates < self.search_top_k:
            raise ConfigError("SEARCH_CANDIDATES 는 SEARCH_TOP_K 이상이어야 한다")
        if self.rerank_provider not in {"none", "rules", "llm"}:
            raise ConfigError(f"RERANK_PROVIDER 는 none|rules|llm 중 하나: {self.rerank_provider!r}")
        if self.grounding_mode not in {"off", "warn", "strict"}:
            raise ConfigError(f"GROUNDING_MODE 는 off|warn|strict 중 하나: {self.grounding_mode!r}")
        if self.pg_pool_max < self.pg_pool_min or self.pg_pool_max < 1:
            raise ConfigError("POSTGRES_POOL_MAX 는 POOL_MIN 이상이고 1 이상이어야 한다")

        warnings: list[str] = []
        if not self.auth_enabled:
            warnings.append(
                "API_KEYS 가 비어 있어 인증이 꺼져 있다. 사내 공유 전에 반드시 설정하라."
            )
        if self.pg_password in {"change-me", "postgres", ""}:
            warnings.append("POSTGRES_PASSWORD 가 기본값이다. 교체하라.")
        if self.embedding_provider == "fake" or self.llm_provider == "fake":
            warnings.append("fake 제공자가 켜져 있다. 품질 측정에 쓰면 안 된다.")
        if self.app_env != "local" and self.log_format != "json":
            warnings.append("LOG_FORMAT=json 을 권장한다(로그 수집 연동).")
        if not Path(self.glossary_path).exists():
            warnings.append(f"용어집을 찾지 못했다: {self.glossary_path}. 한국어 키워드 검색이 약해진다.")
        return warnings


settings = Settings()
