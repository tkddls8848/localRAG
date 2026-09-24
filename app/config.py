"""환경변수 기반 설정. 코드에 값을 하드코딩하지 않는다(decisions.md D6)."""
from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()


def _int(name: str, default: int) -> int:
    return int(os.getenv(name, default))


@dataclass(frozen=True)
class Settings:
    pg_host: str = os.getenv("POSTGRES_HOST", "localhost")
    pg_port: int = _int("POSTGRES_PORT", 5432)
    pg_db: str = os.getenv("POSTGRES_DB", "specrag")
    pg_user: str = os.getenv("POSTGRES_USER", "specrag")
    pg_password: str = os.getenv("POSTGRES_PASSWORD", "change-me")

    embedding_provider: str = os.getenv("EMBEDDING_PROVIDER", "ollama")
    llm_provider: str = os.getenv("LLM_PROVIDER", "ollama")
    llm_model: str = os.getenv("LLM_MODEL", "qwen3:8b")
    llm_context_size: int = _int("LLM_CONTEXT_SIZE", 8192)
    ollama_base_url: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    embedding_model: str = os.getenv("EMBEDDING_MODEL", "bge-m3")
    embedding_dim: int = _int("EMBEDDING_DIM", 1024)

    chunk_target_chars: int = _int("CHUNK_TARGET_CHARS", 1200)
    chunk_overlap_chars: int = _int("CHUNK_OVERLAP_CHARS", 200)
    embed_batch_size: int = _int("EMBED_BATCH_SIZE", 16)

    search_top_k: int = _int("SEARCH_TOP_K", 10)
    search_candidates: int = _int("SEARCH_CANDIDATES", 50)

    @property
    def dsn(self) -> str:
        return (
            f"host={self.pg_host} port={self.pg_port} dbname={self.pg_db} "
            f"user={self.pg_user} password={self.pg_password}"
        )


settings = Settings()
