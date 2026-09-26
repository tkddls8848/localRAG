-- 001: 기본 스키마 (문서 / 청크 / 검색 색인)
-- 설계 근거는 docs/architecture.md, docs/decisions.md 참고.
--
-- 모든 문장은 재실행 가능해야 한다. 마이그레이션 러너와 컨테이너 initdb 가
-- 같은 파일을 쓰기 때문에, 이미 적용된 DB 에 다시 걸려도 안전해야 한다.

CREATE TABLE IF NOT EXISTS schema_migrations (
    version    text        PRIMARY KEY,
    applied_at timestamptz NOT NULL DEFAULT now()
);

CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pg_trgm;

CREATE TABLE IF NOT EXISTS documents (
    id          bigserial PRIMARY KEY,
    source_path text        NOT NULL,
    title       text        NOT NULL,
    products    text[]      NOT NULL DEFAULT '{}',  -- 한 문서가 여러 모델을 다룰 수 있다
    sha256      text        NOT NULL UNIQUE,
    page_count  int,
    status      text        NOT NULL DEFAULT 'pending',  -- pending|indexing|indexed|failed|quarantined
    error       text,
    meta        jsonb       NOT NULL DEFAULT '{}',
    ingested_at timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS chunks (
    id           bigserial PRIMARY KEY,
    document_id  bigint NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    ordinal      int    NOT NULL,
    kind         text   NOT NULL,          -- table_row | prose
    content      text   NOT NULL,
    section_path text,                     -- TOC 에서 확정. 예: "Memory"
    page_from    int,
    page_to      int,
    products     text[] NOT NULL DEFAULT '{}',  -- 본문이 한 모델만 가리키면 그 모델로 좁혀 저장
    embedding    vector(1024),
    UNIQUE (document_id, ordinal)
);

-- 밀집 검색
CREATE INDEX IF NOT EXISTS chunks_embedding_idx
    ON chunks USING hnsw (embedding vector_cosine_ops);

-- 희소 검색: 모델명·파트번호 정확 매칭용
CREATE INDEX IF NOT EXISTS chunks_trgm_idx
    ON chunks USING gin (content gin_trgm_ops);

-- 모델 혼동 차단용 필터. SR650a V4 질문에 SR650i V4 행이 딸려오는 것을 막는다.
CREATE INDEX IF NOT EXISTS chunks_products_idx ON chunks USING gin (products);

INSERT INTO schema_migrations (version) VALUES ('001') ON CONFLICT DO NOTHING;
