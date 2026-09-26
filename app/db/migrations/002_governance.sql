-- 002: 접근 권한, 질의 로그, 색인 작업 큐
--
-- PoC 를 사내에서 돌리는 데 필요한 세 가지를 더한다.
--   1. 문서별 열람 그룹 — 전사 공개가 아닌 문서를 섞어도 되게 만든다
--   2. 질의 로그 — "무엇을 못 찾았는지"를 모으는 것이 품질 개선의 가장 빠른 길이다
--   3. 색인 작업 큐 — 큰 PDF 색인은 HTTP 요청 수명보다 길다

-- 열람 그룹. 빈 배열은 "전사 공개"를 뜻한다.
ALTER TABLE documents ADD COLUMN IF NOT EXISTS acl_groups  text[] NOT NULL DEFAULT '{}';
ALTER TABLE documents ADD COLUMN IF NOT EXISTS source_kind text;
ALTER TABLE documents ADD COLUMN IF NOT EXISTS byte_size   bigint;
-- 파일 서버 동기화에서 변경분만 다시 색인하기 위한 기준.
ALTER TABLE documents ADD COLUMN IF NOT EXISTS source_mtime timestamptz;
ALTER TABLE documents ADD COLUMN IF NOT EXISTS updated_at   timestamptz NOT NULL DEFAULT now();

ALTER TABLE chunks ADD COLUMN IF NOT EXISTS token_count  int;
-- 사내 문서는 같은 표·머리말이 여러 문서에 복사돼 있다. 같은 내용을 여러 건
-- 돌려주면 컨텍스트 자리만 낭비하므로 검색 결과에서 접을 수 있게 해시를 둔다.
ALTER TABLE chunks ADD COLUMN IF NOT EXISTS content_hash text;

CREATE TABLE IF NOT EXISTS query_log (
    id           bigserial PRIMARY KEY,
    asked_at     timestamptz NOT NULL DEFAULT now(),
    request_id   text,
    principal    text,
    question     text        NOT NULL,
    products     text[]      NOT NULL DEFAULT '{}',
    top_k        int,
    used_rag     boolean     NOT NULL DEFAULT true,
    answered     boolean     NOT NULL DEFAULT false,  -- 근거를 찾아 답했는가
    grounded     boolean,                             -- 수치 근거 검증 결과
    unsupported  jsonb       NOT NULL DEFAULT '[]',   -- 근거 없는 수치 목록
    chunk_ids    bigint[]    NOT NULL DEFAULT '{}',
    answer       text,
    model        text,
    latency_ms   jsonb       NOT NULL DEFAULT '{}',   -- {"embed":..,"search":..,"generate":..,"total":..}
    rating       smallint,                            -- 사용자 평가 -1 / +1
    comment      text
);

CREATE TABLE IF NOT EXISTS ingest_jobs (
    id          bigserial PRIMARY KEY,
    created_at  timestamptz NOT NULL DEFAULT now(),
    started_at  timestamptz,
    finished_at timestamptz,
    kind        text        NOT NULL,                 -- upload | sync | reindex
    source      text        NOT NULL,
    status      text        NOT NULL DEFAULT 'queued',-- queued|running|done|failed
    principal   text,
    detail      text,
    stats       jsonb       NOT NULL DEFAULT '{}'
);

INSERT INTO schema_migrations (version) VALUES ('002') ON CONFLICT DO NOTHING;
