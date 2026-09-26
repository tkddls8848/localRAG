-- 003: 검색 경로 정리
--
-- 희소 검색을 함수 색인 대신 생성 컬럼으로 바꾼다. ts_rank 를 정렬에 쓰면
-- 함수 색인만으로는 매 행마다 to_tsvector 를 다시 계산한다. 저장 컬럼으로
-- 두면 색인과 정렬이 같은 값을 쓴다.
--
-- to_tsvector(regconfig, text) 는 설정을 명시하면 immutable 이므로 생성
-- 컬럼에 쓸 수 있다. 설정을 생략한 1인자 버전은 stable 이라 쓸 수 없다.

ALTER TABLE chunks
    ADD COLUMN IF NOT EXISTS content_tsv tsvector
    GENERATED ALWAYS AS (to_tsvector('simple'::regconfig, content)) STORED;

CREATE INDEX IF NOT EXISTS chunks_tsv_idx ON chunks USING gin (content_tsv);
DROP INDEX IF EXISTS chunks_fts_idx;

-- 검색은 항상 status='indexed' 문서만 본다.
CREATE INDEX IF NOT EXISTS documents_status_idx ON documents (status);
CREATE INDEX IF NOT EXISTS documents_acl_idx ON documents USING gin (acl_groups);
CREATE INDEX IF NOT EXISTS documents_mtime_idx ON documents (source_path, source_mtime);

-- 실패한 질문을 모으는 것이 품질 개선의 가장 빠른 길이다(architecture.md 8.4).
CREATE INDEX IF NOT EXISTS query_log_asked_at_idx ON query_log (asked_at DESC);
CREATE INDEX IF NOT EXISTS query_log_unanswered_idx
    ON query_log (asked_at DESC) WHERE answered = false;
CREATE INDEX IF NOT EXISTS query_log_rating_idx
    ON query_log (rating) WHERE rating IS NOT NULL;
CREATE INDEX IF NOT EXISTS ingest_jobs_status_idx ON ingest_jobs (status, created_at DESC);

INSERT INTO schema_migrations (version) VALUES ('003') ON CONFLICT DO NOTHING;
