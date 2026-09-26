"""RUN_DB_TESTS=1로 실행. 별도 임시 스키마에서 실제 pgvector 경로를 검증한다.

검증 대상은 SQL 과 파이프라인의 정합성이다. 모델 품질은 여기서 재지 않는다
(그건 `python -m app.evaluation.run` 의 일이다). 그래서 fake 제공자를 쓴다.
"""
import os
from contextlib import contextmanager
from pathlib import Path
from uuid import uuid4

import pytest
from docx import Document
from fastapi.testclient import TestClient

pytestmark = pytest.mark.skipif(
    os.getenv('RUN_DB_TESTS') != '1', reason='PostgreSQL integration opt-in'
)


@contextmanager
def _isolated_schema():
    """테스트 전용 스키마를 만들고 마이그레이션을 적용한다."""
    from app.db.migrate import apply_migrations, applied, available
    from app.db.session import connect

    schema = 'test_' + uuid4().hex
    with connect() as conn:
        conn.execute(f'CREATE SCHEMA {schema}')
        conn.commit()

    @contextmanager
    def isolated():
        with connect() as conn:
            conn.execute(f'SET search_path TO {schema}, public')
            yield conn

    try:
        with isolated() as conn:
            ran = apply_migrations(conn)
            assert ran == [v for v, _ in available()]
            # 마이그레이션은 재실행해도 안전해야 한다. 이미 색인된 DB 를 지우지
            # 않고 스키마를 바꾸는 것이 목적이기 때문이다.
            assert apply_migrations(conn) == []
            assert applied(conn) == set(ran)
        yield isolated
    finally:
        with connect() as conn:
            conn.execute(f'DROP SCHEMA {schema} CASCADE')
            conn.commit()


def _docx(path: Path, *paragraphs: str) -> Path:
    doc = Document()
    for text in paragraphs:
        doc.add_paragraph(text)
    doc.save(path)
    return path


def test_ingest_search_and_api(tmp_path, monkeypatch):
    from app.api import main
    from app.ingest import jobs, pipeline, sync
    from app.providers.fake import EchoLLMProvider, FakeEmbeddingProvider
    from app.retrieval import audit
    from app.retrieval.answer import NOT_FOUND, answer_question
    from app.retrieval.search import Filters, search

    with _isolated_schema() as isolated:
        monkeypatch.setattr(pipeline, 'connect', isolated)
        monkeypatch.setattr(main, 'connect', isolated)
        monkeypatch.setattr(jobs, 'connect', isolated)
        monkeypatch.setattr(sync, 'connect', isolated)
        monkeypatch.setattr(audit, 'connect', isolated, raising=False)

        embedder, llm = FakeEmbeddingProvider(), EchoLLMProvider()
        monkeypatch.setattr(main, '_providers', lambda: (embedder, llm))
        monkeypatch.setattr(pipeline, 'get_embedding_provider', lambda: embedder)

        path = _docx(
            tmp_path / 'SR650i V4.docx',
            'ThinkSystem SR650i V4 Inference Configuration feature code CF3G',
            'Memory maximum: Up to 4TB by using 32x 128GB RDIMMs',
        )

        # --- 색인: 신규 / 중복 / 실패 시 기존 색인 보존 ---
        first = pipeline.ingest_document(path, embedder)
        assert first.status == 'indexed' and first.chunks > 0
        assert first.tokens > 0
        assert pipeline.ingest_document(path, embedder).status == 'skipped'

        class BrokenEmbedder:
            def embed(self, texts):
                raise RuntimeError('model unavailable')

        failed = pipeline.ingest_document(path, BrokenEmbedder(), force=True)
        assert failed.status == 'failed'
        with isolated() as conn:
            # 재색인이 실패해도 이전 청크가 검색 가능한 상태로 남아야 한다.
            status, error = conn.execute(
                'SELECT status, error FROM documents WHERE id = %s', (first.document_id,)
            ).fetchone()
            assert status == 'indexed' and 'model unavailable' in error
            assert conn.execute('SELECT count(*) FROM chunks').fetchone()[0] > 0
            # 청크 메타가 채워져야 중복 제거와 컨텍스트 예산이 동작한다.
            assert conn.execute(
                'SELECT count(*) FROM chunks WHERE content_hash IS NULL'
                ' OR token_count IS NULL'
            ).fetchone()[0] == 0

        # --- 한국어 질문이 영문 본문에 닿는가(용어집 + 희소 경로) ---
        with isolated() as conn:
            vector = embedder.embed(['최대 메모리 용량은?'])[0]
            hits = search(conn, '최대 메모리 용량은?', vector, top_k=5)
            assert hits, '용어집 확장이 없으면 영문 문서에서 한 건도 못 찾는다'
            assert any('Memory maximum' in h.content for h in hits)

            # 식별자 정확 일치 경로
            exact = search(conn, 'CF3G feature code',
                           embedder.embed(['CF3G feature code'])[0], top_k=5)
            assert any('CF3G' in h.content for h in exact)
            assert any('exact' in h.paths for h in exact)

        # --- 접근 권한: 열람 그룹이 다르면 보이지 않아야 한다 ---
        restricted = _docx(tmp_path / 'secret.docx',
                           'ThinkSystem SR630 V4 confidential pricing 12345 USD')
        assert pipeline.ingest_document(
            restricted, embedder, acl_groups=['finance']
        ).status == 'indexed'
        with isolated() as conn:
            question = 'confidential pricing'
            vec = embedder.embed([question])[0]
            assert any('confidential' in h.content
                       for h in search(conn, question, vec, top_k=10, groups=None))
            assert not any(
                'confidential' in h.content
                for h in search(conn, question, vec, top_k=10, groups=['sales'])
            )
            assert any('confidential' in h.content
                       for h in search(conn, question, vec, top_k=10, groups=['finance']))

        # --- 답변, 인용, 질의 로그 ---
        with isolated() as conn:
            answer = answer_question(conn, 'SR650i V4 memory', embedder, llm)
            assert answer.sources and '문단' in answer.sources[0].citation
            assert answer.query_id, '질의 로그가 남아야 실패 질문을 모을 수 있다'
            assert answer.latency_ms.get('total') is not None

            missing = answer_question(conn, 'SR999 V9 memory', embedder, llm)
            assert missing.answer == NOT_FOUND
            assert not missing.sources

            assert audit.set_feedback(conn, answer.query_id, -1, '값이 틀렸다')
            failures = audit.recent(conn, only_failed=True)
            assert any(f['id'] == answer.query_id for f in failures)
            assert audit.stats(conn, days=1)['queries'] >= 2

        # --- API ---
        client = TestClient(main.app)
        asked = client.post('/ask', json={'question': 'SR650i V4 memory'})
        assert asked.status_code == 200 and asked.json()['sources']
        assert asked.headers['x-request-id']

        diagnosed = client.post('/search', json={'question': 'CF3G'}).json()
        assert diagnosed['identifiers'] == ['CF3G']
        assert diagnosed['hits'] and diagnosed['hits'][0]['paths']

        streamed = client.post('/ask/stream', json={'question': 'SR650i V4 memory'})
        assert streamed.status_code == 200
        assert 'data:' in streamed.text and '"type": "done"' in streamed.text

        upload = client.post(
            '/ingest?wait=true',
            files={'file': ('SR650i V4.docx', path.read_bytes())},
        )
        assert upload.status_code == 200 and upload.json()['status'] == 'skipped'
        reindexed = client.post(
            '/ingest?wait=true&force=true',
            files={'file': ('SR650i V4.docx', path.read_bytes())},
        )
        assert reindexed.json()['status'] == 'indexed'

        listed = client.get('/documents').json()['documents']
        assert len(listed) == 2
        assert all(d['chunks'] > 0 and d['tokens'] > 0 for d in listed)

        # --- 비동기 색인 작업 ---
        queued = client.post(
            '/ingest', files={'file': ('another.docx', _docx(
                tmp_path / 'another.docx', 'ThinkSystem SR850 V4 four socket'
            ).read_bytes())},
        )
        assert queued.status_code == 202
        job_id = queued.json()['job_id']
        for _ in range(100):
            job = client.get(f'/jobs/{job_id}').json()
            if job['status'] in {'done', 'failed'}:
                break
            import time
            time.sleep(0.2)
        assert job['status'] == 'done', job
        assert job['stats']['status'] == 'indexed'

        # --- 문서 삭제로 오염된 색인을 되돌릴 수 있어야 한다 ---
        target = [d for d in client.get('/documents').json()['documents']
                  if d['title'] == 'secret'][0]
        assert client.delete(f"/documents/{target['id']}").status_code == 200
        assert client.delete(f"/documents/{target['id']}").status_code == 404

        assert client.get('/ready').status_code in (200, 503)
        assert 'http_requests_total' in client.get('/metrics').text


def test_directory_sync_indexes_only_changes(tmp_path, monkeypatch):
    from app.ingest import pipeline, sync
    from app.providers.fake import FakeEmbeddingProvider

    with _isolated_schema() as isolated:
        monkeypatch.setattr(pipeline, 'connect', isolated)
        monkeypatch.setattr(sync, 'connect', isolated)
        embedder = FakeEmbeddingProvider()

        root = tmp_path / 'docs'
        root.mkdir()
        _docx(root / 'a.docx', 'ThinkSystem SR630 V4 one socket')
        _docx(root / 'b.docx', 'ThinkSystem SR650 V4 two sockets')

        first = sync.run_sync(root, provider=embedder)
        assert first['indexed'] == 2
        assert sorted(first['new']) == ['a.docx', 'b.docx']

        # 두 번째 실행은 아무것도 다시 색인하지 않아야 한다.
        again = sync.plan(root)
        assert len(again.unchanged) == 2 and not again.new and not again.changed

        # 내용이 바뀐 파일만 골라낸다.
        _docx(root / 'b.docx', 'ThinkSystem SR650 V4 two sockets and 8TB memory')
        changed = sync.plan(root)
        assert [p.name for p in changed.changed] == ['b.docx']
        assert len(changed.unchanged) == 1

        # 파일이 사라지면 보고만 하고 기본적으로 지우지 않는다.
        (root / 'a.docx').unlink()
        after = sync.run_sync(root, provider=embedder)
        assert after['missing'] == ['a']
        with isolated() as conn:
            assert conn.execute('SELECT count(*) FROM documents').fetchone()[0] == 2
