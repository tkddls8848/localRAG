"""RUN_DB_TESTS=1로 실행. 별도 임시 스키마에서 실제 pgvector 경로를 검증한다."""
import os
from contextlib import contextmanager
from pathlib import Path
from uuid import uuid4

import pytest
from docx import Document
from fastapi.testclient import TestClient

pytestmark = pytest.mark.skipif(os.getenv('RUN_DB_TESTS') != '1', reason='PostgreSQL integration opt-in')


def test_database_ingest_search_api(tmp_path, monkeypatch):
    from app.db.session import connect
    from app.ingest import pipeline
    from app.providers.fake import FakeEmbeddingProvider, EchoLLMProvider
    from app.retrieval.answer import answer_question, NOT_FOUND
    from app.api import main
    schema = 'test_' + uuid4().hex
    with connect() as conn:
        conn.execute(f'CREATE SCHEMA {schema}')

    @contextmanager
    def isolated():
        with connect() as conn:
            conn.execute(f'SET search_path TO {schema}, public')
            yield conn

    try:
        with isolated() as conn:
            conn.execute(Path('app/db/schema.sql').read_text(encoding='utf-8'))
        monkeypatch.setattr(pipeline, 'connect', isolated)
        monkeypatch.setattr(main, 'connect', isolated)
        embedder, llm = FakeEmbeddingProvider(), EchoLLMProvider()
        monkeypatch.setattr(main, '_providers', lambda: (embedder, llm))
        monkeypatch.setattr(pipeline, 'get_embedding_provider', lambda: embedder)
        path = tmp_path / 'SR650i V4.docx'
        doc = Document()
        doc.add_paragraph('ThinkSystem SR650i V4 Inference Configuration memory 4TB')
        doc.save(path)
        assert pipeline.ingest_document(path, embedder).status == 'indexed'
        assert pipeline.ingest_document(path, embedder).status == 'skipped'
        class BrokenEmbedder:
            def embed(self, texts):
                raise RuntimeError('model unavailable')
        assert pipeline.ingest_document(path, BrokenEmbedder(), force=True).status == 'failed'
        with isolated() as conn:
            answer = answer_question(conn, 'SR650i V4 memory', embedder, llm)
            assert answer.sources and '문단' in answer.sources[0].citation
            missing = answer_question(conn, 'SR999 V9 memory', embedder, llm)
            assert missing.answer == NOT_FOUND
            assert not missing.sources
        client = TestClient(main.app)
        assert client.post('/ask', json={'question': 'SR650i V4 memory'}).json()['sources']
        response = client.post('/ingest', files={'file': ('SR650i V4.docx', path.read_bytes())})
        assert response.status_code == 200
        assert response.json()['status'] == 'skipped'
        assert client.post('/ingest?force=true', files={'file': ('SR650i V4.docx', path.read_bytes())}).json()['status'] == 'indexed'
        assert len(client.get('/documents').json()['documents']) == 1
    finally:
        with connect() as conn:
            conn.execute(f'DROP SCHEMA {schema} CASCADE')
