from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient
from docx import Document
from openpyxl import Workbook
from pptx import Presentation
import pymupdf

from app.api.main import app
from app.ingest.documents import parse_document
from app.ingest.chunker import chunk_document
from app.retrieval.answer import answer_question, NOT_FOUND
from app.retrieval.search import Hit
from app.providers.fake import FakeEmbeddingProvider


def test_office_locations(tmp_path):
    word = Document()
    word.add_heading('출장 규정', 1)
    word.add_paragraph('숙박비 상한은 100000원입니다.')
    path = tmp_path / 'rules.docx'
    word.save(path)
    chunks = chunk_document(parse_document(path))
    assert any('100000' in c.content and '문단' in c.section_path for c in chunks)
    assert all(c.page_from == 0 for c in chunks)
    book = Workbook()
    book.active.title = '예산'
    book.active.append(['출장', 100000])
    path = tmp_path / 'budget.xlsx'
    book.save(path)
    block = parse_document(path).blocks[0]
    assert block.section_path == '예산 > 행 1'
    assert ('B1', '100000') in block.pairs
    slides = Presentation()
    slide = slides.slides.add_slide(slides.slide_layouts[1])
    slide.shapes.title.text = '사내 규정'
    path = tmp_path / 'intro.pptx'
    slides.save(path)
    assert parse_document(path).blocks[0].page == 1


def test_pdf_page_and_long_text(tmp_path):
    path = tmp_path / 'guide.pdf'
    with pymupdf.open() as pdf:
        pdf.new_page().insert_text((50, 50), 'Memory maximum: 4TB.')
        pdf.save(path)
    assert chunk_document(parse_document(path))[0].page_from == 1
    from app.ingest.parser import ParsedDoc, ProseBlock
    doc = ParsedDoc('a', 'a', [], 1, 'hash', [ProseBlock('x' * 10000, 1, '')])
    chunks = chunk_document(doc)
    assert len(chunks) > 5
    assert all(len(c.content) <= 1202 for c in chunks)


@pytest.mark.parametrize('text,expected,count', [
    ('4TB [1]', '4TB [1]', 1),
    ('4TB [99]', NOT_FOUND, 0),
    ('4TB', NOT_FOUND, 0),
    (NOT_FOUND, NOT_FOUND, 0),
])
def test_citation_validation(text, expected, count):
    class LLM:
        name = 'test'
        def generate(self, system, user):
            return text
    hit = Hit(1, 'Memory: 4TB', 'Memory', 2, 2, [], 'Guide', 'guide.pdf')
    with patch('app.retrieval.answer.search', return_value=[hit]):
        result = answer_question(None, 'memory?', FakeEmbeddingProvider(), LLM())
    assert result.answer == expected
    assert len(result.sources) == count


def test_api_validation():
    client = TestClient(app)
    assert client.get('/').status_code == 200
    assert client.post('/ask', json={'question': '   '}).status_code == 422
    assert client.post('/ask', json={'question': 'hi', 'top_k': -1}).status_code == 422
    assert client.post('/ingest', files={'file': ('a.exe', b'bad')}).status_code == 415
    assert client.post('/ingest', files={'file': ('a.pdf', b'')}).status_code == 400


def test_cli_windows_glob(tmp_path):
    from app.ingest.cli import main
    for name in ['a.docx', 'b.docx']:
        d = Document()
        d.add_paragraph('Test document')
        d.save(tmp_path / name)
    assert main(['--dry-run', str(tmp_path / '*.docx')]) == 0
