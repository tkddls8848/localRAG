"""Lenovo Press 계열 제품 가이드 PDF 파서.

이 문서군의 특성(검증 결과, docs/architecture.md 참고):
  - 전부 텍스트 기반. 스캔 이미지 페이지 없음
  - TOC 가 페이지 번호까지 포함 → 섹션 경로를 추론이 아니라 확정으로 얻는다
  - 표의 다수가 단순 키-값 사양표 → 행 하나가 자기완결적 청크가 된다
  - 소수의 넓은 표는 헤더가 2단 → 그룹 헤더를 전방 채움으로 병합해야 한다
"""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from pathlib import Path

import pymupdf

# 표 위에 딸려 들어오는 캡션/제목 행을 걸러내는 길이 기준
_CAPTION_MIN_CHARS = 60


@dataclass
class TableRowBlock:
    """표의 데이터 행 하나. 열 이름과 값이 짝지어진 상태.

    group 은 넓은 표 안에서 행 묶음을 나누는 구분 행(예: "Intel Xeon
    6700-series with P-cores")이다. 값이 아니라 맥락이므로 별도로 보관한다.
    """

    pairs: list[tuple[str, str]]
    page: int
    section_path: str
    group: str = ""

    def render(self) -> str:
        body = "\n".join(f"{k}: {v}" for k, v in self.pairs if v)
        return f"[{self.group}]\n{body}" if self.group else body


@dataclass
class ProseBlock:
    text: str
    page: int
    section_path: str

    def render(self) -> str:
        return self.text


@dataclass
class ParsedDoc:
    source_path: str
    title: str
    products: list[str]      # 한 문서가 제품 여러 종을 다루는 경우가 있다
    page_count: int
    sha256: str
    blocks: list[TableRowBlock | ProseBlock] = field(default_factory=list)


def _norm(cell: object) -> str:
    """셀 값을 한 줄 문자열로 정규화."""
    if cell is None:
        return ""
    return re.sub(r"\s+", " ", str(cell)).strip()


def _section_index(doc: pymupdf.Document) -> dict[int, str]:
    """TOC 로부터 페이지 → 섹션 경로 맵을 만든다.

    TOC 항목은 '이 섹션이 시작되는 페이지'를 가리키므로, 다음 항목 직전까지
    같은 섹션으로 본다.
    """
    toc = doc.get_toc()
    if not toc:
        return {}

    index: dict[int, str] = {}
    stack: list[str] = []
    entries: list[tuple[int, str]] = []

    for level, title, page in toc:
        title = _norm(title)
        if not title or page < 1:
            continue
        del stack[level - 1 :]
        stack.append(title)
        entries.append((page, " > ".join(stack)))

    for i, (page, path) in enumerate(entries):
        end = entries[i + 1][0] if i + 1 < len(entries) else doc.page_count + 1
        for p in range(page, end):
            index.setdefault(p, path)
    return index


def _strip_caption_rows(rows: list[list[str]]) -> list[list[str]]:
    """표 위쪽에 섞여 들어온 캡션 행을 제거.

    첫 셀에만 내용이 있고 그 내용이 길면 데이터가 아니라 문장이다.
    """
    out = list(rows)
    while out:
        first, rest = out[0][0], out[0][1:]
        if first and not any(rest) and len(first) >= _CAPTION_MIN_CHARS:
            out.pop(0)
            continue
        break
    return out


def _split_header(rows: list[list[str]]) -> tuple[list[str], list[list[str]]]:
    """헤더와 데이터 행을 분리. 2단(그룹) 헤더면 병합한다.

    그룹 헤더의 특징: 상위 행이 비어 있는 위치를 하위 행이 채운다.
    예)  [... , 'Accelerators', '',    '',    ''   ]
         [... , 'QAT',          'DLB', 'DSA', 'IAA']
    → ['... ', 'Accelerators QAT', 'Accelerators DLB', ...]
    """
    if not rows:
        return [], []

    header = rows[0]
    if len(rows) < 2:
        return header, []

    nxt = rows[1]
    fills = sum(1 for h, n in zip(header, nxt) if not h and n)
    if fills < 2:
        return header, rows[1:]

    merged: list[str] = []
    group = ""
    for h, n in zip(header, nxt):
        if h:
            group = h
        parts = [p for p in (group if not h else h, n) if p]
        merged.append(" ".join(dict.fromkeys(parts)))
    return merged, rows[2:]


def _table_blocks(table, page_no: int, section: str) -> list[TableRowBlock]:
    raw = [[_norm(c) for c in row] for row in table.extract()]
    raw = [r for r in raw if any(r)]
    raw = _strip_caption_rows(raw)
    header, data = _split_header(raw)
    if not header or not data:
        return []

    # 2열짜리는 사실상 키-값 목록이다. 이때 헤더("Components"/"Specification")는
    # 메타 라벨일 뿐 내용이 아니므로, 첫 열의 값 자체를 키로 쓴다.
    key_value = len(header) == 2

    blocks: list[TableRowBlock] = []
    group = ""
    for row in data:
        if not any(row):
            continue

        # 첫 열에만 짧은 내용이 있는 행은 데이터가 아니라 묶음 구분 행이다.
        if row[0] and not any(row[1:]) and len(row[0]) < _CAPTION_MIN_CHARS:
            group = row[0]
            continue

        if key_value:
            pairs = [(row[0], row[1])]
        else:
            pairs = [(h or f"col{i}", v) for i, (h, v) in enumerate(zip(header, row))]

        if not any(v for _, v in pairs):
            continue
        blocks.append(
            TableRowBlock(pairs=pairs, page=page_no, section_path=section, group=group)
        )
    return blocks


def _prose_blocks(page, tables, page_no: int, section: str) -> list[ProseBlock]:
    """표 영역을 제외한 본문 텍스트."""
    table_rects = [pymupdf.Rect(t.bbox) for t in tables]
    out: list[ProseBlock] = []

    for x0, y0, x1, y1, text, *_ in page.get_text("blocks"):
        rect = pymupdf.Rect(x0, y0, x1, y1)
        if any(rect.intersects(tr) for tr in table_rects):
            continue
        text = re.sub(r"[ \t]+", " ", text).strip()
        if not text or text.isdigit():
            continue
        out.append(ProseBlock(text=text, page=page_no, section_path=section))
    return out


# 제목에 모델이 여러 개 나오는 문서가 있다.
# 예: "Lenovo ThinkSystem SR650a V4 and SR650i V4 Servers"
# 이를 하나로 뭉뚱그리면 SR650i 내용이 SR650a 로 태깅되어, 이 프로젝트가
# 막으려는 바로 그 모델 혼동이 색인 단계에서 발생한다.
_MODEL_RE = re.compile(r"\b([A-Z]{2}\d{2,4}[a-z]?)\s+(V\d+)\b")


def _extract_products(title: str) -> list[str]:
    seen = dict.fromkeys(
        f"ThinkSystem {m.group(1)} {m.group(2)}" for m in _MODEL_RE.finditer(title)
    )
    return list(seen)


def match_products(text: str, candidates: list[str]) -> list[str]:
    """본문이 특정 모델만 가리키면 그 모델로 좁힌다.

    문서가 SR650a/SR650i 를 함께 다루더라도, 개별 표 행은 대개 한쪽 이야기다.
    """
    hits = [c for c in candidates if c.split()[1] in text]
    return hits if len(hits) == 1 else candidates


def parse_pdf(path: str | Path) -> ParsedDoc:
    path = Path(path)
    sha = hashlib.sha256(path.read_bytes()).hexdigest()

    with pymupdf.open(path) as doc:
        title = _norm(doc.metadata.get("title")) or path.stem
        sections = _section_index(doc)
        blocks: list[TableRowBlock | ProseBlock] = []

        for i in range(doc.page_count):
            page = doc[i]
            page_no = i + 1
            section = sections.get(page_no, "")
            tables = page.find_tables().tables

            for table in tables:
                blocks.extend(_table_blocks(table, page_no, section))
            blocks.extend(_prose_blocks(page, tables, page_no, section))

        return ParsedDoc(
            source_path=str(path),
            title=title,
            products=_extract_products(title),
            page_count=doc.page_count,
            sha256=sha,
            blocks=blocks,
        )
