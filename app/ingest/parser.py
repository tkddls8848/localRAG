"""Lenovo Press 계열 제품 가이드 PDF 파서.

이 문서군의 특성(검증 결과, docs/architecture.md 참고):
  - 전부 텍스트 기반. 스캔 이미지 페이지 없음
  - TOC 가 페이지 번호까지 포함 → 섹션 경로를 추론이 아니라 확정으로 얻는다
  - 표의 다수가 단순 키-값 사양표 → 행 하나가 자기완결적 청크가 된다
  - 소수의 넓은 표는 헤더가 2단 → 그룹 헤더를 전방 채움으로 병합해야 한다
  - **모든 페이지에 같은 머리말/꼬리말이 반복된다** → 그대로 색인하면 문서당
    수백 개의 동일 청크가 생기고, 검색 결과 상단을 의미 없는 행이 차지한다
"""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from pathlib import Path

import pymupdf

from app.config import settings
from app.taxonomy import extract_products

# 표 위에 딸려 들어오는 캡션/제목 행을 걸러내는 길이 기준
_CAPTION_MIN_CHARS = 60

# 반복 머리말/꼬리말 판정: 전체 페이지의 이 비율 이상에 등장하면 상용구로 본다.
# 실측에서 페이지 꼬리말은 100%, 다음으로 많은 실제 표 헤더는 27% 였다.
# 절반을 기준으로 두면 본문을 잘못 지울 여지가 없다.
_BOILERPLATE_RATIO = 0.5
_BOILERPLATE_MAX_CHARS = 160


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
    # 표의 단위 표기("(단위:천원)"). 표 밖의 캡션에 적혀 있어서 행만 떼어
    # 내면 사라진다. 실제로 겪은 오답: 9,876,543 천원을 "9,876,543원"으로
    # 답했다 — 1000배 차이다. 값 자체가 아니라 값을 읽는 단위이므로 매 행에
    # 붙여야 그 행이 단독으로 읽힌다.
    unit: str = ""

    def render(self) -> str:
        body = "\n".join(f"{k}: {v}" for k, v in self.pairs if v)
        if self.unit:
            body = f"{body}\n({self.unit})"
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
    # 색인을 막지는 않지만 사람이 알아야 하는 사실. documents.meta 에 남긴다.
    warnings: list[str] = field(default_factory=list)
    # 텍스트가 거의 없으면 스캔 PDF 다. 조용히 빈 색인을 만들지 않는다.
    likely_scanned: bool = False


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


def _table_blocks(table, page_no: int, section: str,
                  unit: str = "") -> list[TableRowBlock]:
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
    row_label = ""
    for row in data:
        if not any(row):
            continue

        # 첫 열에만 짧은 내용이 있는 행은 데이터가 아니라 묶음 구분 행이다.
        if row[0] and not any(row[1:]) and len(row[0]) < _CAPTION_MIN_CHARS:
            group = row[0]
            continue

        # 세로로 병합된 첫 열을 앞의 값으로 채운다.
        #
        # 왜 필요한가. 추출된 표에서 세로 병합 셀은 첫 행에만 값이 있고 나머지
        # 행은 비어 있다. 행 단위로 청크를 만드는 구조에서는 그 행이 무엇에
        # 관한 값인지 잃는다. 실제로 겪은 오답:
        #
        #   구분: 매출액 | 시스템구축 | 2025년: 6,543,210   ← 라벨이 있는 행
        #              | 기술용역외 | 2025년:  3,333,333   ← 라벨 유실
        #              | 합계      | 2025년: 9,876,543   ← 라벨 유실(정답)
        #
        # "2025년 매출액"을 물으면 라벨이 남은 첫 행만 걸려, 소계를 합계로
        # 답한다. 출처는 맞는데 값이 틀린 가장 위험한 실패다(decisions.md D13).
        if not row[0] and row_label and any(row[1:]):
            row = [row_label, *row[1:]]
        elif row[0]:
            row_label = row[0]

        if key_value:
            pairs = [(row[0], row[1])]
        else:
            pairs = [(h or f"col{i}", v) for i, (h, v) in enumerate(zip(header, row))]

        if not any(v for _, v in pairs):
            continue
        blocks.append(
            TableRowBlock(pairs=pairs, page=page_no, section_path=section,
                          group=group, unit=unit)
        )
    return blocks


def _page_prose(page, tables, page_no: int, section: str) -> list[ProseBlock]:
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


# 표의 단위 표기. 표 안이 아니라 옆/위 캡션에 적히는 것이 보통이다.
_UNIT_NOTE = re.compile(
    r"\(\s*(?:단위|단위당|Unit|UNIT|unit)\s*[:：]?\s*[^)]{1,24}\)"
)


def unit_note(blocks: list[ProseBlock]) -> str:
    """페이지 본문에서 단위 표기를 찾는다.

    "(단위:천원)" 같은 표기는 표 바깥에 있어서 행 단위 청크가 놓친다. 그러면
    9,876,543 천원을 "9,876,543원" 으로 답한다 — 1000배 오차다. 값을 그대로
    옮기라는 규칙(answer.py 4)을 지켜도 단위가 없으면 값이 틀린다.

    한 페이지에 단위가 다른 표가 둘 있으면 잘못 붙일 수 있다. 값이 아니라
    주석으로 붙이고 원문 확인 경로(출처 표시)를 남겨 두는 선택이다.
    """
    for block in blocks:
        if len(block.text) > 120:
            continue
        found = _UNIT_NOTE.search(block.text)
        if found:
            return _norm(found.group(0)).strip("()")
    return ""


def _boiler_key(text: str) -> str:
    """페이지마다 달라지는 숫자(페이지 번호, 날짜)를 지운 비교용 키."""
    return re.sub(r"\d+", "#", re.sub(r"\s+", " ", text)).strip()


def boilerplate_keys(blocks: list[ProseBlock], page_count: int) -> set[str]:
    """페이지 절반 이상에 반복되는 짧은 텍스트 = 머리말/꼬리말.

    실측: 156페이지 문서에서 꼬리말 한 줄이 156회, 그 다음으로 잦은 실제
    표 헤더가 42회였다. 절반을 기준으로 잡으면 본문을 지울 위험이 없다.
    """
    if page_count < 4:
        return set()
    pages: dict[str, set[int]] = {}
    for block in blocks:
        if len(block.text) > _BOILERPLATE_MAX_CHARS:
            continue
        pages.setdefault(_boiler_key(block.text), set()).add(block.page)
    threshold = max(4, int(page_count * _BOILERPLATE_RATIO))
    return {key for key, seen in pages.items() if key and len(seen) >= threshold}


# 제목에 모델이 여러 개 나오는 문서가 있다.
# 예: "Lenovo ThinkSystem SR650a V4 and SR650i V4 Servers"
# 이를 하나로 뭉뚱그리면 SR650i 내용이 SR650a 로 태깅되어, 이 프로젝트가
# 막으려는 바로 그 모델 혼동이 색인 단계에서 발생한다.
# 모델 식별 규칙은 app/taxonomy.py 에 있다. 검색 단계와 같은 규칙을 써야
# 하기 때문이다. 두 곳에 두면 조용히 어긋나고, 그 결과가 바로 모델 혼동이다.


def parse_pdf(path: str | Path) -> ParsedDoc:
    path = Path(path)
    sha = hashlib.sha256(path.read_bytes()).hexdigest()

    with pymupdf.open(path) as doc:
        title = _norm(doc.metadata.get("title")) or path.stem
        sections = _section_index(doc)

        # 1차: 페이지 순서를 유지한 채 표/산문을 모두 모은다.
        collected: list[TableRowBlock | ProseBlock] = []
        prose: list[ProseBlock] = []
        for i in range(doc.page_count):
            page = doc[i]
            page_no = i + 1
            section = sections.get(page_no, "")
            tables = page.find_tables().tables

            # 본문을 먼저 훑어 단위 표기를 찾는다. 표 행에 붙여야 그 행이
            # 단독으로 읽힌다. 수집 순서(표 -> 본문)는 그대로 둔다.
            page_prose = _page_prose(page, tables, page_no, section)
            unit = unit_note(page_prose)
            for table in tables:
                collected.extend(_table_blocks(table, page_no, section, unit))
            prose.extend(page_prose)
            collected.extend(page_prose)

        # 2차: 문서 전체를 본 뒤에야 무엇이 상용구인지 알 수 있다.
        drop = boilerplate_keys(prose, doc.page_count)
        blocks = [
            b for b in collected
            if not (isinstance(b, ProseBlock) and _boiler_key(b.text) in drop)
        ]

        warnings: list[str] = []
        removed = len(collected) - len(blocks)
        if removed:
            warnings.append(f"반복 머리말/꼬리말 {removed}건 제거")

        text_chars = sum(len(b.render()) for b in blocks)
        per_page = text_chars / doc.page_count if doc.page_count else 0
        scanned = per_page < settings.min_chars_per_page
        if scanned:
            warnings.append(
                f"페이지당 추출 문자가 {per_page:.0f}자뿐이다. 스캔 PDF 로 의심된다."
            )

        return ParsedDoc(
            source_path=str(path),
            title=title,
            products=extract_products(title),
            page_count=doc.page_count,
            sha256=sha,
            blocks=blocks,
            warnings=warnings,
            likely_scanned=scanned,
        )
