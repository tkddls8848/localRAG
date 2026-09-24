"""블록 → 청크.

핵심 원칙: 청크는 **단독으로 읽혀야 한다.**
"Memory maximum: Up to 4TB" 만 떼어놓으면 어느 장비 이야기인지 알 수 없다.
그래서 모든 청크 앞에 제품명과 섹션 경로를 붙인다. 이것이 SR650 V4 와
SR650a V4 를 뒤섞지 않게 만드는 1차 방어선이다(2차는 검색 단계의 product 필터).
"""
from __future__ import annotations

from dataclasses import dataclass

from app.config import settings
from app.ingest.parser import ParsedDoc, ProseBlock, TableRowBlock, match_products


@dataclass
class Chunk:
    ordinal: int
    kind: str           # table_row | prose
    content: str
    section_path: str
    page_from: int
    page_to: int
    products: list[str]


def _with_context(products: list[str], section: str, body: str) -> str:
    label = " / ".join(products)
    head = " > ".join(p for p in (label, section) if p)
    return f"{head}\n{body}" if head else body


def _flush_prose(
    buf: list[ProseBlock], candidates: list[str], out: list[Chunk]
) -> None:
    if not buf:
        return
    body = "\n".join(b.text for b in buf)
    products = match_products(body, candidates)
    out.append(
        Chunk(
            ordinal=len(out),
            kind="prose",
            content=_with_context(products, buf[0].section_path, body),
            section_path=buf[0].section_path,
            page_from=buf[0].page,
            page_to=buf[-1].page,
            products=products,
        )
    )


def chunk_document(doc: ParsedDoc) -> list[Chunk]:
    target = settings.chunk_target_chars
    overlap = settings.chunk_overlap_chars

    chunks: list[Chunk] = []
    buf: list[ProseBlock] = []
    buf_len = 0

    if target <= 0 or not 0 <= overlap < target:
        raise ValueError("Require 0 <= CHUNK_OVERLAP_CHARS < CHUNK_TARGET_CHARS")
    blocks = []
    for block in doc.blocks:
        if isinstance(block, ProseBlock) and len(block.text) > target:
            for start in range(0, len(block.text), target - overlap):
                blocks.append(ProseBlock(block.text[start:start + target], block.page, block.section_path))
                if start + target >= len(block.text):
                    break
        else:
            blocks.append(block)
    for block in blocks:
        if isinstance(block, TableRowBlock):
            # 표 행은 그 자체로 완결된 사실이므로 쪼개지도 합치지도 않는다.
            _flush_prose(buf, doc.products, chunks)
            buf, buf_len = [], 0
            body = block.render()
            products = match_products(body, doc.products)
            chunks.append(
                Chunk(
                    ordinal=len(chunks),
                    kind="table_row",
                    content=_with_context(products, block.section_path, body),
                    section_path=block.section_path,
                    page_from=block.page,
                    page_to=block.page,
                    products=products,
                )
            )
            continue

        # 산문: 섹션이 바뀌면 끊는다. 섹션 경계를 넘는 청크는 맥락이 섞인다.
        if buf and block.section_path != buf[-1].section_path:
            _flush_prose(buf, doc.products, chunks)
            buf, buf_len = [], 0

        buf.append(block)
        buf_len += len(block.text)

        if buf_len >= target:
            _flush_prose(buf, doc.products, chunks)
            # 겹침: 마지막 블록을 다음 청크로 넘겨 문맥 단절을 줄인다
            tail = buf[-1:] if len(buf[-1].text) <= overlap else []
            buf = list(tail)
            buf_len = sum(len(b.text) for b in buf)

    _flush_prose(buf, doc.products, chunks)
    for i, c in enumerate(chunks):
        c.ordinal = i
        if not c.products:
            c.content = f"{doc.title}\n{c.content}"
    return chunks
