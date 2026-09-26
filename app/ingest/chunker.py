"""블록 → 청크.

핵심 원칙: 청크는 **단독으로 읽혀야 한다.**
"Memory maximum: Up to 4TB" 만 떼어놓으면 어느 장비 이야기인지 알 수 없다.
그래서 모든 청크 앞에 제품명과 섹션 경로를 붙인다. 이것이 SR650 V4 와
SR650a V4 를 뒤섞지 않게 만드는 1차 방어선이다(2차는 검색 단계의 product 필터).
"""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass

from app.config import settings
from app.ingest.parser import ParsedDoc, ProseBlock, TableRowBlock
from app.taxonomy import match_products


@dataclass
class Chunk:
    ordinal: int
    kind: str           # table_row | prose
    content: str
    section_path: str
    page_from: int
    page_to: int
    products: list[str]
    # 검색 결과에서 같은 사실을 여러 건 돌려주지 않기 위한 키. 섹션 경로가
    # 달라도 같은 제품의 같은 본문이면 같은 해시가 된다.
    content_hash: str = ""
    # 컨텍스트 예산 계산용 근사값. 정확한 토큰 수가 필요한 용도는 없다.
    token_count: int = 0


def estimate_tokens(text: str) -> int:
    """토크나이저를 불러오지 않고 토큰 수를 근사한다.

    영문은 대략 4자/토큰, 한글은 1.5자/토큰이다. 컨텍스트 예산과 보고용
    수치이므로 이 정도 정확도로 충분하고, 모델별 토크나이저 의존을 피한다.
    """
    ascii_chars = sum(1 for c in text if ord(c) < 128)
    wide = len(text) - ascii_chars
    return max(1, round(ascii_chars / 4 + wide / 1.5))


def _hash(products: list[str], body: str) -> str:
    normalized = re.sub(r"\s+", " ", body).strip().lower()
    key = "|".join(sorted(products)) + "||" + normalized
    return hashlib.blake2b(key.encode("utf-8"), digest_size=16).hexdigest()


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
    content = _with_context(products, buf[0].section_path, body)
    out.append(
        Chunk(
            ordinal=len(out),
            kind="prose",
            content=content,
            section_path=buf[0].section_path,
            page_from=buf[0].page,
            page_to=buf[-1].page,
            products=products,
            content_hash=_hash(products, body),
            token_count=estimate_tokens(content),
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
            content = _with_context(products, block.section_path, body)
            chunks.append(
                Chunk(
                    ordinal=len(chunks),
                    kind="table_row",
                    content=content,
                    section_path=block.section_path,
                    page_from=block.page,
                    page_to=block.page,
                    products=products,
                    content_hash=_hash(products, body),
                    token_count=estimate_tokens(content),
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
            # 제품 태그가 없으면 최소한 문서명은 붙여야 단독으로 읽힌다.
            c.content = f"{doc.title}\n{c.content}"
            c.token_count = estimate_tokens(c.content)
    return chunks
