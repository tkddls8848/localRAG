"""색인 CLI.

사용:
    python -m app.ingest.cli data/pdfs/*.pdf
    python -m app.ingest.cli --force data/pdfs/sr650-v4.pdf
    python -m app.ingest.cli --dry-run data/pdfs/sr650-v4.pdf   # 파싱만, DB·모델 불필요
"""
from __future__ import annotations

import argparse
import glob
import logging
import sys
from pathlib import Path

from app.ingest.chunker import chunk_document
from app.ingest.documents import parse_document, SUPPORTED


def _dry_run(paths: list[Path]) -> int:
    for p in paths:
        doc = parse_document(p)
        chunks = chunk_document(doc)
        tables = sum(1 for c in chunks if c.kind == "table_row")
        print(f"{p.name}")
        print(f"  제품    : {' / '.join(doc.products) or '(미검출)'}")
        print(f"  페이지  : {doc.page_count}")
        print(f"  청크    : {len(chunks)}  (표 {tables} / 산문 {len(chunks) - tables})")
        if chunks:
            print(f"  예시    : {chunks[0].content.splitlines()[0][:70]}")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="제품 가이드 PDF 색인")
    ap.add_argument("paths", nargs="+", type=Path)
    ap.add_argument("--force", action="store_true", help="이미 색인된 문서도 재색인")
    ap.add_argument("--dry-run", action="store_true",
                    help="파싱·청킹까지만 수행. DB 와 임베딩 모델이 필요 없다")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(levelname)s %(message)s",
    )

    paths: list[Path] = []
    for pattern in args.paths:
        matches = [Path(p) for p in sorted(glob.glob(str(pattern)))]
        if not matches:
            print(f"파일 없음: {pattern}", file=sys.stderr)
            return 2
        for p in matches:
            if p.is_dir():
                paths.extend(sorted(f for f in p.rglob('*') if f.suffix.lower() in SUPPORTED))
            else:
                paths.append(p)
    paths = list(dict.fromkeys(paths))
    if not paths:
        print("지원하는 문서가 없습니다.", file=sys.stderr)
        return 2

    if args.dry_run:
        return _dry_run(paths)

    from app.ingest.pipeline import ingest_pdf
    from app.providers.base import get_embedding_provider

    provider = get_embedding_provider()
    failed = 0
    for p in paths:
        try:
            r = ingest_pdf(p, provider=provider, force=args.force)
        except Exception as exc:
            print(f"[!!] {p.name}: {exc}", file=sys.stderr)
            failed += 1
            continue
        mark = {"indexed": "OK", "skipped": "--", "failed": "!!"}[r.status]
        print(f"[{mark}] {p.name}  {' / '.join(r.products or [])}  청크 {r.chunks}  {r.detail}")
        failed += r.status == "failed"
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
