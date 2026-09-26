"""색인 CLI.

사용:
    python -m app.ingest.cli data/pdfs/*.pdf
    python -m app.ingest.cli --force data/pdfs/sr650-v4.pdf
    python -m app.ingest.cli --dry-run data/pdfs/sr650-v4.pdf   # 파싱만, DB·모델 불필요
    python -m app.ingest.cli --sync data/pdfs                   # 변경분만 재색인
    python -m app.ingest.cli --acl-groups finance data/pricing/  # 열람 그룹 지정
"""
from __future__ import annotations

import argparse
import glob
import logging
import sys
from pathlib import Path

from app.ingest.chunker import chunk_document
from app.ingest.documents import SUPPORTED, parse_document
from app.observability import configure_logging

log = logging.getLogger(__name__)


def _dry_run(paths: list[Path]) -> int:
    for p in paths:
        doc = parse_document(p)
        chunks = chunk_document(doc)
        tables = sum(1 for c in chunks if c.kind == "table_row")
        tokens = sum(c.token_count for c in chunks)
        print(f"{p.name}")
        print(f"  제품    : {' / '.join(doc.products) or '(미검출)'}")
        print(f"  페이지  : {doc.page_count}")
        print(f"  청크    : {len(chunks)}  (표 {tables} / 산문 {len(chunks) - tables})")
        print(f"  토큰    : 약 {tokens:,}")
        for warning in doc.warnings:
            print(f"  주의    : {warning}")
        if doc.likely_scanned:
            print("  판정    : 스캔 PDF 로 의심됨. OCR 없이는 내용을 색인할 수 없다")
        if chunks:
            print(f"  예시    : {chunks[0].content.splitlines()[0][:70]}")
    return 0


def _check_migrations() -> None:
    """스키마가 최신인지 확인한다. 색인 중간에 컬럼 부재로 실패하는 것보다 낫다."""
    try:
        from app.db.migrate import applied, available
        from app.db.session import connect

        with connect() as conn:
            pending = [v for v, _ in available() if v not in applied(conn)]
        if pending:
            print(
                f"[!!] 미적용 마이그레이션 {pending} 이 있다. "
                f"먼저 `python -m app.db.migrate` 를 실행하라.",
                file=sys.stderr,
            )
            raise SystemExit(2)
    except SystemExit:
        raise
    except Exception as exc:
        print(f"[!!] DB 확인 실패: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc


def _expand(patterns: list[Path]) -> list[Path]:
    paths: list[Path] = []
    for pattern in patterns:
        matches = [Path(p) for p in sorted(glob.glob(str(pattern)))]
        if not matches:
            print(f"파일 없음: {pattern}", file=sys.stderr)
            raise SystemExit(2)
        for p in matches:
            if p.is_dir():
                paths.extend(
                    sorted(f for f in p.rglob("*") if f.suffix.lower() in SUPPORTED)
                )
            else:
                paths.append(p)
    return list(dict.fromkeys(paths))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="제품 가이드 문서 색인")
    ap.add_argument("paths", nargs="+", type=Path)
    ap.add_argument("--force", action="store_true", help="이미 색인된 문서도 재색인")
    ap.add_argument("--dry-run", action="store_true",
                    help="파싱·청킹까지만 수행. DB 와 임베딩 모델이 필요 없다")
    ap.add_argument("--sync", action="store_true",
                    help="디렉터리를 동기화한다. 변경된 파일만 다시 색인")
    ap.add_argument("--delete-missing", action="store_true",
                    help="--sync 에서 파일이 사라진 문서를 DB 에서도 지운다")
    ap.add_argument("--acl-groups", default="",
                    help="열람 그룹(쉼표 구분). 비우면 전사 공개")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args(argv)

    configure_logging(level="INFO" if args.verbose else "WARNING")
    groups = [g.strip() for g in args.acl_groups.split(",") if g.strip()]

    if args.dry_run:
        return _dry_run(_expand(args.paths))

    _check_migrations()
    from app.providers.base import get_embedding_provider

    provider = get_embedding_provider()

    if args.sync:
        from app.ingest.sync import run_sync

        failed = 0
        for root in args.paths:
            print(f"동기화: {root}")
            stats = run_sync(root, provider=provider, acl_groups=groups,
                             delete_missing=args.delete_missing)
            print(f"  신규 {len(stats['new'])} / 변경 {len(stats['changed'])} / "
                  f"이동 {len(stats['moved'])} / 변경없음 {stats['unchanged']}")
            print(f"  색인 완료 {stats['indexed']}")
            if stats["missing"]:
                note = "삭제됨" if args.delete_missing else "DB 에 남김"
                print(f"  파일 없음 {len(stats['missing'])}건 ({note}): "
                      f"{', '.join(stats['missing'][:5])}")
            for item in stats["failed"]:
                print(f"  [!!] {item['file']}: {item['detail'][:200]}", file=sys.stderr)
                failed += 1
        return 1 if failed else 0

    paths = _expand(args.paths)
    if not paths:
        print("지원하는 문서가 없습니다.", file=sys.stderr)
        return 2

    from app.ingest.pipeline import ingest_document

    failed = 0
    for p in paths:
        try:
            r = ingest_document(p, provider=provider, force=args.force,
                                acl_groups=groups)
        except Exception as exc:
            print(f"[!!] {p.name}: {exc}", file=sys.stderr)
            failed += 1
            continue
        mark = {"indexed": "OK", "skipped": "--", "failed": "!!"}[r.status]
        detail = r.detail or ("; ".join(r.warnings) if r.warnings else "")
        print(f"[{mark}] {p.name}  {' / '.join(r.products or [])}  "
              f"청크 {r.chunks}  토큰 {r.tokens:,}  {r.elapsed_ms / 1000:.0f}s  {detail}")
        failed += r.status == "failed"
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
