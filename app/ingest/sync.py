"""디렉터리 동기화 — 변경분만 다시 색인한다.

수동 색인은 첫 데모까지만 통한다. 사내에 붙이면 문서가 계속 바뀌고, 매번
전량 재색인하면 787페이지에 수십 분이 든다. 그래서 무엇이 바뀌었는지부터
판정한다.

판정 순서(싼 것부터)
  1. 경로·크기·수정시각이 DB 기록과 같으면 **변경 없음**. 파일을 읽지 않는다
  2. 다르면 해시를 계산한다. 해시가 이미 색인돼 있으면 **이동/복사**이므로
     경로만 갱신한다
  3. 그래도 없으면 **신규/변경**이라 색인한다

삭제는 기본으로 하지 않는다. 파일 서버가 잠깐 안 보이는 것과 문서가 삭제된
것을 구별할 수 없고, 잘못 지우면 되돌리는 비용이 재색인보다 크다.
"""
from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from app.config import settings
from app.db.session import connect
from app.ingest.documents import SUPPORTED
from app.ingest.pipeline import delete_document, ingest_document
from app.providers.base import EmbeddingProvider

log = logging.getLogger(__name__)


@dataclass
class SyncPlan:
    new: list[Path] = field(default_factory=list)
    changed: list[Path] = field(default_factory=list)
    unchanged: list[Path] = field(default_factory=list)
    moved: list[tuple[Path, int]] = field(default_factory=list)
    missing: list[tuple[str, int]] = field(default_factory=list)   # (title, document_id)

    def summary(self) -> dict:
        return {
            "new": [p.name for p in self.new],
            "changed": [p.name for p in self.changed],
            "moved": [p.name for p, _ in self.moved],
            "unchanged": len(self.unchanged),
            "missing": [t for t, _ in self.missing],
        }


def scan(root: str | Path) -> list[Path]:
    root = Path(root)
    if not root.exists():
        raise FileNotFoundError(f"문서 디렉터리가 없다: {root}")
    if root.is_file():
        return [root] if root.suffix.lower() in SUPPORTED else []
    return sorted(p for p in root.rglob("*") if p.suffix.lower() in SUPPORTED)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def plan(root: str | Path) -> SyncPlan:
    files = scan(root)
    result = SyncPlan()

    with connect() as conn:
        rows = conn.execute(
            """SELECT id, source_path, sha256, byte_size, source_mtime, status, title
               FROM documents"""
        ).fetchall()

    by_path = {r[1]: r for r in rows}
    by_sha = {r[2]: r for r in rows}
    seen_ids: set[int] = set()

    for path in files:
        key = str(path)
        info = path.stat()
        mtime = datetime.fromtimestamp(info.st_mtime, tz=timezone.utc)
        row = by_path.get(key)

        if (
            row
            and row[5] == "indexed"
            and row[3] == info.st_size
            and row[4] is not None
            and abs((row[4] - mtime).total_seconds()) < 2
        ):
            result.unchanged.append(path)
            seen_ids.add(row[0])
            continue

        sha = _sha256(path)
        known = by_sha.get(sha)
        if known and known[5] == "indexed":
            seen_ids.add(known[0])
            if known[1] == key:
                # 내용은 같고 수정시각만 바뀐 경우. 기록만 맞춰 둔다.
                result.unchanged.append(path)
                _touch(known[0], key, info.st_size, mtime)
            else:
                result.moved.append((path, known[0]))
                _touch(known[0], key, info.st_size, mtime)
            continue

        (result.changed if (row or known) else result.new).append(path)
        if row:
            seen_ids.add(row[0])

    # "사라진 문서"는 **이 디렉터리에서 색인된 것**만 본다. 업로드로 들어온
    # 문서는 source_path 가 파일명뿐이라 여기서 판정하면 존재하지 않는 것으로
    # 보이고, --delete-missing 과 만나면 멀쩡한 문서를 지운다.
    root_path = Path(root).resolve()
    for row in rows:
        if row[0] in seen_ids or row[5] != "indexed":
            continue
        recorded = Path(row[1])
        if not recorded.is_absolute():
            recorded = (Path.cwd() / recorded).resolve()
        if not recorded.is_relative_to(root_path):
            continue
        if not recorded.exists():
            result.missing.append((row[6], row[0]))
    return result


def _touch(document_id: int, source_path: str, size: int, mtime) -> None:
    with connect() as conn:
        conn.execute(
            """UPDATE documents
               SET source_path = %s, byte_size = %s, source_mtime = %s, updated_at = now()
               WHERE id = %s""",
            (source_path, size, mtime, document_id),
        )
        conn.commit()


def run_sync(
    root: str | Path | None = None,
    provider: EmbeddingProvider | None = None,
    acl_groups: list[str] | None = None,
    delete_missing: bool = False,
    on_progress=None,
) -> dict:
    root = root or settings.doc_root
    prepared = plan(root)
    targets = prepared.new + prepared.changed
    stats = prepared.summary()
    stats["indexed"] = 0
    stats["failed"] = []

    for i, path in enumerate(targets, start=1):
        if on_progress:
            on_progress(f"ingest:{path.name}", i - 1, len(targets))
        result = ingest_document(
            path, provider=provider, force=True, acl_groups=acl_groups,
            supersede=True,
        )
        if result.status == "indexed":
            stats["indexed"] += 1
        elif result.status == "failed":
            stats["failed"].append({"file": path.name, "detail": result.detail})

    if delete_missing:
        removed = []
        for title, document_id in prepared.missing:
            if delete_document(document_id):
                removed.append(title)
        stats["deleted"] = removed

    if on_progress:
        on_progress("done", len(targets), len(targets))
    log.info("동기화 완료", extra={"root": str(root), **{k: v for k, v in stats.items()
                                                        if not isinstance(v, list)}})
    return stats
