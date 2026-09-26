"""스키마 마이그레이션 러너.

    python -m app.db.migrate            # 미적용 마이그레이션 적용
    python -m app.db.migrate --status   # 적용 상태만 확인

단일 schema.sql 을 버리고 번호가 붙은 파일로 나눈 이유는 하나다. PoC 가
사내에서 돌기 시작하면 이미 색인된 DB 를 지우지 않고 스키마를 바꿔야 한다.
"지우고 다시 만들기"는 첫 데모까지만 통한다.

규칙
  - 파일명은 `NNN_이름.sql`. NNN 이 버전이다
  - 모든 문장은 재실행 가능해야 한다(IF NOT EXISTS 등)
  - 파일 마지막에 자기 버전을 schema_migrations 에 남긴다.
    컨테이너 initdb 가 이 디렉터리를 직접 실행해도 기록이 남게 하기 위해서다
"""
from __future__ import annotations

import argparse
import logging
import re
from pathlib import Path

log = logging.getLogger(__name__)

MIGRATIONS_DIR = Path(__file__).parent / "migrations"
_NAME = re.compile(r"^(\d{3})_[\w-]+\.sql$")


def available() -> list[tuple[str, Path]]:
    out: list[tuple[str, Path]] = []
    for path in sorted(MIGRATIONS_DIR.glob("*.sql")):
        match = _NAME.match(path.name)
        if not match:
            raise ValueError(f"마이그레이션 파일명 규칙 위반: {path.name} (NNN_이름.sql)")
        out.append((match.group(1), path))
    versions = [v for v, _ in out]
    if len(set(versions)) != len(versions):
        raise ValueError(f"마이그레이션 버전이 중복됐다: {versions}")
    return out


def applied(conn) -> set[str]:
    """이미 적용된 버전. 테이블 자체가 없으면 빈 집합으로 본다."""
    row = conn.execute(
        "SELECT to_regclass(current_schema() || '.schema_migrations') IS NOT NULL"
    ).fetchone()
    if not row[0]:
        return set()
    return {r[0] for r in conn.execute("SELECT version FROM schema_migrations")}


def apply_migrations(conn, verbose: bool = False) -> list[str]:
    """미적용 마이그레이션을 순서대로 적용하고 적용한 버전 목록을 돌려준다."""
    done = applied(conn)
    ran: list[str] = []
    for version, path in available():
        if version in done:
            continue
        sql = path.read_text(encoding="utf-8")
        conn.execute(sql)
        conn.execute(
            "INSERT INTO schema_migrations (version) VALUES (%s) ON CONFLICT DO NOTHING",
            (version,),
        )
        conn.commit()
        ran.append(version)
        if verbose:
            print(f"적용: {path.name}")
        log.info("마이그레이션 적용", extra={"version": version, "file": path.name})
    return ran


def main(argv: list[str] | None = None) -> int:
    from app.db.session import connect
    from app.observability import configure_logging

    ap = argparse.ArgumentParser(description="스키마 마이그레이션")
    ap.add_argument("--status", action="store_true", help="적용 상태만 확인")
    args = ap.parse_args(argv)

    configure_logging()
    try:
        with connect() as conn:
            done = applied(conn)
            if args.status:
                for version, path in available():
                    mark = "적용됨" if version in done else "미적용"
                    print(f"  {version}  {mark}  {path.name}")
                return 0
            ran = apply_migrations(conn, verbose=True)
            print(f"적용한 마이그레이션 {len(ran)}건" if ran else "변경 없음 (최신 상태)")
    except Exception as exc:
        print(f"마이그레이션 실패: {exc}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
