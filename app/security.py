"""인증·권한·요청 제한.

사내 문서는 전사 공개가 아니다. 접근 권한 분리는 실서비스의 전제 조건이고
(architecture.md 8.1), PoC 단계에서도 "이 구조로 권한을 걸 수 있다"를 보여줘야
도입 검토가 다음 단계로 넘어간다. 그래서 세 가지만 넣는다.

  - API 키 -> 주체(역할 + 소속 그룹)
  - 문서의 acl_groups 와 주체 그룹의 교집합으로 검색 범위를 제한
  - 주체별 분당 요청 제한

키를 DB 가 아니라 환경변수에 두는 이유는 PoC 에서 사용자 관리 화면을 만들
이유가 없기 때문이다. 사내 SSO 를 붙일 때 이 모듈만 교체하면 된다.
"""
from __future__ import annotations

import hmac
import logging
import threading
import time
from dataclasses import dataclass

from app.config import settings

log = logging.getLogger(__name__)

ROLES = ("reader", "editor", "admin")
_RANK = {role: i for i, role in enumerate(ROLES)}


@dataclass(frozen=True)
class Principal:
    id: str
    role: str = "reader"
    groups: tuple[str, ...] = ()

    def can(self, role: str) -> bool:
        return _RANK.get(self.role, -1) >= _RANK[role]

    @property
    def acl_groups(self) -> list[str] | None:
        """검색에 쓸 그룹 목록. None 은 제한 없음(관리자)."""
        return None if self.role == "admin" else list(self.groups)


# 인증이 꺼진 상태(로컬 개발)의 주체. 감사 로그에 그대로 남는다.
ANONYMOUS = Principal(id="anonymous", role="admin", groups=())


class AuthError(Exception):
    def __init__(self, status: int, detail: str) -> None:
        super().__init__(detail)
        self.status = status
        self.detail = detail


def parse_keys(raw: str) -> dict[str, Principal]:
    """키:역할:그룹1|그룹2 를 쉼표로 나열한 형식을 읽는다."""
    table: dict[str, Principal] = {}
    for entry in (e.strip() for e in raw.split(",")):
        if not entry:
            continue
        parts = entry.split(":")
        key = parts[0].strip()
        role = parts[1].strip() if len(parts) > 1 and parts[1].strip() else "reader"
        raw_groups = parts[2] if len(parts) > 2 else ""
        groups = tuple(g.strip() for g in raw_groups.split("|") if g.strip())
        if not key:
            continue
        if role not in ROLES:
            raise ValueError("알 수 없는 역할: " + repr(role) + " (가능: " + ", ".join(ROLES) + ")")
        if len(key) < 16:
            log.warning("API 키가 16자 미만이다. 추측 가능한 키를 쓰지 말 것.")
        table[key] = Principal(id=role + ":" + key[:6], role=role, groups=groups)
    return table


_KEYS = parse_keys(settings.api_keys)


def authenticate(header: str | None) -> Principal:
    """Authorization 헤더 또는 X-API-Key 값을 주체로 바꾼다."""
    if not settings.auth_enabled:
        return ANONYMOUS
    if not header:
        raise AuthError(401, "API 키가 필요하다. Authorization: Bearer <key>")
    if header.lower().startswith("bearer "):
        token = header.split(" ", 1)[1].strip()
    else:
        token = header.strip()
    for key, principal in _KEYS.items():
        # 타이밍 차이로 키를 추론하지 못하게 상수시간 비교를 쓴다.
        if hmac.compare_digest(key, token):
            return principal
    raise AuthError(403, "유효하지 않은 API 키다.")


def require(principal: Principal, role: str) -> None:
    if not principal.can(role):
        raise AuthError(403, role + " 권한이 필요하다.")


@dataclass
class _Bucket:
    tokens: float
    updated: float


class RateLimiter:
    """주체별 토큰 버킷.

    단일 프로세스 안의 방어선이다. 인스턴스를 여러 개로 늘리면 공유 저장소가
    필요하지만, 반복 질의가 모델 런타임을 마비시키는 것을 막는 데는 충분하다.
    """

    def __init__(self, per_min: int | None = None) -> None:
        self.per_min = settings.rate_limit_per_min if per_min is None else per_min
        self._buckets: dict[str, _Bucket] = {}
        self._lock = threading.Lock()

    def check(self, key: str) -> None:
        if self.per_min <= 0:
            return
        now = time.monotonic()
        rate = self.per_min / 60.0
        with self._lock:
            bucket = self._buckets.get(key)
            if bucket is None:
                self._buckets[key] = _Bucket(tokens=self.per_min - 1, updated=now)
                return
            refilled = bucket.tokens + (now - bucket.updated) * rate
            bucket.tokens = min(float(self.per_min), refilled)
            bucket.updated = now
            if bucket.tokens < 1:
                retry = int((1 - bucket.tokens) / rate) + 1
                raise AuthError(429, "요청이 너무 많다. " + str(retry) + "초 후 다시 시도하라.")
            bucket.tokens -= 1


limiter = RateLimiter()
