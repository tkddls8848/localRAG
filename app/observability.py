"""로깅·메트릭.

PoC 를 사내에서 돌리면 "왜 이 질문은 답을 못 찾았나"를 사후에 재구성해야 한다.
그래서 두 가지를 최소한으로 갖춘다.

  1. 요청 단위 상관관계 — 모든 로그 줄에 request_id 를 붙인다
  2. 단계별 소요 시간 — 검색과 생성을 분리해서 기록한다. 느린 것을
     품질 문제로 오진하지 않기 위해서다(architecture.md 6).

외부 수집기를 붙일 수 있도록 Prometheus 텍스트 형식으로도 노출한다.
의존성을 늘리지 않기 위해 직접 만든다. 지표 수가 적어 충분하다.
"""
from __future__ import annotations

import json
import logging
import sys
import threading
import time
import uuid
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field

request_id: ContextVar[str] = ContextVar("request_id", default="-")
principal_id: ContextVar[str] = ContextVar("principal_id", default="-")

_RESERVED = set(logging.LogRecord("", 0, "", 0, "", (), None).__dict__) | {
    "asctime",
    "message",
    "taskName",
}


def new_request_id() -> str:
    return uuid.uuid4().hex[:12]


class _JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(record.created))
            + ".%03d" % int(record.msecs),
            "level": record.levelname,
            "logger": record.name,
            "request_id": request_id.get(),
            "principal": principal_id.get(),
            "message": record.getMessage(),
        }
        for key, value in record.__dict__.items():
            if key not in _RESERVED and not key.startswith("_"):
                payload[key] = value
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False, default=str)


class _TextFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        rid = request_id.get()
        head = self.formatTime(record, "%H:%M:%S") + " " + record.levelname.ljust(7)
        tail = " [" + rid + "]" if rid != "-" else ""
        body = head + tail + " " + record.name + ": " + record.getMessage()
        if record.exc_info:
            body += "\n" + self.formatException(record.exc_info)
        return body


def configure_logging(level: str | None = None, fmt: str | None = None) -> None:
    from app.config import settings

    handler = logging.StreamHandler(sys.stderr)
    use_json = (fmt or settings.log_format) == "json"
    handler.setFormatter(_JsonFormatter() if use_json else _TextFormatter())
    root = logging.getLogger()
    root.handlers[:] = [handler]
    root.setLevel(getattr(logging, (level or settings.log_level), logging.INFO))
    # uvicorn 이 자기 핸들러를 심어 두면 형식이 섞인다.
    for name in ("uvicorn", "uvicorn.access", "uvicorn.error"):
        logging.getLogger(name).handlers[:] = []
        logging.getLogger(name).propagate = True


_BUCKETS_MS = (50, 100, 250, 500, 1000, 2500, 5000, 10_000, 30_000)


@dataclass
class _Histogram:
    buckets: dict[float, int] = field(
        default_factory=lambda: {b: 0 for b in _BUCKETS_MS}
    )
    count: int = 0
    total: float = 0.0

    def observe(self, value_ms: float) -> None:
        self.count += 1
        self.total += value_ms
        for bound in self.buckets:
            if value_ms <= bound:
                self.buckets[bound] += 1


class Metrics:
    """프로세스 수명 동안의 누적 지표. 재시작하면 초기화된다."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._counters: dict[tuple[str, tuple], int] = {}
        self._histograms: dict[tuple[str, tuple], _Histogram] = {}

    def inc(self, name: str, value: int = 1, **labels) -> None:
        key = (name, tuple(sorted(labels.items())))
        with self._lock:
            self._counters[key] = self._counters.get(key, 0) + value

    def observe(self, name: str, value_ms: float, **labels) -> None:
        key = (name, tuple(sorted(labels.items())))
        with self._lock:
            self._histograms.setdefault(key, _Histogram()).observe(value_ms)

    def snapshot(self) -> dict:
        with self._lock:
            return {
                "counters": {
                    self._render_key(n, lb): v for (n, lb), v in self._counters.items()
                },
                "latency_ms": {
                    self._render_key(n, lb): {
                        "count": h.count,
                        "avg": round(h.total / h.count, 1) if h.count else None,
                    }
                    for (n, lb), h in self._histograms.items()
                },
            }

    @staticmethod
    def _render_key(name: str, labels: tuple) -> str:
        if not labels:
            return name
        inner = ",".join(k + '="' + str(v) + '"' for k, v in labels)
        return name + "{" + inner + "}"

    def render_prometheus(self) -> str:
        lines: list[str] = []
        with self._lock:
            for (name, labels), value in sorted(self._counters.items()):
                lines.append("# TYPE " + name + " counter")
                lines.append(self._render_key(name, labels) + " " + str(value))
            for (name, labels), hist in sorted(self._histograms.items()):
                lines.append("# TYPE " + name + " histogram")
                base = list(labels)
                for bound, count in hist.buckets.items():
                    key = self._render_key(
                        name + "_bucket", tuple(base + [("le", bound)])
                    )
                    lines.append(key + " " + str(count))
                inf = self._render_key(name + "_bucket", tuple(base + [("le", "+Inf")]))
                lines.append(inf + " " + str(hist.count))
                lines.append(
                    self._render_key(name + "_sum", labels) + " %.1f" % hist.total
                )
                lines.append(
                    self._render_key(name + "_count", labels) + " " + str(hist.count)
                )
        return "\n".join(lines) + "\n"


metrics = Metrics()


@contextmanager
def timed(name: str, **labels):
    """블록 소요 시간을 히스토그램에 기록하고 결과를 dict 로 돌려준다."""
    started = time.perf_counter()
    result = {"ms": 0.0}
    try:
        yield result
    finally:
        result["ms"] = (time.perf_counter() - started) * 1000
        metrics.observe(name, result["ms"], **labels)
