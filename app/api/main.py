"""질의응답 API.

PoC 를 사내에서 돌리는 데 필요한 것만 더했다. 화면이 아니라 운영에 쓰이는
경로들이다.

  - 인증·권한·요청 제한 (`app/security.py`)
  - 요청 상관 ID 와 단계별 지연 기록 (`app/observability.py`)
  - 색인은 작업 큐로. 큰 PDF 는 HTTP 요청 수명보다 오래 걸린다
  - `/search` — 답변 없이 검색 후보만 본다. "왜 못 찾았나"를 설명하는 창구
  - `/admin/queries` — 실패한 질문 목록. 다음 개선 대상을 목록으로 고른다
"""
from __future__ import annotations

import json
import logging
import shutil
import tempfile
import time
from contextlib import asynccontextmanager, nullcontext
from dataclasses import asdict
from pathlib import Path

import httpx
from fastapi import Depends, FastAPI, HTTPException, Query, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, PlainTextResponse, StreamingResponse
from pydantic import BaseModel, Field, field_validator

from app.config import ConfigError, settings
from app.db.session import connect
from app.observability import (
    configure_logging, metrics, new_request_id, principal_id, request_id,
)
from app.providers.base import get_embedding_provider, get_llm_provider
from app.retrieval import audit
from app.retrieval.answer import answer_question, answer_stream, retrieve
from app.security import AuthError, Principal, authenticate, limiter, require

log = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    configure_logging()
    try:
        warnings = settings.validate()
    except ConfigError as exc:
        # 잘못된 설정으로 절반만 동작하는 서버를 띄우지 않는다.
        log.error("설정 오류로 기동을 중단한다: %s", exc)
        raise
    for warning in warnings:
        log.warning("설정 경고: %s", warning)
    log.info(
        "기동", extra={"env": settings.app_env, "llm": settings.llm_model,
                       "embedding": settings.embedding_model,
                       "auth": settings.auth_enabled,
                       "reranker": settings.rerank_provider}
    )
    yield
    from app.db.session import close_pool

    close_pool()


app = FastAPI(title="사내 지식 RAG", version="0.3.0", lifespan=lifespan)

if settings.cors_origins:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=False,
        allow_methods=["GET", "POST", "DELETE"],
        allow_headers=["authorization", "content-type", "x-api-key"],
    )


@app.middleware("http")
async def context(request: Request, call_next):
    """요청 상관 ID 부여와 접근 로그.

    로그 한 줄만 보고 어떤 요청의 어느 단계인지 알 수 있어야 사후 분석이 된다.
    """
    token = request_id.set(request.headers.get("x-request-id") or new_request_id())
    who = principal_id.set("-")
    started = time.perf_counter()
    try:
        response = await call_next(request)
    finally:
        principal_id.reset(who)
    elapsed = (time.perf_counter() - started) * 1000
    response.headers["x-request-id"] = request_id.get()
    # 브라우저에서 직접 열리는 화면이므로 최소한의 보호 헤더는 붙인다.
    response.headers["x-content-type-options"] = "nosniff"
    response.headers["referrer-policy"] = "no-referrer"
    # 지표 라벨에는 실제 경로가 아니라 라우트 틀(`/jobs/{job_id}`)을 쓴다.
    # `/jobs/12` 를 그대로 쓰면 요청마다 새 시계열이 생겨 메모리가 계속 늘어난다.
    route = request.scope.get("route")
    label = getattr(route, "path", None) or request.url.path
    if label not in {"/metrics", "/health"}:
        metrics.observe("http_request_ms", elapsed, path=label)
        metrics.inc("http_requests_total", path=label,
                    status=response.status_code)
        log.info(
            "%s %s -> %s (%.0fms)", request.method, request.url.path,
            response.status_code, elapsed,
        )
    request_id.reset(token)
    return response


@app.exception_handler(AuthError)
async def auth_error(request, exc: AuthError):
    return JSONResponse(status_code=exc.status, content={"detail": exc.detail})


@app.exception_handler(Exception)
async def unavailable(request, exc):
    log.exception("요청 처리 실패", exc_info=exc)
    metrics.inc("http_unhandled_total")
    return JSONResponse(
        status_code=503,
        content={"detail": "처리에 실패했습니다. DB·모델 연결 및 서버 로그를 확인하세요."},
    )


def caller(request: Request) -> Principal:
    """요청 주체를 확인하고 요청 제한을 적용한다."""
    header = request.headers.get("authorization") or request.headers.get("x-api-key")
    principal = authenticate(header)
    principal_id.set(principal.id)
    limiter.check(principal.id)
    return principal


_embedder = None
_llm = None


def _providers():
    global _embedder, _llm
    if _embedder is None:
        _embedder = get_embedding_provider()
    if _llm is None:
        _llm = get_llm_provider()
    return _embedder, _llm


@app.get("/", include_in_schema=False)
def home():
    return FileResponse(Path(__file__).with_name("index.html"))


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=4000)
    top_k: int | None = Field(default=None, ge=1, le=50)
    products: list[str] | None = Field(
        default=None,
        description="모델 필터를 직접 지정. 생략하면 질문에서 자동 추출한다.",
    )
    document_ids: list[int] | None = Field(
        default=None, description="특정 문서 안에서만 검색한다."
    )
    reranker: str | None = Field(
        default=None, description="none|rules|llm. 생략하면 서버 기본값."
    )
    use_rag: bool = Field(
        default=True,
        description="False 면 검색 없이 같은 모델로 답한다. 대비군 시연·평가용.",
    )

    @field_validator("question")
    @classmethod
    def nonblank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("질문을 입력하세요.")
        return value.strip()

    @field_validator("reranker")
    @classmethod
    def known_reranker(cls, value):
        if value is not None and value not in {"none", "rules", "llm"}:
            raise ValueError("reranker 는 none|rules|llm 중 하나여야 합니다.")
        return value


class FeedbackRequest(BaseModel):
    query_id: int = Field(ge=1)
    rating: int = Field(description="-1 또는 1")
    comment: str | None = Field(default=None, max_length=2000)

    @field_validator("rating")
    @classmethod
    def valid(cls, value: int) -> int:
        if value not in (-1, 1):
            raise ValueError("rating 은 -1 또는 1 이어야 합니다.")
        return value


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "version": app.version,
        "embedding_model": settings.embedding_model,
        "llm_model": settings.llm_model,
        "reranker": settings.rerank_provider,
        "auth": settings.auth_enabled,
    }


@app.get("/ready")
def ready():
    """의존성 점검. 하나라도 준비되지 않으면 503 을 준다."""
    checks: dict[str, object] = {}
    try:
        with connect() as conn:
            conn.execute("SELECT embedding FROM chunks LIMIT 0")
            from app.db.migrate import applied, available

            pending = [v for v, _ in available() if v not in applied(conn)]
        checks["database"] = True
        checks["migrations"] = "최신" if not pending else f"미적용 {pending}"
        if pending:
            checks["database"] = False
    except Exception as exc:
        checks["database"] = False
        checks["database_error"] = str(exc)[:200]

    needed = []
    if settings.embedding_provider == "ollama":
        needed.append(settings.embedding_model)
    if settings.llm_provider == "ollama":
        needed.append(settings.llm_model)
    try:
        if needed:
            response = httpx.get(f"{settings.ollama_base_url}/api/tags", timeout=5)
            response.raise_for_status()
            names = {m["name"] for m in response.json()["models"]}
            missing = [m for m in needed
                       if m not in names and f"{m}:latest" not in names]
            checks["models"] = not missing
            if missing:
                checks["models_missing"] = missing
        else:
            checks["models"] = True
    except Exception:
        checks["models"] = False

    if settings.llm_provider == "ollama" or settings.embedding_provider == "ollama":
        from app.providers.ollama import breaker_state

        checks["model_breaker"] = breaker_state()
        if checks["model_breaker"] == "open":
            checks["models"] = False

    ok = all(v is not False for v in checks.values())
    return JSONResponse(
        status_code=200 if ok else 503,
        content={"status": "ready" if ok else "unavailable", "checks": checks},
    )


@app.get("/metrics", include_in_schema=False)
def prometheus():
    if not settings.metrics_enabled:
        raise HTTPException(404, "메트릭이 꺼져 있습니다.")
    return PlainTextResponse(metrics.render_prometheus(), media_type="text/plain")


@app.post("/ask")
def ask(req: AskRequest, principal: Principal = Depends(caller)) -> dict:
    embedder, llm = _providers()
    try:
        with connect() if req.use_rag else nullcontext(None) as conn:
            result = answer_question(
                conn, req.question, embedder, llm,
                top_k=req.top_k, products=req.products, use_rag=req.use_rag,
                groups=principal.acl_groups, document_ids=req.document_ids,
                reranker=req.reranker,
            )
    except Exception as exc:
        log.exception("질의응답 실패")
        raise HTTPException(503, "DB·모델 연결 및 서버 로그를 확인하세요.") from exc
    return asdict(result)


@app.post("/ask/stream")
def ask_stream(req: AskRequest, principal: Principal = Depends(caller)):
    """SSE 스트리밍 답변.

    검증은 마지막 `done` 이벤트에 담는다. 마지막 토큰을 받기 전에는 인용과
    근거를 확인할 수 없기 때문이다(answer.answer_stream 주석 참고).
    """
    embedder, llm = _providers()
    if not hasattr(llm, "generate_stream"):
        raise HTTPException(501, "이 생성 제공자는 스트리밍을 지원하지 않습니다.")

    def events():
        try:
            with connect() as conn:
                for event in answer_stream(
                    conn, req.question, embedder, llm, top_k=req.top_k,
                    products=req.products, groups=principal.acl_groups,
                    reranker=req.reranker,
                ):
                    yield f"data: {json.dumps(event, ensure_ascii=False, default=str)}\n\n"
        except Exception as exc:
            log.exception("스트리밍 실패")
            payload = {"type": "error", "detail": "처리에 실패했습니다."}
            yield f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"

    return StreamingResponse(
        events(),
        media_type="text/event-stream",
        headers={"cache-control": "no-cache", "x-accel-buffering": "no"},
    )


@app.post("/search")
def search_only(req: AskRequest, principal: Principal = Depends(caller)) -> dict:
    """검색 후보만 돌려준다. 생성 모델을 거치지 않는다.

    "왜 이 질문에 답을 못 했나"를 설명하려면 검색이 무엇을 가져왔는지부터
    봐야 한다. 생성 품질과 검색 품질을 분리해서 보는 창구다.
    """
    embedder, _ = _providers()
    with connect() as conn:
        hits, aq, timings = retrieve(
            conn, req.question, embedder, top_k=req.top_k,
            products=req.products, groups=principal.acl_groups,
            document_ids=req.document_ids, reranker=req.reranker,
        )
    return {
        "question": req.question,
        "intent": aq.intent,
        "products": aq.products,
        "identifiers": aq.identifiers,
        "terms": aq.terms,
        "latency_ms": timings,
        "hits": [
            {
                "chunk_id": h.chunk_id, "citation": h.citation, "kind": h.kind,
                "section_path": h.section_path, "products": h.products,
                "score": round(h.score, 4), "paths": h.paths,
                "excerpt": h.content[:400],
            }
            for h in hits
        ],
    }


@app.post("/feedback")
def feedback(req: FeedbackRequest, principal: Principal = Depends(caller)) -> dict:
    with connect() as conn:
        ok = audit.set_feedback(conn, req.query_id, req.rating, req.comment)
    if not ok:
        raise HTTPException(404, "해당 질의 기록을 찾을 수 없습니다.")
    metrics.inc("rag_feedback_total", rating=req.rating)
    return {"status": "ok"}


@app.post("/ingest")
def ingest(
    file: UploadFile,
    principal: Principal = Depends(caller),
    force: bool = False,
    wait: bool = Query(False, description="true 면 색인이 끝날 때까지 기다린다"),
    acl_groups: str = Query("", description="열람 그룹. 쉼표로 구분. 비우면 전사 공개"),
):
    require(principal, "editor")
    from app.ingest.documents import SUPPORTED
    from app.ingest.pipeline import ingest_document

    name = Path((file.filename or "").replace("\\", "/")).name
    if Path(name).suffix.lower() not in SUPPORTED:
        raise HTTPException(415, "PDF, DOCX, XLSX, PPTX만 지원합니다.")
    groups = [g.strip() for g in acl_groups.split(",") if g.strip()]

    # 업로드 파일은 임시 디렉터리에 격리한다. 색인이 끝나면 원본을 지우고
    # DB 의 청크만 남긴다. 비동기 경로에서는 작업이 끝낸 뒤 지운다.
    limit = settings.max_upload_mb * 1024 * 1024
    holder = Path(tempfile.mkdtemp(prefix="ingest-"))
    path = holder / name
    size = 0
    try:
        with path.open("wb") as target:
            while block := file.file.read(1024 * 1024):
                size += len(block)
                if size > limit:
                    raise HTTPException(
                        413, f"파일은 {settings.max_upload_mb}MB 이하여야 합니다."
                    )
                target.write(block)
        if not size:
            raise HTTPException(400, "빈 파일입니다.")
    except Exception:
        shutil.rmtree(holder, ignore_errors=True)
        raise

    if wait:
        try:
            result = ingest_document(path, force=force, source_name=name,
                                     acl_groups=groups)
        except (ValueError, RuntimeError) as exc:
            raise HTTPException(422, "문서를 읽을 수 없습니다. 파일 형식을 확인하세요.") from exc
        finally:
            shutil.rmtree(holder, ignore_errors=True)
        if result.status == "failed":
            raise HTTPException(422, "색인에 실패했습니다. 문서 목록과 서버 로그를 확인하세요.")
        result.path = name
        return asdict(result)

    from app.ingest import jobs

    job_id = jobs.create("upload", name, principal.id)

    def work(progress):
        try:
            result = ingest_document(path, force=force, source_name=name,
                                     acl_groups=groups, on_progress=progress)
            if result.status == "failed":
                raise RuntimeError(result.detail or "색인 실패")
            return {"status": result.status, "chunks": result.chunks,
                    "products": result.products, "warnings": result.warnings}
        finally:
            shutil.rmtree(holder, ignore_errors=True)

    jobs.submit(job_id, work, principal.id)
    return JSONResponse(
        status_code=202,
        content={"job_id": job_id, "status": "queued", "file": name,
                 "poll": f"/jobs/{job_id}"},
    )


@app.post("/admin/sync")
def admin_sync(
    principal: Principal = Depends(caller),
    root: str = Query("", description="비우면 DOC_ROOT"),
    delete_missing: bool = False,
    dry_run: bool = False,
    acl_groups: str = Query(""),
):
    """서버에 있는 문서 디렉터리를 동기화한다. 변경분만 다시 색인한다."""
    require(principal, "admin")
    from app.ingest import jobs, sync

    target = root or settings.doc_root
    groups = [g.strip() for g in acl_groups.split(",") if g.strip()]
    try:
        prepared = sync.plan(target)
    except FileNotFoundError as exc:
        raise HTTPException(404, str(exc)) from exc
    if dry_run:
        return {"root": target, "plan": prepared.summary()}

    job_id = jobs.create("sync", target, principal.id)
    jobs.submit(
        job_id,
        lambda progress: sync.run_sync(
            target, acl_groups=groups, delete_missing=delete_missing,
            on_progress=progress,
        ),
        principal.id,
    )
    return JSONResponse(
        status_code=202,
        content={"job_id": job_id, "status": "queued", "root": target,
                 "plan": prepared.summary(), "poll": f"/jobs/{job_id}"},
    )


@app.get("/jobs")
def list_jobs(principal: Principal = Depends(caller), limit: int = Query(20, ge=1, le=200)):
    from app.ingest import jobs

    return {"jobs": jobs.recent(limit)}


@app.get("/jobs/{job_id}")
def job_status(job_id: int, principal: Principal = Depends(caller)):
    from app.ingest import jobs

    found = jobs.get(job_id)
    if not found:
        raise HTTPException(404, "작업을 찾을 수 없습니다.")
    return found


@app.get("/documents")
def documents(principal: Principal = Depends(caller)) -> dict:
    from app.ingest.pipeline import document_rows

    with connect() as conn:
        rows = document_rows(conn)
    groups = principal.acl_groups
    if groups is not None:
        # 열람 권한이 없는 문서는 목록에서도 보이지 않아야 한다.
        rows = [r for r in rows
                if not r["acl_groups"] or set(r["acl_groups"]) & set(groups)]
    return {"documents": rows}


@app.delete("/documents/{document_id}")
def delete_document(document_id: int, principal: Principal = Depends(caller)) -> dict:
    require(principal, "editor")
    from app.ingest.pipeline import delete_document as remove

    if not remove(document_id):
        raise HTTPException(404, "문서를 찾을 수 없습니다.")
    return {"status": "deleted", "document_id": document_id}


@app.get("/admin/queries")
def admin_queries(
    principal: Principal = Depends(caller),
    limit: int = Query(50, ge=1, le=500),
    only_failed: bool = False,
):
    """질의 로그. 실패한 질문을 모으는 것이 품질 개선의 가장 빠른 길이다."""
    require(principal, "admin")
    with connect() as conn:
        return {"queries": audit.recent(conn, limit, only_failed)}


@app.get("/admin/stats")
def admin_stats(principal: Principal = Depends(caller), days: int = Query(7, ge=1, le=365)):
    require(principal, "admin")
    with connect() as conn:
        summary = audit.stats(conn, days)
    return {"queries": summary, "process": metrics.snapshot()}
