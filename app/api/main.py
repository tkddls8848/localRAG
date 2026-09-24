"""질의응답 API."""
from __future__ import annotations

from dataclasses import asdict
from contextlib import nullcontext
from pathlib import Path
import logging
import tempfile

from fastapi import FastAPI, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field, field_validator
import httpx

from app.config import settings
from app.db.session import connect
from app.providers.base import get_embedding_provider, get_llm_provider
from app.retrieval.answer import answer_question

app = FastAPI(title="사내 지식 RAG", version="0.2.0")
log = logging.getLogger(__name__)


@app.exception_handler(Exception)
async def unavailable(request, exc):
    log.exception("요청 처리 실패", exc_info=exc)
    return JSONResponse(status_code=503, content={"detail": "처리에 실패했습니다. DB·모델 연결 및 서버 로그를 확인하세요."})


@app.get("/", include_in_schema=False)
def home():
    return FileResponse(Path(__file__).with_name("index.html"))

_embedder = None
_llm = None


def _providers():
    global _embedder, _llm
    if _embedder is None:
        _embedder = get_embedding_provider()
    if _llm is None:
        _llm = get_llm_provider()
    return _embedder, _llm


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=4000)
    top_k: int | None = Field(default=None, ge=1, le=50)

    @field_validator("question")
    @classmethod
    def nonblank(cls, value):
        if not value.strip():
            raise ValueError("질문을 입력하세요.")
        return value.strip()
    products: list[str] | None = Field(
        default=None,
        description="모델 필터를 직접 지정. 생략하면 질문에서 자동 추출한다.",
    )
    use_rag: bool = Field(
        default=True,
        description="False 면 검색 없이 같은 모델로 답한다. 대비군 시연·평가용.",
    )


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "embedding_model": settings.embedding_model,
            "llm_model": settings.llm_model}


@app.get("/ready")
def ready():
    checks = {}
    try:
        with connect() as conn:
            conn.execute("SELECT embedding FROM chunks LIMIT 0")
        checks["database"] = True
    except Exception:
        checks["database"] = False
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
            checks["models"] = all(m in names or f"{m}:latest" in names for m in needed)
        else:
            checks["models"] = True
    except Exception:
        checks["models"] = False
    ok = all(checks.values())
    return JSONResponse(status_code=200 if ok else 503,
                        content={"status": "ready" if ok else "unavailable", "checks": checks})


@app.post("/ingest")
def ingest(file: UploadFile, force: bool = False):
    from app.ingest.documents import SUPPORTED
    from app.ingest.pipeline import ingest_document
    name = Path((file.filename or "").replace("\\", "/")).name
    if Path(name).suffix.lower() not in SUPPORTED:
        raise HTTPException(415, "PDF, DOCX, XLSX, PPTX만 지원합니다.")
    # 요청 파일을 격리하고 색인 후 제거한다. 원문은 DB의 청크로 확인한다.
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / name
        size = 0
        with path.open("wb") as target:
            while block := file.file.read(1024 * 1024):
                size += len(block)
                if size > 50 * 1024 * 1024:
                    raise HTTPException(413, "파일은 50MB 이하여야 합니다.")
                target.write(block)
        if not size:
            raise HTTPException(400, "빈 파일입니다.")
        try:
            result = ingest_document(path, force=force, source_name=name)
        except (ValueError, RuntimeError) as exc:
            raise HTTPException(422, "문서를 읽을 수 없습니다. 파일 형식을 확인하세요.") from exc
        if result.status == "failed":
            raise HTTPException(422, "색인에 실패했습니다. 문서 목록과 서버 로그를 확인하세요.")
        result.path = name
        return asdict(result)


@app.get("/documents")
def documents() -> dict:
    with connect() as conn, conn.cursor() as cur:
        cur.execute(
            """SELECT d.id, d.title, d.products, d.page_count, d.status, d.error,
                      count(c.id) AS chunks
               FROM documents d LEFT JOIN chunks c ON c.document_id = d.id
               GROUP BY d.id ORDER BY d.title"""
        )
        rows = cur.fetchall()
    return {
        "documents": [
            {"id": r[0], "title": r[1], "products": list(r[2] or []),
             "pages": r[3], "status": r[4], "error": r[5], "chunks": r[6]}
            for r in rows
        ]
    }


@app.post("/ask")
def ask(req: AskRequest) -> dict:
    embedder, llm = _providers()
    try:
        with connect() if req.use_rag else nullcontext(None) as conn:
            result = answer_question(
                conn, req.question, embedder, llm,
                top_k=req.top_k, products=req.products, use_rag=req.use_rag,
            )
    except Exception as exc:
        log.exception("질의응답 실패")
        raise HTTPException(status_code=503, detail="DB·모델 연결 및 서버 로그를 확인하세요.") from exc
    return asdict(result)
