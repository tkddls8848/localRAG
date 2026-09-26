# 사내 지식 RAG PoC

사내에 흩어져 있는 문서(규정집, 제안서, 장비 스펙, 보고서)를 색인하고, 한국어
자연어 질문에 **출처와 근거 검증을 붙여** 답하는 사내 특화 질의응답 시스템의
개념검증(PoC).

컨테이너 3개(app / PostgreSQL+pgvector / Ollama)로 끝나고, 문서가 외부로 나가지
않는다. 사내 서버로 옮길 때 버릴 것이 없도록 만들었다.

```
PDF/Office → 파싱 → 청킹 → 임베딩 → PostgreSQL(pgvector)
                                          ↓
  질문 → 질의 분석(용어집) → 4경로 하이브리드 검색 → 리랭킹 → LLM
                                          ↓
                     답변 + 출처(문서·페이지) + 수치 근거 검증
```

| 문서 | 내용 |
|---|---|
| [docs/architecture.md](docs/architecture.md) | 전체 구조, 구성 요소, 데이터 모델, 평가 방법 |
| [docs/decisions.md](docs/decisions.md) | 기술 선택의 근거와 트레이드오프 (ADR 14건) |
| [docs/operations.md](docs/operations.md) | **운영 절차서** — 기동·색인·권한·장애 대응·백업 |
| [docs/eval-report.md](docs/eval-report.md) | 실측 품질 보고서 (자동 생성) |
| [docs/verification.md](docs/verification.md) | 실제 로컬 색인·질의응답 검증 기록 |

---

## 현재 상태

| 영역 | 상태 |
|---|---|
| PDF·Office 색인 | 완료. 실제 Lenovo 제품 가이드 6건(787p)로 검증 |
| 반복 머리말·꼬리말 제거 | 완료. 787건 제거(문서당 전 페이지 꼬리말) |
| 하이브리드 검색 | 4경로(dense / sparse / exact / trigram) + 가중 RRF |
| 한국어 질의 확장 | 용어집 기반. **영문·한국어 문서가 섞인 코퍼스**를 함께 지원 |
| 리랭킹 | 규칙 기반 기본, LLM 리랭커 선택 |
| 답변 신뢰성 | 인용 번호 검증 + **수치 근거 대조**(발췌에 없는 값을 표시/보류) |
| 접근 권한 | API 키 → 역할 3단계 + 문서별 열람 그룹(검색 SQL 에서 판정) |
| 관측 | 요청 상관 ID, 단계별 지연, Prometheus 지표, 질의 로그·피드백 |
| 색인 운영 | 작업 큐 + 디렉터리 동기화(변경분만) + 문서 삭제 |
| 스키마 관리 | 번호 붙은 마이그레이션. 색인을 지우지 않고 스키마를 바꾼다 |
| 평가 | Recall/nDCG/MRR/출처 정확도/정직도 + 경로별 ablation + 실행 간 비교 |

---

## 실측 결과

문서: `data/pdfs/` 의 Lenovo ThinkSystem 제품 가이드 6건 (787페이지, 5,361청크).
**문서 원본은 저장소에 없다**(아래 참고). 재현하려면 같은 문서를 `data/pdfs/` 에 둔다.
정답표: `data/eval/questions.poc.yaml` (42문항 — 정답 있음 37 / 함정 5).
임베딩 `bge-m3`. 재현: `python -m app.evaluation.run --modes dense,sparse,exact,trigram,hybrid,hybrid+rerank`

### 검색 경로별 (top_k=10)

| 모드 | Recall@1 | Recall@3 | Recall@5 | Recall@10 | MRR | nDCG@10 |
|---|---|---|---|---|---|---|
| `dense` (벡터 단독) | 0.486 | 0.595 | 0.676 | 0.784 | 0.571 | 0.590 |
| `sparse` (전문검색 단독) | 0.243 | 0.514 | 0.622 | 0.811 | 0.419 | 0.515 |
| `exact` (식별자 AND 단독) | 0.054 | 0.081 | 0.108 | 0.108 | 0.074 | 0.083 |
| `trigram` (유사도 단독) | 0.243 | 0.270 | 0.297 | 0.297 | 0.264 | 0.271 |
| `hybrid` (4경로 RRF) | 0.432 | 0.730 | 0.892 | **0.973** | 0.613 | 0.693 |
| `hybrid+rerank` | **0.784** | **0.838** | **0.919** | **0.973** | **0.826** | **0.811** |

읽는 법.

- **하이브리드가 단독 경로보다 확실히 낫다.** Recall@10 이 0.78/0.81 → 0.97.
  이 표가 없으면 "하이브리드가 필요하다"는 주장에 근거가 없다(decisions.md D5).
- **리랭킹은 순서를 고친다.** Recall@10 은 그대로인데 Recall@1 이 0.43 → 0.78,
  MRR 이 0.61 → 0.83. 후보 안에 있어도 10위에 있으면 LLM 이 놓친다.
- `exact` 단독 점수가 낮은 것은 정상이다. 파트번호·피처코드가 있는 질문
  4~5개에서만 동작하고, 그 질문에서는 결정적이다(단독으로 쓸 경로가 아니다).
- `trigram` 은 기여가 가장 작다. 가중치를 0.6 으로 낮춰 둔 이유다.

### 튜닝 이력 (무엇이 실제로 효과가 있었나)

같은 정답표로 측정한 `hybrid+rerank` Recall@10 의 변화다.

| 조치 | Recall@10 | Recall@1 | 비고 |
|---|---|---|---|
| 초기(MMR 다양화를 후보 단계에 적용) | 0.541 | 0.270 | 다양성 항이 관련도를 압도 |
| 다양화를 리랭킹 뒤로 옮기고 근접 중복만 정리 | 0.757 | 0.459 | |
| 희소 검색에서 모델명 토큰 제외 + 길이 정규화 | 0.757 | 0.622 | `sparse` 단독 0.19→0.60 |
| 용어집에 실패 질문 어휘 보강 | 0.865 | 0.757 | 재색인 불필요 |
| 섹션별 상한(한 표가 상위 독점 방지) | 0.919 | 0.757 | |
| 정답 위치 보정(원문 확인 후 복수 페이지 인정) | **0.973** | **0.784** | 아래 주의 참고 |

> **주의 — 이 숫자는 낙관적이다.** 용어집 어휘를 이 42문항의 실패에서 보강했고,
> 정답 위치도 결과를 보고 보정했다(원문을 읽어 실제로 답이 있는 페이지만 추가).
> 즉 **본 적 없는 질문에서는 이보다 낮게 나온다.** 정직한 검증은 `query_log` 에
> 쌓인 실무자의 새 질문으로 다시 재는 것이다. 그 경로를 위해 질의 로그와
> 피드백을 넣었다(decisions.md D14).

### 색인 실적

| 문서 | 페이지 | 청크 | 토큰(근사) | 소요 | 제거한 상용구 |
|---|---:|---:|---:|---:|---:|
| SR630 V4 | 156 | 1,089 | 114,425 | 119s | 156 |
| SR650 V4 | 200 | 1,582 | 173,526 | 154s | 200 |
| SR650a V4 + SR650i V4 | 132 | 917 | 98,705 | 84s | 132 |
| SR680a V4 | 65 | 331 | 33,989 | 22s | 65 |
| SR850 V4 | 112 | 662 | 64,268 | 66s | 112 |
| SR860 V4 | 122 | 780 | 75,614 | 76s | 122 |
| **합계** | **787** | **5,361** | **560,527** | **~8.5분** | **787** |

- TOC 에서 섹션 경로를 확정 → **섹션 미할당 청크 0개**
- 2단 그룹 헤더 병합 (`Accelerators` + `QAT` → `Accelerators QAT`)
- 한 문서가 제품 2종을 다루면 본문이 가리키는 쪽으로 좁혀 태깅
  (SR650a/SR650i 합본 문서 — 모델 혼동을 막는 1차 방어선)

---

## 색인 대상 문서

**문서 원본은 이 저장소에 들어가지 않는다.** 색인 대상에는 단가·인사·계약처럼
민감한 내용이 섞일 수 있고, 한 번 커밋하면 히스토리에서 지우기가 번거롭다.
`.gitignore` 가 `data/pdfs/` 하위와 `data/` 아래 문서 확장자를 모두 무시한다
(`.gitkeep` 만 남겨 디렉터리를 유지한다). 컨테이너 이미지에도 들어가지 않는다
(`.dockerignore` 에 `data`).

```bash
# 색인할 문서를 여기에 둔다 (DOC_ROOT 로 다른 경로 지정 가능)
cp /path/to/사내문서/*.pdf data/pdfs/
```

운영에서는 이 디렉터리를 파일 서버 마운트로 두고 `POST /admin/sync` 로 변경분만
색인한다. 원본의 단일 출처는 파일 서버이고, 저장소에는 코드와 설정만 둔다.

**영문 문서와 한국어 문서를 한 색인에 섞어도 된다.** 질의 확장이 영문 용어와
한국어 어간을 함께 검색어로 쓴다(decisions.md D15). 사내 문서는 언어가 문서
단위가 아니라 문단 단위로 섞이는 경우가 많아서 언어별 색인 분리를 택하지 않았다.

## 실행

### Docker Compose

```bash
cp .env.example .env          # POSTGRES_PASSWORD, API_KEYS 를 채운다
docker compose up -d                    # db → migrate → ollama → app
docker compose run --rm model-pull      # 임베딩·생성 모델 내려받기
curl -s localhost:8000/ready | jq
```

`app` 은 마이그레이션이 **성공으로 끝난 뒤에만** 뜬다.

### 로컬 파이썬 (Windows PowerShell)

```powershell
python -m venv .venv
.\.venv\Scripts\python -m pip install -r requirements-dev.txt
Copy-Item .env.example .env
docker compose up -d db
.\.venv\Scripts\python -m app.db.migrate
ollama pull bge-m3 ; ollama pull qwen3:8b
.\.venv\Scripts\python -m app.ingest.cli -v --force data/pdfs
.\.venv\Scripts\python -m uvicorn app.api.main:app --host 127.0.0.1 --port 8000
```

브라우저에서 <http://localhost:8000> — 운영 콘솔(질문 / 진단 / 문서 / 운영).
API 명세는 `/docs`.

기존 PostgreSQL 이 5432 를 쓰면 `.env` 에서 `POSTGRES_PORT=55432` 로 옮긴다.

### 모델 없이 경로만 확인

```bash
EMBEDDING_PROVIDER=fake LLM_PROVIDER=fake python -m app.ingest.cli data/pdfs
```

`fake` 제공자는 결정적 해시 임베딩과 더미 생성기다. 모델 없이 색인·검색·API
경로를 통째로 돌려볼 수 있다. **운영에는 쓰지 않는다**(기동 시 경고가 뜬다).

### 파싱만 확인 (DB·모델 불필요)

```bash
python -m app.ingest.cli --dry-run data/pdfs/*.pdf
```

---

## API

| 엔드포인트 | 최소 역할 | 용도 |
|---|---|---|
| `GET /` | – | 운영 콘솔 |
| `POST /ask` | reader | 질문 → 답변 + 출처 + 근거 검증. `use_rag:false` 로 무검색 대비군 |
| `POST /ask/stream` | reader | 같은 동작의 SSE 스트리밍 |
| `POST /search` | reader | **답변 없이 검색 후보만.** 왜 못 찾았나를 조사 |
| `POST /feedback` | reader | 답변 평가(±1). 실패 질문 수집 |
| `POST /ingest` | editor | 파일 업로드(기본 비동기 작업, `?wait=true` 로 동기) |
| `GET /jobs`, `/jobs/{id}` | reader | 색인 작업 진행 상황 |
| `GET /documents` | reader | 색인 상태(열람 권한 범위 안에서) |
| `DELETE /documents/{id}` | editor | 잘못 색인된 문서 제거 |
| `POST /admin/sync` | admin | 서버 디렉터리 동기화(`?dry_run=true`) |
| `GET /admin/queries` | admin | 질의 로그(`?only_failed=true`) |
| `GET /admin/stats` | admin | 응답률·지연 백분위 |
| `GET /health`, `/ready` | – | 상태 확인. `/ready` 는 스키마·모델·차단기까지 |
| `GET /metrics` | – | Prometheus. **인증 없음 → 외부 노출 금지** |

```bash
curl -X POST localhost:8000/ask -H 'content-type: application/json' \
  -H "authorization: Bearer $API_KEY" \
  -d '{"question":"SR650 V4 최대 메모리 용량은?"}'
```

답변의 `[번호]`와 반환한 출처를 연결한다. 출처가 없거나 범위를 벗어난 번호가
있으면 근거 부족으로 처리한다. 그다음 답변의 **수치·식별자가 인용 발췌에 실제로
있는지** 대조해 `grounded` 와 `unsupported` 로 알린다. 이것은 표기 검증이며
문장별 사실 검증을 대신하지 않는다.

---

## 접근 권한

```ini
# .env — "키:역할:그룹1|그룹2" 를 쉼표로 나열. 비우면 인증이 꺼진다(로컬 전용)
API_KEYS=9f2c8b1a4e7d6c5f:admin,3a7b2c9d8e1f4a6b:reader:sales|eng
```

| 역할 | 가능한 일 |
|---|---|
| `reader` | 질문, 검색 진단, 자기 열람 범위의 문서 목록 |
| `editor` | + 문서 색인·삭제 |
| `admin` | + 디렉터리 동기화, 질의 로그·통계, 전체 문서 열람 |

문서의 열람 그룹은 색인할 때 정한다(비우면 전사 공개).

```bash
python -m app.ingest.cli --acl-groups finance data/pricing/
```

판정은 검색 SQL 안에서 일어나므로 경로를 우회할 수 없다. 메타데이터와 벡터를
한 DB 에 둔 선택의 직접적인 이득이다(decisions.md D2).

---

## 평가

```bash
# 정답표 (원문 확인 필수 — 자동 초안을 그대로 쓰면 점수가 부풀려진다)
cp data/eval/questions.poc.yaml data/eval/questions.yaml
python tools/draft_questions.py data/pdfs/*.pdf -n 5 > data/eval/draft.yaml

python -m app.evaluation.run                                   # 검색만(빠름)
python -m app.evaluation.run --modes dense,sparse,hybrid,hybrid+rerank
python -m app.evaluation.run --answers --baseline --label v1    # 답변 + 대비군
python -m app.evaluation.run --compare data/eval/results/<이전>.json
python -m app.evaluation.run --answers --report docs/eval-report.md
```

| 지표 | 의미 |
|---|---|
| Recall@k | 정답 문단이 상위 k개에 들어온 비율. **생성 품질의 천장** |
| nDCG@k / MRR | 몇 번째로 맞혔는가 |
| Precision@k | 가져온 것 중 쓸모 있는 비율 |
| 답변 정확도 | `expect` 문자열이 답변에 있는가 |
| 출처 정확도 | 제시한 출처가 실제 정답 위치인가 |
| 근거 일치율 | 답변의 수치가 인용 발췌 안에 있는가 |
| 정직도 | 답 없는 질문에 "찾지 못했다"고 답한 비율 |
| 지연 | 품질과 분리해 단계별 p50/p95 |

검색이 실패하면 생성은 무조건 실패한다. **Recall 을 먼저 본다.**
2%p 미만의 차이는 개선이라고 부르지 않는다(질문 하나가 그만큼을 움직인다).

---

## 자동 검증

```powershell
.\.venv\Scripts\python -m pytest -q                 # 32건, DB·모델 불필요
$env:RUN_DB_TESTS='1'
.\.venv\Scripts\python -m pytest -q                 # + 실제 pgvector 통합 2건
```

DB 통합 테스트는 임시 스키마를 만들고 마이그레이션을 적용한 뒤(재실행 안전성까지
확인) fake 제공자로 색인·중복·재색인 실패 시 이전 색인 보존·한국어 질의의 영문
본문 도달·식별자 정확 일치·**열람 권한 격리**·질의 로그·피드백·스트리밍·비동기
작업·디렉터리 동기화·문서 삭제를 검증하고 스키마를 지운다.

서버 기동 후 실제 요청 점검:

```bash
python tools/check_local.py --key $API_KEY
```

준비 상태 · 색인 실패 문서 · 검색 후보 · 실제 답변과 출처 · 근거 검증 ·
**미등록 모델에 지어내지 않는지**를 한 번에 본다.

---

## 이 PoC 가 아직 하지 않는 것

도입 검토에서 반드시 나오는 질문이므로 미리 적는다. 자세한 내용은
[operations.md 11](docs/operations.md).

- **OCR** — 스캔 PDF 는 색인하지 않고 `검토 필요` 로 표시만 한다
- **SSO 연동** — 환경변수 API 키까지. `app/security.py` 가 교체 지점
- **여러 인스턴스** — 작업 큐와 요청 제한이 프로세스 안에 있다
- **외부 시스템 연동** — Confluence·그룹웨어. 현재는 파일 디렉터리 동기화만
- **IaC** — 배포 대상 확정 후. 지금은 설정 외부화와 상태 분리만 지킨다
- **한국어 문서 코퍼스** — 용어집 방식은 "문서가 영문"을 전제한다
