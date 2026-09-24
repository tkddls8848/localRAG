# 사내 지식 RAG PoC

사내에 흩어져 있는 문서(규정집, 제안서, 장비 스펙, 보고서)를 모아 색인하고,
자연어 질문에 **출처와 함께** 답변하는 사내 특화 질의응답 시스템의 개념검증(PoC).

## 현재 상태

**로컬 RAG 구현 완료.** PDF/Office 색인, 하이브리드 검색, 출처를 포함한 답변 API와 웹 화면을 제공합니다.

| 단계 | 상태 |
|---|---|
| PDF 파싱 · 청킹 | 완료. 실제 Lenovo Press 문서로 검증 |
| 임베딩 · DB 적재 | Ollama + pgvector. 별도 스키마를 사용하는 실제 DB 통합 테스트 제공 |
| Office 파싱 | DOCX 문단·표, XLSX 시트·셀, PPTX 슬라이드·표 |
| 웹 화면 · 업로드 | `/`에서 파일 업로드, 문서 목록, 질문·출처 확인 |
| 검색 · 질의응답 API | 완료. 실제 PostgreSQL 로 검증 |
| 평가 스크립트 | 완료. 정답표는 직접 작성해야 함 |

| 문서 | 내용 |
|---|---|
| [docs/architecture.md](docs/architecture.md) | 전체 구조, 구성 요소, 데이터 모델, 평가 방법 |
| [docs/decisions.md](docs/decisions.md) | 기술 선택의 근거와 트레이드오프 (ADR) |
| [docs/verification.md](docs/verification.md) | 실제 로컬 색인·질의응답 검증 결과 |

## 한 줄 요약

```
PDF/Office 문서 → 파싱 → 청킹 → 임베딩 → PostgreSQL(pgvector)
                                              ↓
                    질문 → 하이브리드 검색 → LLM → 답변 + 출처
```

## 결정된 스택

| 영역 | 선택 | 한 줄 이유 |
|---|---|---|
| 언어/프레임워크 | Python 3.12 + FastAPI | 문서 파싱 생태계가 Python에 집중 |
| 저장소 | PostgreSQL 16 + pgvector | 메타데이터·본문·벡터를 한 DB에서 관리 |
| 모델 런타임 | Ollama | 임베딩·생성을 한 런타임으로 처리 |
| 임베딩 | `bge-m3` (1024차원) | 한국어 지원, 로컬 고정 |
| 생성 | 로컬 모델 우선, 교체 가능 | 문서가 외부로 나가지 않는 상태로 먼저 검증 |
| 실행 환경 | Docker Compose (로컬) | PoC 단계는 노트북에서 완결 |

## 범위

**이번 PoC에서 검증할 것**

1. 사내 PDF/Office 문서를 자동으로 색인할 수 있는가
2. 한국어 질문에 대해 관련 문단을 제대로 찾아오는가
3. 답변의 출처(문서명·페이지)를 정확히 제시하는가
4. 로컬 모델만으로 실무에 쓸 만한 품질이 나오는가

**이번 PoC에서 다루지 않는 것**

- 사용자 인증, 문서별 접근 권한 분리
- Confluence·그룹웨어 등 외부 시스템 연동
- 다중 사용자 동시 접속, 운영 수준의 가용성
- 사내 서버 배포를 위한 IaC (구조만 준비, 구현은 이후)

## 실행

### Windows PowerShell (설치된 Ollama 사용)

```powershell
python -m venv .venv
.\.venv\Scripts\python -m pip install -r requirements.txt
Copy-Item .env.example .env   # 이미 있으면 기존 설정을 유지
docker compose up -d db
ollama pull bge-m3
ollama pull qwen3:8b
.\.venv\Scripts\python -m app.ingest.cli -v "data/pdfs/*.pdf"
.\.venv\Scripts\python -m uvicorn app.api.main:app --host 127.0.0.1 --port 8000
```

브라우저에서 http://localhost:8000 을 엽니다. API 명세는 `/docs`에 있습니다.
기존 PostgreSQL이 5432를 사용하면 `.env`에서 `POSTGRES_PORT=55432`로 지정합니다.
기존 Ollama 모델을 사용할 경우 `LLM_MODEL`을 설치된 모델명으로 변경할 수 있습니다.
임베딩은 `bge-m3`와 스키마의 1024차원을 유지해야 합니다.

CLI는 와일드카드와 디렉터리 재귀 색인을 지원합니다. 웹 업로드는 파일당 50MB까지이며
동기 처리이므로 큰 PDF는 시간이 걸립니다. 업로드 원본은 처리 후 제거하고 추출한 청크를 DB에 보관합니다.
Office의 실제 페이지를 추정하지 않습니다. DOCX는 문단·표 위치, XLSX는 시트·행·셀,
PPTX는 슬라이드 번호를 출처로 표시합니다. XLSX 수식은 수식 문자열로 읽으며 계산하지 않습니다.
구형 `.doc/.xls/.ppt`, 암호화 문서, 이미지 OCR은 지원하지 않습니다.

### 자동 검증

```powershell
.\.venv\Scripts\python -m pip install -r requirements-dev.txt
.\.venv\Scripts\python -m pytest -q
$env:RUN_DB_TESTS='1'
.\.venv\Scripts\python -m pytest -q
```

DB 통합 테스트는 임시 스키마를 생성·제거하며 fake 모델로 업로드, 중복 색인,
재색인, 실제 벡터·전문검색, 모델 필터, API를 검증합니다. 모델의 답변 품질 평가는 별도로 필요합니다.
문서 색인 및 서버 실행 후 `python tools/check_local.py`로 실제 한국어 답변과
미등록 모델의 답변 거절도 확인할 수 있습니다.

### 파싱만 확인 (DB·모델 불필요)

```bash
python3 -m venv .venv && ./.venv/bin/pip install -r requirements.txt
./.venv/bin/python -m app.ingest.cli --dry-run data/pdfs/*.pdf
```

### 전체 색인

```bash
cp .env.example .env          # 비밀번호 수정
docker compose up -d
docker compose exec ollama ollama pull bge-m3
./.venv/bin/python -m app.ingest.cli -v data/pdfs/*.pdf
```

`--force` 로 재색인, `-v` 로 진행 상황 출력.

### 질의응답

```bash
docker compose exec ollama ollama pull qwen3:8b
./.venv/bin/uvicorn app.api.main:app --reload      # 또는 docker compose up app
```

```bash
curl -X POST localhost:8000/ask -H 'content-type: application/json' \
  -d '{"question":"SR650 V4 최대 메모리 용량은?"}'
```

| 엔드포인트 | 용도 |
|---|---|
| `GET /` | 문서 업로드·질문 웹 화면 |
| `POST /ingest` | multipart `file` 업로드. `?force=true`로 재색인 |
| `POST /ask` | 질문 → 답변 + 출처. `use_rag:false` 로 무검색 대비군 |
| `GET /documents` | 색인된 문서와 상태 |
| `GET /health` | 상태 확인 |
| `GET /ready` | DB 스키마·Ollama 모델 설치 확인. 준비되지 않았으면 503 |

답변의 `[번호]`와 반환하는 출처를 연결합니다. 출처가 없거나 범위를 벗어난 번호가 있으면
근거 부족 답변으로 처리합니다. 이 검증은 인용 형식 검증이며 문장별 사실 검증을 대신하지 않습니다.

### 모델 없이 경로만 확인

```bash
EMBEDDING_PROVIDER=fake LLM_PROVIDER=fake ./.venv/bin/python -m app.ingest.cli data/pdfs/*.pdf
```

`fake` 제공자는 결정적 해시 임베딩과 더미 생성기입니다. 모델 다운로드 없이
색인·검색·API 경로를 통째로 돌려볼 수 있습니다. 운영에는 쓰지 않습니다.

## 검증된 동작

기존 PDF 파서 기준 제품 가이드 6건(787페이지, 청크 5,415개) 파싱 결과입니다.
현재 구현은 짧은 본문도 보존하고 긴 산문을 분할하므로 청크 수는 아래 기록과 다를 수 있습니다.

| 문서 | 페이지 | 청크 (표 / 산문) |
|---|---|---|
| SR630 V4 | 156 | 1,086 (1,012 / 74) |
| SR650 V4 | 200 | 1,574 (1,493 / 81) |
| SR650a V4 + SR650i V4 | 132 | 999 (841 / 158) |
| SR680a V4 | 65 | 326 (275 / 51) |
| SR850 V4 | 112 | 656 (588 / 68) |
| SR860 V4 | 122 | 774 (707 / 67) |

처리한 문서 특성:

- TOC 에서 섹션 경로 확정 → **섹션 미할당 청크 0개**
- 표 캡션 행과 그룹 구분 행을 데이터에서 분리
- 2단 그룹 헤더 병합 (`Accelerators` + `QAT` → `Accelerators QAT`)
- **한 문서가 제품 2종을 다루는 경우 분리 태깅**

청크는 단독으로 읽히도록 제품명과 섹션을 포함합니다.

```
ThinkSystem SR680a V4 > Standard specifications
Memory maximum: Up to 4TB by using 32x 128GB RDIMMs
```

SR650a/SR650i 합본 문서는 본문이 한쪽만 가리킬 때 그 모델로 좁힙니다.
이것이 모델 혼동을 막는 1차 방어선이고, 2차는 검색 단계의 `products` 필터입니다.

```
ThinkSystem SR650i V4 > Inference Model
Description: ThinkSystem SR650i V4 Inference Configuration
```

## 검색 동작

밀집(pgvector 코사인) + 희소(전문검색) + 단어 유사도(pg_trgm) 경로를 RRF 로 결합합니다.
질문에서 모델명을 감지해 해당 모델 청크로 범위를 좁힙니다.

실제 PostgreSQL 에 SR650 V4(1,574청크) 와 SR650a/SR650i V4(999청크) 를
색인하고 확인한 결과:

```
질문: "SR650i V4 Inference Configuration"
  자동 필터: ['ThinkSystem SR650i V4']
  1위: ... p15  ['ThinkSystem SR650i V4']      ← SR650i 전용 청크
  → SR650 V4 청크 1,574개 중 단 한 건도 섞이지 않음

질문: "존재하지 않는 모델 SR999 V9 스펙"
  → "제공된 문서에서 찾지 못했습니다."  (출처 0건)
```

## 평가

바꾼 것이 나아졌는지 숫자로 판단하기 위한 도구입니다.

### 정답표 작성

초안을 자동으로 뽑아 시작할 수 있습니다.

```bash
python tools/draft_questions.py data/pdfs/*.pdf -n 5 > data/eval/draft.yaml
```

표의 "키: 값" 스펙 행에서 질문을 만들고 정답 문서·페이지를 채워 넣습니다.
**초안을 그대로 쓰면 안 됩니다.** 색인 파이프라인이 만든 문제를 같은
파이프라인이 푸는 구조라 점수가 실제보다 높게 나옵니다. 초안은 출발점이고,
실무에서 실제로 받는 질문을 직접 섞어야 의미가 있습니다.

```bash
cp data/eval/draft.yaml data/eval/questions.yaml   # 또는 example 에서 시작
cp data/eval/questions.example.yaml data/eval/questions.yaml
```

질문마다 정답이 실린 문서와 페이지를 직접 확인해 적습니다.
이 작업이 평가의 전부입니다 — 없으면 무엇을 바꿔도 나아졌는지 알 수 없습니다.

```yaml
- id: q001
  question: SR680a V4 최대 메모리 용량은?
  answers:
    - document: Lenovo ThinkSystem SR680a V4 Server
      pages: [10]
  expect: "4TB"          # 답변에 이 문자열이 있는지 자동 채점
  tags: [spec, memory]
```

문서에 답이 없는 함정 질문은 `unanswerable: true` 로 표시합니다.
지어내지 않고 "찾지 못했습니다"라고 답하는지 보는 용도입니다.

### 실행

```bash
python -m app.evaluation.run                        # 검색만. 임베딩 모델 필요, 생성 모델 불필요
python -m app.evaluation.run --answers              # 답변 생성까지 채점
python -m app.evaluation.run --answers --baseline   # 무검색 대비군과 비교
```

### 지표

| 지표 | 의미 |
|---|---|
| Recall@k | 정답 문단이 상위 k개에 들어온 비율. **생성 품질의 천장** |
| MRR | 정답을 몇 번째로 맞혔는지 |
| 답변 정확도 | `expect` 문자열이 답변에 있는가 |
| 정직도 | 답 없는 질문에 "찾지 못했다"고 답한 비율 |
| 태그별 Recall | 어떤 질문 유형에서 깨지는지 |

검색이 실패하면 생성은 무조건 실패합니다. **Recall 을 먼저 봅니다.**
답변이 나쁠 때 프롬프트부터 손대면 대개 시간을 낭비합니다.

무검색 대비군에는 검색 단계가 없으므로 Recall 을 표시하지 않습니다.
비교 대상은 답변 정확도와 정직도입니다.

결과는 `data/eval/results/` 에 타임스탬프 JSON 으로 저장되어 변경 전후를
비교할 수 있습니다.

## 다음 단계

1. 정답표 작성 (질문 30~50개)
2. 나머지 문서 색인 후 전체 측정
3. 측정 결과를 보고 개선 지점 결정
