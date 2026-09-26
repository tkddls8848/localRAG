# 운영 절차서 (PoC)

PoC 기간에 이 시스템을 실제로 돌리는 사람을 위한 문서다. 설계 근거는
[architecture.md](architecture.md), 선택의 이유는 [decisions.md](decisions.md)
에 있다. 여기에는 **무엇을 어떤 순서로 실행하고, 깨졌을 때 어디를 보는지**만 적는다.

---

## 1. 기동

### 1.1 Docker Compose (권장)

```bash
cp .env.example .env          # POSTGRES_PASSWORD, API_KEYS 를 채운다
docker compose up -d          # db → migrate → ollama → app
docker compose run --rm model-pull    # 임베딩·생성 모델 내려받기
curl -s localhost:8000/ready | jq
```

`migrate` 와 `model-pull` 은 한 번 돌고 끝나는 준비 작업이다. `app` 은
`migrate` 가 성공으로 끝난 뒤에만 뜬다. 스키마가 덜 적용된 상태로 서버가 뜨는
경우를 없애기 위한 구성이다.

### 1.2 로컬 파이썬 (개발·검증)

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

기존 PostgreSQL 이 5432 를 쓰면 `.env` 에서 `POSTGRES_PORT=55432` 로 옮긴다.

---

## 2. 준비 상태 점검

```bash
curl -s localhost:8000/ready | jq
```

| 항목 | 실패 시 조치 |
|---|---|
| `database: false` | DB 연결 또는 미적용 마이그레이션. `python -m app.db.migrate --status` |
| `migrations: 미적용 [...]` | `python -m app.db.migrate` |
| `models: false` | `models_missing` 목록을 `ollama pull` |
| `model_breaker: open` | 모델 런타임이 연속 실패 중. 쿨다운(기본 30초) 후 자동 복귀 |

기동 로그의 `설정 경고:` 줄을 반드시 읽는다. 다음 세 가지는 사내 공유 전에
반드시 해소해야 한다.

- `API_KEYS 가 비어 있어 인증이 꺼져 있다`
- `POSTGRES_PASSWORD 가 기본값이다`
- `fake 제공자가 켜져 있다`

---

## 3. 접근 권한 발급

`.env` 의 `API_KEYS` 에 `키:역할:그룹1|그룹2` 를 쉼표로 나열한다.

```bash
# 키 생성
python -c "import secrets; print(secrets.token_hex(16))"
```

```ini
API_KEYS=9f2c…:admin,3a7b…:editor:eng,c1d4…:reader:sales|eng
```

| 역할 | 가능한 일 |
|---|---|
| `reader` | 질문, 검색 진단, 자기 열람 범위의 문서 목록 |
| `editor` | + 문서 색인·삭제 |
| `admin` | + 디렉터리 동기화, 질의 로그·통계 열람, 전체 문서 열람 |

문서의 열람 그룹은 색인할 때 정한다. 비우면 전사 공개다.

```bash
python -m app.ingest.cli --acl-groups finance data/pricing/
curl -X POST "localhost:8000/ingest?acl_groups=finance" -H "authorization: Bearer $KEY" -F file=@price.xlsx
```

권한 판정은 검색 SQL 안에서 일어난다(`acl_groups && :groups`). 애플리케이션이
걸러내는 방식이 아니므로 경로를 우회할 수 없다.

---

## 4. 문서 색인

> **문서 원본은 저장소에 커밋하지 않는다.** `.gitignore` 가 `data/pdfs/` 하위와
> `data/` 아래 문서 확장자를 무시한다. 색인 대상에 단가·인사·계약이 섞일 수
> 있고, 한 번 커밋하면 히스토리에서 지우려면 재작성(`git filter-repo`)과
> 강제 푸시가 필요해진다. 원본의 단일 출처는 파일 서버로 둔다.
>
> 새로 클론한 곳에는 문서가 없다. `data/pdfs/` 에 문서를 두거나 `DOC_ROOT` 로
> 파일 서버 마운트 경로를 지정한 뒤 아래를 실행한다.

### 4.1 처음 전량 색인

```bash
python -m app.ingest.cli -v --force data/pdfs
```

### 4.2 변경분만 (운영 중)

```bash
python -m app.ingest.cli --sync data/pdfs           # CLI
curl -X POST "localhost:8000/admin/sync?dry_run=true" -H "authorization: Bearer $ADMIN"
curl -X POST "localhost:8000/admin/sync" -H "authorization: Bearer $ADMIN"
```

판정 순서는 경로·크기·수정시각 → 해시 → 색인이다. 바뀌지 않은 파일은 읽지도
않는다. 내용이 바뀐 파일은 **이전 판을 지우고** 새로 넣는다(그러지 않으면 옛
스펙이 영원히 검색된다).

파일이 사라진 문서는 기본적으로 지우지 않고 보고만 한다. 파일 서버가 잠깐
안 보이는 것과 문서가 삭제된 것을 구별할 수 없기 때문이다. 확인한 뒤
`--delete-missing` 을 쓴다.

### 4.3 파싱만 확인 (DB·모델 불필요)

```bash
python -m app.ingest.cli --dry-run data/pdfs/*.pdf
```

`주의 :` 줄과 `스캔 PDF 로 의심됨` 판정을 본다. 스캔 PDF 는 OCR 없이 내용을
색인할 수 없고, 이 시스템은 OCR 을 하지 않는다. 문서 목록에서 `검토 필요`
로 표시된다.

### 4.4 색인 작업 상태

```bash
curl -s localhost:8000/jobs -H "authorization: Bearer $KEY" | jq
curl -s localhost:8000/jobs/12 -H "authorization: Bearer $KEY" | jq
```

---

## 5. 품질 측정 루프

PoC 의 결론은 "돌아간다"가 아니라 "쓸 만한가"다. 다음 순서로 돌린다.

```bash
# 1) 정답표 초안 (그대로 쓰지 말 것 — 편향된다)
python tools/draft_questions.py data/pdfs/*.pdf -n 5 > data/eval/draft.yaml
cp data/eval/draft.yaml data/eval/questions.yaml

# 2) 실무자가 실제로 던진 실패 질문을 정답표에 추가한다
curl -s "localhost:8000/admin/queries?only_failed=true" -H "authorization: Bearer $ADMIN" | jq

# 3) 검색만 측정 (빠름. 생성 모델 불필요)
python -m app.evaluation.run

# 4) 경로별 기여 확인 — 하이브리드가 필요한지 숫자로 본다
python -m app.evaluation.run --modes dense,sparse,exact,hybrid,hybrid+rerank

# 5) 답변까지, 무검색 대비군과 함께
python -m app.evaluation.run --answers --baseline --label v1

# 6) 바꾼 뒤 이전 결과와 비교
python -m app.evaluation.run --compare data/eval/results/20260926-101500-v1.json

# 7) 보고서
python -m app.evaluation.run --answers --baseline --report docs/eval-report.md
```

**Recall 을 먼저 본다.** 검색이 실패하면 생성은 무조건 실패한다. 답변이 나쁠
때 프롬프트부터 손대면 대개 시간을 낭비한다.

`app/evaluation/metrics.py` 의 `NOISE = 0.02` 보다 작은 차이는 개선이라고
부르지 않는다. 질문 30~50개 규모에서 질문 하나가 2~3%p 를 움직인다.

---

## 6. 무엇이 안 찾아지는지 조사하기

```bash
curl -X POST localhost:8000/search -H 'content-type: application/json' \
  -H "authorization: Bearer $KEY" \
  -d '{"question":"SR650a V4 최대 메모리 용량은?"}' | jq
```

응답의 `terms` / `identifiers` / `intent` 를 먼저 본다.

| 증상 | 원인과 조치 |
|---|---|
| `terms` 에 영문 용어가 없다 | 용어집에 그 한국어 단어가 없다. `app/retrieval/glossary.yaml` 에 한 줄 추가 |
| `hits` 가 0건 | 제품 필터가 과하게 좁혀졌는지 본다. `products: []` 를 넣어 다시 시도 |
| 정답이 있는데 순위가 낮다 | `paths` 를 본다. 특정 경로만 올렸다면 가중치(`WEIGHT_*`)를 평가와 함께 조정 |
| 다른 모델 청크가 섞인다 | 색인 단계 태깅 문제. 해당 문서를 `--force` 로 재색인 |

용어집 수정은 **재색인이 필요 없다.** 질의 단계에서만 쓰기 때문이다.

---

## 7. 관측

| 대상 | 방법 |
|---|---|
| 로그 | `LOG_FORMAT=json` 이면 한 줄 JSON. 모든 줄에 `request_id` 가 있다 |
| 지표 | `GET /metrics` (Prometheus 텍스트). 인증 없이 열리므로 외부 노출 금지 |
| 질의 기록 | `GET /admin/queries`, `GET /admin/stats?days=7` |
| 응답 시간 | `/ask` 응답의 `latency_ms` 에 embed/search/rerank/generate 가 분리돼 있다 |

느린 것을 품질 문제로 오진하지 않기 위해 지연은 품질 지표와 분리해서 본다.
첫 질문이 느린 것은 대개 모델 로딩이다. `OLLAMA_KEEP_ALIVE` 를 늘린다.

---

## 8. 장애 대응

### 모델 런타임이 죽었다

증상: `/ready` 의 `models: false` 또는 `model_breaker: open`, 답변 503.

- 색인 중이었다면 **이미 색인된 문서는 그대로 검색된다.** 재색인 실패 시
  이전 청크를 보존하도록 만들어져 있다(`documents.error` 에 원인이 남는다).
- 일시적 실패는 재시도(기본 3회, 지수 백오프)로 흡수된다. 연속 실패가
  임계를 넘으면 차단기가 열려 쿨다운 동안 즉시 실패한다. 죽은 런타임에
  요청을 쌓지 않기 위한 동작이다.
- `docker compose restart ollama` 후 `/ready` 재확인.

### DB 가 죽었다

앱은 살아 있고 `/ready` 가 503 을 준다. 연결 풀은 지연 생성이므로 DB 가 돌아오면
재기동 없이 복구된다.

### 색인이 실패한 문서가 있다

```bash
curl -s localhost:8000/documents -H "authorization: Bearer $KEY" \
  | jq '.documents[] | select(.status != "indexed")'
```

`error` 필드에 원인이 있다. 색인된 줄 알았는데 빠져 있는 문서가 가장 위험하므로
이 목록은 비어 있어야 한다.

### 답변에 근거 없는 수치가 섞인다

`/admin/queries` 에서 `grounded: false` 인 항목을 본다. `unsupported` 에 어떤
값이 발췌에 없었는지 남는다. 반복되면 `GROUNDING_MODE=strict` 로 올려 답변을
보류시킨다.

---

## 9. 백업과 복구

상태는 전부 `pgdata` 볼륨에 있다. 업로드 원본은 색인 후 지우므로 백업 대상이
아니다(원본은 파일 서버에 있다).

```bash
# 백업
docker compose exec -T db pg_dump -U specrag -Fc specrag > backup-$(date +%F).dump

# 복구
docker compose exec -T db pg_restore -U specrag -d specrag --clean --if-exists < backup-2026-09-26.dump
python -m app.db.migrate      # 덤프가 오래됐으면 스키마를 맞춘다
```

임베딩은 덤프에 포함되므로 복구 후 재색인이 필요 없다. 단, **임베딩 모델을
바꾸면 덤프는 쓸 수 없다**(아래).

---

## 10. 재색인이 필요한 변경 / 필요 없는 변경

이 표를 착각하면 며칠을 잃는다.

| 변경 | 재색인 |
|---|---|
| `EMBEDDING_MODEL`, `EMBEDDING_DIM` | **필요** (벡터 공간이 달라진다) |
| `CHUNK_TARGET_CHARS`, `CHUNK_OVERLAP_CHARS` | **필요** |
| 파서·청커 코드 수정 | **필요** |
| 문서 내용 변경 | **필요** (해당 문서만. `--sync` 가 골라낸다) |
| `LLM_MODEL`, 프롬프트 | 불필요 |
| 용어집(`glossary.yaml`) | 불필요 |
| `RERANK_PROVIDER`, `WEIGHT_*`, `SEARCH_*` | 불필요 |
| `GROUNDING_MODE`, `API_KEYS` | 불필요 |
| 문서 열람 그룹 변경 | 불필요 (`documents.acl_groups` UPDATE 또는 재색인) |

임베딩과 생성의 결정을 분리해 둔 이유가 이 표다(decisions.md D3).

---

## 11. 이 PoC 가 아직 하지 않는 것

도입 검토에서 반드시 나오는 질문이므로 미리 적어 둔다.

- **OCR** — 스캔 PDF 는 색인하지 않고 `검토 필요` 로 표시만 한다
- **SSO 연동** — 환경변수 API 키까지만. `app/security.py` 교체 지점
- **여러 인스턴스** — 작업 큐와 요청 제한이 프로세스 안에 있다. 인스턴스를
  늘리려면 워커 분리와 공유 저장소가 필요하다(decisions.md D12)
- **외부 시스템 연동** — Confluence·그룹웨어. 현재는 파일 디렉터리 동기화만
- **IaC** — 배포 대상 확정 후(decisions.md D6). 지금은 모든 설정이
  환경변수로 외부화되어 있고 상태가 볼륨에 있다는 조건만 갖춰 둔다
- **한국어 문서 코퍼스** — 용어집 방식은 "문서가 영문"을 전제로 한다(D9)
