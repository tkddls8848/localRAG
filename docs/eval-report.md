# 로컬 RAG 품질 측정 보고서

- 측정 시각: 2026-09-26T10:26:44
- 정답표: `data\eval\questions.yaml`
- 임베딩 모델: `bge-m3`
- 생성 모델: `qwen3.5:4b`
- top_k: 10
- 질문 42개 (정답 있음 37 / 함정 5)

## 1. 검색 경로별 비교

하이브리드 검색이 필요한지는 주장이 아니라 이 표로 판단한다. `dense` 는 벡터 단독, `sparse` 는 전문검색 단독, `exact` 는 식별자 정확 일치, `hybrid` 는 RRF 결합, `+rerank` 는 재정렬까지다.

| 모드 | Recall@1 | Recall@3 | Recall@5 | Recall@10 | nDCG@10 | MRR |
|---|---|---|---|---|---|---|
| `dense` | 0.486 | 0.595 | 0.676 | 0.784 | 0.590 | 0.571 |
| `sparse` | 0.270 | 0.486 | 0.595 | 0.811 | 0.519 | 0.428 |
| `hybrid` | 0.486 | 0.757 | 0.892 | 0.973 | 0.714 | 0.647 |
| `hybrid+rerank` | 0.784 | 0.838 | 0.919 | 0.973 | 0.811 | 0.826 |

## 2. 답변 품질

| 모드 | 답변 정확도 | 출처 정확도 | 근거 일치율 | 정직도 |
|---|---|---|---|---|
| `hybrid+rerank` | 0.618 | 0.590 | 0.973 | 1.000 |
| `무검색 대비군` | 0.000 | – | – | 0.000 |

대비군은 같은 모델에 검색만 끄고 같은 질문을 물은 결과다. RAG 의 효과는 이 차이로만 말할 수 있다.

## 3. 응답 시간

품질과 분리해서 기록한다. 느린 것을 나쁜 것으로 오진하지 않기 위해서다.

| 모드 | 단계 | p50(ms) | p95(ms) |
|---|---|---|---|
| `dense` | embed | 20 | 80 |
| `dense` | search | 10 | 22 |
| `dense` | rerank | 1 | 1 |
| `sparse` | embed | 19 | 22 |
| `sparse` | search | 5 | 17 |
| `sparse` | rerank | 1 | 2 |
| `hybrid` | embed | 21 | 29 |
| `hybrid` | search | 59 | 135 |
| `hybrid` | rerank | 1 | 1 |
| `hybrid+rerank` | embed | 18 | 22 |
| `hybrid+rerank` | search | 52 | 159 |
| `hybrid+rerank` | rerank | 1 | 2 |
| `hybrid+rerank` | generate | 2,352 | 6,635 |
| `hybrid+rerank` | total | 2,329 | 6,780 |

## 4. 질문 유형별 Recall@10

어떤 유형에서 깨지는지 보면 다음에 무엇을 고칠지 정할 수 있다.

| 모드 | 태그 | Recall |
|---|---|---|
| `dense` | comparison | 0.000 |
| `dense` | cooling | 0.500 |
| `dense` | gpu | 1.000 |
| `dense` | identifier | 0.500 |
| `dense` | long-tail | 0.800 |
| `dense` | memory | 0.857 |
| `dense` | model-disambiguation | 0.700 |
| `dense` | physical | 1.000 |
| `dense` | power | 1.000 |
| `dense` | spec | 0.917 |
| `dense` | storage | 0.500 |
| `sparse` | comparison | 0.667 |
| `sparse` | cooling | 1.000 |
| `sparse` | gpu | 0.000 |
| `sparse` | identifier | 0.833 |
| `sparse` | long-tail | 0.600 |
| `sparse` | memory | 0.857 |
| `sparse` | model-disambiguation | 0.900 |
| `sparse` | physical | 1.000 |
| `sparse` | power | 0.667 |
| `sparse` | spec | 0.833 |
| `sparse` | storage | 1.000 |
| `hybrid` | comparison | 1.000 |
| `hybrid` | cooling | 1.000 |
| `hybrid` | gpu | 1.000 |
| `hybrid` | identifier | 0.833 |
| `hybrid` | long-tail | 1.000 |
| `hybrid` | memory | 1.000 |
| `hybrid` | model-disambiguation | 1.000 |
| `hybrid` | physical | 1.000 |
| `hybrid` | power | 1.000 |
| `hybrid` | spec | 1.000 |
| `hybrid` | storage | 1.000 |
| `hybrid+rerank` | comparison | 1.000 |
| `hybrid+rerank` | cooling | 1.000 |
| `hybrid+rerank` | gpu | 0.750 |
| `hybrid+rerank` | identifier | 1.000 |
| `hybrid+rerank` | long-tail | 1.000 |
| `hybrid+rerank` | memory | 1.000 |
| `hybrid+rerank` | model-disambiguation | 0.900 |
| `hybrid+rerank` | physical | 1.000 |
| `hybrid+rerank` | power | 1.000 |
| `hybrid+rerank` | spec | 0.958 |
| `hybrid+rerank` | storage | 1.000 |

## 5. 검색 실패 질문

정답 문단을 상위 후보에 올리지 못한 질문이다. **여기부터 고친다.** 검색이 실패하면 생성은 무조건 실패한다.

- `dense`: q001, q030, q051, q054, q070, q080, q081, q082
- `sparse`: q040, q041, q042, q043, q052, q062, q081
- `hybrid`: q053
- `hybrid+rerank`: q041

## 6. 해석 시 주의

- **표본이 작다.** 정답 있는 질문 37개이므로 질문 하나가 0.027 만큼 지표를 움직인다. 2%p 미만의 차이를 개선이라고 부르면 안 된다.
- **정답표를 보고 튜닝하면 점수가 부풀려진다.** 용어집 어휘나 가중치를 이 정답표의 실패 사례에서 고쳤다면, 본 적 없는 질문에서는 이보다 낮게 나온다. 정직한 검증은 `query_log` 에 쌓인 새 질문으로 다시 재는 것이다.
- **출처 정확도는 엄격한 정의다.** 답변이 인용한 발췌 **각각**이 정답 위치를 덮는 비율이므로, 여러 발췌를 함께 인용하면 낮게 나온다. 답이 맞았는지와 별개로 읽어야 한다.
- **응답 시간은 품질과 분리해서 본다.** 첫 질문이 느린 것은 대개 모델 로딩이다(`OLLAMA_KEEP_ALIVE`).

## 7. 재현 정보

```json
{
  "chunk_target_chars": 1200,
  "search_candidates": 50,
  "rerank_candidates": 30,
  "weights": {
    "dense": 1.0,
    "sparse": 1.0,
    "trigram": 0.6
  },
  "query_expansion": true,
  "grounding_mode": "warn"
}
```
