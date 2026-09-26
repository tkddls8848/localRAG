"""검색 전처리·리랭킹·근거 검증의 단위 테스트. DB·모델이 필요 없다."""
from __future__ import annotations

import pytest

from app.ingest.parser import ProseBlock, boilerplate_keys
from app.retrieval import grounding
from app.retrieval.query import analyze, extract_identifiers, strip_particles
from app.retrieval.rerank import LLMReranker, NoopReranker, RulesReranker
from app.retrieval.search import Hit, dedupe, diversify, fuse
from app.security import AuthError, Principal, RateLimiter, authenticate, parse_keys


def hit(chunk_id, content, **kw):
    base = dict(
        section_path=kw.pop("section_path", "Memory"),
        page_from=kw.pop("page_from", 1),
        page_to=kw.pop("page_to", 1),
        products=kw.pop("products", []),
        doc_title=kw.pop("doc_title", "Guide"),
        source_path=kw.pop("source_path", "guide.pdf"),
    )
    return Hit(chunk_id=chunk_id, content=content, **base, **kw)


# --- 질의 분석 -------------------------------------------------------------

def test_korean_question_is_translated_to_document_terms():
    """이 문서군은 영문이고 질문은 한국어다. 용어집이 없으면 희소 검색이 죽는다."""
    aq = analyze("SR650a V4 최대 메모리 용량은?")
    assert aq.products == ["ThinkSystem SR650a V4"]
    assert "memory" in aq.terms and "maximum" in aq.terms
    assert "SR650a" in aq.identifiers
    # OR 로 묶어야 질문이 길어져도 결과가 0 이 되지 않는다.
    assert " OR " in aq.websearch_or
    assert aq.intent == "spec"


def test_comparison_and_procedure_intents():
    assert analyze("SR630 V4 와 SR650 V4 의 차이는?").intent == "comparison"
    assert analyze("액체 냉각 설치 방법").intent == "procedure"
    assert analyze("SR650 V4 소개").intent == "general"


def test_identifier_extraction_keeps_exact_tokens():
    found = extract_identifiers("DDR5-5600 4TB 4X97B17362 M.2 를 지원하나")
    assert "DDR5-5600" in found
    assert "4TB" in found
    assert "4X97B17362" in found
    # 식별자 조각만 잘린 토큰이 검색어로 새지 않아야 한다.
    assert "X97B17362" not in analyze("4X97B17362 는?").terms


def test_particles_are_stripped():
    assert strip_particles("메모리의") == "메모리"
    assert strip_particles("용량은") == "용량"
    assert strip_particles("메모리") == "메모리"


def test_korean_stems_survive_expansion_for_korean_documents():
    """코퍼스에 한국어 문서가 섞이면 영문 확장만으로는 희소 검색이 0건이다."""
    aq = analyze("가나다정보통신의 주요 사업 분야는?")
    assert "사업" in aq.terms and "분야" in aq.terms
    # 영문 확장이 일어나는 질문에서도 어간은 남아야 한다.
    mixed = analyze("회사 인증 현황은?")
    assert "인증" in mixed.terms and "certifications" in mixed.terms


def test_korean_compound_decomposes_to_glossary_entry():
    """한국어는 명사를 붙여 쓴다. "총매출액" 이 "매출액" 을 못 찾으면 답을 놓친다."""
    from app.retrieval.query import load_glossary

    glossary = load_glossary()
    assert glossary.decompose("총매출액") == "매출액"
    assert glossary.decompose("연매출") == "매출"
    assert glossary.decompose("매출액") is None      # 자기 자신은 분해하지 않는다
    # 실제로 겪은 실패: 이 질문에서 정답 행이 12위로 밀렸다.
    terms = analyze("가나다정보통신 2025년 총매출액").terms
    assert "매출액" in terms and "합계" in terms


def test_copula_endings_are_stripped():
    """조사 제거만으로는 "서버인가요" 가 남는다. 서술격 어미도 떼어낸다."""
    assert strip_particles("서버인가요") == "서버"
    assert strip_particles("얼마인가요") == "얼마"
    assert "서버인가요" not in analyze("SR630 V4 는 몇 U 랙 서버인가요?").terms


def test_corpus_wide_words_never_become_search_terms():
    """모든 청크에 있는 어휘는 변별력이 없다. 용어집 확장도 우회하지 못해야 한다."""
    terms = [t.lower() for t in analyze("SR630 V4 는 몇 U 랙 서버인가요?").terms]
    assert "server" not in terms and "lenovo" not in terms
    # 한국어 어간은 남는다(한국어 문서에서 쓰인다).
    assert "서버" in analyze("SR630 V4 는 몇 U 랙 서버인가요?").terms


def test_question_scaffolding_words_are_dropped():
    """의문·요청 표현은 검색어로 넣으면 아무 문서나 조금씩 맞는다."""
    terms = analyze("SR630 V4 의 머신 타입 번호는 무엇인가요?").terms
    assert "번호" in terms
    assert "무엇인가요" not in terms and "무엇" not in terms
    assert "알려주세요" not in analyze("무게를 알려주세요.").terms


def test_trigram_text_picks_script_by_question_language():
    """pg_trgm 은 문자열 하나를 본문과 비교한다. 스크립트가 섞이면 유사도가 무너진다.

    어느 쪽을 쓸지는 확장 결과가 아니라 **질문의 언어**로 정해야 한다. 확장
    결과로 정하면 용어집이 한국어 질문에 영문 어휘를 넣는 순간 한국어 문서를
    향한 질의가 영문으로 비교되어 이 경로가 통째로 죽는다.
    """
    english = analyze("SR630 V4 의 머신 타입 번호는 무엇인가요?")
    assert "번호" in english.terms                      # 희소 검색에는 남고
    assert "번호" not in english.trigram_text           # 유사도 비교에는 빠진다

    korean = analyze("2025년 총매출액은 얼마인가요?")
    assert "revenue" in korean.terms                    # 영문 확장은 일어나지만
    assert "revenue" not in korean.trigram_text         # 비교는 한국어로 한다
    assert "매출액" in korean.trigram_text


def test_reranker_ignores_terms_absent_from_every_candidate():
    """어디에도 없는 어휘가 분모에 끼면 모든 후보가 똑같이 희석된다."""
    aq = analyze("SR850 V4 서버 무게를 알려주세요.")
    assert "무게" in aq.terms and "weight" in aq.terms
    hits = [hit(1, "ThinkSystem SR850 V4 > Models\nWeight: Maximum weight: 42 kg")]
    idents, terms = RulesReranker.discriminating(aq, hits)
    assert "weight" in terms
    assert "무게" not in terms and "서버" not in terms


def test_exact_path_ands_identifiers_only():
    aq = analyze("SR650i V4 CF3G feature code")
    assert '"' not in aq.websearch_exact or " " in aq.websearch_exact
    assert "CF3G" in aq.websearch_exact
    assert "OR" not in aq.websearch_exact


# --- 결합·중복 제거·다양화 -------------------------------------------------

def test_weighted_fusion_prefers_trusted_path():
    a, b = hit(1, "A"), hit(2, "B")
    fused = fuse({"dense": [a], "exact": [b]}, top_k=2, weights={"dense": 1.0, "exact": 5.0})
    assert [h.chunk_id for h in fused] == [2, 1]
    # 어느 경로가 올렸는지 남아야 검색 실패를 진단할 수 있다.
    assert fused[0].paths == {"exact": 1}


def test_dedupe_collapses_identical_content():
    first = hit(1, "Memory maximum: 4TB", content_hash="x")
    second = hit(2, "Memory maximum: 4TB", content_hash="x")
    assert [h.chunk_id for h in dedupe([first, second])] == [1]


def test_diversify_demotes_near_duplicates_without_reordering():
    hits = [
        hit(1, "head\nDrive bay SAS/SATA 2.5-inch", score=0.9),
        hit(2, "head\nDrive bay SAS/SATA 2.5-inch", score=0.8),
        hit(3, "head\nMemory maximum: Up to 8TB", score=0.7),
    ]
    # 거의 같은 두 행 중 하나만 자리를 차지해야 한다.
    assert [h.chunk_id for h in diversify(hits, top_k=2, threshold=0.9)] == [1, 3]
    # 자리가 남으면 버리지 않고 원래 순서대로 채운다.
    assert [h.chunk_id for h in diversify(hits, top_k=3, threshold=0.9)] == [1, 3, 2]


def test_diversify_ignores_shared_chunk_header():
    """모든 청크가 같은 머리(제품 > 섹션)를 달고 있어도 서로 비슷하다고 보지 않는다."""
    head = "ThinkSystem SR630 V4 > Standard specifications"
    hits = [
        hit(1, head + "\nForm factor: 1U rack", score=0.9),
        hit(2, head + "\nMemory maximum: Up to 8TB", score=0.8),
    ]
    assert len(diversify(hits, top_k=2, threshold=0.9)) == 2


# --- 리랭킹 ----------------------------------------------------------------

def test_rules_reranker_puts_matching_product_first():
    aq = analyze("SR650a V4 최대 메모리 용량은?")
    wrong = hit(1, "ThinkSystem SR650i V4 > Memory\nMemory maximum: Up to 4TB",
                products=["ThinkSystem SR650i V4"], kind="table_row", score=0.02)
    right = hit(2, "ThinkSystem SR650a V4 > Memory\nMemory maximum: Up to 8TB",
                products=["ThinkSystem SR650a V4"], kind="table_row", score=0.01)
    ranked = RulesReranker().rerank(aq, [wrong, right], top_k=2)
    assert [h.chunk_id for h in ranked] == [2, 1]


def test_rules_reranker_prefers_table_rows_for_spec_questions():
    aq = analyze("최대 메모리 용량은?")
    prose = hit(1, "This server supports large memory configurations.", kind="prose",
                score=0.02)
    row = hit(2, "Memory maximum: Up to 8TB", kind="table_row", score=0.02)
    ranked = RulesReranker().rerank(aq, [prose, row], top_k=2)
    assert ranked[0].chunk_id == 2


def test_llm_reranker_survives_garbage_output():
    class Broken:
        name = "broken"

        def generate(self, system, user):
            return "죄송하지만 정렬할 수 없습니다"

    aq = analyze("메모리")
    hits = [hit(1, "a"), hit(2, "b")]
    # 리랭커가 검색을 망치는 것이 가장 나쁜 결과다. 원래 순위를 유지해야 한다.
    assert [h.chunk_id for h in LLMReranker(Broken()).rerank(aq, hits, 2)] == [1, 2]


def test_llm_reranker_appends_missing_indices():
    class Partial:
        name = "partial"

        def generate(self, system, user):
            return "2"

    aq = analyze("메모리")
    hits = [hit(1, "a"), hit(2, "b"), hit(3, "c")]
    ranked = LLMReranker(Partial()).rerank(aq, hits, 3)
    assert [h.chunk_id for h in ranked] == [2, 1, 3]


def test_noop_reranker_truncates_only():
    aq = analyze("메모리")
    hits = [hit(i, str(i)) for i in range(1, 5)]
    assert len(NoopReranker().rerank(aq, hits, 2)) == 2


# --- 근거 검증 -------------------------------------------------------------

def test_grounding_flags_values_absent_from_evidence():
    evidence = ["Memory maximum: Up to 8TB by using 32x 256GB 3DS RDIMMs"]
    assert grounding.check("최대 8TB 입니다 [1]", evidence) == []
    assert grounding.check("최대 4TB 입니다 [1]", evidence) == ["4TB"]


def test_grounding_tolerates_spacing_and_case():
    evidence = ["Memory maximum: Up to 8TB"]
    assert grounding.check("최대 8 TB [1]", evidence) == []


def test_grounding_ignores_citation_numbers():
    assert grounding.check("메모리는 충분합니다 [1][2]", ["아무 내용"]) == []


def test_grounding_catches_invented_part_number():
    unsupported = grounding.check("파트번호는 4X97B99999 입니다 [1]",
                                 ["Part number: 4X97B17362"])
    assert unsupported == ["4X97B99999"]


# --- 파서 상용구 제거 -------------------------------------------------------

def test_boilerplate_detection_needs_majority_of_pages():
    footer = [ProseBlock("Lenovo ThinkSystem SR630 V4 Server 7", p, "") for p in range(1, 11)]
    body = [ProseBlock("Memory maximum: Up to 8TB", 3, "")]
    keys = boilerplate_keys(footer + body, page_count=10)
    assert any("Lenovo ThinkSystem" in k for k in keys)
    assert not any("Memory maximum" in k for k in keys)


def test_boilerplate_skipped_for_short_documents():
    blocks = [ProseBlock("반복 문구", p, "") for p in range(1, 4)]
    assert boilerplate_keys(blocks, page_count=3) == set()


# --- 인증·요청 제한 ---------------------------------------------------------

def test_api_key_parsing_maps_roles_and_groups():
    table = parse_keys("0123456789abcdef:admin:eng|sales, fedcba9876543210:reader:sales")
    roles = {p.role: p.groups for p in table.values()}
    assert roles["admin"] == ("eng", "sales")
    assert roles["reader"] == ("sales",)
    with pytest.raises(ValueError):
        parse_keys("0123456789abcdef:wizard")


def test_admin_sees_everything_reader_is_scoped():
    assert Principal("a", "admin", ("eng",)).acl_groups is None
    assert Principal("r", "reader", ("eng",)).acl_groups == ["eng"]
    assert Principal("r", "reader").can("admin") is False
    assert Principal("a", "admin").can("editor") is True


def test_auth_disabled_returns_anonymous(monkeypatch):
    # API_KEYS 가 비어 있으면 로컬 개발용으로 인증을 끈다.
    assert authenticate(None).id == "anonymous"


def test_rate_limiter_rejects_burst():
    limiter = RateLimiter(per_min=2)
    limiter.check("k")
    limiter.check("k")
    with pytest.raises(AuthError) as exc:
        limiter.check("k")
    assert exc.value.status == 429
    # 주체가 다르면 서로 영향이 없어야 한다.
    limiter.check("other")
