"""질의 분석.

검색에 넣기 전에 질문에서 네 가지를 뽑는다.

  1. **제품** — 모델을 특정한 질문에 다른 모델 행이 섞이는 것을 막는다
  2. **식별자** — 파트번호·규격코드·수치. 정확 일치가 의미 유사도보다 중요하다
  3. **검색어** — 조사를 떼고 용어집으로 영문 문서 표현까지 넓힌다
  4. **의도** — 스펙 조회인지 절차 설명인지 비교인지. 리랭킹 기준이 달라진다

왜 여기까지 하는가. 이 문서군은 영문이고 질문은 한국어다. 원문 질문을
그대로 전문검색에 넣으면 `websearch_to_tsquery` 가 모든 어절을 AND 로 묶어
한 건도 맞히지 못한다. 하이브리드 검색이라 불러 놓고 실제로는 밀집 검색
하나로 돌고 있는 상태가 되며, 하필 죽은 쪽이 식별자에 강한 경로다.
"""
from __future__ import annotations

import functools
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from app.config import settings
from app.taxonomy import extract_products

log = logging.getLogger(__name__)

# 조사와 서술격 어미. 긴 것부터 떼어낸다. 형태소 분석기 없이도 명사 어간은
# 대체로 남는다. 길이 내림차순으로 정렬해 두어야 "인가요" 가 "인가" 보다 먼저
# 시도된다("서버인가요" -> "서버").
_PARTICLES = tuple(sorted((
    "으로서", "으로써", "에서의", "에게서", "이라는", "이었나요", "인가요", "일까요",
    "입니까", "이에요", "라는", "이란", "으로", "에서", "에게", "까지", "부터",
    "보다", "처럼", "마다", "이나", "든지", "이며", "이고", "인가", "인지", "예요",
    "이다", "은", "는", "이", "가", "을", "를", "의", "와", "과", "도",
    "만", "로", "에", "야", "랑",
), key=len, reverse=True))

# 이 코퍼스의 모든 청크에 등장하는 어휘. 검색어로 넣으면 변별력이 없는데
# `discriminating` 은 "후보에 존재한다"는 이유로 살려 두므로 순위를 흔든다.
# ASCII 토큰 경로에서는 걸러내고 있었는데, 용어집 확장이 우회해 다시 들여왔다.
# 그래서 걸러내는 지점을 검색어 추가 함수 한 곳으로 모은다.
_LOW_VALUE_TERMS = frozenset({
    "thinksystem", "lenovo", "server", "servers", "the", "of", "and", "or",
    "product", "guide", "company",
})

# 식별자: 영숫자가 섞인 토큰, 또는 단위가 붙은 수치.
_IDENT_RE = re.compile(
    r"\b(?:"
    r"[A-Za-z]{1,6}[0-9][A-Za-z0-9./\-]*"      # SR650a, DDR5-5600, CF3G, M.2
    r"|[0-9]{1,4}(?:\.[0-9]+)?\s?"
    r"(?:TB|GB|MB|PB|KB|W|V|A|U|mm|cm|kg|lb|MHz|GHz|MT/s|Gb/s|Gbps|inch|nm)"
    r"|[0-9][A-Za-z][A-Za-z0-9]{2,}"           # 4X97B17362 같은 파트번호
    r")\b",
    re.IGNORECASE,
)

_HANGUL = re.compile(r"[가-힣]+")
# "V4" 처럼 판만 가리키는 토큰은 식별자로서 변별력이 없다.
_VERSION_ONLY = re.compile(r"^[Vv]\d+$")

# 질문을 만드는 데만 쓰이는 어휘. 검색어로 넣으면 아무 문서나 조금씩 맞는다.
# 조사 제거(strip_particles)로는 걸러지지 않는 의문·요청 표현이 대부분이다.
_KO_STOPWORDS = frozenset({
    "무엇", "무엇인가요", "무엇인가", "어떻게", "어떤", "어느", "언제", "어디",
    "누구", "왜", "얼마", "얼마인가요", "얼마인가", "몇", "알려주세요", "알려줘",
    "알려", "주세요", "해주세요", "있나요", "하나요", "인가요", "입니까", "입니다",
    "합니까", "됩니까", "대해", "대한", "관해", "관한", "경우", "때문", "그리고",
    "또는", "또한", "그러나", "하지만", "이것", "그것", "저것", "여기", "거기",
    "비교해", "설명해", "정리해", "찾아", "보여", "가능한", "가능", "관련",
})
_ASCII_WORD = re.compile(r"[A-Za-z][A-Za-z0-9./\-]{1,}")

# 의도 판정용 신호어.
_SPEC_HINTS = ("용량", "최대", "최소", "몇", "얼마", "크기", "무게", "속도", "전력",
               "개수", "수량", "사양", "스펙", "지원", "온도", "높이", "효율")
_PROCEDURE_HINTS = ("방법", "절차", "설치", "설정", "구성하", "어떻게", "순서",
                    "how to", "install", "configure")
_COMPARE_HINTS = ("비교", "차이", "대비", "vs", "versus", "어느 쪽", "중 어떤")


@dataclass(frozen=True)
class Glossary:
    phrases: dict[str, list[str]] = field(default_factory=dict)
    terms: dict[str, list[str]] = field(default_factory=dict)
    synonyms: dict[str, list[str]] = field(default_factory=dict)
    # 복합어 분해에 쓸 한국어 단어 키. 긴 것부터 정렬해 둔다.
    compound_keys: tuple[str, ...] = ()

    @property
    def phrase_keys(self) -> list[str]:
        # 긴 구부터 매칭해야 "최대 메모리 용량" 이 "메모리" 로 쪼개지지 않는다.
        return sorted(self.phrases, key=len, reverse=True)

    def decompose(self, token: str) -> str | None:
        """한국어 복합어 안에 든 용어집 표제어를 찾는다(가장 긴 것).

        한국어는 명사를 붙여 쓴다. "총매출액"·"연매출"·"매출액이" 는 모두
        "매출액"을 담고 있지만 문자열로는 다르다. 실제로 겪은 실패:
        "2025년 **총매출액**" 질문에서 문서의 "매출액" 행이 12위로 밀렸다.
        `총매출액` 이 어느 후보에도 없어 리랭커가 그 어휘를 버렸고, 남은
        어휘는 모든 후보에 있어 재무 행 전체가 동점이 됐다.

        형태소 분석기를 넣는 대신 표제어 포함 관계로 처리한다. 표제어를
        늘리는 것은 실무자가 할 수 있는 일이고 재색인이 필요 없다.

        토큰 자체가 표제어면 분해하지 않는다("매출액" -> None). 그 경우는
        호출부가 이미 직접 조회로 처리했고, 더 짧은 표제어("매출")로 내려가면
        뜻이 넓어질 뿐이다.
        """
        if token in self.terms:
            return None
        for key in self.compound_keys:
            if key != token and key in token:
                return key
        return None


def _as_list(value) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [t for t in value.split() if t]
    return [str(v) for v in value]


@functools.lru_cache(maxsize=4)
def load_glossary(path: str | None = None) -> Glossary:
    """용어집을 읽는다. 없으면 빈 용어집으로 계속 진행한다.

    용어집이 없다고 검색을 멈출 이유는 없다. 다만 한국어 희소 검색 품질이
    떨어지므로 설정 검증에서 경고로 알린다.
    """
    target = Path(path or settings.glossary_path)
    if not target.exists():
        log.warning("용어집 없음: %s", target)
        return Glossary()
    raw = yaml.safe_load(target.read_text(encoding="utf-8")) or {}
    terms = {k.strip(): _as_list(v) for k, v in (raw.get("terms") or {}).items()}
    # 한 글자 표제어는 아무 복합어에나 걸리므로 분해에 쓰지 않는다.
    compound = tuple(
        sorted((k for k in terms if len(k) >= 2 and _HANGUL.fullmatch(k)),
               key=len, reverse=True)
    )
    return Glossary(
        phrases={k.strip(): _as_list(v) for k, v in (raw.get("phrases") or {}).items()},
        terms=terms,
        synonyms={
            k.strip().lower(): _as_list(v) for k, v in (raw.get("synonyms") or {}).items()
        },
        compound_keys=compound,
    )


# 색인 단계와 같은 규칙을 쓴다(app/taxonomy.py). 이름만 검색 문맥에 맞춘다.
detect_products = extract_products


def strip_particles(token: str) -> str:
    """한국어 어절에서 조사를 떼어낸다.

    형태소 분석기를 넣지 않는 이유는 구성요소를 늘리기 때문이다(decisions.md D5).
    용어집이 명사 어간만 알면 되므로 이 정도로 충분히 동작한다.
    """
    for particle in _PARTICLES:
        if token.endswith(particle) and len(token) - len(particle) >= 1:
            return token[: -len(particle)]
    return token


def extract_identifiers(text: str) -> list[str]:
    found = [m.group(0).strip() for m in _IDENT_RE.finditer(text)]
    # "4 TB" 처럼 공백이 섞인 표기를 문서 표기("4TB")와 함께 취급한다.
    out: list[str] = []
    for item in found:
        out.append(item)
        squeezed = re.sub(r"\s+", "", item)
        if squeezed != item:
            out.append(squeezed)
    return list(dict.fromkeys(out))


@dataclass(frozen=True)
class AnalyzedQuery:
    raw: str
    products: list[str]
    identifiers: list[str]
    terms: list[str]          # 희소 검색에 쓸 어휘. 공백이 있으면 구(句)다
    intent: str               # spec | procedure | comparison | general

    @property
    def comparison(self) -> bool:
        return self.intent == "comparison"

    def _quote(self, term: str) -> str:
        # websearch_to_tsquery 는 따옴표를 구 검색으로 해석한다.
        return f'"{term}"' if " " in term else term

    @property
    def websearch_or(self) -> str:
        """어휘를 OR 로 묶은 websearch 문자열.

        AND 로 묶으면 질문이 길어질수록 결과가 0 이 된다. OR 로 묶고 순위는
        ts_rank_cd 에 맡긴다. 많이 겹치는 문서가 위로 온다.
        """
        return " OR ".join(self._quote(t) for t in self.terms)

    def sparse_query(self, product_filtered: bool) -> str:
        """희소 검색에 넣을 문자열.

        제품 필터가 이미 걸려 있으면 모델명 토큰을 뺀다. 모델명은 그 문서의
        모든 청크 머리에 있어서(청킹 규칙) 필터 안에서는 변별력이 0 인데,
        ts_rank_cd 는 등장 횟수를 세므로 모델명을 여러 번 쓴 긴 청크가 위로
        올라간다. 실측에서 희소 경로 Recall 을 끌어내린 주된 원인이었다.
        """
        if not product_filtered:
            return self.websearch_or
        model_parts = {p.lower() for product in self.products for p in product.split()}
        terms = [
            t for t in self.terms
            if t.lower() not in model_parts and not _VERSION_ONLY.match(t)
        ]
        return " OR ".join(self._quote(t) for t in (terms or self.terms))

    @property
    def exact_identifiers(self) -> list[str]:
        """제품 모델에서 유도된 토큰을 뺀 식별자.

        모델명은 그 문서의 **모든** 청크 머리에 붙어 있으므로(청킹 규칙),
        모델명만으로 AND 검색하면 문서 전체가 같은 취급을 받는다. 정확 일치
        경로는 파트번호·피처코드처럼 소수의 청크에만 있는 토큰에 써야 뜻이 있다.
        """
        model_parts = {p.lower() for product in self.products
                       for p in product.split()}
        return [
            i for i in self.identifiers
            if i.lower() not in model_parts and not _VERSION_ONLY.match(i)
        ]

    @property
    def websearch_exact(self) -> str:
        """식별자만 AND 로 묶는다. 정확 일치 경로용."""
        return " ".join(self._quote(t) for t in self.exact_identifiers)

    @property
    def trigram_text(self) -> str:
        """pg_trgm 유사도 비교에 넣을 문자열.

        이 경로는 문자열 하나를 본문 전체와 비교하므로 스크립트가 섞이면
        유사도가 무너진다. 실측: 영문 본문을 향한 질의에 한국어 어간을 섞자
        `word_similarity` 가 임계값을 넘지 못해 후보가 0건이 됐다.

        어느 쪽을 쓸지는 **질문의 언어**로 정한다. 확장된 어휘로 정하면,
        용어집이 한국어 질문에 영문 어휘를 넣는 순간 한국어 문서를 향한 질의가
        영문으로 비교되어 이 경로가 통째로 죽는다(실제로 그렇게 됐다).
        질문에 영문 토큰이 있으면 영문 문서를 향한 질문으로 본다.
        """
        if _ASCII_WORD.search(self.raw):
            chosen = [t for t in self.terms if not _HANGUL.search(t)]
        else:
            chosen = [t for t in self.terms if _HANGUL.search(t)]
        return " ".join(chosen or self.terms) or self.raw


def _classify(raw: str, products: list[str]) -> str:
    low = raw.lower()
    if len(products) > 1 or any(h in low for h in _COMPARE_HINTS):
        return "comparison"
    if any(h in low for h in _PROCEDURE_HINTS):
        return "procedure"
    if any(h in raw for h in _SPEC_HINTS) or raw.rstrip().endswith(("는?", "은?", "까?")):
        return "spec"
    return "general"


def analyze(raw: str, glossary: Glossary | None = None) -> AnalyzedQuery:
    glossary = glossary if glossary is not None else load_glossary()
    products = detect_products(raw)
    identifiers = extract_identifiers(raw)

    terms: list[str] = []

    def add(*values: str) -> None:
        for value in values:
            value = value.strip()
            if not value or value in terms:
                continue
            if value.lower() in _LOW_VALUE_TERMS:
                continue
            terms.append(value)

    # 1) 식별자가 가장 중요하다. 먼저 넣는다.
    add(*identifiers)

    # 2) 제품 모델 코드. "ThinkSystem" 접두는 모든 문서에 있어 변별력이 없다.
    for product in products:
        parts = product.split()
        add(*parts[1:])

    working = raw
    if settings.query_expansion:
        # 3) 구 단위 치환. 긴 표현부터 소비해 짧은 단어로 쪼개지지 않게 한다.
        for phrase in glossary.phrase_keys:
            if phrase in working:
                add(*glossary.phrases[phrase])
                working = working.replace(phrase, " ")

        # 4) 남은 한국어 어절을 조사 제거 후 용어집으로 찾는다.
        for token in _HANGUL.findall(working):
            stem = strip_particles(token)
            for candidate in (token, stem):
                if candidate in glossary.terms:
                    add(*glossary.terms[candidate])
                    break
            else:
                # 표제어가 없으면 복합어 안에 든 표제어를 찾는다.
                # "총매출액" -> "매출액". 어간 자체와 그 영문 대응을 함께 넣는다.
                inner = glossary.decompose(stem)
                if inner:
                    add(inner)
                    add(*glossary.terms.get(inner, []))
            # 영문으로 바꿨더라도 한국어 어간을 함께 남긴다. 코퍼스에 한국어
            # 문서(사내 소개서·규정집)가 섞이면 영문 용어만으로는 희소 검색이
            # 한 건도 맞히지 못한다. 반대로 영문 문서에는 한국어 어휘가 없어
            # 매칭에 영향을 주지 않으므로 넣어 두는 쪽이 안전하다.
            if len(stem) >= 2 and stem not in _KO_STOPWORDS:
                add(stem)

    # 5) 영문 토큰은 그대로 쓰되 동의어를 함께 넣는다.
    for token in _ASCII_WORD.findall(working):
        low = token.lower()
        # 식별자의 일부만 잘린 토큰("4X97B17362" 안의 "X97B17362")은 버린다.
        if any(token != ident and token in ident for ident in identifiers):
            continue
        add(token)
        add(*glossary.synonyms.get(low, []))

    if not terms:
        # 용어집이 비어 있거나 한국어만 남은 경우. 원문 어절이라도 넣는다.
        add(*[t for t in raw.split() if len(t) > 1])

    return AnalyzedQuery(
        raw=raw,
        products=products,
        identifiers=identifiers,
        terms=terms,
        intent=_classify(raw, products),
    )
