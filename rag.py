from pathlib import Path
import copy
import re

import numpy as np

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from llm_loader import init_custom_llm


# =========================================================
# 1. 기본 설정
# =========================================================

BASE_DIR = Path(__file__).resolve().parent
DB_PATH = BASE_DIR / "chroma_db_J-Eats"
COLLECTION_NAME = "jeonju_restaurants"

DEFAULT_COUNT = 3
MAX_COUNT = 10


# =========================================================
# 2. LLM / 임베딩 / Chroma DB
# =========================================================

llm = init_custom_llm()

# embedding.py에서 DB를 만들 때 사용한 모델과 같아야 합니다.
embedding = HuggingFaceEmbeddings(
    model_name="BAAI/bge-m3"
)

db = Chroma(
    persist_directory=str(DB_PATH),
    collection_name=COLLECTION_NAME,
    embedding_function=embedding,
)

print("=" * 60)
print("RAG DB 경로:", DB_PATH)
print("DB 폴더 존재:", DB_PATH.exists())
print("컬렉션:", COLLECTION_NAME)
print("전체 DB 문서 수:", db._collection.count())
print("=" * 60)


# =========================================================
# 3. 문자열 정규화
# =========================================================

TEXT_REPLACEMENTS = {
    "돈까스": "돈가스",
    "돈카츠": "돈가스",
    "스시": "초밥",
    "순댓국": "순대국",
    "아구찜": "아귀찜",
}


def normalize_text(value):
    """검색 비교를 위해 공백과 기호, 표기 차이를 정리합니다."""

    if value is None:
        return ""

    text = str(value).strip().lower()

    for before, after in TEXT_REPLACEMENTS.items():
        text = text.replace(before, after)

    # 마침표나 쉼표가 포함돼도 부정 표현을 인식하도록 모든 기호 제거
    return re.sub(r"[^0-9a-z가-힣]+", "", text)


# =========================================================
# 4. Chroma 문서 파싱 및 전체 식당 로드
# =========================================================


def parse_document(content):
    data = {}

    if not content:
        return data

    for line in content.splitlines():
        if ":" not in line:
            continue

        key, value = line.split(":", 1)
        data[key.strip()] = value.strip()

    return data


def is_valid_restaurant(data):
    return bool(
        data.get("상호명")
        and data.get("주소")
        and data.get("카테고리")
    )


def load_all_restaurants():
    result = db.get(
        include=["documents", "metadatas", "embeddings"]
    )

    documents = result.get("documents") or []
    metadatas = result.get("metadatas") or []
    stored_embeddings = result.get("embeddings")

    restaurants = []

    for index, content in enumerate(documents):
        data = parse_document(content)

        if not is_valid_restaurant(data):
            continue

        metadata = (
            metadatas[index]
            if index < len(metadatas) and metadatas[index]
            else {}
        )

        vector = None
        if stored_embeddings is not None and index < len(stored_embeddings):
            vector = np.asarray(
                stored_embeddings[index],
                dtype=np.float32,
            )

        restaurants.append(
            {
                "content": content,
                "data": data,
                "metadata": metadata,
                "embedding": vector,
            }
        )

    return restaurants


ALL_RESTAURANTS = load_all_restaurants()
print("정상 식당 데이터 수:", len(ALL_RESTAURANTS))


# =========================================================
# 5. 검색용 문자열 구성
# =========================================================


def make_food_search_text(data):
    return normalize_text(
        " ".join(
            [
                data.get("상호명", ""),
                data.get("카테고리", ""),
                data.get("업종", ""),
                data.get("메인메뉴", ""),
            ]
        )
    )


def make_category_search_text(data):
    # 카테고리뿐 아니라 '요리주점', '일식당' 같은 업종도 함께 검사
    return normalize_text(
        " ".join(
            [
                data.get("카테고리", ""),
                data.get("업종", ""),
            ]
        )
    )


def make_region_search_text(data):
    return normalize_text(
        " ".join(
            [
                data.get("지역권", ""),
                data.get("주소", ""),
                data.get("상호명", ""),
            ]
        )
    )


for restaurant in ALL_RESTAURANTS:
    data = restaurant["data"]
    restaurant["food_search_text"] = make_food_search_text(data)
    restaurant["category_search_text"] = make_category_search_text(data)
    restaurant["region_search_text"] = make_region_search_text(data)


# =========================================================
# 6. 카테고리 및 지역 검색어 구성
# =========================================================

CATEGORY_VALUES = sorted(
    {
        restaurant["data"].get("카테고리", "").strip()
        for restaurant in ALL_RESTAURANTS
        if restaurant["data"].get("카테고리", "").strip()
    },
    key=len,
    reverse=True,
)

# 사용자가 실제 데이터의 표현과 다르게 말해도 대표 카테고리로 연결합니다.
CATEGORY_EXTRA_ALIASES = {
    "한식": {"한식", "한식집", "백반집"},
    "일식": {"일식", "일식집", "일본식", "일식당"},
    "중식": {"중식", "중식집", "중국집", "중국요리"},
    "양식": {"양식", "양식집", "서양식"},
    "분식": {"분식", "분식집"},
    "고기": {"고기", "고깃집", "고기집", "육류"},
    "해산물": {"해산물", "횟집", "수산물"},
    "국물 요리": {"국물요리", "국물음식"},
    "주점": {
        "주점",
        "술집",
        "포차",
        "호프집",
        "펍",
        "요리주점",
    },
    "패스트 푸드": {"패스트푸드", "패스트푸드점"},
}


def build_category_aliases():
    aliases = {}

    for category in CATEGORY_VALUES:
        category_aliases = {
            normalize_text(category),
            normalize_text(category.replace(" ", "")),
        }

        for alias in CATEGORY_EXTRA_ALIASES.get(category, set()):
            normalized_alias = normalize_text(alias)
            if len(normalized_alias) >= 2:
                category_aliases.add(normalized_alias)

        aliases[category] = category_aliases

    return aliases


CATEGORY_ALIASES = build_category_aliases()


def build_region_aliases():
    """지역권 컬럼에서 검색 가능한 지역 별칭을 자동 생성합니다."""

    aliases = {}

    region_values = {
        restaurant["data"].get("지역권", "").strip()
        for restaurant in ALL_RESTAURANTS
        if restaurant["data"].get("지역권", "").strip()
    }

    for region_value in region_values:
        without_suffix = region_value[:-1] if region_value.endswith("권") else region_value

        candidates = {
            normalize_text(region_value),
            normalize_text(without_suffix),
        }

        for part in re.split(r"[·,/]", without_suffix):
            normalized_part = normalize_text(part)
            if len(normalized_part) >= 2:
                candidates.add(normalized_part)

        for alias in candidates:
            if len(alias) < 2:
                continue
            aliases.setdefault(alias, set()).add(region_value)

    # 데이터에 없는 일상 표현 보완
    if any("전북대" in value for value in region_values):
        aliases.setdefault("전대", set()).update(
            value for value in region_values if "전북대" in value
        )

    return aliases


REGION_ALIASES = build_region_aliases()

print("검색 가능한 카테고리:", CATEGORY_VALUES)
print("검색 가능한 지역어:", sorted(REGION_ALIASES.keys()))


# =========================================================
# 7. 검색 상태
# =========================================================


def initial_filter_state():
    return {
        "parking": False,
        "solo": False,
        "family": False,
        "group": False,
        "include_categories": [],
        "exclude_categories": [],
        "include_regions": [],
        "exclude_regions": [],
        "search_terms": [],
        "exclude_terms": [],
        "count": DEFAULT_COUNT,
    }


# =========================================================
# 8. 질문 단어 정리
# =========================================================

STOPWORDS = {
    "추천",
    "추천해줘",
    "추천해주세요",
    "알려줘",
    "알려주세요",
    "찾아줘",
    "찾아주세요",
    "어디",
    "어디야",
    "좋은",
    "괜찮은",
    "갈만한",
    "먹을만한",
    "맛있는",
    "전주",
    "전주시",
    "근처",
    "주변",
    "있는",
    "가능한",
    "가능",
    "곳",
    "메뉴",
    "음식",
    "요리",
    "해줘",
    "해주세요",
    "그중",
    "중에서",
    "다른",
    "말고",
    "빼고",
    "제외",
    "아닌",
    "좀",
    "한번",
    "싫어",
    "싫어요",
    "별로",
    "별로야",
    "안땡겨",
    "안당겨",
    "식당",
    "맛집",
    "음식점",
    "가게",
    "주차장",
}

GENERIC_SUFFIXES = [
    "음식점",
    "맛집",
    "식당",
    "가게",
    "집",
]

# 조사 때문에 '국밥을', '주점은'이 검색되지 않는 문제를 방지합니다.
TOKEN_PARTICLES = sorted(
    [
        "에서는",
        "으로는",
        "한테는",
        "에게는",
        "부터",
        "까지",
        "에서",
        "으로",
        "에게",
        "한테",
        "이랑",
        "하고",
        "처럼",
        "보다",
        "은",
        "는",
        "이",
        "가",
        "을",
        "를",
        "도",
        "만",
        "로",
        "과",
        "와",
        "랑",
    ],
    key=len,
    reverse=True,
)


def clean_query_token(token):
    token = normalize_text(token)

    changed = True
    while changed and len(token) >= 2:
        changed = False

        # 조사부터 제거: 국밥집은 -> 국밥집
        for particle in TOKEN_PARTICLES:
            normalized_particle = normalize_text(particle)
            if token.endswith(normalized_particle):
                candidate = token[: -len(normalized_particle)]
                if len(candidate) >= 2:
                    token = candidate
                    changed = True
                    break

        if changed:
            continue

        # 일반 접미사 제거: 국밥집 -> 국밥
        for suffix in GENERIC_SUFFIXES:
            normalized_suffix = normalize_text(suffix)
            if token.endswith(normalized_suffix):
                candidate = token[: -len(normalized_suffix)]
                if len(candidate) >= 2:
                    token = candidate
                    changed = True
                    break

    return token


def term_exists_in_data(term):
    normalized_term = normalize_text(term)

    if len(normalized_term) < 2:
        return False

    return any(
        normalized_term in restaurant["food_search_text"]
        for restaurant in ALL_RESTAURANTS
    )


# =========================================================
# 9. 부정 표현 판별
# =========================================================

NEGATIVE_EXPRESSIONS = [
    "말고",
    "빼고",
    "빼줘",
    "제외",
    "제외해줘",
    "아닌",
    "아니고",
    "싫어",
    "싫어요",
    "싫습니다",
    "별로",
    "별로야",
    "안땡겨",
    "안당겨",
    "원하지않아",
    "원하지않아요",
    "원치않아",
    "피하고싶어",
    "추천하지마",
    "추천하지말아줘",
    "안돼",
    "안됨",
]

NEGATIVE_PREFIXES = [
    "싫은",
    "싫어하는",
    "제외할",
    "피하고싶은",
    "원하지않는",
]

NEGATIVE_PARTICLES = sorted(
    set(TOKEN_PARTICLES + ["같은", "쪽은", "쪽은", "종류는"]),
    key=len,
    reverse=True,
)


def is_negative_expression(question, term):
    """
    '주점은 싫어', '주점은 진짜 별로야', '싫은 주점' 등을
    제외 조건으로 판별합니다. 문장 부호를 기준으로 절을 나눠
    다른 절의 부정어가 엉뚱한 단어에 적용되지 않게 합니다.
    """

    normalized_term = normalize_text(term)

    if not normalized_term:
        return False

    # '주차장 있는 식당. 주점은 싫어'에서 식당까지 부정되는
    # 오검출을 막기 위해 문장/절 단위로 검사합니다.
    clauses = re.split(
        r"[.!?,;:\n]+|(?:그리고|하지만|그런데|근데)",
        str(question),
    )

    term_variants = {
        normalized_term,
        normalized_term + "집",
    }

    normalized_negatives = [normalize_text(value) for value in NEGATIVE_EXPRESSIONS]
    normalized_prefixes = [normalize_text(value) for value in NEGATIVE_PREFIXES]
    normalized_particles = [normalize_text(value) for value in NEGATIVE_PARTICLES]

    # 부정어 앞에 올 수 있는 짧은 강조 표현만 허용합니다.
    allowed_fillers = {
        "",
        "진짜",
        "정말",
        "너무",
        "좀",
        "그닥",
        "딱히",
        "아예",
        "별로",
    }

    for clause in clauses:
        normalized_clause = normalize_text(clause)

        for variant in term_variants:
            start = 0

            while True:
                index = normalized_clause.find(variant, start)
                if index < 0:
                    break

                before = normalized_clause[max(0, index - 10):index]
                after_start = index + len(variant)
                after = normalized_clause[after_start:after_start + 24]

                # '싫은 주점', '제외할 주점'
                if any(before.endswith(prefix) for prefix in normalized_prefixes):
                    return True

                trimmed_after = after
                for particle in normalized_particles:
                    if trimmed_after.startswith(particle):
                        trimmed_after = trimmed_after[len(particle):]
                        break

                for negative in normalized_negatives:
                    negative_index = trimmed_after.find(negative)

                    if negative_index < 0:
                        continue

                    between = trimmed_after[:negative_index]

                    if between in allowed_fillers:
                        return True

                start = index + len(variant)

    return False


# =========================================================
# 10. 추천 개수 / 카테고리 / 지역 / 메뉴 조건 추출
# =========================================================


def extract_count(question):
    numeric_match = re.search(
        r"(?<![0-9가-힣a-zA-Z])(\d+)\s*(곳|개|군데)(?:만)?",
        question,
    )

    if numeric_match:
        return max(1, min(int(numeric_match.group(1)), MAX_COUNT))

    korean_number_map = {
        "한": 1,
        "두": 2,
        "세": 3,
        "네": 4,
        "다섯": 5,
    }

    korean_match = re.search(
        r"(?<![가-힣a-zA-Z0-9])(한|두|세|네|다섯)\s*(곳|개|군데)(?:만)?",
        question,
    )

    if korean_match:
        return korean_number_map[korean_match.group(1)]

    return None


def extract_category_conditions(question):
    normalized_question = normalize_text(question)

    included = []
    excluded = []
    matched_aliases = []

    for category, aliases in CATEGORY_ALIASES.items():
        mentioned = [alias for alias in aliases if alias in normalized_question]

        if not mentioned:
            continue

        matched_aliases.extend(mentioned)

        if any(is_negative_expression(question, alias) for alias in mentioned):
            excluded.append(category)
        else:
            included.append(category)

    return (
        list(dict.fromkeys(included)),
        list(dict.fromkeys(excluded)),
        list(dict.fromkeys(matched_aliases)),
    )


def keep_most_specific_aliases(aliases):
    """긴 지역명이 잡히면 그 안에 포함된 짧은 중복 별칭은 제거합니다."""

    result = []

    for alias in sorted(set(aliases), key=len, reverse=True):
        if any(alias in selected for selected in result):
            continue
        result.append(alias)

    return result


def extract_region_conditions(question):
    normalized_question = normalize_text(question)

    included = []
    excluded = []
    matched_aliases = []

    for alias in sorted(REGION_ALIASES.keys(), key=len, reverse=True):
        if alias not in normalized_question:
            continue

        matched_aliases.append(alias)

        if is_negative_expression(question, alias):
            excluded.append(alias)
        else:
            included.append(alias)

    return (
        keep_most_specific_aliases(included),
        keep_most_specific_aliases(excluded),
        keep_most_specific_aliases(matched_aliases),
    )


def extract_search_terms(question, blocked_aliases):
    tokens = re.findall(r"[가-힣a-zA-Z0-9]+", question)

    included = []
    excluded = []

    normalized_stopwords = {normalize_text(word) for word in STOPWORDS}
    normalized_blocked = [normalize_text(value) for value in blocked_aliases]

    condition_words = {
        "주차",
        "주차장",
        "혼밥",
        "혼자",
        "가족",
        "부모님",
        "아이",
        "단체",
        "회식",
        "모임",
        "1인",
    }

    for token in tokens:
        term = clean_query_token(token)

        if len(term) < 2 or term in normalized_stopwords:
            continue

        if any(
            normalize_text(word) in term or term in normalize_text(word)
            for word in condition_words
        ):
            continue

        if any(term in blocked or blocked in term for blocked in normalized_blocked):
            continue

        if not term_exists_in_data(term):
            continue

        if is_negative_expression(question, term):
            excluded.append(term)
        else:
            included.append(term)

    return (
        list(dict.fromkeys(included)),
        list(dict.fromkeys(excluded)),
    )


# =========================================================
# 11. 대화 상태 업데이트
# =========================================================


def is_new_standalone_search(question, previous_state):
    """
    완전한 새 질문이면 이전 메뉴/지역 조건이 남지 않도록 합니다.
    짧은 후속 질문('그중', '일식 말고')은 기존 조건을 유지합니다.
    """

    if previous_state is None:
        return False

    normalized_question = normalize_text(question)

    follow_up_markers = [
        "그중",
        "거기서",
        "그곳에서",
        "그럼",
        "그거",
        "방금",
        "아까",
        "이번에는",
    ]

    if any(marker in normalized_question for marker in follow_up_markers):
        return False

    # '주점은 싫어', '일식 말고'처럼 제외 조건만 말한 경우는 후속 질문
    has_negative = any(
        normalize_text(negative) in normalized_question
        for negative in NEGATIVE_EXPRESSIONS
    )

    generic_search_words = ["맛집", "식당", "음식점", "추천해줘", "찾아줘"]

    if any(word in normalized_question for word in generic_search_words):
        return True

    return not has_negative and len(normalized_question) >= 8


def update_filter_state(question, previous_state=None):
    # 새 검색으로 판단되면 오래된 메뉴·지역 조건을 먼저 제거합니다.
    if is_new_standalone_search(question, previous_state):
        state = initial_filter_state()
    else:
        state = copy.deepcopy(previous_state or initial_filter_state())

    normalized_question = normalize_text(question)

    reset_words = [
        "조건초기화",
        "대화초기화",
        "처음부터",
        "새로검색",
        "새로추천",
        "조건다지워",
    ]

    if any(word in normalized_question for word in reset_words):
        state = initial_filter_state()

    requested_count = extract_count(question)
    if requested_count is not None:
        state["count"] = requested_count

    # 주차 조건
    if any(
        word in normalized_question
        for word in ["주차상관없", "주차필요없", "주차장없어도", "주차조건빼"]
    ):
        state["parking"] = False
    elif "주차" in normalized_question:
        state["parking"] = True

    # 혼밥 조건
    if any(word in normalized_question for word in ["혼밥", "혼자", "1인"]):
        state["solo"] = True
    if any(word in normalized_question for word in ["혼밥상관없", "혼자아니어도"]):
        state["solo"] = False

    # 가족식사 조건
    if any(word in normalized_question for word in ["가족", "부모님", "아이와"]):
        state["family"] = True
    if "가족식사상관없" in normalized_question:
        state["family"] = False

    # 단체수용 조건
    if any(word in normalized_question for word in ["단체", "회식", "모임"]):
        state["group"] = True
    if any(word in normalized_question for word in ["단체상관없", "회식아니어도"]):
        state["group"] = False

    (
        included_categories,
        excluded_categories,
        matched_category_aliases,
    ) = extract_category_conditions(question)

    (
        included_regions,
        excluded_regions,
        matched_region_aliases,
    ) = extract_region_conditions(question)

    included_terms, excluded_terms = extract_search_terms(
        question,
        matched_category_aliases + matched_region_aliases,
    )

    # 새 포함 카테고리는 이전 포함 카테고리를 대체합니다.
    if included_categories:
        state["include_categories"] = included_categories

        for category in included_categories:
            if category in state["exclude_categories"]:
                state["exclude_categories"].remove(category)

        # 구체 메뉴가 없는 새 카테고리 질문은 이전 메뉴 조건을 해제
        if not included_terms:
            state["search_terms"] = []

    # 제외 카테고리는 누적하되 과거 포함 조건에서는 제거합니다.
    for category in excluded_categories:
        if category not in state["exclude_categories"]:
            state["exclude_categories"].append(category)

        if category in state["include_categories"]:
            state["include_categories"].remove(category)

    # 새 지역은 이전 지역을 대체합니다.
    if included_regions:
        state["include_regions"] = included_regions

        for region in included_regions:
            if region in state["exclude_regions"]:
                state["exclude_regions"].remove(region)

    for region in excluded_regions:
        if region not in state["exclude_regions"]:
            state["exclude_regions"].append(region)

        if region in state["include_regions"]:
            state["include_regions"].remove(region)

    if any(
        word in normalized_question
        for word in ["지역상관없", "아무지역", "전주전체", "지역조건빼"]
    ):
        state["include_regions"] = []
        state["exclude_regions"] = []

    # 새 구체 메뉴는 이전 구체 메뉴를 대체합니다.
    if included_terms:
        state["search_terms"] = included_terms

        for term in included_terms:
            if term in state["exclude_terms"]:
                state["exclude_terms"].remove(term)

        # 새 메뉴만 지정하면 이전 카테고리 제한은 해제
        if not included_categories:
            state["include_categories"] = []

    for term in excluded_terms:
        if term not in state["exclude_terms"]:
            state["exclude_terms"].append(term)

        if term in state["search_terms"]:
            state["search_terms"].remove(term)

    if any(
        word in normalized_question
        for word in ["메뉴상관없", "음식상관없", "종류상관없", "아무거나"]
    ):
        state["search_terms"] = []
        state["exclude_terms"] = []
        state["include_categories"] = []
        state["exclude_categories"] = []

    print("포함 카테고리:", state["include_categories"])
    print("제외 카테고리:", state["exclude_categories"])
    print("포함 지역:", state["include_regions"])
    print("검색 메뉴:", state["search_terms"])

    return state


# =========================================================
# 12. 조건 판별 및 필터링
# =========================================================


def is_parking_available(value):
    normalized_value = normalize_text(value)

    if not normalized_value:
        return False

    # 공영/유료 주차가 함께 적힌 경우는 '전용 없음'이어도 주차 가능
    available_keywords = [
        "주차가능",
        "주차장",
        "공영",
        "유료",
        "노상",
        "지하주차",
        "전용주차",
        "주차지원",
        "매장앞",
        "매장옆",
        "천변주차",
        "골목주차",
    ]

    exact_unavailable = {
        "없음",
        "전용없음",
        "전용없음골목",
        "주차불가",
    }

    if normalized_value in exact_unavailable:
        return False

    if "주차불가" in normalized_value:
        return False

    return any(keyword in normalized_value for keyword in available_keywords)


def matches_category(restaurant, category):
    aliases = CATEGORY_ALIASES.get(category, {normalize_text(category)})
    category_text = restaurant["category_search_text"]

    return any(alias in category_text for alias in aliases)


def matches_region(restaurant, selected_regions):
    if not selected_regions:
        return True

    region_text = restaurant["region_search_text"]
    return any(normalize_text(region) in region_text for region in selected_regions)


def matches_all_search_terms(restaurant, search_terms):
    if not search_terms:
        return True

    search_text = restaurant["food_search_text"]
    return all(normalize_text(term) in search_text for term in search_terms)


def matches_any_search_term(restaurant, search_terms):
    if not search_terms:
        return False

    search_text = restaurant["food_search_text"]
    return any(normalize_text(term) in search_text for term in search_terms)


def is_available_flag(value):
    return normalize_text(value) in {"추천", "가능"}


def filter_restaurants(state, require_all_terms=True):
    filtered = []

    for restaurant in ALL_RESTAURANTS:
        data = restaurant["data"]

        parking = data.get("주차유형", "")
        solo = data.get("혼밥", "")
        family = data.get("가족식사", "")
        group = data.get("단체수용", "")

        if state["parking"] and not is_parking_available(parking):
            continue

        if state["include_categories"] and not any(
            matches_category(restaurant, category)
            for category in state["include_categories"]
        ):
            continue

        # 카테고리뿐 아니라 업종의 '요리주점'도 주점 제외에 포함
        if any(
            matches_category(restaurant, category)
            for category in state["exclude_categories"]
        ):
            continue

        if state["include_regions"] and not matches_region(
            restaurant,
            state["include_regions"],
        ):
            continue

        if state["exclude_regions"] and matches_region(
            restaurant,
            state["exclude_regions"],
        ):
            continue

        if state["search_terms"]:
            if require_all_terms:
                matched = matches_all_search_terms(
                    restaurant,
                    state["search_terms"],
                )
            else:
                matched = matches_any_search_term(
                    restaurant,
                    state["search_terms"],
                )

            if not matched:
                continue

        if state["exclude_terms"] and matches_any_search_term(
            restaurant,
            state["exclude_terms"],
        ):
            continue

        if state["solo"] and not is_available_flag(solo):
            continue

        if state["family"] and not is_available_flag(family):
            continue

        if state["group"] and not is_available_flag(group):
            continue

        filtered.append(restaurant)

    return filtered


# =========================================================
# 13. 임베딩 유사도 및 최종 선택
# =========================================================


def cosine_similarity(vector_a, vector_b):
    if vector_a is None or vector_b is None:
        return 0.0

    norm_a = np.linalg.norm(vector_a)
    norm_b = np.linalg.norm(vector_b)

    if norm_a == 0 or norm_b == 0:
        return 0.0

    return float(np.dot(vector_a, vector_b) / (norm_a * norm_b))


def build_ranking_query(question, state):
    """
    '주점은 싫어'의 '주점'이 유사도 점수를 올리지 않도록
    긍정 조건만으로 순위 계산용 질문을 새로 만듭니다.
    """

    parts = []

    if state["parking"]:
        parts.append("주차 가능")
    if state["solo"]:
        parts.append("혼밥 가능")
    if state["family"]:
        parts.append("가족 식사 가능")
    if state["group"]:
        parts.append("단체 이용 가능")

    parts.extend(state["include_regions"])
    parts.extend(state["include_categories"])
    parts.extend(state["search_terms"])

    return " ".join(parts + ["전주 맛집"]) if parts else question


def calculate_keyword_bonus(restaurant, state):
    data = restaurant["data"]
    menu_text = normalize_text(
        " ".join([data.get("업종", ""), data.get("메인메뉴", "")])
    )
    name_text = normalize_text(data.get("상호명", ""))

    bonus = 0.0

    for term in state["search_terms"]:
        normalized_term = normalize_text(term)

        if normalized_term in menu_text:
            bonus += 0.20
        if normalized_term in name_text:
            bonus += 0.10

    return bonus


def rank_restaurants(question, restaurants, state):
    ranking_query = build_ranking_query(question, state)

    try:
        query_vector = np.asarray(
            embedding.embed_query(ranking_query),
            dtype=np.float32,
        )
    except Exception as error:
        print("질문 임베딩 오류:", error)
        query_vector = None

    ranked = []

    for restaurant in restaurants:
        semantic_score = cosine_similarity(
            query_vector,
            restaurant.get("embedding"),
        )
        keyword_bonus = calculate_keyword_bonus(restaurant, state)

        ranked.append(
            {
                **restaurant,
                "score": semantic_score + keyword_bonus,
            }
        )

    ranked.sort(key=lambda item: item["score"], reverse=True)
    return ranked


def select_restaurants(ranked, count, state):
    if not ranked:
        return []

    if state["search_terms"] or state["include_categories"]:
        return ranked[:count]

    selected = []
    selected_names = set()
    used_categories = set()

    # 특정 음식 조건이 없으면 서로 다른 카테고리를 우선 추천
    for restaurant in ranked:
        data = restaurant["data"]
        name = data.get("상호명", "")
        category = data.get("카테고리", "")

        if name in selected_names or category in used_categories:
            continue

        selected.append(restaurant)
        selected_names.add(name)
        used_categories.add(category)

        if len(selected) >= count:
            return selected

    for restaurant in ranked:
        name = restaurant["data"].get("상호명", "")

        if name in selected_names:
            continue

        selected.append(restaurant)
        selected_names.add(name)

        if len(selected) >= count:
            break

    return selected


# =========================================================
# 14. 답변 생성
# =========================================================


def format_filter_state(state):
    conditions = []

    if state["parking"]:
        conditions.append("주차 가능")
    if state["solo"]:
        conditions.append("혼밥 가능")
    if state["family"]:
        conditions.append("가족식사 가능")
    if state["group"]:
        conditions.append("단체수용 가능")
    if state["include_categories"]:
        conditions.append("카테고리: " + ", ".join(state["include_categories"]))
    if state["exclude_categories"]:
        conditions.append("제외 카테고리: " + ", ".join(state["exclude_categories"]))
    if state["include_regions"]:
        conditions.append("지역: " + ", ".join(state["include_regions"]))
    if state["exclude_regions"]:
        conditions.append("제외 지역: " + ", ".join(state["exclude_regions"]))
    if state["search_terms"]:
        conditions.append("검색 음식: " + ", ".join(state["search_terms"]))
    if state["exclude_terms"]:
        conditions.append("제외 음식: " + ", ".join(state["exclude_terms"]))

    return " / ".join(conditions) if conditions else "별도 조건 없음"


def format_context(restaurants):
    documents = []

    for index, restaurant in enumerate(restaurants, start=1):
        data = restaurant["data"]

        documents.append(
            f"""
===== 선정 식당 {index} =====
상호명: {data.get("상호명", "정보 없음")}
카테고리: {data.get("카테고리", "정보 없음")}
업종: {data.get("업종", "정보 없음")}
주소: {data.get("주소", "정보 없음")}
지역권: {data.get("지역권", "정보 없음")}
메인메뉴: {data.get("메인메뉴", "정보 없음")}
가격_원: {data.get("가격_원", "정보 없음")}
주차유형: {data.get("주차유형", "정보 없음")}
혼밥: {data.get("혼밥", "정보 없음")}
가족식사: {data.get("가족식사", "정보 없음")}
단체수용: {data.get("단체수용", "정보 없음")}
""".strip()
        )

    return "\n\n".join(documents)


prompt = ChatPromptTemplate.from_template(
    """
당신은 직접 구축한 전주 맛집 데이터를 기반으로 안내하는 AI입니다.

현재 적용된 검색 조건:
{filter_description}

아래 식당 {selected_count}곳은 Python 코드에서 조건 검사를 통과한 최종 결과입니다.
선정된 식당을 모두 빠짐없이 안내하세요.

[답변 규칙]
1. 제공된 식당 데이터만 사용하세요.
2. 같은 식당을 중복하지 마세요.
3. 데이터에 없는 정보를 만들지 마세요.
4. 제외 조건에 해당하는 식당을 추가하지 마세요.
5. 가격은 가능한 경우 원 단위로 보기 좋게 작성하세요.
6. 추천 이유는 제공된 데이터와 질문 조건만 근거로 짧게 작성하세요.
7. [식당이름] 같은 예시 문구를 그대로 출력하지 마세요.

[출력 형식]

조건에 맞는 전주 맛집은 다음과 같습니다.

### 1. 실제 상호명
- 카테고리:
- 업종:
- 주소:
- 지역권:
- 메인메뉴:
- 가격:
- 주차:
- 혼밥:
- 가족식사:
- 단체수용:
- 추천 이유:

[선정 식당 데이터]
{context}

[사용자 질문]
{question}
"""
)

chain = prompt | llm | StrOutputParser()


def format_price(value):
    if value is None:
        return "정보 없음"

    text = str(value).strip()

    if not text or text.lower() in {"nan", "none"}:
        return "정보 없음"

    numeric_text = text.replace(",", "").replace("원", "").strip()

    try:
        return f"{int(float(numeric_text)):,}원"
    except ValueError:
        return text if text.endswith("원") else f"{text}원"


def build_direct_answer(restaurants):
    lines = ["조건에 맞는 전주 맛집은 다음과 같습니다.", ""]

    for index, restaurant in enumerate(restaurants, start=1):
        data = restaurant["data"]

        lines.extend(
            [
                f"### {index}. {data.get('상호명', '정보 없음')}",
                "",
                f"- **카테고리:** {data.get('카테고리', '정보 없음')}",
                f"- **업종:** {data.get('업종', '정보 없음')}",
                f"- **주소:** {data.get('주소', '정보 없음')}",
                f"- **지역권:** {data.get('지역권', '정보 없음')}",
                f"- **메인메뉴:** {data.get('메인메뉴', '정보 없음')}",
                f"- **가격:** {format_price(data.get('가격_원'))}",
                f"- **주차:** {data.get('주차유형', '정보 없음')}",
                f"- **혼밥:** {data.get('혼밥', '정보 없음')}",
                f"- **가족식사:** {data.get('가족식사', '정보 없음')}",
                f"- **단체수용:** {data.get('단체수용', '정보 없음')}",
                "",
            ]
        )

    return "\n".join(lines)


# =========================================================
# 15. 최종 질문 처리
# =========================================================


def ask(question, previous_state=None):
    question = question.strip()

    if not question:
        state = copy.deepcopy(previous_state or initial_filter_state())
        return "질문을 입력해 주세요.", state

    state = update_filter_state(question, previous_state)

    candidates = filter_restaurants(
        state,
        require_all_terms=True,
    )

    # 두 개 이상의 메뉴 검색어를 모두 만족하는 식당이 없을 때만 OR 검색
    if not candidates and len(state["search_terms"]) >= 2:
        candidates = filter_restaurants(
            state,
            require_all_terms=False,
        )

    print("\n" + "=" * 60)
    print("사용자 질문:", question)
    print("현재 조건:", format_filter_state(state))
    print("조건 통과 식당 수:", len(candidates))

    if not candidates:
        return (
            "현재 적용된 조건에 맞는 식당을 데이터에서 찾지 못했습니다.\n\n"
            f"적용 조건: {format_filter_state(state)}",
            state,
        )

    ranked = rank_restaurants(
        question,
        candidates,
        state,
    )

    count = min(state["count"], len(ranked))
    selected = select_restaurants(
        ranked,
        count,
        state,
    )

    print("최종 추천 식당 수:", len(selected))

    for index, restaurant in enumerate(selected, start=1):
        data = restaurant["data"]
        print(
            f"{index}. {data.get('상호명')} "
            f"/ 카테고리: {data.get('카테고리')} "
            f"/ 업종: {data.get('업종')} "
            f"/ 지역: {data.get('지역권')} "
            f"/ 주차: {data.get('주차유형')} "
            f"/ 점수: {restaurant.get('score', 0):.4f}"
        )

    print("=" * 60)

    context = format_context(selected)

    try:
        answer = chain.invoke(
            {
                "question": question,
                "context": context,
                "selected_count": len(selected),
                "filter_description": format_filter_state(state),
            }
        )
    except Exception as error:
        print("LLM 답변 생성 오류:", error)
        answer = build_direct_answer(selected)

    # LLM이 최종 선정 식당을 하나라도 누락하면 안전한 직접 출력으로 전환
    if any(
        restaurant["data"].get("상호명", "") not in answer
        for restaurant in selected
    ):
        print("LLM이 선정 식당을 누락하여 직접 출력 방식으로 전환합니다.")
        answer = build_direct_answer(selected)

    return answer, state


# =========================================================
# 16. rag.py 단독 테스트
# =========================================================

if __name__ == "__main__":
    state = initial_filter_state()

    while True:
        user_question = input("\n질문: ").strip()

        if user_question in {"종료", "exit", "quit"}:
            break

        response, state = ask(user_question, state)

        print("\n답변:")
        print(response)