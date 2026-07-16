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
# 2. LLM 및 임베딩 모델
# =========================================================

llm = init_custom_llm()

# embedding.py에서 사용한 모델과 같아야 합니다.
embedding = HuggingFaceEmbeddings(
    model_name="BAAI/bge-m3"
)


# =========================================================
# 3. Chroma DB 연결
# =========================================================

db = Chroma(
    persist_directory=str(DB_PATH),
    collection_name=COLLECTION_NAME,
    embedding_function=embedding
)

print("=" * 60)
print("RAG DB 경로:", DB_PATH)
print("DB 폴더 존재:", DB_PATH.exists())
print("컬렉션:", COLLECTION_NAME)
print("전체 DB 문서 수:", db._collection.count())
print("=" * 60)


# =========================================================
# 4. 문자열 정규화
# =========================================================

TEXT_REPLACEMENTS = {
    "돈까스": "돈가스",
    "돈카츠": "돈가스",
    "스시": "초밥",
    "순댓국": "순대국",
    "아구찜": "아귀찜",
}


def normalize_text(value):
    """
    띄어쓰기와 기호 차이를 제거해 검색 정확도를 높입니다.

    예:
    패스트 푸드 → 패스트푸드
    콩나물 국밥 → 콩나물국밥
    돈까스 → 돈가스
    """

    if value is None:
        return ""

    text = str(value).strip().lower()

    for before, after in TEXT_REPLACEMENTS.items():
        text = text.replace(before, after)

    text = re.sub(
        r"[\s·,/()\[\]{}_\-]+",
        "",
        text
    )

    return text


# =========================================================
# 5. Chroma 문서 파싱
# =========================================================

def parse_document(content):
    """
    CSVLoader로 저장된 문서를 딕셔너리로 변환합니다.

    예:
    상호명: 금암피순대
    카테고리: 국물 요리
    업종: 순대,순댓국
    """

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
    """
    빈 행이나 정상적인 식당이 아닌 문서를 제외합니다.
    """

    return bool(
        data.get("상호명")
        and data.get("주소")
        and data.get("카테고리")
    )


# =========================================================
# 6. DB 전체 식당 불러오기
# =========================================================

def load_all_restaurants():
    result = db.get(
        include=[
            "documents",
            "metadatas",
            "embeddings"
        ]
    )

    documents = result.get("documents") or []
    metadatas = result.get("metadatas") or []
    stored_embeddings = result.get("embeddings")

    restaurants = []

    for index, content in enumerate(documents):
        data = parse_document(content)

        if not is_valid_restaurant(data):
            continue

        metadata = {}

        if index < len(metadatas):
            if metadatas[index]:
                metadata = metadatas[index]

        vector = None

        if stored_embeddings is not None:
            if index < len(stored_embeddings):
                vector = np.asarray(
                    stored_embeddings[index],
                    dtype=np.float32
                )

        restaurants.append(
            {
                "content": content,
                "data": data,
                "metadata": metadata,
                "embedding": vector
            }
        )

    return restaurants


ALL_RESTAURANTS = load_all_restaurants()

print("정상 식당 데이터 수:", len(ALL_RESTAURANTS))


# =========================================================
# 7. 식당 검색용 문자열
# =========================================================

def make_food_search_text(data):
    """
    카테고리뿐 아니라 업종과 메인메뉴까지 모두 검색합니다.
    """

    return normalize_text(
        " ".join(
            [
                data.get("상호명", ""),
                data.get("카테고리", ""),
                data.get("업종", ""),
                data.get("메인메뉴", "")
            ]
        )
    )


def make_region_search_text(data):
    """
    지역권과 실제 주소를 함께 검색합니다.
    """

    return normalize_text(
        " ".join(
            [
                data.get("지역권", ""),
                data.get("주소", ""),
                data.get("상호명", "")
            ]
        )
    )


for restaurant in ALL_RESTAURANTS:
    restaurant["food_search_text"] = make_food_search_text(
        restaurant["data"]
    )

    restaurant["region_search_text"] = make_region_search_text(
        restaurant["data"]
    )


# =========================================================
# 8. DB에서 카테고리 목록 자동 생성
# =========================================================

CATEGORY_VALUES = sorted(
    {
        restaurant["data"].get("카테고리", "").strip()
        for restaurant in ALL_RESTAURANTS
        if restaurant["data"].get("카테고리", "").strip()
    },
    key=len,
    reverse=True
)

print("검색 가능한 카테고리:", CATEGORY_VALUES)


# =========================================================
# 9. DB에서 지역 검색어 자동 생성
# =========================================================

def build_region_aliases():
    """
    지역권 데이터에서 검색 가능한 지역명을 자동 생성합니다.

    예:
    전북대·덕진공원권
        → 전북대
        → 덕진공원
        → 전북대덕진공원
    """

    aliases = {}

    region_values = {
        restaurant["data"].get("지역권", "").strip()
        for restaurant in ALL_RESTAURANTS
        if restaurant["data"].get("지역권", "").strip()
    }

    for region_value in region_values:
        region_without_suffix = region_value.replace("권", "")

        candidate_aliases = {
            normalize_text(region_value),
            normalize_text(region_without_suffix)
        }

        parts = re.split(
            r"[·,/]",
            region_without_suffix
        )

        for part in parts:
            part = part.strip()

            if part:
                candidate_aliases.add(
                    normalize_text(part)
                )

        for alias in candidate_aliases:
            if len(alias) < 2:
                continue

            if alias not in aliases:
                aliases[alias] = set()

            aliases[alias].add(region_value)

    return aliases


REGION_ALIASES = build_region_aliases()

print(
    "검색 가능한 지역어:",
    sorted(REGION_ALIASES.keys())
)


# =========================================================
# 10. 대화 조건 초기값
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

        # 업종·메인메뉴에서 직접 검색할 일반 음식어
        "search_terms": [],
        "exclude_terms": [],

        "count": DEFAULT_COUNT
    }


# =========================================================
# 11. 일반 검색어 설정
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
    "한번"
}

GENERIC_SUFFIXES = [
    "음식점",
    "맛집",
    "식당",
    "가게",
    "집"
]


def remove_generic_suffix(token):
    """
    국밥집 → 국밥
    파스타집 → 파스타
    곱창 맛집 → 곱창
    """

    token = normalize_text(token)

    changed = True

    while changed:
        changed = False

        for suffix in GENERIC_SUFFIXES:
            normalized_suffix = normalize_text(suffix)

            if (
                token.endswith(normalized_suffix)
                and len(token) > len(normalized_suffix) + 1
            ):
                token = token[:-len(normalized_suffix)]
                changed = True
                break

    return token


def term_exists_in_data(term):
    """
    질문의 단어가 실제 DB 식당 데이터에 존재하는지 확인합니다.

    따라서 국밥만 따로 등록하지 않아도
    곱창, 파스타, 오코노미야끼, 우육면 등
    DB에 존재하는 메뉴는 모두 검색됩니다.
    """

    normalized_term = normalize_text(term)

    if len(normalized_term) < 2:
        return False

    return any(
        normalized_term in restaurant["food_search_text"]
        for restaurant in ALL_RESTAURANTS
    )


# =========================================================
# 12. 부정 표현 판별
# =========================================================

def is_negative_expression(
    normalized_question,
    normalized_term
):
    """
    다음과 같은 표현을 판별합니다.

    일식집 말고
    국밥 빼고
    전북대 제외
    파스타 아닌 곳
    """

    patterns = [
        normalized_term + "말고",
        normalized_term + "집말고",
        normalized_term + "빼고",
        normalized_term + "집빼고",
        normalized_term + "제외",
        normalized_term + "집제외",
        normalized_term + "아닌",
        normalized_term + "집아닌"
    ]

    return any(
        pattern in normalized_question
        for pattern in patterns
    )


# =========================================================
# 13. 추천 개수 추출
# =========================================================

def extract_count(question):
    match = re.search(
        r"(\d+)\s*(곳|개|군데)",
        question
    )

    if match:
        count = int(match.group(1))

        return max(
            1,
            min(count, MAX_COUNT)
        )

    normalized_question = normalize_text(question)

    korean_counts = {
        "한곳": 1,
        "한군데": 1,
        "두곳": 2,
        "두군데": 2,
        "세곳": 3,
        "세군데": 3,
        "네곳": 4,
        "네군데": 4,
        "다섯곳": 5,
        "다섯군데": 5
    }

    for expression, count in korean_counts.items():
        if expression in normalized_question:
            return count

    return None


# =========================================================
# 14. 카테고리 조건 추출
# =========================================================

def extract_category_conditions(question):
    normalized_question = normalize_text(question)

    included = []
    excluded = []

    for category in CATEGORY_VALUES:
        normalized_category = normalize_text(category)

        if normalized_category not in normalized_question:
            continue

        if is_negative_expression(
            normalized_question,
            normalized_category
        ):
            excluded.append(category)

        else:
            included.append(category)

    return (
        list(dict.fromkeys(included)),
        list(dict.fromkeys(excluded))
    )


# =========================================================
# 15. 지역 조건 추출
# =========================================================

def extract_region_conditions(question):
    normalized_question = normalize_text(question)

    included = []
    excluded = []

    # 긴 지역명부터 검사
    aliases = sorted(
        REGION_ALIASES.keys(),
        key=len,
        reverse=True
    )

    for alias in aliases:
        if alias not in normalized_question:
            continue

        if is_negative_expression(
            normalized_question,
            alias
        ):
            excluded.append(alias)

        else:
            included.append(alias)

    # 전북대덕진공원과 전북대가 동시에 잡히는 경우 중복 정리
    included = remove_contained_terms(included)
    excluded = remove_contained_terms(excluded)

    return included, excluded


def remove_contained_terms(terms):
    """
    긴 표현과 짧은 표현이 동시에 감지되면
    실제 검색에 유용한 짧은 지역 단위를 유지합니다.

    예:
    전북대덕진공원, 전북대
    """

    unique_terms = list(dict.fromkeys(terms))

    result = []

    for term in sorted(unique_terms, key=len):
        if term not in result:
            result.append(term)

    return result


# =========================================================
# 16. 자유 음식 검색어 추출
# =========================================================

def extract_search_terms(
    question,
    detected_categories,
    excluded_categories,
    detected_regions,
    excluded_regions
):
    """
    카테고리나 지역이 아닌 나머지 질문 단어 중
    실제 DB의 업종·메인메뉴에 있는 단어를 자동 검색합니다.
    """

    normalized_question = normalize_text(question)

    tokens = re.findall(
        r"[가-힣a-zA-Z0-9]+",
        question
    )

    blocked_values = []

    for value in (
        detected_categories
        + excluded_categories
        + detected_regions
        + excluded_regions
    ):
        blocked_values.append(
            normalize_text(value)
        )

    included = []
    excluded = []

    normalized_stopwords = {
        normalize_text(word)
        for word in STOPWORDS
    }

    condition_words = [
        "주차",
        "혼밥",
        "혼자",
        "가족",
        "부모님",
        "아이",
        "단체",
        "회식",
        "모임",
        "1인"
    ]

    for token in tokens:
        term = remove_generic_suffix(token)

        if len(term) < 2:
            continue

        if term in normalized_stopwords:
            continue

        if any(
            normalize_text(word) in term
            or term in normalize_text(word)
            for word in condition_words
        ):
            continue

        # 이미 카테고리나 지역으로 인식한 단어는 제외
        if any(
            term in blocked
            or blocked in term
            for blocked in blocked_values
        ):
            continue

        # DB에 실제 존재하지 않는 단어는 구조 필터로 사용하지 않음
        # 이런 표현은 이후 임베딩 유사도 검색이 처리합니다.
        if not term_exists_in_data(term):
            continue

        if is_negative_expression(
            normalized_question,
            term
        ):
            excluded.append(term)

        else:
            included.append(term)

    return (
        list(dict.fromkeys(included)),
        list(dict.fromkeys(excluded))
    )


# =========================================================
# 17. 대화 조건 업데이트
# =========================================================

def update_filter_state(
    question,
    previous_state=None
):
    state = copy.deepcopy(
        previous_state or initial_filter_state()
    )

    normalized_question = normalize_text(question)

    # -----------------------------------------------------
    # 조건 초기화
    # -----------------------------------------------------

    reset_words = [
        "조건초기화",
        "대화초기화",
        "처음부터",
        "새로검색",
        "새로추천",
        "조건다지워"
    ]

    if any(
        word in normalized_question
        for word in reset_words
    ):
        state = initial_filter_state()

    # -----------------------------------------------------
    # 추천 개수
    # -----------------------------------------------------

    requested_count = extract_count(question)

    if requested_count is not None:
        state["count"] = requested_count

    # -----------------------------------------------------
    # 주차 조건
    # -----------------------------------------------------

    parking_reset_words = [
        "주차상관없",
        "주차필요없",
        "주차장없어도",
        "주차조건빼"
    ]

    if any(
        word in normalized_question
        for word in parking_reset_words
    ):
        state["parking"] = False

    elif "주차" in normalized_question:
        state["parking"] = True

    # -----------------------------------------------------
    # 혼밥 조건
    # -----------------------------------------------------

    if (
        "혼밥" in normalized_question
        or "혼자" in normalized_question
        or "1인" in normalized_question
    ):
        state["solo"] = True

    if (
        "혼밥상관없" in normalized_question
        or "혼자아니어도" in normalized_question
    ):
        state["solo"] = False

    # -----------------------------------------------------
    # 가족식사 조건
    # -----------------------------------------------------

    if (
        "가족" in normalized_question
        or "부모님" in normalized_question
        or "아이와" in normalized_question
    ):
        state["family"] = True

    if "가족식사상관없" in normalized_question:
        state["family"] = False

    # -----------------------------------------------------
    # 단체수용 조건
    # -----------------------------------------------------

    if (
        "단체" in normalized_question
        or "회식" in normalized_question
        or "모임" in normalized_question
    ):
        state["group"] = True

    if (
        "단체상관없" in normalized_question
        or "회식아니어도" in normalized_question
    ):
        state["group"] = False

    # -----------------------------------------------------
    # 카테고리 및 지역 추출
    # -----------------------------------------------------

    included_categories, excluded_categories = (
        extract_category_conditions(question)
    )

    included_regions, excluded_regions = (
        extract_region_conditions(question)
    )

    # -----------------------------------------------------
    # 모든 업종·메인메뉴를 대상으로 음식어 자동 추출
    # -----------------------------------------------------

    included_terms, excluded_terms = extract_search_terms(
        question,
        included_categories,
        excluded_categories,
        included_regions,
        excluded_regions
    )

    # -----------------------------------------------------
    # 카테고리 상태 갱신
    # -----------------------------------------------------

    if included_categories:
        state["include_categories"] = included_categories

        # 새로운 카테고리를 직접 요청했다면
        # 이전 메뉴 조건은 제거합니다.
        if not included_terms:
            state["search_terms"] = []

        for category in included_categories:
            if category in state["exclude_categories"]:
                state["exclude_categories"].remove(category)

    for category in excluded_categories:
        if category not in state["exclude_categories"]:
            state["exclude_categories"].append(category)

        if category in state["include_categories"]:
            state["include_categories"].remove(category)

    # -----------------------------------------------------
    # 지역 상태 갱신
    # -----------------------------------------------------

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

    region_reset_words = [
        "지역상관없",
        "아무지역",
        "전주전체",
        "지역조건빼"
    ]

    if any(
        word in normalized_question
        for word in region_reset_words
    ):
        state["include_regions"] = []
        state["exclude_regions"] = []

    # -----------------------------------------------------
    # 메뉴·업종 검색어 상태 갱신
    # -----------------------------------------------------

    if included_terms:
        state["search_terms"] = included_terms

        # 국밥처럼 구체적인 새 메뉴를 요청하면서
        # 이번 질문에 카테고리가 없다면 이전 카테고리를 해제
        if not included_categories:
            state["include_categories"] = []

        for term in included_terms:
            if term in state["exclude_terms"]:
                state["exclude_terms"].remove(term)

    for term in excluded_terms:
        if term not in state["exclude_terms"]:
            state["exclude_terms"].append(term)

        if term in state["search_terms"]:
            state["search_terms"].remove(term)

    food_reset_words = [
        "메뉴상관없",
        "음식상관없",
        "종류상관없",
        "아무거나"
    ]

    if any(
        word in normalized_question
        for word in food_reset_words
    ):
        state["search_terms"] = []
        state["exclude_terms"] = []
        state["include_categories"] = []
        state["exclude_categories"] = []

    return state


# =========================================================
# 18. 주차 가능 여부
# =========================================================

def is_parking_available(value):
    normalized_value = normalize_text(value)

    if not normalized_value:
        return False

    # 주차시설이 전혀 없는 경우
    unavailable_values = {
        "없음",
        "전용없음",
        "전용없음골목",
        "주차불가"
    }

    if normalized_value in unavailable_values:
        return False

    if "주차불가" in normalized_value:
        return False

    # 공영주차장이나 유료주차도 주차 가능으로 처리
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
        "천변주차"
    ]

    return any(
        keyword in normalized_value
        for keyword in available_keywords
    )


# =========================================================
# 19. 지역 일치 여부
# =========================================================

def matches_region(
    restaurant,
    selected_regions
):
    if not selected_regions:
        return True

    region_text = restaurant["region_search_text"]

    return any(
        normalize_text(region) in region_text
        for region in selected_regions
    )


# =========================================================
# 20. 음식 검색어 일치 여부
# =========================================================

def matches_all_search_terms(
    restaurant,
    search_terms
):
    """
    전북대 콩나물 국밥처럼 검색어가 여러 개면
    모든 음식어가 포함된 식당을 우선 검색합니다.
    """

    if not search_terms:
        return True

    search_text = restaurant["food_search_text"]

    return all(
        normalize_text(term) in search_text
        for term in search_terms
    )


def matches_any_search_term(
    restaurant,
    search_terms
):
    if not search_terms:
        return True

    search_text = restaurant["food_search_text"]

    return any(
        normalize_text(term) in search_text
        for term in search_terms
    )


# =========================================================
# 21. 조건 필터링
# =========================================================

def filter_restaurants(
    state,
    require_all_terms=True
):
    filtered = []

    for restaurant in ALL_RESTAURANTS:
        data = restaurant["data"]

        category = data.get("카테고리", "")
        parking = data.get("주차유형", "")
        solo = data.get("혼밥", "")
        family = data.get("가족식사", "")
        group = data.get("단체수용", "")

        # 주차
        if state["parking"]:
            if not is_parking_available(parking):
                continue

        # 포함 카테고리
        if state["include_categories"]:
            if category not in state["include_categories"]:
                continue

        # 제외 카테고리
        if category in state["exclude_categories"]:
            continue

        # 포함 지역
        if state["include_regions"]:
            if not matches_region(
                restaurant,
                state["include_regions"]
            ):
                continue

        # 제외 지역
        if state["exclude_regions"]:
            if matches_region(
                restaurant,
                state["exclude_regions"]
            ):
                continue

        # 음식·업종·메인메뉴 직접 검색
        if state["search_terms"]:
            if require_all_terms:
                matched = matches_all_search_terms(
                    restaurant,
                    state["search_terms"]
                )
            else:
                matched = matches_any_search_term(
                    restaurant,
                    state["search_terms"]
                )

            if not matched:
                continue

        # 제외 메뉴
        if state["exclude_terms"]:
            if matches_any_search_term(
                restaurant,
                state["exclude_terms"]
            ):
                continue

        # 혼밥
        if state["solo"]:
            if solo not in ["추천", "가능"]:
                continue

        # 가족식사
        if state["family"]:
            if family not in ["추천", "가능"]:
                continue

        # 단체수용
        if state["group"]:
            if group not in ["추천", "가능"]:
                continue

        filtered.append(restaurant)

    return filtered


# =========================================================
# 22. 코사인 유사도 계산
# =========================================================

def cosine_similarity(vector_a, vector_b):
    if vector_a is None or vector_b is None:
        return 0.0

    norm_a = np.linalg.norm(vector_a)
    norm_b = np.linalg.norm(vector_b)

    if norm_a == 0 or norm_b == 0:
        return 0.0

    return float(
        np.dot(vector_a, vector_b)
        / (norm_a * norm_b)
    )


# =========================================================
# 23. 검색 결과 순위 계산
# =========================================================

def calculate_keyword_bonus(
    restaurant,
    state
):
    data = restaurant["data"]

    menu_text = normalize_text(
        " ".join(
            [
                data.get("업종", ""),
                data.get("메인메뉴", "")
            ]
        )
    )

    name_text = normalize_text(
        data.get("상호명", "")
    )

    bonus = 0.0

    for term in state["search_terms"]:
        normalized_term = normalize_text(term)

        if normalized_term in menu_text:
            bonus += 0.20

        if normalized_term in name_text:
            bonus += 0.10

    return bonus


def rank_restaurants(
    question,
    restaurants,
    state
):
    try:
        query_vector = np.asarray(
            embedding.embed_query(question),
            dtype=np.float32
        )

    except Exception as error:
        print("질문 임베딩 오류:", error)
        query_vector = None

    ranked = []

    for restaurant in restaurants:
        semantic_score = cosine_similarity(
            query_vector,
            restaurant.get("embedding")
        )

        keyword_bonus = calculate_keyword_bonus(
            restaurant,
            state
        )

        ranked.append(
            {
                **restaurant,
                "score": semantic_score + keyword_bonus
            }
        )

    ranked.sort(
        key=lambda item: item["score"],
        reverse=True
    )

    return ranked


# =========================================================
# 24. 최종 식당 선택
# =========================================================

def select_restaurants(
    ranked,
    count,
    state
):
    if not ranked:
        return []

    # 사용자가 특정 음식이나 카테고리를 요청했다면
    # 관련도 순서대로 선택
    if (
        state["search_terms"]
        or state["include_categories"]
    ):
        return ranked[:count]

    # 특정 음식 요청이 없으면 같은 카테고리만
    # 반복되지 않도록 여러 종류를 우선 선택
    selected = []
    selected_names = set()
    used_categories = set()

    for restaurant in ranked:
        data = restaurant["data"]

        name = data.get("상호명", "")
        category = data.get("카테고리", "")

        if name in selected_names:
            continue

        if category in used_categories:
            continue

        selected.append(restaurant)
        selected_names.add(name)
        used_categories.add(category)

        if len(selected) >= count:
            return selected

    # 개수가 부족하면 카테고리 중복 허용
    for restaurant in ranked:
        name = restaurant["data"].get(
            "상호명",
            ""
        )

        if name in selected_names:
            continue

        selected.append(restaurant)
        selected_names.add(name)

        if len(selected) >= count:
            break

    return selected


# =========================================================
# 25. 현재 검색 조건 표시
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
        conditions.append(
            "카테고리: "
            + ", ".join(state["include_categories"])
        )

    if state["exclude_categories"]:
        conditions.append(
            "제외 카테고리: "
            + ", ".join(state["exclude_categories"])
        )

    if state["include_regions"]:
        conditions.append(
            "지역: "
            + ", ".join(state["include_regions"])
        )

    if state["exclude_regions"]:
        conditions.append(
            "제외 지역: "
            + ", ".join(state["exclude_regions"])
        )

    if state["search_terms"]:
        conditions.append(
            "검색 음식: "
            + ", ".join(state["search_terms"])
        )

    if state["exclude_terms"]:
        conditions.append(
            "제외 음식: "
            + ", ".join(state["exclude_terms"])
        )

    if not conditions:
        return "별도 조건 없음"

    return " / ".join(conditions)


# =========================================================
# 26. LLM에 전달할 데이터 생성
# =========================================================

def format_context(restaurants):
    documents = []

    for index, restaurant in enumerate(
        restaurants,
        start=1
    ):
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


# =========================================================
# 27. 답변 프롬프트
# =========================================================

prompt = ChatPromptTemplate.from_template(
"""
당신은 전주 맛집 데이터를 기반으로 식당을 안내하는 AI입니다.

현재 적용된 검색 조건:
{filter_description}

조건 검색과 임베딩 유사도 검색을 통해 식당
{selected_count}곳이 최종 선정되었습니다.

반드시 아래 선정 식당을 모두 안내하세요.

[답변 규칙]

1. 제공된 식당 데이터만 사용하세요.
2. 선정된 식당을 빠뜨리지 마세요.
3. 같은 식당을 중복하지 마세요.
4. 데이터에 없는 정보를 만들지 마세요.
5. 제외 조건에 해당하는 식당은 추천하지 마세요.
6. 가격은 가능한 경우 원 단위로 작성하세요.
7. 각 식당이 질문 조건에 맞는 이유를 짧게 설명하세요.
8. 자연스럽고 보기 좋게 정리하세요.

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

[현재 사용자 질문]

{question}
"""
)

chain = prompt | llm | StrOutputParser()


# =========================================================
# 28. 가격 표시
# =========================================================

def format_price(value):
    if value is None:
        return "정보 없음"

    text = str(value).strip()

    if not text or text.lower() == "nan":
        return "정보 없음"

    try:
        return f"{int(float(text)):,}원"

    except ValueError:
        return text


# =========================================================
# 29. LLM 실패 시 직접 출력
# =========================================================

def build_direct_answer(restaurants):
    lines = [
        "조건에 맞는 전주 맛집은 다음과 같습니다.",
        ""
    ]

    for index, restaurant in enumerate(
        restaurants,
        start=1
    ):
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
                ""
            ]
        )

    return "\n".join(lines)


# =========================================================
# 30. 최종 질문 처리
# =========================================================

def ask(
    question,
    previous_state=None
):
    question = question.strip()

    state = update_filter_state(
        question,
        previous_state
    )

    if not question:
        return "질문을 입력해 주세요.", state

    # -----------------------------------------------------
    # 1. 모든 검색어가 포함된 식당부터 검색
    # -----------------------------------------------------

    candidates = filter_restaurants(
        state,
        require_all_terms=True
    )

    # 검색어가 여러 개인데 결과가 없다면
    # 하나 이상의 검색어가 포함된 식당으로 완화
    if (
        not candidates
        and len(state["search_terms"]) >= 2
    ):
        candidates = filter_restaurants(
            state,
            require_all_terms=False
        )

    print("\n" + "=" * 60)
    print("사용자 질문:", question)
    print("현재 조건:", format_filter_state(state))
    print("조건 통과 식당 수:", len(candidates))

    if not candidates:
        return (
            "현재 적용된 조건에 맞는 식당을 "
            "데이터에서 찾지 못했습니다.\n\n"
            f"적용 조건: {format_filter_state(state)}",
            state
        )

    # -----------------------------------------------------
    # 2. 임베딩과 정확한 키워드로 순위 결정
    # -----------------------------------------------------

    ranked = rank_restaurants(
        question,
        candidates,
        state
    )

    count = min(
        state["count"],
        len(ranked)
    )

    selected = select_restaurants(
        ranked,
        count,
        state
    )

    print("최종 추천 식당 수:", len(selected))

    for index, restaurant in enumerate(
        selected,
        start=1
    ):
        data = restaurant["data"]

        print(
            f"{index}. {data.get('상호명')} "
            f"/ 카테고리: {data.get('카테고리')} "
            f"/ 지역: {data.get('지역권')} "
            f"/ 업종: {data.get('업종')} "
            f"/ 메뉴: {data.get('메인메뉴')} "
            f"/ 점수: {restaurant.get('score', 0):.4f}"
        )

    print("=" * 60)

    # -----------------------------------------------------
    # 3. LLM 답변 생성
    # -----------------------------------------------------

    context = format_context(selected)

    try:
        answer = chain.invoke(
            {
                "question": question,
                "context": context,
                "selected_count": len(selected),
                "filter_description": format_filter_state(state)
            }
        )

    except Exception as error:
        print("LLM 답변 생성 오류:", error)

        answer = build_direct_answer(selected)

    # -----------------------------------------------------
    # 4. LLM이 식당을 누락하면 직접 출력
    # -----------------------------------------------------

    for restaurant in selected:
        name = restaurant["data"].get(
            "상호명",
            ""
        )

        if name and name not in answer:
            print(
                "LLM이 선정 식당을 누락하여 "
                "직접 출력 방식으로 전환합니다."
            )

            answer = build_direct_answer(selected)
            break

    return answer, state


# =========================================================
# 31. rag.py 단독 테스트
# =========================================================

if __name__ == "__main__":
    state = initial_filter_state()

    while True:
        user_question = input("\n질문: ").strip()

        if user_question in [
            "종료",
            "exit",
            "quit"
        ]:
            break

        response, state = ask(
            user_question,
            state
        )

        print("\n답변:")
        print(response)