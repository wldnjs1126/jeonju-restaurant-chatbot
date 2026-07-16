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
# 2. LLM과 임베딩
# =========================================================

llm = init_custom_llm()

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
print("DB 존재 여부:", DB_PATH.exists())
print("DB 문서 수:", db._collection.count())
print("=" * 60)


# =========================================================
# 4. 문서 파싱
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
        and data.get("카테고리")
        and data.get("주소")
    )


# =========================================================
# 5. DB 전체 식당 불러오기
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

        if index < len(metadatas) and metadatas[index]:
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
                "data": data,
                "content": content,
                "metadata": metadata,
                "embedding": vector
            }
        )

    return restaurants


ALL_RESTAURANTS = load_all_restaurants()

print("유효 식당 수:", len(ALL_RESTAURANTS))


# =========================================================
# 6. 문자열 정리
# =========================================================

def normalize_text(text):
    if text is None:
        return ""

    text = str(text).lower().strip()
    text = re.sub(r"\s+", "", text)

    return text


# =========================================================
# 7. 대화 조건 기본값
# =========================================================

def initial_filter_state():
    return {
        "parking": False,
        "solo": False,
        "family": False,
        "group": False,
        "include_categories": [],
        "exclude_categories": [],
        "regions": [],
        "count": DEFAULT_COUNT
    }


# =========================================================
# 8. 카테고리 표현
# =========================================================

CATEGORY_ALIASES = {
    "한식": [
        "한식",
        "한식집",
        "백반"
    ],
    "일식": [
        "일식",
        "일식집",
        "초밥",
        "스시",
        "돈카츠",
        "돈까스",
        "라멘",
        "우동"
    ],
    "중식": [
        "중식",
        "중식집",
        "중국집",
        "짜장면",
        "짬뽕"
    ],
    "양식": [
        "양식",
        "양식집",
        "파스타",
        "스테이크"
    ],
    "분식": [
        "분식",
        "떡볶이",
        "김밥"
    ],
    "고기": [
        "고기",
        "고깃집",
        "삼겹살",
        "곱창",
        "막창",
        "갈비",
        "닭갈비"
    ],
    "해산물": [
        "해산물",
        "횟집",
        "회",
        "생선"
    ],
    "국물 요리": [
        "국물요리",
        "국밥",
        "찌개",
        "전골",
        "감자탕",
        "순두부"
    ],
    "주점": [
        "주점",
        "술집",
        "술안주"
    ],
    "패스트 푸드": [
        "패스트푸드",
        "햄버거"
    ]
}


# =========================================================
# 9. 주차 가능 여부
# =========================================================

def is_parking_available(value):
    value = normalize_text(value)

    if not value:
        return False

    # 확실하게 주차가 없는 경우
    unavailable_values = {
        "없음",
        "전용없음",
        "전용없음·골목",
        "주차불가"
    }

    if value in unavailable_values:
        return False

    if "불가" in value:
        return False

    # 전용주차장이 없어도 공영주차장 등이 있으면 가능으로 판단
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
        "매장주차",
        "천변주차"
    ]

    return any(
        keyword in value
        for keyword in available_keywords
    )


# =========================================================
# 10. 요청 개수 추출
# =========================================================

def extract_count(question):
    match = re.search(
        r"(\d+)\s*(곳|개|군데)",
        question
    )

    if match:
        count = int(match.group(1))
        return max(1, min(count, MAX_COUNT))

    korean_count = {
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

    normalized = normalize_text(question)

    for expression, count in korean_count.items():
        if expression in normalized:
            return count

    return None


# =========================================================
# 11. 제외 표현 확인
# =========================================================

def is_excluded_expression(question, aliases):
    normalized_question = normalize_text(question)

    negative_words = [
        "말고",
        "빼고",
        "제외",
        "아닌",
        "말고는",
        "싫어",
        "말고다른"
    ]

    for alias in aliases:
        normalized_alias = normalize_text(alias)

        for negative_word in negative_words:
            patterns = [
                normalized_alias + negative_word,
                normalized_alias + "집" + negative_word,
                negative_word + normalized_alias
            ]

            if any(
                pattern in normalized_question
                for pattern in patterns
            ):
                return True

    return False


# =========================================================
# 12. 대화 조건 업데이트
# =========================================================

def update_filter_state(question, previous_state=None):
    state = copy.deepcopy(
        previous_state or initial_filter_state()
    )

    normalized_question = normalize_text(question)

    # 사용자가 조건 초기화를 요청한 경우
    reset_words = [
        "조건초기화",
        "처음부터",
        "새로추천",
        "다초기화"
    ]

    if any(
        word in normalized_question
        for word in reset_words
    ):
        state = initial_filter_state()

    # 추천 개수
    requested_count = extract_count(question)

    if requested_count is not None:
        state["count"] = requested_count

    # 주차 조건
    parking_negative_words = [
        "주차상관없",
        "주차필요없",
        "주차장없어도",
        "주차제외"
    ]

    if any(
        word in normalized_question
        for word in parking_negative_words
    ):
        state["parking"] = False

    elif "주차" in normalized_question:
        state["parking"] = True

    # 혼밥
    if (
        "혼밥" in normalized_question
        or "혼자" in normalized_question
        or "1인" in normalized_question
    ):
        state["solo"] = True

    # 가족식사
    if (
        "가족" in normalized_question
        or "부모님" in normalized_question
        or "아이와" in normalized_question
    ):
        state["family"] = True

    # 단체
    if (
        "단체" in normalized_question
        or "회식" in normalized_question
        or "모임" in normalized_question
    ):
        state["group"] = True

    positive_categories = []
    negative_categories = []

    for category, aliases in CATEGORY_ALIASES.items():

        # "일식집 말고"와 같은 표현
        if is_excluded_expression(question, aliases):
            negative_categories.append(category)
            continue

        # 일반적인 카테고리 요청
        category_mentioned = any(
            normalize_text(alias) in normalized_question
            for alias in aliases
        )

        if category_mentioned:
            positive_categories.append(category)

    # 새롭게 카테고리를 지정했다면 포함 조건 교체
    if positive_categories:
        state["include_categories"] = list(
            dict.fromkeys(positive_categories)
        )

        for category in positive_categories:
            if category in state["exclude_categories"]:
                state["exclude_categories"].remove(category)

    # 제외 조건 저장
    for category in negative_categories:
        if category not in state["exclude_categories"]:
            state["exclude_categories"].append(category)

        if category in state["include_categories"]:
            state["include_categories"].remove(category)

    return state


# =========================================================
# 13. 조건 필터링
# =========================================================

def filter_restaurants(state):
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

        # 혼밥
        if state["solo"]:
            if solo not in ["추천", "가능"]:
                continue

        # 가족식사
        if state["family"]:
            if family not in ["추천", "가능"]:
                continue

        # 단체
        if state["group"]:
            if group not in ["추천", "가능"]:
                continue

        filtered.append(restaurant)

    return filtered


# =========================================================
# 14. 유사도 계산
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


def rank_restaurants(question, restaurants):
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
        score = cosine_similarity(
            query_vector,
            restaurant.get("embedding")
        )

        ranked.append(
            {
                **restaurant,
                "score": score
            }
        )

    ranked.sort(
        key=lambda item: item["score"],
        reverse=True
    )

    return ranked


# =========================================================
# 15. 카테고리가 겹치지 않게 선택
# =========================================================

def select_diverse_restaurants(ranked, count, state):
    if not ranked:
        return []

    # 사용자가 특정 카테고리를 요청한 경우에는 점수순 선택
    if state["include_categories"]:
        return ranked[:count]

    selected = []
    selected_names = set()
    used_categories = set()

    # 1차: 서로 다른 카테고리 우선
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

    # 2차: 부족하면 카테고리 중복 허용
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
# 16. 현재 조건 문자열
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
            "포함 카테고리: "
            + ", ".join(state["include_categories"])
        )

    if state["exclude_categories"]:
        conditions.append(
            "제외 카테고리: "
            + ", ".join(state["exclude_categories"])
        )

    if not conditions:
        return "별도 조건 없음"

    return " / ".join(conditions)


# =========================================================
# 17. LLM 문서 구성
# =========================================================

def format_context(restaurants):
    documents = []

    for index, restaurant in enumerate(restaurants, start=1):
        data = restaurant["data"]

        documents.append(
            f"""
===== 추천 식당 {index} =====
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
# 18. 프롬프트
# =========================================================

prompt = ChatPromptTemplate.from_template(
"""
당신은 전주 맛집을 안내하는 AI 챗봇입니다.

현재 유지되고 있는 대화 조건:
{filter_description}

아래에는 조건을 정확하게 적용한 식당이 {selected_count}곳 있습니다.
선정된 식당을 모두 안내하세요.

규칙:
1. 제공된 식당 정보만 사용하세요.
2. 선정된 식당을 빠뜨리지 마세요.
3. 제외 카테고리에 포함된 식당은 절대 추천하지 마세요.
4. 같은 식당을 중복하지 마세요.
5. 없는 정보는 만들지 마세요.
6. 사용자의 현재 질문이 짧은 후속 질문이어도 유지 조건을 반영하세요.
7. 간단하고 보기 좋게 작성하세요.

출력 형식:

조건에 맞는 전주 맛집은 다음과 같습니다.

### 1. 상호명

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

선정된 식당 데이터:

{context}

현재 사용자 질문:

{question}
"""
)

chain = prompt | llm | StrOutputParser()


# =========================================================
# 19. 직접 답변 생성
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


def build_direct_answer(restaurants):
    lines = [
        "조건에 맞는 전주 맛집은 다음과 같습니다.",
        ""
    ]

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
                ""
            ]
        )

    return "\n".join(lines)


# =========================================================
# 20. 최종 질문 처리
# =========================================================

def ask(question, previous_state=None):
    question = question.strip()

    state = update_filter_state(
        question,
        previous_state
    )

    if not question:
        return "질문을 입력해 주세요.", state

    candidates = filter_restaurants(state)

    print("\n" + "=" * 60)
    print("사용자 질문:", question)
    print("현재 조건:", format_filter_state(state))
    print("조건 통과 식당 수:", len(candidates))

    if not candidates:
        return (
            "현재 조건에 맞는 식당을 데이터에서 찾지 못했습니다.",
            state
        )

    ranked = rank_restaurants(
        question,
        candidates
    )

    count = min(
        state["count"],
        len(ranked)
    )

    selected = select_diverse_restaurants(
        ranked,
        count,
        state
    )

    print("최종 추천 식당 수:", len(selected))

    for index, restaurant in enumerate(selected, start=1):
        data = restaurant["data"]

        print(
            f"{index}. {data.get('상호명')} "
            f"/ {data.get('카테고리')} "
            f"/ 주차: {data.get('주차유형')}"
        )

    print("=" * 60)

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

    # LLM이 식당을 누락하면 직접 출력
    for restaurant in selected:
        name = restaurant["data"].get("상호명", "")

        if name and name not in answer:
            answer = build_direct_answer(selected)
            break

    return answer, state


# =========================================================
# 21. 단독 테스트
# =========================================================

if __name__ == "__main__":
    state = initial_filter_state()

    while True:
        question = input("\n질문: ").strip()

        if question == "종료":
            break

        answer, state = ask(question, state)

        print("\n답변:")
        print(answer)