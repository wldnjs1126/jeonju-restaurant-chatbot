from pathlib import Path

from dotenv import load_dotenv
import streamlit as st


# =========================================================
# 1. 기본 경로 설정
# =========================================================

# 현재 streamlit_app.py가 있는 폴더
BASE_DIR = Path(__file__).resolve().parent

# 환경변수 파일 경로
ENV_PATH = BASE_DIR / ".env"

# 상단 배너 이미지 경로
IMAGE_PATH = BASE_DIR / "images" / "전주 맛집 이미지.png"


# =========================================================
# 2. 환경변수 불러오기
# =========================================================

# rag.py를 import하기 전에 반드시 .env를 먼저 불러옵니다.
load_dotenv(
    dotenv_path=ENV_PATH,
    override=True,
    encoding="utf-8-sig"
)


# =========================================================
# 3. RAG 함수 불러오기
# =========================================================

# .env를 불러온 뒤 import해야 API 키가 정상적으로 인식됩니다.
from rag import ask, initial_filter_state


# =========================================================
# 4. Streamlit 페이지 설정
# =========================================================

st.set_page_config(
    page_title="명예 전주인 맛집 도감",
    page_icon="🍽️",
    layout="wide",
    initial_sidebar_state="expanded"
)


# =========================================================
# 5. 세션 상태 초기화
# =========================================================

# 화면에 표시할 채팅 기록
if "messages" not in st.session_state:
    st.session_state.messages = []


# RAG 대화 조건 저장
#
# 예:
# 첫 질문: 주차장 있는 곳
# 두 번째 질문: 일식 말고
#
# 최종 유지 조건:
# 주차 가능 + 일식 제외
if "rag_filter_state" not in st.session_state:
    st.session_state.rag_filter_state = initial_filter_state()


# =========================================================
# 6. 상태 초기화 함수
# =========================================================

def reset_chat():
    """
    채팅 기록과 RAG 검색 조건을 모두 초기화합니다.
    """

    st.session_state.messages = []
    st.session_state.rag_filter_state = initial_filter_state()


# =========================================================
# 7. 사이드바
# =========================================================

with st.sidebar:
    st.header("검색 안내")

    st.markdown(
        """
        다음과 같이 질문할 수 있습니다.

        - 주차 가능한 맛집 알려줘
        - 혼밥하기 좋은 곳
        - 가족끼리 갈 만한 한식집
        - 단체 모임 가능한 곳
        - 전북대 근처 맛집
        """
    )

    st.divider()

    # 현재 적용 중인 검색 조건
    st.subheader("현재 검색 조건")

    current_state = st.session_state.rag_filter_state

    active_conditions = []

    if current_state.get("parking"):
        active_conditions.append("주차 가능")

    if current_state.get("solo"):
        active_conditions.append("혼밥 가능")

    if current_state.get("family"):
        active_conditions.append("가족식사 가능")

    if current_state.get("group"):
        active_conditions.append("단체수용 가능")

    include_categories = current_state.get(
        "include_categories",
        []
    )

    exclude_categories = current_state.get(
        "exclude_categories",
        []
    )

    regions = current_state.get(
        "regions",
        []
    )

    if include_categories:
        active_conditions.append(
            "포함: " + ", ".join(include_categories)
        )

    if exclude_categories:
        active_conditions.append(
            "제외: " + ", ".join(exclude_categories)
        )

    if regions:
        active_conditions.append(
            "지역: " + ", ".join(regions)
        )

    if active_conditions:
        for condition in active_conditions:
            st.markdown(f"- {condition}")
    else:
        st.caption("현재 유지 중인 조건이 없습니다.")

    st.caption(
        f"기본 추천 개수: "
        f"{current_state.get('count', 3)}곳"
    )

    st.divider()

    # 채팅과 검색 조건을 함께 초기화
    st.button(
        "대화 및 조건 초기화",
        on_click=reset_chat,
        use_container_width=True
    )


# =========================================================
# 8. 상단 이미지
# =========================================================

if IMAGE_PATH.exists():
    st.image(
        str(IMAGE_PATH),
        use_container_width=True
    )

else:
    st.warning(
        f"배너 이미지를 찾을 수 없습니다.\n\n"
        f"`{IMAGE_PATH}`"
    )


# =========================================================
# 9. 제목 및 설명
# =========================================================

st.title("명예 전주인 맛집 도감 🍽️")

st.caption(
    "전주 맛집 데이터를 기반으로 "
    "주차·혼밥·가족식사·단체수용 등의 조건을 검색합니다."
)


# =========================================================
# 10. 첫 화면 안내 메시지
# =========================================================

if not st.session_state.messages:
    with st.chat_message("assistant"):
        st.markdown(
            """
            안녕하세요. 명예 전주인 맛집 도감입니다. 
            드시고 싶은 음식의 조건을 검색해주세요.
            """
        )


# =========================================================
# 11. 이전 대화 출력
# =========================================================

for message in st.session_state.messages:
    role = message.get("role", "assistant")
    content = message.get("content", "")

    with st.chat_message(role):
        st.markdown(content)


# =========================================================
# 12. 사용자 입력
# =========================================================

question = st.chat_input(
    "예: 주차 가능한 맛집 3곳 알려줘"
)


# =========================================================
# 13. 질문 처리
# =========================================================

if question:
    question = question.strip()

    # 빈 문자열 방지
    if question:

        # -------------------------------------------------
        # 사용자 메시지 저장
        # -------------------------------------------------
        st.session_state.messages.append(
            {
                "role": "user",
                "content": question
            }
        )

        # 사용자 메시지 출력
        with st.chat_message("user"):
            st.markdown(question)

        # -------------------------------------------------
        # AI 답변 생성
        # -------------------------------------------------
        with st.chat_message("assistant"):
            with st.spinner("조건에 맞는 맛집을 찾는 중입니다..."):

                try:
                    # 이전 대화의 검색 조건을 함께 전달
                    result = ask(
                        question,
                        st.session_state.rag_filter_state
                    )

                    # 새 rag.py에서는
                    # (답변, 갱신된 조건)을 반환합니다.
                    if (
                        isinstance(result, tuple)
                        and len(result) == 2
                    ):
                        answer, updated_state = result

                        # 갱신된 조건을 세션에 저장
                        st.session_state.rag_filter_state = (
                            updated_state
                        )

                    # 이전 rag.py와 임시 호환
                    elif isinstance(result, str):
                        answer = result

                    else:
                        answer = (
                            "답변 결과를 처리할 수 없습니다. "
                            "rag.py의 ask() 반환값을 확인해 주세요."
                        )

                except Exception as error:
                    print("Streamlit 답변 생성 오류:", error)

                    answer = (
                        "답변 생성 중 오류가 발생했습니다.\n\n"
                        "터미널의 오류 내용을 확인해 주세요.\n\n"
                        f"```text\n{error}\n```"
                    )

            st.markdown(answer)

        # -------------------------------------------------
        # AI 답변 저장
        # -------------------------------------------------
        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": answer
            }
        )

        # 사이드바의 현재 조건을 바로 갱신
        st.rerun()