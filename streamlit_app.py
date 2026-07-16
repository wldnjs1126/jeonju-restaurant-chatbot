from pathlib import Path

from dotenv import load_dotenv


# =========================================================
# 1. 기본 경로
# =========================================================

# 현재 streamlit_app.py 파일이 있는 폴더
BASE_DIR = Path(__file__).resolve().parent

# 환경변수 파일
ENV_PATH = BASE_DIR / ".env"

# 이미지 경로
IMAGE_PATH = BASE_DIR / "images" / "전주 맛집 이미지.png"


# =========================================================
# 2. 환경변수 불러오기
# =========================================================

# rag.py를 불러오기 전에 .env를 먼저 읽어야 합니다.
load_dotenv(
    dotenv_path=ENV_PATH,
    override=True,
    encoding="utf-8-sig"
)


# =========================================================
# 3. 라이브러리 및 RAG 함수
# =========================================================

import streamlit as st

from rag import ask, initial_filter_state


# =========================================================
# 4. Streamlit 페이지 설정
# =========================================================

st.set_page_config(
    page_title="명예 전주인 맛집 도감",
    page_icon="🍽️",
    layout="wide"
)


# =========================================================
# 5. 세션 상태 초기화
# =========================================================

# 화면에 표시할 대화 기록
if "messages" not in st.session_state:
    st.session_state.messages = []


# 이전 질문의 검색 조건 저장
#
# 예:
# 첫 질문: 주차장 있는 곳
# 두 번째 질문: 일식집 말고
#
# 주차 조건을 유지하면서 일식을 제외합니다.
if "rag_filter_state" not in st.session_state:
    st.session_state.rag_filter_state = initial_filter_state()


# =========================================================
# 6. 대화 초기화 함수
# =========================================================

def reset_chat():
    """
    화면의 대화 기록과 검색 조건을 모두 초기화합니다.
    """

    st.session_state.messages = []
    st.session_state.rag_filter_state = initial_filter_state()


# =========================================================
# 7. 사이드바
# =========================================================

with st.sidebar:
    st.header("맛집 검색")

    st.markdown(
        """
        **질문 예시**

        - 객사 파스타 맛집
        - 주차 가능한 식당
        - 혼밥하기 좋은 곳
        - 가족끼리 가기 좋은 한식집
        - 단체 식사가 가능한 식당
        """
    )

    st.divider()

    st.button(
        "대화 및 조건 초기화",
        on_click=reset_chat,
        use_container_width=True
    )


# =========================================================
# 8. 상단 배너 이미지
# =========================================================

if IMAGE_PATH.exists():
    st.image(
        str(IMAGE_PATH),
        use_container_width=True
    )

else:
    st.warning(
        f"배너 이미지를 찾을 수 없습니다: {IMAGE_PATH}"
    )


# =========================================================
# 9. 제목
# =========================================================

st.title("명예 전주인 맛집 도감 🍽️")

st.caption(
    "전주 맛집 데이터를 기반으로 지역·메뉴·주차·혼밥·"
    "가족식사·단체수용 조건을 검색합니다."
)


# =========================================================
# 10. 첫 화면 안내
# =========================================================

if not st.session_state.messages:
    with st.chat_message("assistant"):
        st.markdown(
            """
            안녕하세요. 명예 전주인 맛집 도감입니다.
            드시고 싶은 음식의 조건을 검색해주세요!
            """
        )


# =========================================================
# 11. 이전 대화 출력
# =========================================================

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


# =========================================================
# 12. 사용자 입력
# =========================================================

question = st.chat_input(
    "예: 혼밥 가능한 식당 찾아줘"
)


# =========================================================
# 13. 질문 처리
# =========================================================

if question:
    question = question.strip()

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

        # -------------------------------------------------
        # 사용자 메시지 출력
        # -------------------------------------------------

        with st.chat_message("user"):
            st.markdown(question)

        # -------------------------------------------------
        # AI 답변 생성
        # -------------------------------------------------

        with st.chat_message("assistant"):
            with st.spinner("조건에 맞는 맛집을 찾는 중입니다..."):

                try:
                    # 이전 검색 조건을 rag.py에 전달
                    answer, updated_state = ask(
                        question,
                        st.session_state.rag_filter_state
                    )

                    # 이번 질문에서 변경된 조건 저장
                    st.session_state.rag_filter_state = (
                        updated_state
                    )

                except Exception as error:
                    print("답변 생성 오류:", error)

                    answer = (
                        "답변 생성 중 오류가 발생했습니다.\n\n"
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