from pathlib import Path
from dotenv import load_dotenv

# 현재 streamlit_app.py 파일이 있는 폴더
BASE_DIR = Path(__file__).resolve().parent

# .env를 rag.py보다 먼저 불러오기
ENV_PATH = BASE_DIR / ".env"

load_dotenv(
    dotenv_path=ENV_PATH,
    override=True,
    encoding="utf-8-sig"
)

import streamlit as st
from rag import ask


# 이미지 경로
IMAGE_PATH = BASE_DIR / "images" / "전주 맛집 이미지.png"


st.set_page_config(
    page_title="명예 전주인 맛집 도감",
    page_icon="🍽️",
    layout="wide"
)


# 상단 배너 이미지
if IMAGE_PATH.exists():
    st.image(
        str(IMAGE_PATH),
        use_container_width=True
    )
else:
    st.warning(
        f"배너 이미지를 찾을 수 없습니다: {IMAGE_PATH}"
    )


st.title("명예 전주인 맛집 도감 🍽️")


# 대화 기록 저장
if "messages" not in st.session_state:
    st.session_state.messages = []


# 이전 대화 출력
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


# 사용자 입력
question = st.chat_input("질문을 입력하세요.")


if question:
    # 사용자 메시지 저장
    st.session_state.messages.append(
        {
            "role": "user",
            "content": question
        }
    )

    # 사용자 메시지 출력
    with st.chat_message("user"):
        st.markdown(question)

    # AI 응답 생성 및 출력
    with st.chat_message("assistant"):
        with st.spinner("생각하는 중..."):
            try:
                answer = ask(question)

            except Exception as e:
                answer = (
                    "답변 생성 중 오류가 발생했습니다.\n\n"
                    f"```text\n{e}\n```"
                )

        st.markdown(answer)

    # AI 응답 저장
    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer
        }
    )