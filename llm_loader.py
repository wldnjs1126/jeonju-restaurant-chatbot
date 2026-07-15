import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI


BASE_DIR = Path(__file__).resolve().parent
ENV_PATH = BASE_DIR / ".env"

# 로컬 컴퓨터에서는 .env 사용
load_dotenv(
    dotenv_path=ENV_PATH,
    override=False,
    encoding="utf-8-sig"
)


def init_custom_llm():
    model_name = os.getenv("LLM_AI_MODEL", "gpt-4o-mini")
    api_key = os.getenv("OPENAI_API_KEY")

    # Streamlit Cloud의 Secrets도 확인
    if not api_key:
        try:
            import streamlit as st

            api_key = st.secrets.get("OPENAI_API_KEY")
            model_name = st.secrets.get(
                "LLM_AI_MODEL",
                model_name
            )
        except Exception:
            pass

    if not api_key:
        raise ValueError(
            "OPENAI_API_KEY가 설정되지 않았습니다."
        )

    return ChatOpenAI(
        model=model_name,
        api_key=api_key,
        temperature=0
    )