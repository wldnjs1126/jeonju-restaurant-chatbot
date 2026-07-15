from pathlib import Path
from langchain_community.document_loaders import CSVLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

import sys
from pathlib import Path

# # 현재 파일 기준으로 4단계 위 폴더를 import 경로에 추가
# sys.path.append(str(Path(__file__).resolve().parent.parent.parent.parent))

from llm_loader import init_custom_llm

llm = init_custom_llm()

# 이미 만들어진 크로마 DB 객체 생성
embedding = HuggingFaceEmbeddings(
    model_name = "BAAI/bge-m3"
)

# 현재 rag.py가 있는 폴더
BASE_DIR = Path(__file__).resolve().parent

# chroma db 폴더
DB_PATH = BASE_DIR / "chroma_db_J-Eats"
COLLECTION_NAME = "jeonju_restaurants"

db = Chroma(
    persist_directory=str(DB_PATH),
    collection_name=COLLECTION_NAME,
    embedding_function=embedding
)
print("RAG DB 경로:", DB_PATH)
print("RAG 컬렉션:", COLLECTION_NAME)
print("RAG DB 문서 수:", db._collection.count())

# 검색기
retriever = db.as_retriever(
    search_kwargs={"k": 4}
)

# 프롬프트 만들고 llm 연결 준비
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

prompt = ChatPromptTemplate.from_template(
"""
당신은 전주 맛집을 가이드 해주는 AI입니다.

아래 문서를 참고하여 답변하세요.

문서

{context}

질문

{question}
"""
)

chain = prompt | llm | StrOutputParser()

# 관련문서 검색 후 llm 연결
def format_docs(docs):
    result = ""
    for doc in docs :
        result = result + doc.page_content
        result = result + "|n|n"
    return result
# 문서와 문서 사이를 명확하게 구분하기 위해서, LLM도 문서가 구분되어 있다는 것을 더 명확하게 인식하기 위해서 


def ask(question):
    
    # 1. 관련 문서 검색
    docs = retriever.invoke(question)
    
    # 2. 문자열로 변환
    context = format_docs(docs)
    
    #3. Chain 실행
    answer = chain.invoke(
        {
        "context":context,
        "question": question
        }
    )

    return answer


# question = input("질문:")

# # 1. 관련 문서 검색
# docs = retriever.invoke(question)
# #print(docs)

# # 2. 문자열로 변환
# context = format_docs(docs)
# print(context)

# #3. Chain 실행
# answer = chain.invoke(
#     {
#        "context":context,
#        "question": question
#     }
# )
# print(answer)