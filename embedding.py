# pdf 파일 읽어 들이기

from pathlib import Path
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

# 현재 실행 중인 파이썬 파일이 있는 폴더의 절대 경로
BASE_DIR = Path(__file__).resolve().parent

# 1. CSV 를 document 객체로 변환
from pathlib import Path
from langchain_community.document_loaders import CSVLoader

BASE_DIR = Path(__file__).resolve().parent

documents = []

for csv_file in BASE_DIR.glob("*.csv"):
    loader = CSVLoader(
        file_path=str(csv_file),
        encoding="utf-8-sig"  # 한글 CSV 파일에 주로 사용
    )
    documents.extend(loader.load())

print("문서 수:", len(documents))
print(documents)

# 2. 문서 분할
splitter = RecursiveCharacterTextSplitter(
    chunk_size = 500,  
    chunk_overlap = 50 
)

docs = splitter.split_documents(documents)
print("청크 갯수",len(docs))  # 청크 갯수 : 28

# 3. 임베딩 작업
embedding = HuggingFaceEmbeddings(
    model_name = "BAAI/bge-m3"
)

# 4. Chroma DB 생성
DB_PATH = BASE_DIR / "chroma_db_J-Eats"
COLLECTION_NAME = "jeonju_restaurants"

db = Chroma.from_documents(
    documents=docs,
    embedding=embedding,
    persist_directory=str(DB_PATH),
    collection_name=COLLECTION_NAME
)

print("전주 맛집 Vector DB 저장 완료")
