import os
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

# 저장된 ChromaDB 경로
PERSIST_DIRECTORY = "./hyuwiki_vectorstore"

# 임베딩 모델 로딩
embedding_model = OpenAIEmbeddings()

# 저장된 벡터스토어 로딩
vectorstore = Chroma(
    persist_directory=PERSIST_DIRECTORY,
    embedding_function=embedding_model,
)

# 검색 테스트
# query = "원영준이 누구야?"
# docs = vectorstore.similarity_search(query, k=3)

# for i, d in enumerate(docs):
#     print(f"\n---- 문서 {i+1} ----")
#     print(d.page_content[:500])  # 길면 일부만 출력
#     print(d.metadata)

docs = vectorstore.similarity_search("서울캠퍼스 정보시스템 교수", k=3)
print(docs)
