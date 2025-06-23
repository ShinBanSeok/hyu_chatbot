import os
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv

# 환경변수에서 OPENAI_API_KEY 로드
load_dotenv()

# 1. ChromaDB 불러오기
persist_directory = "./hyuwiki_vectorstore"  # 실제 vectorstore 경로
embedding_model = OpenAIEmbeddings()

vectordb = Chroma(
    persist_directory=persist_directory, embedding_function=embedding_model
)

# 2. 유사 문서 검색
query = "원영준이 누구야?"
similar_docs = vectordb.similarity_search(query, k=3)

# 3. 결과 출력
for i, doc in enumerate(similar_docs, 1):
    print(f"\n📄 [문서 {i}]")
    print("🔹 제목:", doc.metadata.get("title"))
    print("🔗 URL:", doc.metadata.get("url"))
    print("📝 내용:\n", doc.page_content[:500], "...")  # 처음 500자만 출력
