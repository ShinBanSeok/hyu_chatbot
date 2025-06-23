import os
import json
from typing import List
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain.docstore.document import Document
from dotenv import load_dotenv

# 환경 변수 로드
load_dotenv()

# 파일 경로
current_dir = os.path.dirname(__file__)
file_path = os.path.join(current_dir, "./docu.json")

# JSON 로드
with open(file_path, "r", encoding="utf-8") as f:
    data = json.load(f)

# 개별 문서 단위로 Document 생성 (split 없이)
documents: List[Document] = []
for item in data:
    content = item.get("content", "").strip()
    metadata = {
        "title": item.get("title", ""),
        "url": item.get("url", ""),
        "source": item.get("url_title", ""),
    }
    if content:
        documents.append(Document(page_content=content, metadata=metadata))

# 임베딩 및 저장
embedding_model = OpenAIEmbeddings()
persist_directory = "./hyuwiki_vectorstore"

vectorstore = Chroma.from_documents(
    documents,
    embedding_model,
    persist_directory=persist_directory,
)
vectorstore.persist()

print("✅ 문서 단위로 임베딩 완료")
