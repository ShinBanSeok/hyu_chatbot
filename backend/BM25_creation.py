import os
import json
import pickle
from typing import List
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document

print("--- BM25 인덱스 생성 전용 스크립트를 시작합니다. ---")
print("INFO: 이 스크립트는 기존 Vector DB('hyuwiki_vectorstore')를 건드리지 않습니다.")

# --- 1. 설정 ---
# 파일 경로 설정
BASE_DIR = "content"
DOC_FILE_PATH = os.path.join(BASE_DIR, "hyuwiki_documents_20250621_234549.json")
BM25_INDEX_PATH = "./bm25_index.pkl"   # 스크립트 실행 위치에 생성됩니다.

# --- 2. 기존 BM25 인덱스 파일 자동 삭제 ---
if os.path.exists(BM25_INDEX_PATH):
    os.remove(BM25_INDEX_PATH)
    print(f"\n[단계 1/3] 기존 BM25 인덱스 파일 '{BM25_INDEX_PATH}'을 삭제했습니다.")

# --- 3. 원본 문서 로드 ---
print(f"\n[단계 2/3] 원본 문서 파일 '{DOC_FILE_PATH}'을 로드합니다...")
with open(DOC_FILE_PATH, "r", encoding="utf-8") as f:
    data = json.load(f)

# 개별 문서 단위로 Document 객체 생성 (Vector DB와 동일한 단위)
documents: List[Document] = []
for item in data:
    content = item.get("content", "").strip()
    # 메타데이터도 함께 저장합니다. 검색 결과에 포함됩니다.
    metadata = {key: value for key, value in item.items() if key != 'content'}
    
    if content:
        documents.append(Document(page_content=content, metadata=metadata))

print(f"   -> 총 {len(documents)}개의 문서를 로드했습니다.")


# --- 4. BM25 키워드 DB 생성 ---
print("\n[단계 3/3] BM25 키워드 인덱스를 생성합니다...")
# BM25Retriever는 문서 리스트만 있으면 바로 인덱스를 생성합니다.
# 이 과정은 CPU 기반으로 작동하며, API 호출이 없어 비용이 발생하지 않습니다.
bm25_retriever = BM25Retriever.from_documents(
    documents=documents
)

# 생성된 retriever 객체를 파일로 저장 (직렬화)
with open(BM25_INDEX_PATH, "wb") as f:
    pickle.dump(bm25_retriever, f)

print(f"   -> BM25 인덱스 생성 완료! '{BM25_INDEX_PATH}' 파일에 저장되었습니다.")
print("\n--- 모든 작업이 성공적으로 완료되었습니다. ---")
