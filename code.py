import os
import json
import pickle
from collections import OrderedDict
from langchain_community.vectorstores import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from typing import List
import google.generativeai as genai

# --- 0. 래퍼 클래스 및 API 키 설정 ---
class GoogleGenAIEmbeddingsLC(Embeddings):
    def __init__(self, model="models/text-embedding-004"): self.model = model
    def embed_documents(self, texts: List[str]) -> List[List[float]]: return []
    def embed_query(self, text: str) -> List[float]:
        return genai.embed_content(model=self.model, content=text, task_type="RETRIEVAL_QUERY")['embedding']

API_KEY = ""
genai.configure(api_key=API_KEY)

# --- 1. 검색 시스템 및 전체 문서 데이터 로드 ---
VECTOR_DB_PATH, BM25_INDEX_PATH = "./chroma_db_hyu", "bm25_index.pkl"
DOC_FILE_PATH = "hyuwiki_documents_20250621_234549.json"
QA_FILE_PATH = "qa_random_200_samples_20250622_203907.json"

print("1. 검색 시스템 및 전체 문서 데이터를 로드합니다...")
embedding_model = GoogleGenAIEmbeddingsLC()
vector_retriever = Chroma(persist_directory=VECTOR_DB_PATH, embedding_function=embedding_model).as_retriever(search_kwargs={"k": 10})
with open(BM25_INDEX_PATH, "rb") as f: bm25_retriever = pickle.load(f)
hybrid_retriever = EnsembleRetriever(retrievers=[bm25_retriever, vector_retriever], weights=[0.5, 0.5])
with open(DOC_FILE_PATH, 'r', encoding='utf-8') as f: all_docs_data = json.load(f)
title_to_doc_map = {item['title']: Document(page_content=item['content'], metadata=item) for item in all_docs_data}
all_titles = list(title_to_doc_map.keys())
print("   -> 로드 완료!")

# --- 2. Few-shot 예시 생성 ---
with open(QA_FILE_PATH, 'r', encoding='utf-8') as f: qa_samples = json.load(f)
few_shot_examples = random.sample(qa_samples, 2)
few_shot_prompt_part = "\n".join([f"예시 질문: {ex['question']}\n예시 답변: {ex['answer']}" for ex in few_shot_examples])

# --- 3. 최종 RAG 답변 생성 함수 (프롬프트 재구성 적용) ---
def get_final_response(original_query: str):
    print("\n[단계 1: AI를 이용한 프롬프트 재구성]")
    
    # [핵심] 프롬프트 재구성용 프롬프트
    REWRITE_PROMPT = """[지시]
당신은 사용자의 질문 의도를 파악하여, 정보 검색에 더 적합한 명확하고 상세한 질문으로 재구성하는 전문가입니다.
아래 [사용자 원본 질문]을 '한양대학교' 관련 정보를 찾는다는 맥락에 맞게, 완전한 문장의 상세한 질문으로 한 문장만 생성해주세요. 다른 설명은 붙이지 마세요.

[사용자 원본 질문]
{user_query}

[재구성된 질문]"""

    model = genai.GenerativeModel('gemini-2.5-flash')
    try:
        formatted_prompt = REWRITE_PROMPT.format(user_query=original_query)
        response = model.generate_content(formatted_prompt)
        rewritten_query = response.text.strip()
        print(f"   -> 원본 질문: '{original_query}'")
        print(f"   -> AI가 재구성한 질문: '{rewritten_query}'")
    except Exception as e:
        print(f"   -> 프롬프트 재구성 실패: {e}. 원본 질문으로 계속 진행합니다.")
        rewritten_query = original_query
    
    # --- 이제부터는 재구성된 질문(rewritten_query)을 사용 ---
    
    print("\n[단계 2: 확정적 Title 검색 시작]")
    query_no_space = rewritten_query.replace(" ", "")
    golden_docs = []
    for title in all_titles:
        title_no_space = title.replace(" ", "")
        if title in rewritten_query or title_no_space in query_no_space:
            golden_docs.append(title_to_doc_map[title])
    if golden_docs:
        print(f"   -> '황금 티켓' 발견! Title 일치 문서: {[doc.metadata['title'] for doc in golden_docs]}")

    print("[단계 3: 보조 하이브리드 검색 시작]")
    hybrid_docs = hybrid_retriever.invoke(rewritten_query) # 재구성된 질문으로 검색
    
    print("[단계 4: 결과 종합 및 정제]")
    combined_docs_dict = OrderedDict()
    for doc in golden_docs: combined_docs_dict[doc.metadata['id']] = doc
    for doc in hybrid_docs:
        if doc.metadata['id'] not in combined_docs_dict:
            combined_docs_dict[doc.metadata['id']] = doc
    final_retrieved_docs = list(combined_docs_dict.values())[:7]
    
    if not final_retrieved_docs: return "관련 정보를 찾을 수 없습니다.", []
        
    print(f"   -> 최종적으로 {len(final_retrieved_docs)}개의 관련 문서를 선별했습니다.")
    
    # 5. LLM 답변 생성
    context_str = "\n\n---\n\n".join([f"문서 제목: {doc.metadata.get('title')}\n내용: {doc.page_content}" for doc in final_retrieved_docs])
    source_info = [doc.metadata for doc in final_retrieved_docs]
    
    # 답변 생성 시에도 재구성된 질문을 사용
    rag_prompt = f"""[지시]
당신은 여러 문서 조각을 종합하여 하나의 완성된 글로 재구성하는 '글쓰기 전문가'입니다.
주어진 [검색된 문서 내용]을 바탕으로, 아래 [질문]에 대한 답변을 매우 상세하고, 논리적이며, 잘 다듬어진 설명문 형태로 작성하세요.

[답변 예시]
{few_shot_prompt_part}
---
[검색된 문서 내용]
{context_str}
---
[질문]
{rewritten_query}
[답변]"""
    
    print("\n[단계 5: 최종 답변 생성 시작]")
    try:
        response = model.generate_content(rag_prompt)
        return response.text, source_info
    except Exception as e:
        return f"답변 생성 중 오류 발생: {e}", []

# --- 4. 메인 실행 부분 ---
if __name__ == "__main__":
    print("\n챗봇을 시작합니다. (종료하려면 'exit' 또는 '종료' 입력)")
    
    while True:
        user_query = input("\n🤔 질문을 입력하세요: ")
        if user_query.lower() in ['exit', '종료']:
            print("🤖 챗봇을 종료합니다."); break
        
        answer, sources = get_final_response(user_query)
        
        print("\n" + "="*50)
        print(f"🤖 답변:\n{answer}")
        
        if sources:
            print("\n📚 참고 자료:")
            for src in sources:
                print(f"  - {src.get('title')} ({src.get('url')})")
        print("="*50)
