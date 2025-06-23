import os
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

# 🔹 .env에 OPENAI_API_KEY가 들어 있어야 함
load_dotenv()

# 1. DB 로딩
persist_directory = "./hyuwiki_vectorstore"  # 벡터 DB 저장된 경로
embedding_model = OpenAIEmbeddings()
vectordb = Chroma(
    persist_directory=persist_directory, embedding_function=embedding_model
)
retriever = vectordb.as_retriever()

docs = vectordb.similarity_search("서울캠퍼스 정보시스템 교수", k=3)
# 2. 프롬프트 템플릿 정의 (Few-shot 없이)
prompt_template = PromptTemplate(
    input_variables=["context", "question"],
    template="""
당신은 한양대학교에 대한 정보를 알려주는 챗봇입니다.
다음은 검색된 문서 내용입니다. 이를 참고하여 질문에 정답을 답하세요.

검색 문서 내용:
{context}

질문: {question}

정확하고 간결하게 답변하세요:
""",
)

# 3. QA 체인 구성
qa_chain = RetrievalQA.from_chain_type(
    llm=ChatOpenAI(temperature=0),
    chain_type="stuff",
    retriever=retriever,
    chain_type_kwargs={"prompt": prompt_template},
)

# 4. 예시 질문
query = "정보시스템 학과 교수가 누구야?"
response = qa_chain.run(query)
print("🤖 답변:", response)
