import os
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

# 환경변수 로드
load_dotenv()

# 1. 벡터 DB 로드
persist_directory = "./hyuwiki_vectorstore"
embedding_model = OpenAIEmbeddings()
vectordb = Chroma(
    persist_directory=persist_directory, embedding_function=embedding_model
)

# 2. 유사한 문서 검색
query = "무신사 언제 생겼나요?"
docs = vectordb.similarity_search(query, k=3)

# 3. context 추출
context = "\n\n".join([doc.page_content for doc in docs])

# 4. 프롬프트 구성
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

prompt = prompt_template.format(context=context, question=query)

# 5. LLM 호출
llm = ChatOpenAI(temperature=0)
response = llm.invoke(prompt)

# 6. 출력
print("🤖 답변:", response.content)
