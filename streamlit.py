import streamlit as st
from langchain_community.vectorstores import Chroma  # Updated import path
from langchain_openai import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain_community.chat_models import ChatOpenAI


# API 키 설정
api_key = 'Your_key'

# OpenAIEmbeddings 객체 생성
embedding = OpenAIEmbeddings(api_key=api_key)

# Chroma 벡터 스토어 설정
vectorstore = Chroma(
    persist_directory="./chroma_db",
    embedding_function=embedding
)

# Retriever 생성
retriever = vectorstore.as_retriever()

# RetrievalQA 생성
qa_chain = RetrievalQA.from_chain_type(
    llm=ChatOpenAI(model_name="gpt-3.5-turbo", api_key=api_key),
    retriever=retriever
)

def get_answer(question: str) -> str:
    result = qa_chain({"query": question})
    return result["result"]

# Streamlit 애플리케이션
st.title("Chroma DB 기반 질문 응답 시스템")

question = st.text_input("질문을 입력하세요:")

if st.button("질문 제출"):
    if question:
        answer = get_answer(question)
        st.write("답변:", answer)
    else:
        st.write("질문을 입력해 주세요.")
