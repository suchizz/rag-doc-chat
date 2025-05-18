import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains.question_answering import load_qa_chain
from langchain.docstore.document import Document

openai_api_key = st.text_input("Enter your OpenAI API Key:", type="password")

if openai_api_key:
    st.title("🧠 Simple RAG Chatbot")

    uploaded_file = st.file_uploader("Upload a text file", type="txt")

    if uploaded_file:
        text = uploaded_file.read().decode("utf-8")

        # Split text
        text_splitter = CharacterTextSplitter(chunk_size=400, chunk_overlap=50)
        chunks = text_splitter.split_text(text)

        # Embed
        embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
        vectorstore = FAISS.from_texts(chunks, embedding=embeddings)

        # Get user query
        query = st.text_input("Ask something about the document")

        if query:
            docs = vectorstore.similarity_search(query)
            llm = ChatOpenAI(openai_api_key=openai_api_key)
            chain = load_qa_chain(llm, chain_type="stuff")
            answer = chain.run(input_documents=docs, question=query)
            st.markdown(f"### 🤖 Answer:\n{answer}")



