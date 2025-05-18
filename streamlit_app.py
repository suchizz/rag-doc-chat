import streamlit as st
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS as CommunityFAISS
import os
import time

# ===== Embedding selection =====
USE_OPENAI = True

if USE_OPENAI:
    from langchain_openai import OpenAIEmbeddings, ChatOpenAI
    import openai
    openai.api_key = os.getenv("OPENAI_API_KEY")
    embeddings = OpenAIEmbeddings()
    llm = ChatOpenAI()
else:
    from langchain.embeddings import HuggingFaceEmbeddings
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    from langchain.chat_models import ChatOpenAI  # Placeholder if using HF

# ===== Document Upload =====
st.title("📄 RAG Doc Chat App")
uploaded_file = st.file_uploader("Upload a PDF", type="pdf")

if uploaded_file:
    loader = PyPDFLoader(uploaded_file.name)
    docs = loader.load()

    # ===== Text Splitting =====
    text_splitter = CharacterTextSplitter(chunk_size=300, chunk_overlap=30)
    chunks = text_splitter.split_documents(docs)

    # ===== FAISS Handling =====
    INDEX_PATH = "faiss_index"

    if os.path.exists(INDEX_PATH):
        st.info("📁 Loading cached FAISS index...")
        vectorstore = CommunityFAISS.load_local(INDEX_PATH, embeddings, allow_dangerous_deserialization=True)
    else:
        st.info("📚 Creating vectorstore (can take time)...")

        # Retry mechanism for embedding API rate limits
        texts = [chunk.page_content for chunk in chunks]
        embedded_chunks = []
        for text in texts:
            try:
                embedded_chunks.append(text)
                time.sleep(1)
            except Exception as e:
                st.error(f"Error embedding: {e}")
                time.sleep(5)

        vectorstore = CommunityFAISS.from_texts(texts=embedded_chunks, embedding=embeddings)
        vectorstore.save_local(INDEX_PATH)

    retriever = vectorstore.as_retriever()
    chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

    query = st.text_input("Ask a question based on the PDF:")

    if query:
        result = chain.run(query)
        st.write("### Answer:")
        st.write(result)




