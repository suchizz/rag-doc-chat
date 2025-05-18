import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
import tempfile
import os

# Set your OpenAI API key
os.environ["OPENAI_API_KEY"] = "your-openai-api-key"  # <-- Replace this

st.set_page_config(page_title="RAG PDF Chatbot", layout="wide")
st.title("📄 RAG PDF Chatbot")
st.markdown("Upload a PDF and chat with it using LangChain + OpenAI")

uploaded_file = st.file_uploader("Upload a PDF", type="pdf")

if uploaded_file is not None:
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(uploaded_file.read())
            tmp_file_path = tmp_file.name

        # Load and split the document
        loader = PyPDFLoader(tmp_file_path)
        docs = loader.load()

        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = text_splitter.split_documents(docs)

        # Generate embeddings
        embeddings = OpenAIEmbeddings()
        vectorstore = FAISS.from_documents(chunks, embeddings)

        # Build retriever
        retriever = vectorstore.as_retriever()

        # Setup QA chain
        qa_chain = RetrievalQA.from_chain_type(
            llm=ChatOpenAI(model_name="gpt-3.5-turbo"),
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=True
        )

        # User query input
        query = st.text_input("Ask a question about the PDF")

        if query:
            with st.spinner("Thinking..."):
                result = qa_chain({"query": query})
                st.write("### Answer")
                st.write(result["result"])

    except Exception as e:
        st.error(f"Error: {e}")




