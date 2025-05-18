import streamlit as st
import os
from langchain_openai import OpenAIEmbeddings
from PyPDF2 import PdfReader
import docx
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI

st.set_page_config(page_title="RAG Chat with Docs")
st.title("📚 Chat with Your Documents - RAG App")

# 1. Get OpenAI API key input from user
openai_api_key = st.text_input("Enter your OpenAI API Key:", type="password")

# 2. Validate the API key before proceeding
if not openai_api_key or not openai_api_key.startswith("sk-"):
    st.error("Please enter a valid OpenAI API key that starts with 'sk-'")
    st.stop()  # stops the app here if invalid key
os.environ["OPENAI_API_KEY"] = openai_api_key
embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")
uploaded_files = st.file_uploader(
    "Upload PDF, DOCX, or TXT files",
    type=['pdf', 'docx', 'txt'],
    accept_multiple_files=True
)

@st.cache_data(show_spinner=False)
def load_document(file):
    text = ""
    if file.type == "application/pdf":
        reader = PdfReader(file)
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text
    elif file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
        doc = docx.Document(file)
        for para in doc.paragraphs:
            text += para.text + "\n"
    elif file.type == "text/plain":
        text = file.read().decode("utf-8")
    return text

if uploaded_files:
    raw_text = ""
    for file in uploaded_files:
        raw_text += load_document(file)

    # Text splitting
    text_splitter = CharacterTextSplitter(separator="\n", chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_text(raw_text)

    # Vector store using Chroma
    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
    vectorstore = Chroma.from_texts(chunks, embedding=embeddings)
    retriever = vectorstore.as_retriever()

    qa_chain = RetrievalQA.from_chain_type(
        llm=ChatOpenAI(openai_api_key=openai_api_key),
        retriever=retriever
    )

    st.success("✅ Documents processed. You can now ask questions below.")
    query = st.text_input("Ask something about your documents:")

    if query:
        with st.spinner("Searching..."):
            result = qa_chain.run(query)
            st.markdown(f"**Answer:** {result}")
else:
    st.info("Please upload documents and provide your OpenAI API key to continue.")

