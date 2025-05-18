import streamlit as st
from PyPDF2 import PdfReader
import docx
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA

st.set_page_config(page_title="📚 Chat with Your Documents", page_icon="📚", layout="wide")
st.title("📚 Chat with Your Documents - RAG App")

# Input for OpenAI API Key
openai_api_key = st.text_input("Enter your OpenAI API Key:", type="password")

if not openai_api_key or not openai_api_key.startswith("sk-"):
    st.error("Please enter a valid OpenAI API Key.")
    st.stop()

uploaded_files = st.file_uploader(
    "Upload PDF, DOCX, or TXT files",
    type=['pdf', 'docx', 'txt'],
    accept_multiple_files=True
)

@st.cache_data(show_spinner=False)
def load_document(file):
    text = ""
    if file.type == "application/pdf":
        try:
            reader = PdfReader(file)
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
        except Exception as e:
            st.warning(f"Could not read PDF file {file.name}: {e}")
    elif file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
        try:
            doc = docx.Document(file)
            for para in doc.paragraphs:
                text += para.text + "\n"
        except Exception as e:
            st.warning(f"Could not read DOCX file {file.name}: {e}")
    elif file.type == "text/plain":
        try:
            text = file.read().decode("utf-8")
        except Exception as e:
            st.warning(f"Could not read TXT file {file.name}: {e}")
    else:
        st.warning(f"Unsupported file type {file.type} for file {file.name}")
    return text

if uploaded_files:
    with st.spinner("Loading and processing documents..."):
        raw_text = ""
        for file in uploaded_files:
            raw_text += load_document(file) + "\n"

        if not raw_text.strip():
            st.error("Failed to extract text from the uploaded documents.")
            st.stop()

        text_splitter = CharacterTextSplitter(separator="\n", chunk_size=1000, chunk_overlap=200)
        chunks = text_splitter.split_text(raw_text)

        embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)

        vectorstore = FAISS.from_texts(chunks, embeddings)

        retriever = vectorstore.as_retriever()

        qa_chain = RetrievalQA.from_chain_type(
            llm=ChatOpenAI(openai_api_key=openai_api_key, temperature=0),
            retriever=retriever,
            return_source_documents=False,
        )

    st.success("✅ Documents processed! You can now ask questions about your documents.")

    query = st.text_input("Ask a question about your documents:")

    if query:
        with st.spinner("Generating answer..."):
            answer = qa_chain.run(query)
        st.markdown(f"**Answer:** {answer}")

else:
    st.info("Please upload at least one document (PDF, DOCX, or TXT) to start chatting.")

st.markdown(
    """
    ---
    Powered by OpenAI & LangChain.  
    Upload documents and chat with their content using Retrieval Augmented Generation!
    """
)



