import streamlit as st
from PyPDF2 import PdfReader
import docx
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains import RetrievalQA

# Set up page
st.set_page_config(page_title="📚 Chat with Your Documents")
st.title("📚 Chat with Your Documents - RAG App")

# Load API Key securely from Streamlit Secrets
openai_api_key = st.secrets["OPENAI_API_KEY"]

# Upload documents
uploaded_files = st.file_uploader(
    "Upload PDF, DOCX, or TXT files",
    type=["pdf", "docx", "txt"],
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

    # Split text into chunks
    text_splitter = CharacterTextSplitter(separator="\n", chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_text(raw_text)

    # Create embeddings and vectorstore
    embeddings = OpenAIEmbeddings(api_key=openai_api_key)
    vectorstore = Chroma.from_texts(chunks, embedding=embeddings)

    # Set up QA chain
    retriever = vectorstore.as_retriever()
    qa_chain = RetrievalQA.from_chain_type(
        llm=ChatOpenAI(api_key=openai_api_key),
        retriever=retriever
    )

    st.success("✅ Documents processed. Ask your question below!")

    # User input
    query = st.text_input("Ask something about your documents:")

    if query:
        with st.spinner("Thinking..."):
            answer = qa_chain.run(query)
            st.markdown(f"**Answer:** {answer}")
else:
    st.info("Upload at least one document to get started.")

