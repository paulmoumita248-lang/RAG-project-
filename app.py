import streamlit as st
from dotenv import load_dotenv
import tempfile
import os

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# ---------------- LOAD ENV ----------------
load_dotenv()

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="📚 Mistral RAG Chatbot",
    page_icon="🤖",
    layout="wide"
)

# ---------------- HEADER ----------------
st.markdown(
    """
    <h1 style='text-align: center; color: #4CAF50;'>
        🤖📚 Mistral RAG Chatbot 📚🤖
    </h1>

    <h4 style='text-align: center;'>
        Upload your PDF and ask questions ✨
    </h4>
    """,
    unsafe_allow_html=True
)

st.divider()

# ---------------- SIDEBAR ----------------
with st.sidebar:

    st.header("📂 Upload PDF")

    uploaded_file = st.file_uploader(
        "Choose PDF File",
        type=["pdf"]
    )

    st.markdown("---")

    st.markdown("### 🌟 Features")

    st.markdown("""
    ✅ Upload PDF Books  
    ✅ Fast Mistral Small Model  
    ✅ Chroma Vector Database  
    ✅ Ask Questions  
    ✅ AI Powered Answers  
    """)

# ---------------- CACHE EMBEDDING MODEL ----------------
@st.cache_resource
def load_embedding_model():

    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

embedding_model = load_embedding_model()

# ---------------- CACHE LLM ----------------
@st.cache_resource
def load_llm():

    return ChatMistralAI(
        model="mistral-small-2506",
        temperature=0.7
    )

llm = load_llm()

# ---------------- PROMPT ----------------
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are a helpful AI assistant.

Use ONLY the provided context to answer the question.

If the answer is not present in the context,
say:
"I could not find the answer in the document."
"""
        ),

        (
            "human",
            """Context:
{context}

Question:
{question}
"""
        )
    ]
)

# ---------------- SESSION STATE ----------------
if "messages" not in st.session_state:
    st.session_state.messages = []

if "retriever" not in st.session_state:
    st.session_state.retriever = None

# ---------------- FILE PROCESSING ----------------
if uploaded_file is not None:

    file_name = uploaded_file.name

    db_path = f"chroma_db/{file_name}"

    # ---------------- CHECK EXISTING DATABASE ----------------
    if os.path.exists(db_path):

        st.success(
            f"✅ '{file_name}' already exists in database.\n\n"
            f"You can directly ask questions without re-uploading."
        )

        vectorstore = Chroma(
            persist_directory=db_path,
            embedding_function=embedding_model
        )

    else:

        with st.spinner("📖 Processing PDF..."):

            with tempfile.NamedTemporaryFile(
                delete=False,
                suffix=".pdf"
            ) as tmp_file:

                tmp_file.write(uploaded_file.getvalue())

                pdf_path = tmp_file.name

            loader = PyPDFLoader(pdf_path)

            documents = loader.load()

            splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200
            )

            chunks = splitter.split_documents(documents)

            vectorstore = Chroma.from_documents(
                documents=chunks,
                embedding=embedding_model,
                persist_directory=db_path
            )

        st.success("✅ PDF Uploaded & Stored Successfully")

    # ---------------- RETRIEVER ----------------
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={
            "k": 2
        }
    )

    st.session_state.retriever = retriever

# ---------------- CHAT SECTION ----------------
if st.session_state.retriever is not None:

    st.markdown("## 💬 Chat With Your PDF")

    # ---------------- DISPLAY CHAT ----------------
    for role, content in st.session_state.messages:

        with st.chat_message(role):
            st.write(content)

    # ---------------- CHAT INPUT ----------------
    query = st.chat_input(
        "Ask something from your PDF... 📚"
    )

    if query:

        # ---------------- EXIT OPTION ----------------
        if query == "0":

            st.session_state.messages = []

            st.success("✅ Chat Ended")

            st.stop()

        # ---------------- USER MESSAGE ----------------
        st.session_state.messages.append(
            ("user", query)
        )

        with st.chat_message("user"):
            st.write(query)

        # ---------------- SIMPLE GREETINGS ----------------
        greetings = [
            "hi",
            "hello",
            "hey",
            "hii"
        ]

        if query.lower() in greetings:

            response_text = (
                "Hello 👋\n\n"
                "Ask me anything from your PDF 📚"
            )

        else:

            with st.spinner("🔍 Searching Document..."):

                docs = st.session_state.retriever.invoke(query)

                context = "\n\n".join(
                    [doc.page_content for doc in docs]
                )

                final_prompt = prompt.invoke(
                    {
                        "context": context,
                        "question": query
                    }
                )

            with st.spinner("🤖 Generating Answer..."):

                response = llm.invoke(final_prompt)

                response_text = response.content

        # ---------------- STORE RESPONSE ----------------
        st.session_state.messages.append(
            ("assistant", response_text)
        )

        # ---------------- DISPLAY RESPONSE ----------------
        with st.chat_message("assistant"):
            st.write("🤖 " + response_text)

else:

    st.info(
        "📚 Upload a PDF file to start chatting."
    )
