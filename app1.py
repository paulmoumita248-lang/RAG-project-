import streamlit as st
from dotenv import load_dotenv

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate

# ---------------- LOAD ENV ----------------
load_dotenv()

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Mistral RAG Chatbot",
    page_icon="🤖",
    layout="centered"
)

# ---------------- TITLE ----------------
st.markdown(
    """
    <h1 style='text-align: center; color: #4CAF50;'>
        🤖 Mistral RAG Chatbot
    </h1>
    <h4 style='text-align: center;'>
        📚 Ask Questions From Your Documents
    </h4>
    """,
    unsafe_allow_html=True
)

# ---------------- LOAD EMBEDDING MODEL ----------------
@st.cache_resource
def load_embedding_model():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

embedding_model = load_embedding_model()

# ---------------- LOAD VECTORSTORE ----------------
@st.cache_resource
def load_vectorstore():
    return Chroma(
        persist_directory="chroma_db",
        embedding_function=embedding_model
    )

vectorstore = load_vectorstore()

# ---------------- RETRIEVER ----------------
retriever = vectorstore.as_retriever(
    search_type = "mmr",
    search_kwargs = {
        "k" : 4,
        "fetch_k":10,
        "lambda_mult" :0.5
    }
)

# ---------------- LOAD LLM ----------------
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

# ---------------- DISPLAY CHAT ----------------
for role, content in st.session_state.messages:

    with st.chat_message(role):
        st.write(content)

# ---------------- USER INPUT ----------------
query = st.chat_input("Ask a question...")

if query:

    # Exit Option
    if query == "0":
        st.session_state.messages = []
        st.success("✅ Chat Ended")
        st.stop()

    # Store User Message
    st.session_state.messages.append(("user", query))

    with st.chat_message("user"):
        st.write(query)

    # ---------------- SIMPLE CHAT HANDLING ----------------
    greetings = [
        "hi",
        "hello",
        "hey",
        "hii",
        "good morning",
        "good evening"
    ]

    if query.lower() in greetings:

        response_text = (
            "Hello 👋\n\n"
            "How can I help you with your document?"
        )

    else:

        # ---------------- RETRIEVE DOCUMENTS ----------------
        with st.spinner("🔍 Searching Document..."):

            docs = retriever.invoke(query)

            context = "\n\n".join(
                [doc.page_content for doc in docs]
            )

            final_prompt = prompt.invoke(
                {
                    "context": context,
                    "question": query
                }
            )

        # ---------------- LLM RESPONSE ----------------
        with st.spinner("🤖 Generating Answer..."):

            response = llm.invoke(final_prompt)

            response_text = response.content

    # ---------------- STORE RESPONSE ----------------
    st.session_state.messages.append(
        ("assistant", response_text)
    )

    # ---------------- DISPLAY RESPONSE ----------------
    with st.chat_message("assistant"):
        st.write(response_text)