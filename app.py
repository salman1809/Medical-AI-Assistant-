import streamlit as st
import os
import time

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain_mistralai import ChatMistralAI
from dotenv import load_dotenv

# -----------------------------
# LOAD ENV
# -----------------------------
load_dotenv()

# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(
    page_title="AI Medical Assistant",
    page_icon="🩺",
    layout="wide"
)

# -----------------------------
# MODERN CSS
# -----------------------------
st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #0f172a, #020617);
    color: white;
}
.main-title {
    text-align: center;
    font-size: 40px;
    font-weight: bold;
    color: #22c55e;
}
.sub-title {
    text-align: center;
    color: #94a3b8;
}
</style>
""", unsafe_allow_html=True)

# -----------------------------
# HEADER
# -----------------------------
st.markdown("<div class='main-title'>🩺 AI Medical Assistant</div>", unsafe_allow_html=True)
st.markdown("<div class='sub-title'>Ask medical questions with AI-powered insights</div>", unsafe_allow_html=True)

# -----------------------------
# LOAD LLM
# -----------------------------
@st.cache_resource
def load_llm():
    return ChatMistralAI(
        model="mistral-small",
        temperature=0.2
    )

llm = load_llm()

# -----------------------------
# LOAD VECTOR DB
# -----------------------------
@st.cache_resource
def load_vectorstore():
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    db_path = os.path.join(BASE_DIR, "faiss_medical_db")
    data_path = os.path.join(BASE_DIR, "full_dataset_clean.txt")

    if os.path.exists(db_path):
        return FAISS.load_local(
            db_path,
            embeddings,
            allow_dangerous_deserialization=True
        )

    loader = TextLoader(data_path, encoding="utf-8")
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=50
    )
    chunks = splitter.split_documents(docs)

    vector_store = FAISS.from_documents(chunks, embeddings)
    vector_store.save_local(db_path)

    return vector_store

vector_store = load_vectorstore()

retriever = vector_store.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 4}
)

# -----------------------------
# UPDATED PROMPT
# -----------------------------
prompt = ChatPromptTemplate.from_template("""
You are an expert AI medical assistant.

RULES:
- Answer ONLY from context
- If missing info say: "Not available in provided data"
- Do NOT hallucinate

FORMAT:
Condition / Medicine:
Overview:
Symptoms:
Treatment:
Precautions:

Context:
{context}

Question:
{question}

Answer:
""")

# -----------------------------
# CHAT STATE
# -----------------------------
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# -----------------------------
# DISPLAY CHAT
# -----------------------------
for role, message in st.session_state.chat_history:
    with st.chat_message(role):
        st.markdown(message)

# -----------------------------
# INPUT
# -----------------------------
user_input = st.chat_input("💬 Ask your medical question...")

# -----------------------------
# PROCESS
# -----------------------------
if user_input:
    st.session_state.chat_history.append(("user", user_input))

    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        with st.spinner("🤖 Thinking..."):

            docs = retriever.invoke(user_input)

            if not docs:
                response_text = "Not available in provided data."
            else:
                context = "\n\n".join([doc.page_content for doc in docs])

                final_prompt = prompt.invoke({
                    "context": context,
                    "question": user_input
                })

                response = llm.invoke(final_prompt)
                response_text = response.content

                # 🎯 FORMAT ENHANCEMENT (icons)
                response_text = response_text.replace("Condition / Medicine:", "🩺 **Condition / Medicine:**")
                response_text = response_text.replace("Overview:", "📖 **Overview:**")
                response_text = response_text.replace("Symptoms:", "⚠️ **Symptoms:**")
                response_text = response_text.replace("Treatment:", "💊 **Treatment:**")
                response_text = response_text.replace("Precautions:", "🛡️ **Precautions:**")

        # Typing animation
        placeholder = st.empty()
        full_text = ""

        for word in response_text.split():
            full_text += word + " "
            placeholder.markdown(full_text)
            time.sleep(0.02)

    st.session_state.chat_history.append(("assistant", response_text))

# -----------------------------
# FOOTER
# -----------------------------
st.markdown("""
<hr>
<p style='text-align:center;color:gray;'>
 Developed by Salman Raju
</p>
""", unsafe_allow_html=True)
