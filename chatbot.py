import os, time, streamlit as st
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.vectorstores import FAISS
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from google.api_core.exceptions import ResourceExhausted

# ─── Load API Key ───────────────────────────────────────────────────────────────
load_dotenv()
if "GOOGLE_API_KEY" not in os.environ:
    st.error("🔑 Please set GOOGLE_API_KEY in your .env")
    st.stop()

# ─── App Config ─────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Smart AI Chatbot", layout="centered")
st.title("🤖 Smart AI Chatbot")

# ─── Session State ──────────────────────────────────────────────────────────────
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
    st.session_state.chat_history = []

# ─── Init & Setup ───────────────────────────────────────────────────────────────
def initialize_vectorstore():
    emb = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vs  = FAISS.load_local("faiss_index", emb, allow_dangerous_deserialization=True)
    st.session_state.vectorstore = vs

def setup_retriever():
    return st.session_state.vectorstore.as_retriever(search_type="similarity", search_kwargs={"k":10})

def setup_llm():
    return ChatGoogleGenerativeAI(
        model="models/gemini-1.5-flash",  # <- swapped in here
        temperature=0,
        max_tokens=None,
        timeout=None
    )

def setup_prompt():
    system = (
        "You are an assistant for question-answering tasks. "
        "Use the retrieved context to answer concisely (max 3 sentences). "
        "If you don't know, say so.\n\n{context}"
    )
    return ChatPromptTemplate.from_messages([("system", system), ("human", "{input}")])

def process_query(query, retriever, llm, prompt, retries=5):
    qa_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(retriever, qa_chain)
    delay = 1
    for i in range(1, retries+1):
        try:
            return rag_chain.invoke({"input": query})
        except ResourceExhausted:
            st.warning(f"429 on attempt {i}, retrying in {delay}s…")
            time.sleep(delay)
            delay = min(delay*2, 30)
    st.error("Quota exhausted after retries. Check your plan.")
    st.stop()

# ─── Bootstrap if needed ────────────────────────────────────────────────────────
if st.session_state.vectorstore is None:
    initialize_vectorstore()

retriever = setup_retriever()
llm       = setup_llm()
prompt    = setup_prompt()

# ─── Chat UI ────────────────────────────────────────────────────────────────────
msg = st.chat_input("Ask me anything…")
if msg:
    st.session_state.chat_history.append({"role":"user","content":msg})
    with st.spinner("Thinking…"):
        res = process_query(msg, retriever, llm, prompt)
        ans = res.get("answer","<no answer>")
    st.session_state.chat_history.append({"role":"assistant","content":ans})

for turn in st.session_state.chat_history:
    st.chat_message(turn["role"]).write(turn["content"])
