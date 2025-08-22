import asyncio

# Set event loop policy to allow event loops in any thread (required for grpc_asyncio in Streamlit)
try:
    import tornado.platform.asyncio
    asyncio.set_event_loop_policy(tornado.platform.asyncio.AnyThreadEventLoopPolicy())
except ImportError:
    if hasattr(asyncio, "WindowsSelectorEventLoopPolicy"):
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.vectorstores import FAISS
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
import faiss
import os

# Toggle backend-only debug logs
DEBUG = True  # set to False to silence prints

# Load environment variables from .env file
load_dotenv()

# Streamlit app title
st.title("Airceleo AI Support Chatbot")

# Initialize session state variables
if 'vectorstore' not in st.session_state:
    st.session_state.vectorstore = None
    st.session_state.chat_history = []

def initialize_vectorstore():
    """Initialize the vector store and load it into session state."""
    try:
        embedding_model = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        vectorstore = FAISS.load_local("faiss_index", embedding_model, allow_dangerous_deserialization=True)
        st.session_state.vectorstore = vectorstore
        if DEBUG:
            print("[DEBUG] Vectorstore initialized from 'faiss_index'")
    except Exception as e:
        st.error(f"Error initializing vector store: {e}")
        raise

def setup_retriever():
    """Set up the retriever for the vector store."""
    if st.session_state.vectorstore is None:
        st.error("Vectorstore is not initialized.")
        st.stop()
    try:
        retriever = st.session_state.vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 3})
        if DEBUG:
            print("[DEBUG] Retriever set with search_type='similarity', k=3")
        return retriever
    except Exception as e:
        st.error(f"Error setting up retriever: {e}")
        raise

def setup_llm():
    """Initialize the Language Model (LLM) for generating responses."""
    try:
        llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro", temperature=0, max_tokens=None, timeout=None)
        if DEBUG:
            print("[DEBUG] LLM initialized: gemini-1.5-pro, temperature=0")
        return llm
    except Exception as e:
        st.error(f"Error initializing LLM: {e}")
        raise

def setup_prompt_template():
    """Set up the system and human prompt template for the chat."""
    try:
        system_prompt = (
            "You are an assistant for question-answering tasks. "
            "Use the following pieces of retrieved context to answer "
            "the question. If you don't know the answer, say that you "
            "don't know. Use three sentences maximum and keep the "
            "answer concise."
            "\n\n"
            "{context}"
        )
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "{input}"),
        ])
        if DEBUG:
            print("[DEBUG] Prompt template prepared")
        return prompt
    except Exception as e:
        st.error(f"Error setting up prompt template: {e}")
        raise

def format_context_docs(docs):
    """Create a joined context string and a debug-friendly summary of docs."""
    joined = "\n\n".join(d.page_content for d in docs)
    if DEBUG:
        print("[DEBUG] Retrieved context documents:")
        for idx, d in enumerate(docs, start=1):
            meta = getattr(d, "metadata", {}) or {}
            snippet = (d.page_content or "")[:400].replace("\n", " ")
            print(f"  - Context #{idx} | metadata={meta} | text_snippet='{snippet}...'")
    return joined

def render_final_prompt_text(prompt_template, context_text, user_input):
    """Render the final prompt string for debugging."""
    messages = prompt_template.format_messages(context=context_text, input=user_input)
    if DEBUG:
        print("[DEBUG] Final rendered prompt:")
        for m in messages:
            role = getattr(m, "type", "message").upper()
            content = getattr(m, "content", "")
            print(f"[{role}]\n{content}\n")
    return messages

def process_query(query, retriever, llm, prompt):
    """Process the user query through the retrieval and generation pipeline with backend-only logs."""
    try:
        # Retrieve docs (so we can log them)
        retrieved_docs = retriever.get_relevant_documents(query)

        # Build context and log
        context_text = format_context_docs(retrieved_docs)

        # Render the final prompt and log
        rendered_messages = render_final_prompt_text(prompt, context_text, query)

        # Build the chains and invoke normally
        question_answer_chain = create_stuff_documents_chain(llm, prompt)
        rag_chain = create_retrieval_chain(retriever, question_answer_chain)
        response = rag_chain.invoke({"input": query})

        if DEBUG:
            print("[DEBUG] Raw LLM response object:")
            try:
                # response often is a dict; printing directly is ok
                print(response)
            except Exception:
                print("[DEBUG] Could not pretty print response")

        return response
    except Exception as e:
        if DEBUG:
            print(f"[DEBUG] Error processing query: {e}")
        st.error(f"Error processing query: {e}")
        raise

# Load or initialize vector store
if st.session_state.vectorstore is None:
    try:
        initialize_vectorstore()
    except Exception as e:
        st.error(f"Initialization failed: {e}")

# Set up retriever, LLM, and prompt template
try:
    retriever = setup_retriever()
    llm = setup_llm()
    prompt = setup_prompt_template()
except Exception as e:
    st.error(f"Setup failed: {e}")

# Chat interface
query = st.chat_input("Ask something:")
if query:
    st.session_state.chat_history.append({"user": query})

    with st.spinner("Processing your request..."):
        try:
            response = process_query(query, retriever, llm, prompt)
            answer = response.get("answer", "<no answer>")
            if DEBUG:
                print("[DEBUG] Assistant answer:", answer)

            st.session_state.chat_history.append({"assistant": answer})

            # Display chat history (no debug info)
            for chat in st.session_state.chat_history:
                if "user" in chat:
                    st.write(f"**You:** {chat['user']}")
                elif "assistant" in chat:
                    st.write(f"**Assistant:** {chat['assistant']}")
        except Exception as e:
            st.error(f"An error occurred during query processing: {e}")
