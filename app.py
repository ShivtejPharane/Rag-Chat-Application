

import os
import tempfile
import streamlit as st
from rag_engine import ingest_files, ask, load_vector_store

import config

st.set_page_config(
    page_title="Local RAG Chat",
    page_icon="X",
    layout="wide",
    initial_sidebar_state="expanded",
)


if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "doc_count" not in st.session_state:
    st.session_state.doc_count = 0
if "chunk_count" not in st.session_state:
    st.session_state.chunk_count = 0

with st.sidebar:
    st.header("Settings")
    st.divider()

    st.subheader(" Model Info")
    st.write(f"**LLM:** `{config.LLM_MODEL}`")
    st.write(f"**Embeddings:** `{config.EMBEDDING_MODEL}`")
    st.write(f"**Vector DB:** FAISS")
    st.write(f"**Chunks retrieved:** `{config.RETRIEVER_K}`")

    st.divider()
    st.subheader("Upload Documents")
    st.caption("Supported: `.txt`, `.pdf`, `.docx`, `.md`")

    uploaded_files = st.file_uploader(
        "Drop your files here",
        type=["txt", "pdf", "docx", "md"],
        accept_multiple_files=True,
        label_visibility="collapsed",
    )

    if uploaded_files:
        if st.button(" Ingest Documents", use_container_width=True):
            with st.spinner("Processing documents..."):
                # Save uploaded files to temp directory
                temp_paths = []
                for uploaded in uploaded_files:
                    tmp = tempfile.NamedTemporaryFile(
                        delete=False,
                        suffix=os.path.splitext(uploaded.name)[1],
                    )
                    tmp.write(uploaded.read())
                    tmp.close()
                    temp_paths.append(tmp.name)

                # Ingest
                vectorstore, doc_count, chunk_count = ingest_files(temp_paths)
                st.session_state.vectorstore = vectorstore
                st.session_state.doc_count = doc_count
                st.session_state.chunk_count = chunk_count

                # Cleanup temp files
                for p in temp_paths:
                    os.unlink(p)

            st.success(" Documents ingested successfully!")

    st.divider()

    # Load existing vector store
    if os.path.exists(config.FAISS_INDEX_DIR):
        if st.button(" Load Existing Vector Store", use_container_width=True):
            with st.spinner("Loading..."):
                st.session_state.vectorstore = load_vector_store()
            st.success("Loaded existing vector store!")

    # Clear chat
    if st.button(" Clear Chat", use_container_width=True):
        st.session_state.chat_history = []
        st.rerun()

st.title(" Local RAG Chat")
st.caption("Chat with your documents using a fully local LLM — no data leaves your machine.")

# Stats row
col1, col2, col3 = st.columns(3)
col1.metric("Documents Loaded", st.session_state.doc_count)
col2.metric("Text Chunks", st.session_state.chunk_count)
col3.metric("System Status", " Ready" if st.session_state.vectorstore else " No docs")

st.divider()


for role, message in st.session_state.chat_history:
    with st.chat_message("user" if role == "user" else "assistant"):
        st.write(message)

question = st.chat_input("Ask a question about your documents...")

if question:
    # Show user message
    st.session_state.chat_history.append(("user", question))
    with st.chat_message("user"):
        st.write(question)

    if st.session_state.vectorstore is None:
        answer = " Please upload and ingest documents first using the sidebar."
    else:
        with st.spinner("Thinking..."):
            try:
                answer = ask(question, st.session_state.vectorstore)
            except Exception as e:
                answer = f" Error: {str(e)}"

    st.session_state.chat_history.append(("bot", answer))
    with st.chat_message("assistant"):
        st.write(answer)
    st.rerun()
