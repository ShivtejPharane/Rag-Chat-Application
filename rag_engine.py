"""
Core RAG engine — handles document ingestion, vector store management,
and question-answering via a local LM Studio LLM.
"""

import os
from typing import List

from langchain_community.document_loaders import TextLoader
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

import config




def load_document(file_path: str):
    """Load a single document based on its file extension."""
    ext = os.path.splitext(file_path)[1].lower()

    if ext == ".pdf":
        loader = PyPDFLoader(file_path)
    elif ext == ".docx":
        loader = Docx2txtLoader(file_path)
    elif ext in (".txt", ".md"):
        loader = TextLoader(file_path, encoding="utf-8")
    else:
        raise ValueError(f"Unsupported file type: {ext}")

    return loader.load()


def load_documents(file_paths: List[str]):
    """Load multiple documents and return a flat list of Document objects."""
    all_docs = []
    for path in file_paths:
        all_docs.extend(load_document(path))
    return all_docs




def split_documents(documents):
    """Split documents into smaller chunks for embedding."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
        length_function=len,
        add_start_index=True,
    )
    return text_splitter.split_documents(documents)




def get_embeddings():
    """Return the LM Studio embedding model instance (OpenAI-compatible API)."""
    return OpenAIEmbeddings(
        model=config.EMBEDDING_MODEL,
        openai_api_key="lm-studio",
        openai_api_base=config.LM_STUDIO_BASE_URL,
        check_embedding_ctx_length=False,
    )


def create_vector_store(chunks):
    """Create a FAISS vector store from chunks and save it to disk."""
    embeddings = get_embeddings()
    vectorstore = FAISS.from_documents(
        documents=chunks,
        embedding=embeddings,
    )
    # Persist to disk
    vectorstore.save_local(config.FAISS_INDEX_DIR)
    return vectorstore


def load_vector_store():
    """Load an existing FAISS vector store from disk."""
    embeddings = get_embeddings()
    vectorstore = FAISS.load_local(
        config.FAISS_INDEX_DIR,
        embeddings,
        allow_dangerous_deserialization=True,
    )
    return vectorstore


RAG_PROMPT_TEMPLATE = """\
You are a helpful AI assistant. Use the following retrieved context to answer \
the user's question. If you don't know the answer based on the context, say \
"I don't have enough information to answer that."

Context:
{context}

Question: {question}

Answer:"""


def format_docs(docs):
    """Join retrieved document pages into a single context string."""
    return "\n\n---\n\n".join(doc.page_content for doc in docs)


def get_rag_chain(vectorstore):
    """Build and return a LangChain LCEL RAG chain."""
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": config.RETRIEVER_K},
    )

    llm = ChatOpenAI(
        model=config.LLM_MODEL,
        base_url=config.LM_STUDIO_BASE_URL,
        api_key="lm-studio",
        temperature=0.3,
    )

    prompt = ChatPromptTemplate.from_template(RAG_PROMPT_TEMPLATE)

    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    return rag_chain

def ingest_files(file_paths: List[str]):
    """Full pipeline: load → split → embed → store.  Returns the vectorstore."""
    documents = load_documents(file_paths)
    chunks = split_documents(documents)
    vectorstore = create_vector_store(chunks)
    return vectorstore, len(documents), len(chunks)


def ask(question: str, vectorstore=None):
    """Ask a question against the vector store and return the answer."""
    if vectorstore is None:
        vectorstore = load_vector_store()
    chain = get_rag_chain(vectorstore)
    return chain.invoke(question)
