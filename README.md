# Local RAG Chat Application

A fully **local** Retrieval-Augmented Generation (RAG) application built with **LangChain**, **LM Studio**, and **FAISS**. No data ever leaves your machine.

## Tech Stack

| Component        | Technology                                |
|------------------|-------------------------------------------|
| LLM              | LM Studio (`google/gemma-3n-e4b`)         |
| Embeddings       | `nomic-ai/nomic-embed-text-v1.5-GGUF`     |
| Vector Database  | FAISS                                      |
| Framework        | LangChain (LCEL) + OpenAI-compatible API   |
| UI               | Streamlit                                  |

## Prerequisites

1. **Install LM Studio** — Download from [lmstudio.ai](https://lmstudio.ai)
2. **Load the required models in LM Studio:**
   - **LLM:** `google/gemma-3n-e4b`
   - **Embeddings:** `nomic-ai/nomic-embed-text-v1.5-GGUF`
3. **Start the LM Studio local server** (default: `http://127.0.0.1:1234/v1`)
4. **Python 3.10+** recommended

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Make sure LM Studio's local server is running on port 1234

# 3. Launch the app
streamlit run app.py
```

## Project Structure

```
langchain_application/
├── app.py              # Streamlit UI & chat interface
├── rag_engine.py       # Core RAG pipeline (load → split → embed → retrieve → generate)
├── config.py           # All configurable settings (models, chunking, paths)
├── requirements.txt    # Python dependencies
├── sample_data.txt     # Sample document for testing
├── faiss_index/        # Auto-generated FAISS index (after first ingestion)
└── README.md           # You are here
```

## How to Use

1. **Upload documents** via the sidebar (`.txt`, `.pdf`, `.docx`, `.md`)
2. Click **Ingest Documents** to process & store them in FAISS
3. **Ask questions** in the chat input — the LLM answers using your documents as context
4. Use **Load Existing Vector Store** to reload a previously ingested index

## Configuration

Edit `config.py` to change:

| Setting              | Default                                    | Description                          |
|----------------------|--------------------------------------------|--------------------------------------|
| `LLM_MODEL`          | `google/gemma-3n-e4b`                      | LM Studio model for generation       |
| `EMBEDDING_MODEL`    | `nomic-ai/nomic-embed-text-v1.5-GGUF`      | Embedding model                      |
| `LM_STUDIO_BASE_URL` | `http://127.0.0.1:1234/v1`                 | LM Studio server endpoint            |
| `CHUNK_SIZE`          | `1000`                                     | Characters per text chunk             |
| `CHUNK_OVERLAP`       | `200`                                      | Overlap between consecutive chunks    |
| `RETRIEVER_K`         | `4`                                        | Number of chunks retrieved per query  |
| `FAISS_INDEX_DIR`     | `./faiss_index`                            | Directory for persisted FAISS index   |

## Dependencies

```
langchain>=0.3.0
langchain-community>=0.3.0
langchain-openai>=0.2.0
faiss-cpu>=1.7.0
streamlit>=1.38.0
pypdf>=4.0.0
python-docx>=1.0.0
```

## License

MIT
