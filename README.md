
# Smart Research Agent

An AI-powered research assistant that combines Retrieval-Augmented Generation (RAG) with an autonomous tool-calling agent.

The system allows users to upload documents, search their content semantically, ask questions, generate summaries, and perform calculations using an LLM-powered agent.

## Features

- PDF and TXT document ingestion
- Text extraction and chunking
- Semantic search using Sentence Transformers
- Vector storage with ChromaDB
- Question answering with source references
- Local LLM integration using Ollama
- Tool-calling agent powered by Qwen 2.5
- Three executable tools:
  - `search_documents`
  - `summarize_document`
  - `calculate`
- REST API built with FastAPI
- Interactive API documentation with Swagger

## Architecture

```text
User
 |
 v
FastAPI
 |
 +----------------------+
 |                      |
 v                      v
RAG Pipeline         Research Agent
 |                      |
 v                      v
ChromaDB             Qwen 2.5
 |                      |
 v                      v
Retriever           Tool Selection
 |                      |
 v                      v
Context             Tool Execution
 |                      |
 +----------+-----------+
            |
            v
       Final Answer
```

## Tech Stack

| Component | Technology |
|---|---|
| Programming Language | Python |
| API Framework | FastAPI |
| LLM | Qwen 2.5 3B |
| LLM Runtime | Ollama |
| Embeddings | Sentence Transformers |
| Vector Database | ChromaDB |
| PDF Processing | PyPDF |
| HTTP Client | HTTPX |

## Project Structure

```text
smart-research-agent/
│
├── app/
│   ├── agent/
│   │   ├── __init__.py
│   │   ├── agent.py
│   │   └── tools.py
│   │
│   ├── api/
│   │   ├── __init__.py
│   │   ├── routes.py
│   │   └── schemas.py
│   │
│   ├── core/
│   │   ├── __init__.py
│   │   └── config.py
│   │
│   ├── llm/
│   │   ├── __init__.py
│   │   └── ollama_client.py
│   │
│   ├── rag/
│   │   ├── __init__.py
│   │   ├── loader.py
│   │   ├── chunker.py
│   │   ├── embeddings.py
│   │   ├── vector_store.py
│   │   ├── retriever.py
│   │   ├── pipeline.py
│   │   └── qa.py
│   │
│   └── main.py
│
├── data/
│   ├── documents/
│   └── chroma/
│
├── tests/
│
├── .env.example
├── .gitignore
├── requirements.txt
└── README.md
```

## Installation

### 1. Clone the repository

```bash
git clone YOUR_REPOSITORY_URL
cd smart-research-agent
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

Activate it on Windows:

```powershell
.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure environment variables

On Windows:

```powershell
Copy-Item .env.example .env
```

Review the configuration in `.env`.

### 5. Install Ollama model

Install Ollama, then run:

```bash
ollama pull qwen2.5:3b
```

### 6. Run the application

```bash
uvicorn app.main:app --reload
```

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/` | Application status |
| GET | `/health` | Health check |
| POST | `/api/v1/documents/upload` | Upload PDF or TXT |
| POST | `/api/v1/documents/search` | Semantic document search |
| POST | `/api/v1/documents/ask` | RAG question answering |
| POST | `/api/v1/agent/run` | Run the research agent |

## API Documentation

After starting the application, open:

```text
http://127.0.0.1:8000/docs
```

## Example Agent Requests

### Calculate

```json
{
  "message": "Calculate 125 * 24"
}
```

### Search Documents

```json
{
  "message": "Search my documents for information about Python"
}
```

### Summarize Documents

```json
{
  "message": "Summarize all my uploaded documents"
}
```

## Current Limitations

- Scanned PDFs and image-only documents are not supported by the current text extraction pipeline.
- The default embedding model is English-oriented; Persian retrieval quality may be limited.
- Long-document summarization currently uses a context-length limit.
- Retrieval relevance thresholds and document deletion management need further improvement.
- The application is currently designed as a local MVP.

## Future Improvements

- Multilingual embedding model support
- Better retrieval relevance filtering
- Document management and deletion
- Automated tests
- Streaming responses
- Frontend interface
- Improved long-document summarization

## License

MIT License
