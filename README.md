# RAGQuery

**RAGQuery** is a document Q&A engine powered by Retrieval-Augmented Generation (RAG). Upload a document or paste text, ask questions, and get answers grounded in the actual source — with evidence and quality scores shown for every response.

Built with OpenAI, LangChain, FAISS, RAGAS, FastAPI, and Streamlit.

---

## Features

- **Document Upload** — supports `.txt`, `.pdf`, `.md`, `.docx`
- **Paste Text** — load raw text directly without a file
- **RAG Pipeline** — retrieves relevant chunks and grounds answers in the source
- **Evidence Display** — every answer shows the exact text it was based on
- **RAGAS Evaluation** — each response is scored for **Faithfulness** and **Answer Relevancy** in real time
- **Chat History** — all questions and answers visible in the session
- **API Key via UI** — enter your OpenAI key in the app, no environment variables needed
- **Docker-ready** — containerised with separate backend and frontend services

---

## Tech Stack

| Layer | Tools |
|---|---|
| LLM | OpenAI GPT-4.1 |
| Embeddings | `all-MiniLM-L6-v2` (HuggingFace, runs locally) |
| Vector Store | FAISS (in-memory) |
| RAG Framework | LangChain |
| Evaluation | RAGAS (faithfulness + answer relevancy) |
| Backend | FastAPI |
| Frontend | Streamlit |
| Containerisation | Docker, Docker Compose |

---

## How It Works

```
Document / Text
      │
      ▼
  Text Extraction (data_loader.py)
      │
      ▼
  Chunking — RecursiveCharacterTextSplitter (512 tokens, 50 overlap)
      │
      ▼
  Embeddings — all-MiniLM-L6-v2 → FAISS vector store
      │
      ▼
  Question → Similarity Search → Top 4 chunks
      │
      ▼
  GPT-4.1 (grounded prompt, context-only answers)
      │
      ▼
  Answer + Evidence + RAGAS scores (faithfulness, answer relevancy)
```

---

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/` | Health check |
| `POST` | `/load` | Upload a file and create a session |
| `POST` | `/load-text` | Load pasted text and create a session |
| `POST` | `/ask` | Ask a question against a loaded session |

### `/ask` Response Shape

```json
{
  "question": "What is the main argument?",
  "answer": "...",
  "evidence": ["chunk 1 text", "chunk 2 text"],
  "scores": {
    "faithfulness": 0.95,
    "answer_relevancy": 0.88
  }
}
```

---

## Setup Instructions

### 1. Clone the Repo

```bash
git clone https://github.com/MITI-Jing/ragquery.git
cd ragquery
```

### 2. Run with Docker (recommended)

```bash
docker compose up --build
```

- Backend: `http://localhost:8000`
- Frontend: `http://localhost:8501`

### 3. Run Locally (without Docker)

Create and activate a virtual environment:

```bash
python -m venv venv
source venv/bin/activate        # Windows: .\venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Start backend:

```bash
uvicorn backend.app:app --reload
```

Start frontend (new terminal):

```bash
streamlit run frontend/streamlit_app.py
```

---

## File Structure

```
ragquery/
├── backend/
│   ├── __init__.py
│   ├── app.py              # FastAPI app — /load, /load-text, /ask endpoints
│   ├── data_loader.py      # Text extraction for .txt, .pdf, .md, .docx
│   ├── qa_agent.py         # RetrievalQA chain with GPT-4.1 and grounding prompt
│   ├── rag_pipeline.py     # Chunking and FAISS vector store creation
│   ├── evaluator.py        # RAGAS evaluation (faithfulness + answer relevancy)
│   └── test_rag_qa.py      # Tests
├── frontend/
│   └── streamlit_app.py    # Streamlit UI with file upload, paste text, chat, scores
├── Dockerfile.backend
├── Dockerfile.frontend
├── docker-compose.yml
├── requirements.txt
└── README.md
```

---

## Author

Built by Jing Li — a career changer passionate about AI for real-world use cases.

## License

MIT License
