import uuid
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from pydantic import BaseModel
from .data_loader import extract_text
from .rag_pipeline import chunk_article, create_vectorstore
from .qa_agent import build_agent
from .evaluator import evaluate_rag

app = FastAPI()

# In-memory session store: session_id -> {executor, retriever, api_key}
sessions = {}


class QuestionRequest(BaseModel):
    session_id: str
    question: str


class LoadTextRequest(BaseModel):
    text: str
    api_key: str


@app.get("/")
def read_root():
    return {"message": "Welcome to RAGQuery API"}


def _create_session(text: str, api_key: str) -> dict:
    chunks = chunk_article(text)
    vectorstore = create_vectorstore(chunks)
    executor, retriever = build_agent(vectorstore, api_key)
    session_id = str(uuid.uuid4())
    sessions[session_id] = {"executor": executor, "retriever": retriever, "api_key": api_key}
    return {"session_id": session_id, "chunks": len(chunks)}


@app.post("/load")
async def load_document(
    file: UploadFile = File(...),
    api_key: str = Form(...)
):
    try:
        file_bytes = await file.read()
        text = extract_text(file_bytes, file.filename)
        if not text.strip():
            raise HTTPException(status_code=400, detail="Document appears to be empty or unreadable.")
        result = _create_session(text, api_key)
        return {**result, "filename": file.filename}
    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to process document: {str(e)}")


@app.post("/load-text")
def load_text(req: LoadTextRequest):
    try:
        if not req.text.strip():
            raise HTTPException(status_code=400, detail="Text is empty.")
        result = _create_session(req.text, req.api_key)
        return {**result, "filename": "pasted text"}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to process text: {str(e)}")


@app.post("/ask")
def ask_question(req: QuestionRequest):
    session = sessions.get(req.session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found. Please load a document first.")
    try:
        result = session["executor"].invoke({"query": req.question})
        answer = result["result"]

        # Extract retrieved contexts from source documents
        contexts = [doc.page_content for doc in result.get("source_documents", [])]

        # RAGAS evaluation
        scores = evaluate_rag(req.question, answer, contexts, session["api_key"])

        return {
            "question": req.question,
            "answer": answer,
            "evidence": contexts,
            "scores": scores,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to answer question: {str(e)}")
