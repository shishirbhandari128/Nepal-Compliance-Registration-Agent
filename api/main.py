"""
api/main.py
-----------
FastAPI application.

docs_dir, mode, max_files are all server-side config —
the frontend never sends these. They come from environment variables.

Endpoints:
  GET  /health                 — liveness + config info
  POST /chat                   — ask a question
  GET  /documents              — list available PDFs with summaries
  GET  /session/{session_id}   — get conversation history
  DELETE /session/{session_id} — clear a session
"""

import os
import uuid
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from src.graph import build_graph
from src.state import PipelineState
from src.memory import get_session, delete_session
from src.tools import get_or_build_summary
from src.pdf_loader import list_pdf_files

from .schemas import (
    ChatRequest, ChatResponse,
    DocumentListResponse, DocumentInfo,
    SessionHistoryResponse, ClearSessionResponse,
    HealthResponse,
)

# ── Server-side config (never sent by the frontend) ──────────────────────────
DOCS_DIR  = os.getenv("DOCS_DIR", ".")       # where PDFs live
MAX_FILES = int(os.getenv("MAX_FILES", "3")) # router ceiling

app = FastAPI(
    title="Nepal Compliance RAG API",
    description="Multi-agent RAG system for Nepal business compliance questions.",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Compile graph once at startup — not per request
_graph = build_graph()


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse, tags=["System"])
def health():
    """Liveness check — also tells the frontend where docs are loaded from."""
    return HealthResponse(
        status="ok",
        version="1.0.0",
        docs_dir=str(Path(DOCS_DIR).resolve()),
    )


@app.post("/chat", response_model=ChatResponse, tags=["Chat"])
def chat(request: ChatRequest):
    """
    Ask a compliance question.
    Pass session_id to continue a previous conversation.
    All routing decisions (mode, max_files, docs_dir) are server-side.
    """
    session_id = request.session_id or str(uuid.uuid4())
    docs_dir   = Path(DOCS_DIR).resolve()

    if not list_pdf_files(docs_dir):
        raise HTTPException(
            status_code=400,
            detail=f"No PDF files found in configured docs directory: {docs_dir}",
        )

    initial_state: PipelineState = {
        "question":           request.question,
        "docs_dir":           str(docs_dir),
        "mode":               "routed",      # always routed
        "max_files":          MAX_FILES,      # from server env
        "save_json":          f"outputs/{session_id}.json",
        "session_id":         session_id,
        "sub_questions":      [],
        "resolved_question":  "",
        "selected_pdfs":      [],
        "routing_reason":     "",
        "per_doc_answers":    [],
        "specialist_outputs": {},
        "final_answer":       "",
        "all_citations":      [],
    }

    try:
        result = _graph.invoke(initial_state)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    files_used = [Path(p).name for p in result.get("selected_pdfs", [])]

    return ChatResponse(
        session_id=         session_id,
        question=           request.question,
        resolved_question=  result.get("resolved_question", request.question),
        answer=             result.get("final_answer", ""),
        sub_questions=      result.get("sub_questions", []),
        citations=          result.get("all_citations", []),
        files_used=         files_used,
        files_used_count=   len(files_used),
        routing_reason=     result.get("routing_reason", ""),
        specialist_outputs= result.get("specialist_outputs", {}),
    )


@app.get("/documents", response_model=DocumentListResponse, tags=["Documents"])
def list_documents():
    """List all available PDFs with their cached content summaries."""
    docs_dir = Path(DOCS_DIR).resolve()
    pdfs     = list_pdf_files(docs_dir)
    if not pdfs:
        raise HTTPException(status_code=404, detail=f"No PDFs found in: {docs_dir}")

    return DocumentListResponse(
        docs_dir=str(docs_dir),
        documents=[
            DocumentInfo(filename=p.name, summary=get_or_build_summary(p))
            for p in pdfs
        ],
    )


@app.get("/session/{session_id}", response_model=SessionHistoryResponse, tags=["Session"])
def get_history(session_id: str):
    session = get_session(session_id)
    history = session.get_history()
    return SessionHistoryResponse(
        session_id=session_id,
        turns=len(history),
        history=history,
    )


@app.delete("/session/{session_id}", response_model=ClearSessionResponse, tags=["Session"])
def clear_session(session_id: str):
    delete_session(session_id)
    return ClearSessionResponse(session_id=session_id, cleared=True)
