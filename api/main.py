"""
api/main.py
-----------
FastAPI application with streaming support.

Endpoints:
  GET  /health                    — liveness check
  POST /chat/stream               — streaming chat (SSE)
  POST /chat                      — non-streaming chat (kept for compatibility)
  GET  /documents                 — list PDFs with summaries
  GET  /session/{id}              — conversation history
  DELETE /session/{id}            — clear session
  POST /register/chat             — registration agent
  DELETE /register/{id}           — clear registration session
"""

import asyncio
import json
import os
import uuid
from pathlib import Path
from typing import AsyncGenerator

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from src.graph import build_graph
from src.state import PipelineState
from src.memory import get_session, delete_session
from src.tools import get_or_build_summary
from src.pdf_loader import list_pdf_files
from src.config import OPENAI_MODEL

from .schemas import (
    ChatRequest, ChatResponse,
    DocumentListResponse, DocumentInfo,
    SessionHistoryResponse, ClearSessionResponse,
    HealthResponse,
    RegistrationMessage, RegistrationResponse,
)

DOCS_DIR  = os.getenv("DOCS_DIR", ".")
MAX_FILES = int(os.getenv("MAX_FILES", "3"))

app = FastAPI(
    title="Nepal Compliance RAG API",
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

_graph = build_graph()


                                                                                

def sse(event: str, data: dict) -> str:
    """Format a single SSE event."""
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


                                                                                 

async def stream_chat(request: ChatRequest) -> AsyncGenerator[str, None]:
    """
    Run the full pipeline, streaming progress events then the final answer
    token by token using OpenAI streaming.

    SSE event types:
      progress  — { stage, message }           pipeline progress updates
      meta      — { session_id, sub_questions, files_used, routing_reason }
      token     — { text }                     answer token
      sources   — { citations, specialist_outputs }
      done      — {}                           stream complete
      error     — { message }                  error occurred
    """
    from langchain_openai import ChatOpenAI
    from langchain_core.prompts import ChatPromptTemplate

    session_id = request.session_id or str(uuid.uuid4())
    docs_dir   = Path(DOCS_DIR).resolve()

    if not list_pdf_files(docs_dir):
        yield sse("error", {"message": f"No PDF files found in: {docs_dir}"})
        return

                                                                                 
                                                                                      

    yield sse("progress", {"stage": "decompose",  "message": "Breaking down your question…"})
    await asyncio.sleep(0)                                 

    initial_state: PipelineState = {
        "question":           request.question,
        "docs_dir":           str(docs_dir),
        "mode":               "routed",
        "max_files":          MAX_FILES,
        "save_json":          f"outputs/{session_id}.json",
        "session_id":         session_id,
        "sub_questions":      [],
        "resolved_question":  "",
        "selected_pdfs":      [],
        "routing_reason":     "",
        "per_doc_answers":    [],
        "all_docs":           [],
        "specialist_outputs": {},
        "final_answer":       "",
        "all_citations":      [],
    }

    try:
                                                                
        loop   = asyncio.get_event_loop()
        result = await loop.run_in_executor(None, _graph.invoke, initial_state)
    except Exception as e:
        yield sse("error", {"message": str(e)})
        return

    sub_questions = result.get("sub_questions", [])
    files_used    = [Path(p).name for p in result.get("selected_pdfs", [])]
    citations     = result.get("all_citations", [])
    spec_outputs  = result.get("specialist_outputs", {})
    resolved_q    = result.get("resolved_question", request.question)

                                                 
    yield sse("progress", {"stage": "route",     "message": f"Selected {len(files_used)} document(s)…"})
    await asyncio.sleep(0)
    yield sse("progress", {"stage": "retrieve",  "message": f"Retrieved and reranked evidence…"})
    await asyncio.sleep(0)
    yield sse("progress", {"stage": "specialists","message": "Specialists analysed the evidence…"})
    await asyncio.sleep(0)

                                                                
    yield sse("meta", {
        "session_id":         session_id,
        "sub_questions":      sub_questions,
        "files_used":         files_used,
        "routing_reason":     result.get("routing_reason", ""),
        "resolved_question":  resolved_q,
    })
    await asyncio.sleep(0)

    yield sse("progress", {"stage": "faithfulness", "message": "Checking answer for accuracy…"})
    await asyncio.sleep(0)
    yield sse("progress", {"stage": "stream", "message": "Generating answer…"})
    await asyncio.sleep(0)

                                                                                
                                                                  
    from src.tools import format_per_doc_block, build_sources_block
    from langchain_core.prompts import ChatPromptTemplate

    per_doc = result.get("per_doc_answers", [])

    MERGER_PROMPT = ChatPromptTemplate.from_template(
        """You are a senior Nepal compliance analyst. Merge the per-document answers
and specialist perspectives into one final authoritative answer.

User question:
{question}

Base answers from documents:
{base_answer}

Specialist perspectives:
{specialist_outputs}

Instructions:
- Decide your own section headings based on what the question needs.
- Integrate specialist insights — do not list them separately.
- Rewrite everything in your own words.
- Do NOT write any citations or filenames — sources are appended automatically.
- Highlight critical steps or risks prominently.
"""
    )

    base_answer  = format_per_doc_block(per_doc) if len(per_doc) > 1 else (per_doc[0]["answer"].split("\nSources:")[0].strip() if per_doc else "")
    spec_block   = "\n\n".join(f"=== {role} ===\n{text}" for role, text in spec_outputs.items())

    llm    = ChatOpenAI(model=OPENAI_MODEL, temperature=0.1, streaming=True)
    prompt = MERGER_PROMPT.format_messages(
        question=resolved_q,
        base_answer=base_answer,
        specialist_outputs=spec_block,
    )

    full_answer = ""
    async for chunk in llm.astream(prompt):
        token = chunk.content
        if token:
            full_answer += token
            yield sse("token", {"text": token})

                    
    sources_text = "\n\nSources:\n" + build_sources_block(per_doc)
    yield sse("token", {"text": sources_text})
    full_answer += sources_text

                    
    session = get_session(session_id)
    session.add_assistant_turn(full_answer, sub_questions, citations)

                                                       
    yield sse("sources", {
        "citations":          citations,
        "specialist_outputs": spec_outputs,
    })
    await asyncio.sleep(0)

    yield sse("done", {})


@app.post("/chat/stream", tags=["Chat"])
async def chat_stream(request: ChatRequest):
    """Streaming chat endpoint using Server-Sent Events."""
    return StreamingResponse(
        stream_chat(request),
        media_type="text/event-stream",
        headers={
            "Cache-Control":               "no-cache",
            "X-Accel-Buffering":           "no",                             
            "Access-Control-Allow-Origin": "*",
        },
    )


                                                                                

@app.post("/chat", response_model=ChatResponse, tags=["Chat"])
def chat(request: ChatRequest):
    session_id = request.session_id or str(uuid.uuid4())
    docs_dir   = Path(DOCS_DIR).resolve()

    if not list_pdf_files(docs_dir):
        raise HTTPException(status_code=400, detail=f"No PDF files found in: {docs_dir}")

    initial_state: PipelineState = {
        "question":           request.question,
        "docs_dir":           str(docs_dir),
        "mode":               "routed",
        "max_files":          MAX_FILES,
        "save_json":          f"outputs/{session_id}.json",
        "session_id":         session_id,
        "sub_questions":      [],
        "resolved_question":  "",
        "selected_pdfs":      [],
        "routing_reason":     "",
        "per_doc_answers":    [],
        "all_docs":           [],
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
    docs_dir = Path(DOCS_DIR).resolve()
    pdfs     = list_pdf_files(docs_dir)
    if not pdfs:
        raise HTTPException(status_code=404, detail=f"No PDFs found in: {docs_dir}")
    return DocumentListResponse(
        docs_dir=str(docs_dir),
        documents=[DocumentInfo(filename=p.name, summary=get_or_build_summary(p)) for p in pdfs],
    )


                                                                                

@app.get("/health", response_model=HealthResponse, tags=["System"])
def health():
    return HealthResponse(status="ok", version="1.0.0", docs_dir=str(Path(DOCS_DIR).resolve()))


@app.get("/session/{session_id}", response_model=SessionHistoryResponse, tags=["Session"])
def get_history(session_id: str):
    session = get_session(session_id)
    history = session.get_history()
    return SessionHistoryResponse(session_id=session_id, turns=len(history), history=history)


@app.delete("/session/{session_id}", response_model=ClearSessionResponse, tags=["Session"])
def clear_session(session_id: str):
    delete_session(session_id)
    return ClearSessionResponse(session_id=session_id, cleared=True)


                                                                                

_reg_sessions: dict = {}


@app.post("/register/chat", response_model=RegistrationResponse, tags=["Registration"])
def registration_chat(req: RegistrationMessage):
    from src.registration_agent import (
        extract_fields, get_missing_fields,
        get_next_question, validate_form,
        build_form_summary, FIELD_LABELS,
        REQUIRED_FIELDS,
    )
    sid = req.session_id or str(uuid.uuid4())
    if sid not in _reg_sessions:
        _reg_sessions[sid] = {"collected": {}, "history": []}

    state   = _reg_sessions[sid]
    history = state["history"]

                                                                            
    is_start = req.message == "Hello, I want to register a company."
    if not is_start:
        history.append({"role": "user", "content": req.message})

                                                          
    if not is_start:
        before = set(k for k, v in state["collected"].items() if v)
        state["collected"] = extract_fields(req.message, state["collected"])
        after  = set(k for k, v in state["collected"].items() if v)

                                                                      
                                                                             
        if before == after:
            missing_now = get_missing_fields(state["collected"])
            if missing_now:
                current_field = missing_now[0]
                raw_val = req.message.strip()

                if current_field == "objectives":
                                                                                 
                    lines = [l.strip() for l in raw_val.replace(";", "\n").splitlines() if l.strip()]
                    state["collected"]["objectives"] = [
                        {"nsic_code": "", "description": line} for line in lines
                    ] if lines else [{"nsic_code": "", "description": raw_val}]
                elif current_field == "shareholders":
                                                                                
                    state["collected"]["shareholders"] = raw_val
                else:
                                                            
                    clean = raw_val.replace("NPR", "").replace("Rs.", "").replace(",", "").strip()
                    state["collected"][current_field] = clean

    missing = get_missing_fields(state["collected"])

    if not missing:
        validation = validate_form(state["collected"])
        if not validation.get("valid", True) and validation.get("issues"):
            issues = validation["issues"]
            reply  = "I found a few issues:\n" + "\n".join(f"• {i}" for i in issues) + "\n\nCould you please correct these?"
            history.append({"role": "assistant", "content": reply})
            return RegistrationResponse(session_id=sid, reply=reply, collected=state["collected"], missing=missing, complete=False, issues=issues)

        form_summary = build_form_summary(state["collected"])
        reply = "All details collected! Here is your filled registration form. Please review carefully before submitting."
        history.append({"role": "assistant", "content": reply})
        return RegistrationResponse(session_id=sid, reply=reply, collected=state["collected"], missing=[], complete=True, form_summary=form_summary)

    reply = get_next_question(state["collected"], history)
    history.append({"role": "assistant", "content": reply})
    return RegistrationResponse(session_id=sid, reply=reply, collected=state["collected"], missing=[FIELD_LABELS.get(f, f) for f in missing], complete=False)


@app.delete("/register/{session_id}", tags=["Registration"])
def clear_registration_session(session_id: str):
    _reg_sessions.pop(session_id, None)
    return {"session_id": session_id, "cleared": True}


@app.post("/register/launch-bot", tags=["Registration"])
def launch_bot(req: dict):
    """Launch OCR bot in a background thread. Returns immediately."""
    import threading
    sid = req.get("session_id")
    if not sid or sid not in _reg_sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    collected = _reg_sessions[sid].get("collected", {})
    if not collected:
        raise HTTPException(status_code=400, detail="No collected data for this session")

    collected_snapshot = dict(collected)

    def _run_bot():
        try:
            from src.ocr_bot import OCRBot
            bot = OCRBot(headless=False)
            bot.run(collected_snapshot)
        except Exception as e:
            print(f"[OCRBot] Error: {e}")

    threading.Thread(target=_run_bot, daemon=True).start()
    return {"status": "launched", "session_id": sid}