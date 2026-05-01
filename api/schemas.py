"""
api/schemas.py
--------------
Pydantic request/response models for the FastAPI endpoints.
"""

from typing import List, Optional, Dict
from pydantic import BaseModel, Field


# ── Chat ──────────────────────────────────────────────────────────────────────

class ChatRequest(BaseModel):
    question:   str           = Field(..., description="Compliance question")
    session_id: Optional[str] = Field(None, description="Session ID — auto-generated if not provided")


class ChatResponse(BaseModel):
    session_id:         str
    question:           str
    resolved_question:  str
    answer:             str
    sub_questions:      List[str]
    citations:          List[str]
    files_used:         List[str]       # actual files selected by router
    files_used_count:   int             # how many files were selected
    routing_reason:     str
    specialist_outputs: Dict[str, str]


# ── Documents ─────────────────────────────────────────────────────────────────

class DocumentInfo(BaseModel):
    filename: str
    summary:  str


class DocumentListResponse(BaseModel):
    docs_dir:  str
    documents: List[DocumentInfo]


# ── Session ───────────────────────────────────────────────────────────────────

class SessionHistoryResponse(BaseModel):
    session_id: str
    turns:      int
    history:    List[Dict]


class ClearSessionResponse(BaseModel):
    session_id: str
    cleared:    bool


# ── Health ────────────────────────────────────────────────────────────────────

class HealthResponse(BaseModel):
    status:   str
    version:  str
    docs_dir: str          # tells the frontend where PDFs are loaded from


# ── Registration Agent ────────────────────────────────────────────────────────

class RegistrationMessage(BaseModel):
    message:    str            = Field(..., description="User's message to the registration agent")
    session_id: Optional[str]  = Field(None, description="Registration session ID")


class RegistrationResponse(BaseModel):
    session_id:   str
    reply:        str                    # agent's next question or confirmation
    collected:    Dict                   # fields collected so far
    missing:      List[str]              # field names still needed
    complete:     bool                   # True when all fields collected
    form_summary: Optional[str] = None  # filled form text (when complete)
    issues:       List[str]    = []     # validation issues if any