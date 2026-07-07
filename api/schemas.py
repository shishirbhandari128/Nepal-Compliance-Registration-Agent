"""
api/schemas.py
--------------
Pydantic request/response models for the FastAPI endpoints.
"""

from typing import List, Optional, Dict
from pydantic import BaseModel, Field


                                                                                

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
    files_used:         List[str]                                        
    files_used_count:   int                                           
    routing_reason:     str
    specialist_outputs: Dict[str, str]


                                                                                

class DocumentInfo(BaseModel):
    filename: str
    summary:  str


class DocumentListResponse(BaseModel):
    docs_dir:  str
    documents: List[DocumentInfo]


                                                                                

class SessionHistoryResponse(BaseModel):
    session_id: str
    turns:      int
    history:    List[Dict]


class ClearSessionResponse(BaseModel):
    session_id: str
    cleared:    bool


                                                                                

class HealthResponse(BaseModel):
    status:   str
    version:  str
    docs_dir: str                                                         


                                                                                

class RegistrationMessage(BaseModel):
    message:    str            = Field(..., description="User's message to the registration agent")
    session_id: Optional[str]  = Field(None, description="Registration session ID")


class RegistrationResponse(BaseModel):
    session_id:   str
    reply:        str                                                           
    collected:    Dict                                            
    missing:      List[str]                                        
    complete:     bool                                                   
    form_summary: Optional[str] = None                                    
    issues:       List[str]    = []                               