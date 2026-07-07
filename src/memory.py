"""
memory.py
---------
Session-based conversation memory with two backends:
  - Redis  (production — fast, expiring, multi-instance safe)
  - JSON on disk (fallback — no Redis needed for local dev)

Each session stores a list of turns:
  { role, content, sub_questions, citations, timestamp }

The context string is injected into prompts so the LLM can
resolve follow-up questions against previous answers.
"""

import json
import os
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional

                                                                                
                                                                      

REDIS_URL   = os.getenv("REDIS_URL", "")
SESSION_TTL = int(os.getenv("SESSION_TTL_SECONDS", 3600))                   
MEMORY_DIR  = Path("memory")


def _get_redis():
    """Return a Redis client or None if Redis is not configured."""
    if not REDIS_URL:
        return None
    try:
        import redis
        client = redis.from_url(REDIS_URL, decode_responses=True)
        client.ping()
        return client
    except Exception:
        return None


class ConversationMemory:
    """
    Conversation memory for one session.
    Automatically uses Redis if available, otherwise JSON on disk.
    """

    def __init__(self, session_id: str):
        self.session_id = session_id
        self._redis     = _get_redis()
        self._key       = f"session:{session_id}"

                            
        MEMORY_DIR.mkdir(exist_ok=True)
        self._disk_path = MEMORY_DIR / f"{session_id}.json"

                                                                                

    def _read(self) -> List[Dict]:
        if self._redis:
            raw = self._redis.get(self._key)
            return json.loads(raw) if raw else []
        if self._disk_path.exists():
            try:
                return json.loads(self._disk_path.read_text(encoding="utf-8"))
            except Exception:
                return []
        return []

    def _write(self, history: List[Dict]):
        if self._redis:
            self._redis.setex(self._key, SESSION_TTL, json.dumps(history))
        else:
            self._disk_path.write_text(
                json.dumps(history, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )

                                                                                

    def add_user_turn(self, question: str):
        history = self._read()
        history.append({
            "role":      "user",
            "content":   question,
            "timestamp": datetime.utcnow().isoformat(),
        })
        self._write(history)

    def add_assistant_turn(
        self,
        answer: str,
        sub_questions: List[str],
        citations: List[str],
    ):
        history = self._read()
        history.append({
            "role":          "assistant",
            "content":       answer,
            "sub_questions": sub_questions,
            "citations":     citations,
            "timestamp":     datetime.utcnow().isoformat(),
        })
        self._write(history)

    def get_history(self) -> List[Dict]:
        return self._read()

    def get_context_string(self, max_turns: int = 4) -> str:
        """
        Last N turns formatted as a string for injection into LLM prompts.
        Truncates long assistant answers to keep context window manageable.
        """
        history = self._read()
        recent  = history[-(max_turns * 2):]
        if not recent:
            return ""

        lines = ["Previous conversation:"]
        for turn in recent:
            role    = turn["role"].capitalize()
            content = turn["content"]
            if role == "Assistant" and len(content) > 500:
                content = content[:500] + "..."
            lines.append(f"{role}: {content}")
        return "\n".join(lines)

    def clear(self):
        if self._redis:
            self._redis.delete(self._key)
        elif self._disk_path.exists():
            self._disk_path.unlink()

    @property
    def is_followup(self) -> bool:
        return len(self._read()) > 0

    @property
    def backend(self) -> str:
        return "redis" if self._redis else "disk"


                                                                                

_sessions: Dict[str, ConversationMemory] = {}


def get_session(session_id: str) -> ConversationMemory:
    if session_id not in _sessions:
        _sessions[session_id] = ConversationMemory(session_id)
    return _sessions[session_id]


def delete_session(session_id: str):
    if session_id in _sessions:
        _sessions[session_id].clear()
        del _sessions[session_id]
    else:
                                                                       
        ConversationMemory(session_id).clear()
