"""
registration_agent.py
---------------------
Conversational agent that collects company registration details
step by step and returns a filled registration form.

The agent maintains its own state (collected fields) and decides
what to ask next based on what's still missing.

Fields collected:
  - company_name
  - company_type  (Private Limited / Public Limited / Partnership)
  - objectives    (business purpose)
  - registered_address
  - directors     (list of name + citizenship_no)
  - share_capital
  - contact_email
  - contact_phone
"""

import json
import re
from typing import Dict, List, Optional
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from .config import OPENAI_MODEL


# All fields the form needs
REQUIRED_FIELDS = [
    "company_name",
    "company_type",
    "objectives",
    "registered_address",
    "directors",
    "share_capital",
    "contact_email",
    "contact_phone",
]

FIELD_LABELS = {
    "company_name":       "Company Name",
    "company_type":       "Company Type",
    "objectives":         "Business Objectives",
    "registered_address": "Registered Address",
    "directors":          "Directors",
    "share_capital":      "Share Capital (NPR)",
    "contact_email":      "Contact Email",
    "contact_phone":      "Contact Phone",
}


# ── Prompts ───────────────────────────────────────────────────────────────────

_EXTRACT_PROMPT = ChatPromptTemplate.from_template(
    """You are extracting company registration details from a user's message.

Current collected fields (already known):
{collected}

User's latest message:
{message}

Extract any new registration details from the message and return ONLY valid JSON.
Use null for fields not mentioned. For directors, return a list of objects with
"name" and "citizenship_no" keys.

Return exactly this structure:
{{
  "company_name": null,
  "company_type": null,
  "objectives": null,
  "registered_address": null,
  "directors": null,
  "share_capital": null,
  "contact_email": null,
  "contact_phone": null
}}
"""
)

_NEXT_QUESTION_PROMPT = ChatPromptTemplate.from_template(
    """You are a friendly Nepal company registration assistant helping a user
fill their company registration form step by step.

Fields already collected:
{collected}

Fields still needed:
{missing}

Conversation so far:
{history}

Ask for the NEXT most important missing field in a friendly, clear way.
- Ask only ONE field at a time.
- Give a brief example or hint when helpful.
- If asking for directors, ask for full name and citizenship number.
- Keep it concise — 1-3 sentences max.
- Do NOT mention field names like "company_type" — use natural language.

Return only the question, nothing else.
"""
)

_VALIDATE_PROMPT = ChatPromptTemplate.from_template(
    """You are validating a Nepal company registration form.

Collected fields:
{collected}

Check each field for obvious issues:
- Company name: should not be too generic or contain special characters
- Company type: must be one of Private Limited, Public Limited, Partnership
- Share capital: must be a positive number in NPR
- Email: must look like a valid email
- Phone: should be a Nepal phone number format
- Directors: each must have name and citizenship number

Return ONLY valid JSON:
{{
  "valid": true/false,
  "issues": ["issue 1", "issue 2"]
}}
"""
)


# ── Core agent functions ──────────────────────────────────────────────────────

def extract_fields(message: str, collected: Dict) -> Dict:
    """Extract registration fields from user message using LLM."""
    llm = ChatOpenAI(model=OPENAI_MODEL, temperature=0)
    collected_str = json.dumps(collected, indent=2)
    raw = llm.invoke(
        _EXTRACT_PROMPT.format_messages(collected=collected_str, message=message)
    ).content.strip()

    raw = re.sub(r"^```[a-z]*\n?", "", raw, flags=re.IGNORECASE)
    raw = re.sub(r"\n?```$", "", raw, flags=re.IGNORECASE)

    try:
        extracted = json.loads(raw)
        # Merge: only update fields that are non-null in extracted
        for key, val in extracted.items():
            if val is not None and key in REQUIRED_FIELDS:
                collected[key] = val
    except json.JSONDecodeError:
        pass

    return collected


def get_missing_fields(collected: Dict) -> List[str]:
    """Return list of fields not yet collected."""
    return [f for f in REQUIRED_FIELDS if not collected.get(f)]


def get_next_question(collected: Dict, history: List[Dict]) -> str:
    """Ask for the next missing field in a friendly way."""
    missing = get_missing_fields(collected)
    if not missing:
        return "all_collected"

    llm = ChatOpenAI(model=OPENAI_MODEL, temperature=0.2)
    history_str = "\n".join(
        f"{t['role'].capitalize()}: {t['content']}" for t in history[-6:]
    )
    collected_str = json.dumps(
        {FIELD_LABELS.get(k, k): v for k, v in collected.items() if v}, indent=2
    )
    missing_str = ", ".join(FIELD_LABELS.get(f, f) for f in missing)

    return llm.invoke(
        _NEXT_QUESTION_PROMPT.format_messages(
            collected=collected_str,
            missing=missing_str,
            history=history_str,
        )
    ).content.strip()


def validate_form(collected: Dict) -> Dict:
    """Validate collected fields and return issues."""
    llm = ChatOpenAI(model=OPENAI_MODEL, temperature=0)
    raw = llm.invoke(
        _VALIDATE_PROMPT.format_messages(
            collected=json.dumps(collected, indent=2)
        )
    ).content.strip()

    raw = re.sub(r"^```[a-z]*\n?", "", raw, flags=re.IGNORECASE)
    raw = re.sub(r"\n?```$", "", raw, flags=re.IGNORECASE)

    try:
        return json.loads(raw)
    except Exception:
        return {"valid": True, "issues": []}


def build_form_summary(collected: Dict) -> str:
    """Build a human-readable form summary."""
    lines = ["COMPANY REGISTRATION FORM — NEPAL", "=" * 40]
    for field in REQUIRED_FIELDS:
        label = FIELD_LABELS.get(field, field)
        value = collected.get(field)
        if field == "directors" and isinstance(value, list):
            lines.append(f"{label}:")
            for i, d in enumerate(value, 1):
                if isinstance(d, dict):
                    lines.append(f"  Director {i}: {d.get('name','?')} (Citizenship: {d.get('citizenship_no','?')})")
                else:
                    lines.append(f"  Director {i}: {d}")
        else:
            lines.append(f"{label}: {value or '—'}")
    lines.append("=" * 40)
    lines.append("Note: This form summary is for review. Submit at ocr.gov.np")
    return "\n".join(lines)
