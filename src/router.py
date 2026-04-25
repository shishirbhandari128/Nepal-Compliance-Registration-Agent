import json
import re
from pathlib import Path
from typing import List, Dict

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from .config import OPENAI_MODEL


def choose_relevant_pdfs(question: str, pdf_paths: List[Path], max_files: int = 3) -> Dict:
    """
    Supervisor router: given a user question and a list of available PDFs,
    ask the LLM to select only the files that are genuinely relevant.
    Returns a dict with 'selected_files' (list of names) and 'reason' (str).
    """
    llm = ChatOpenAI(model=OPENAI_MODEL, temperature=0)

    # Build an annotated file list so the LLM has more signal than bare filenames.
    options_lines = []
    for p in pdf_paths:
        stem_hint = p.stem.replace("_", " ").replace("-", " ")
        options_lines.append(f"- {p.name}  (topic hint: {stem_hint})")
    options = "\n".join(options_lines)

    prompt = ChatPromptTemplate.from_template(
        """You are the routing supervisor for a Nepal compliance RAG system.
Your only job is to decide which PDF documents are worth loading for the user's question.

User question:
{question}

Available PDFs (with topic hints derived from their filenames):
{options}

Think carefully:
- Which files are most likely to contain information directly relevant to the question?
- Prefer precision over coverage — it is better to load 1 perfect file than 3 vague ones.
- Select at most {max_files} files.
- If no file seems relevant, pick the single closest match.

Respond with ONLY a valid JSON object, no markdown, no extra text:
{{
  "selected_files": ["filename1.pdf", "filename2.pdf"],
  "reason": "one or two sentences explaining why these files were chosen"
}}
"""
    )
    msg = prompt.format_messages(question=question, options=options, max_files=max_files)
    raw = llm.invoke(msg).content.strip()

    # Strip accidental markdown fences before parsing.
    raw = re.sub(r"^```[a-z]*\n?", "", raw, flags=re.IGNORECASE)
    raw = re.sub(r"\n?```$", "", raw, flags=re.IGNORECASE)

    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        # Graceful fallback: use the first file.
        return {
            "selected_files": [pdf_paths[0].name] if pdf_paths else [],
            "reason": "JSON parse failed — defaulting to first available file.",
        }

    selected_names = parsed.get("selected_files", []) if isinstance(parsed, dict) else []
    selected_names = [n for n in selected_names if isinstance(n, str)]

    # Validate that the LLM only returned real filenames.
    valid_names = {p.name for p in pdf_paths}
    filtered = [n for n in selected_names if n in valid_names]

    if not filtered and pdf_paths:
        filtered = [pdf_paths[0].name]

    return {
        "selected_files": filtered[:max_files],
        "reason": parsed.get("reason", ""),
    }