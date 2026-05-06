"""
router.py
---------
Supervisor router that selects the most relevant PDFs for a question.
Uses cached one-line summaries of each PDF's content for smarter routing
instead of relying solely on filenames.
"""

import json
import re
from pathlib import Path
from typing import List, Dict

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from tools import build_pdf_options_block

from .config import OPENAI_MODEL


# SUMMARY_PROMPT = ChatPromptTemplate.from_template(
#     """Read the following text extracted from a legal/regulatory PDF document.
# Write ONE sentence (max 30 words) describing what topics this document covers.
# Be specific — mention the exact subject matter (e.g. income tax rates, VAT registration,
# company incorporation, labour rights, tourism licensing).

# Document text (first portion):
# {text}

# Return only the one-sentence summary, nothing else.
# """
# )


# def _summary_cache_path(pdf_path: Path) -> Path:
#     """Each PDF gets a .summary file next to its vectorstore."""
#     persist = Path("vectorstore") / pdf_path.stem
#     persist.mkdir(parents=True, exist_ok=True)
#     return persist / ".pdf_summary"


# def _get_or_build_summary(pdf_path: Path) -> str:
#     """
#     Load cached summary if it exists, otherwise read the first 3000 chars
#     of the PDF and ask the LLM to summarise it. Cache the result.
#     """
#     cache = _summary_cache_path(pdf_path)
#     if cache.exists():
#         return cache.read_text(encoding="utf-8").strip()

#     # Read raw text from the PDF
#     try:
#         from langchain_community.document_loaders import PyPDFLoader
#         pages = PyPDFLoader(str(pdf_path)).load()
#         sample_text = " ".join(p.page_content for p in pages[:3])[:3000]
#     except Exception:
#         sample_text = pdf_path.stem.replace("_", " ").replace("-", " ")

#     llm = ChatOpenAI(model=OPENAI_MODEL, temperature=0)
#     msg = SUMMARY_PROMPT.format_messages(text=sample_text)
#     summary = llm.invoke(msg).content.strip()

#     cache.write_text(summary, encoding="utf-8")
#     return summary


# def _build_options_block(pdf_paths: List[Path]) -> str:
#     """Build a rich description block for the router LLM."""
#     lines = []
#     for p in pdf_paths:
#         summary = _get_or_build_summary(p)
#         lines.append(f"- {p.name}\n    Summary: {summary}")
#     return "\n".join(lines)


ROUTER_PROMPT = ChatPromptTemplate.from_template(
    """You are a routing supervisor for a Nepal compliance document search system.
Your job is to select ONLY the PDF files that are directly relevant to the user's question.

User question:
{question}

Available PDFs with summaries of their content:
{options}

Rules:
- Read each PDF summary carefully before deciding.
- Select ONLY files whose summary directly matches the topic of the question.
- Be precise — if the question is about tax, only select tax-related documents.
- If the question is about VAT, select VAT documents. If about income tax, select income tax documents.
- Do NOT select files just because they are loosely related (e.g. do not include \
Company Act for a tax question just because companies pay tax).
- Select at most {max_files} files.
- If only one file is relevant, select only that one.

Return ONLY valid JSON, no markdown:
{{
  "selected_files": ["file1.pdf", "file2.pdf"],
  "reason": "one sentence explaining why these files were chosen"
}}
"""
)


def choose_relevant_pdfs(
    question: str,
    pdf_paths: List[Path],
    max_files: int = 3,
) -> Dict:
    """Select the most relevant PDFs for a question using content summaries."""

    print("  Building/loading PDF summaries for routing...")
    options = build_pdf_options_block(pdf_paths)

    llm = ChatOpenAI(model=OPENAI_MODEL, temperature=0)
    msg = ROUTER_PROMPT.format_messages(
        question=question,
        options=options,
        max_files=max_files,
    )
    raw = llm.invoke(msg).content.strip()

    # Strip markdown fences
    raw = re.sub(r"^```[a-z]*\n?", "", raw, flags=re.IGNORECASE)
    raw = re.sub(r"\n?```$", "", raw, flags=re.IGNORECASE)

    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return {
            "selected_files": [pdf_paths[0].name] if pdf_paths else [],
            "reason": "JSON parse failed — defaulting to first file.",
        }

    selected_names = parsed.get("selected_files", []) if isinstance(parsed, dict) else []
    selected_names = [n for n in selected_names if isinstance(n, str)]

    valid_names = {p.name for p in pdf_paths}
    filtered = [n for n in selected_names if n in valid_names]

    if not filtered and pdf_paths:
        filtered = [pdf_paths[0].name]

    return {
        "selected_files": filtered[:max_files],
        "reason": parsed.get("reason", ""),
    }