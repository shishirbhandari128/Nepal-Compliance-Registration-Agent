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