"""
agents.py
---------
Three specialist LangGraph nodes that analyse the base RAG answer
through their own domain lens:
  - Legal Strategist
  - Tax Consultant
  - Document Auditor
"""

from typing import Dict
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from .config import OPENAI_MODEL


_SPECIALIST_PROMPT = ChatPromptTemplate.from_template(
    """You are a specialist expert — specifically, the {role}.

Your lens: {lens}

The user asked:
{question}

Evidence passages from the PDF(s) (each passage includes file + page):
{evidence}

Rules (strict):
- You may ONLY state facts that are directly supported by the passages above.
- For every factual statement, include an inline evidence tag like: [file.pdf p.X]
- If something is important but not supported, write: "Not found in provided documents."
- Do NOT use general knowledge to fill gaps.

Write a concise specialist briefing with clear headings.
"""
)


def _run_specialist(role: str, lens: str, question: str, evidence: str) -> str:
    llm = ChatOpenAI(model=OPENAI_MODEL, temperature=0.1)
    return llm.invoke(
        _SPECIALIST_PROMPT.format_messages(
            role=role,
            lens=lens,
            question=question,
            evidence=evidence,
        )
    ).content


def run_all_specialists(question: str, evidence: str) -> Dict[str, str]:
    """Run all three specialists and return role → answer dict."""
    from concurrent.futures import ThreadPoolExecutor

    specialists = [
        (
            "Legal Strategist",
            "Legal entity structure, registration requirements, compliance obligations, "
            "and regulatory risk under Nepal's applicable laws and acts.",
        ),
        (
            "Tax Consultant",
            "PAN registration, VAT thresholds and obligations, income tax filing, "
            "advance tax, withholding tax, and any other tax triggers under Nepal's tax law.",
        ),
        (
            "Document Auditor",
            "All documents that must be prepared, submitted, or maintained — including "
            "what is missing, the sequence they are needed, and common pitfalls.",
        ),
    ]

    results: Dict[str, str] = {}

    with ThreadPoolExecutor(max_workers=3) as executor:
        futures = {
            executor.submit(_run_specialist, role, lens, question, evidence): role
            for role, lens in specialists
        }
        for future in futures:
            role = futures[future]
            try:
                results[role] = future.result()
            except Exception as e:
                results[role] = f"Error: {e}"

    return results