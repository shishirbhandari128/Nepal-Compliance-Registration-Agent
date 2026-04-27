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

A base RAG system extracted the following evidence from official documents:
{base_answer}

Your task:
- Analyse the evidence strictly through your specialist lens.
- Organise your response in whatever sections make sense for this question and domain.
- Explain in plain language as if briefing a business owner.
- Rewrite evidence in your own words — do NOT copy sentences.
- Be direct and actionable. Omit anything not relevant to your lens.
- If evidence is missing for something important in your domain, flag it as \
  "Needs further verification."
- Do NOT write citations or page numbers.
"""
)


def _run_specialist(role: str, lens: str, question: str, base_answer: str) -> str:
    llm = ChatOpenAI(model=OPENAI_MODEL, temperature=0.1)
    return llm.invoke(
        _SPECIALIST_PROMPT.format_messages(
            role=role,
            lens=lens,
            question=question,
            base_answer=base_answer,
        )
    ).content


def run_all_specialists(question: str, base_answer: str) -> Dict[str, str]:
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
            executor.submit(_run_specialist, role, lens, question, base_answer): role
            for role, lens in specialists
        }
        for future in futures:
            role = futures[future]
            try:
                results[role] = future.result()
            except Exception as e:
                results[role] = f"Error: {e}"

    return results