"""
query_decomposer.py
-------------------
Generates focused sub-questions from a user question and retrieves
documents for all of them in parallel.
"""

import json
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from .config import OPENAI_MODEL


DECOMPOSE_PROMPT = ChatPromptTemplate.from_template(
    """You are an expert at breaking down a user's question into focused sub-questions \
for searching a legal document database.

User question:
{question}

Your task:
- Generate sub-questions that are STRICTLY within the scope of what the user asked.
- Do NOT expand into related topics the user did not ask about.
- Each sub-question must target one concrete detail that, together, covers the user's question end-to-end.
- Make the set of sub-questions COMPREHENSIVE for the asked topic:
  - If the user asked about tax rules: cover tax types/thresholds/registration, returns & filing frequency, rates, withholding, advance tax, exemptions, penalties, timelines.
  - If the user asked about company registration/incorporation: cover eligibility, directors, name approval, registered office, required constitutional documents (MOA/AOA if applicable), application/submission steps, fees, timelines, post-registration obligations.
    - You MUST include at least ONE sub-question about shareholders (e.g. minimum/maximum, required shareholder details, share capital/ownership requirements) whenever the user is asking about incorporating/registering a company.
  - If the user asked about labour law: cover wages, hours, leave, contracts, termination, safety, required records, penalties.
  - If the user asked about licensing/permits: cover issuing authority, prerequisites, required documents, process, fees, timelines, renewals, penalties.
- If the user asked about tax rules, only generate sub-questions about tax rules — \
do NOT generate sub-questions about registration, licenses, or documents unless \
the user explicitly asked about those.
- Generate between 5 and 8 sub-questions.
- Do NOT rephrase the original question as a sub-question.
- Use user terms (e.g. "private limited", "PAN", "VAT") to improve search recall.

Examples of CORRECT decomposition for "What are the tax rules for tourism businesses?":
  - "What income tax rate applies to tourism businesses in Nepal?"
  - "What VAT obligations apply to tourism service providers?"
  - "Are there tax exemptions or concessions for tourism businesses?"
  - "What are the advance tax payment rules for tourism businesses?"
  - "What are the penalties for late tax filing in the tourism sector?"

Examples of WRONG decomposition (too broad, out of scope):
  - "What licenses are required to operate a tourism business?" ← not about tax
  - "What are the registration requirements for tourism businesses?" ← not about tax

Return ONLY a valid JSON array of strings, no markdown, no extra text:
["sub-question 1", "sub-question 2", ...]
"""
)


def decompose_question(question: str) -> List[str]:
    """Break a user question into focused in-scope sub-questions."""
    llm = ChatOpenAI(model=OPENAI_MODEL, temperature=0)
    msg = DECOMPOSE_PROMPT.format_messages(question=question)
    raw = llm.invoke(msg).content.strip()

    raw = re.sub(r"^```[a-z]*\n?", "", raw, flags=re.IGNORECASE)
    raw = re.sub(r"\n?```$", "", raw, flags=re.IGNORECASE)

    try:
        sub_questions = json.loads(raw)
        if isinstance(sub_questions, list):
            subs = [q for q in sub_questions if isinstance(q, str)]

                                                                                               
                                                                        
            ql = question.lower()
            is_company_registration = (
                "company" in ql
                and any(k in ql for k in ["register", "registration", "incorporat", "incorporation", "shareholder", "shareholdar"])
            )
            if is_company_registration:
                has_shareholder = any("shareholder" in s.lower() for s in subs)
                has_incorporate = any("incorporat" in s.lower() for s in subs)
                if not (has_shareholder and has_incorporate):
                    subs.append(
                        "What are the shareholder requirements (minimum number, such as singly or jointly, and maximum number) for incorporating a private limited company under the Company Act?"
                    )

            return subs
    except json.JSONDecodeError:
        pass

    return [question]


def retrieve_for_sub_questions(
    sub_questions: List[str],
    retriever,
    max_docs_per_sub: int = 4,
) -> List[Document]:
    """
    Run all sub-question retrievals in PARALLEL using threads.
    Returns a deduplicated union of all retrieved documents.
    """
    seen_keys = set()
    all_docs: List[Document] = []
    lock_docs = []

    def _retrieve(sub_q: str) -> List[Document]:
        return retriever.invoke(sub_q)

    with ThreadPoolExecutor(max_workers=len(sub_questions)) as executor:
        futures = {executor.submit(_retrieve, sq): sq for sq in sub_questions}
        for future in as_completed(futures):
            try:
                docs = future.result()
                for d in docs[:max_docs_per_sub]:
                    key = (d.metadata.get("source_file"), d.metadata.get("page_index"))
                    if key not in seen_keys:
                        seen_keys.add(key)
                        all_docs.append(d)
            except Exception as e:
                print(f"  Warning: retrieval failed for a sub-question: {e}")

    return all_docs


def decompose_and_retrieve(
    question: str,
    retriever,
    max_docs_per_sub: int = 4,
) -> Dict:
    sub_questions = decompose_question(question)
    docs = retrieve_for_sub_questions(sub_questions, retriever, max_docs_per_sub)
    return {"sub_questions": sub_questions, "docs": docs}