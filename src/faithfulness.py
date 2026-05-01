"""
faithfulness.py
---------------
Compares the final merged answer against the original retrieved chunks
and corrects any claim that contradicts or is absent from the source text.

Runs as the last node in the graph before save.
"""

from typing import List, Dict
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from .config import OPENAI_MODEL


_CHECK_PROMPT = ChatPromptTemplate.from_template(
    """You are a strict fact-checker for a Nepal legal compliance system.

Compare the DRAFT ANSWER against the SOURCE CHUNKS line by line.

Rules:
- SOURCE CHUNKS are the ground truth. They override everything else.
- If the draft states a number, threshold, or requirement that CONTRADICTS \
  a chunk, replace it with the correct value from the chunk.
- If the draft states a fact that cannot be found anywhere in the chunks, \
  remove that sentence entirely.
- If the draft is fully supported, return it unchanged.
- Do NOT add new information.
- Do NOT change headings or structure.
- Do NOT touch the "Sources:" block at the bottom — leave it exactly as-is.
- Return ONLY the corrected answer, no explanation.

SOURCE CHUNKS:
{chunks}

DRAFT ANSWER:
{answer}
"""
)


def check_and_fix(answer: str, docs: List[Document]) -> Dict:
    """
    Verify every claim in `answer` against `docs`.
    Returns corrected answer and whether any fix was made.
    """
    if not docs:
        return {"answer": answer, "was_corrected": False}

    # Build evidence block — capped to avoid context overflow
    chunk_text = "\n\n".join(
        f"[{d.metadata.get('source_file','?')} p.{d.metadata.get('page_index','?')}]\n"
        f"{d.page_content}"
        for d in docs[:15]
    )

    # Only check the answer body — preserve Sources block untouched
    parts        = answer.split("\nSources:")
    answer_body  = parts[0].strip()
    sources_tail = ("\nSources:" + parts[1]) if len(parts) > 1 else ""

    llm = ChatOpenAI(model=OPENAI_MODEL, temperature=0)
    corrected = llm.invoke(
        _CHECK_PROMPT.format_messages(chunks=chunk_text, answer=answer_body)
    ).content.strip()

    was_corrected = corrected != answer_body
    if was_corrected:
        print("  [faithfulness] corrections applied")

    return {
        "answer":        corrected + sources_tail,
        "was_corrected": was_corrected,
    }