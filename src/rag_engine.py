from typing import Dict, List, Optional
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from .config import OPENAI_MODEL, TOP_K
from .query_decomposer import decompose_question, retrieve_for_sub_questions


RAG_PROMPT = ChatPromptTemplate.from_template(
    """You are a Nepal business compliance assistant. Synthesise the evidence below \
into one clear, well-structured answer.

User question:
{question}

Specific aspects that were researched:
{sub_questions}

Evidence passages:
{context}

Rules:
- Decide your own section headings based on what the question needs.
- Cover every aspect listed above — if evidence exists for it, include it; \
if not, write "Not found in provided documents" for that point.
- Rewrite the evidence in your own words. Do not copy sentences verbatim.
- Do not add any fact not present in the evidence.
- Do NOT write any citations, page numbers, or source references in your answer — \
sources will be appended automatically.
"""
)


def format_context(docs: List[Document]) -> str:
    blocks = []
    for i, d in enumerate(docs, start=1):
        src = d.metadata.get("source_file", "unknown")
        page = d.metadata.get("page_index", "?")
        blocks.append(f"[passage {i} | {src} p.{page}]\n{d.page_content}")
    return "\n\n".join(blocks)


def _build_sources(docs: List[Document]) -> str:
    seen = set()
    lines = []
    for d in docs:
        src = d.metadata.get("source_file", "unknown")
        page = d.metadata.get("page_index", "?")
        label = f"{src} • p.{page}"
        if label not in seen:
            seen.add(label)
            lines.append(f"  {label}")
    return "\n".join(lines)


def answer_with_retriever(
    question: str,
    retriever,
    sub_questions: Optional[List[str]] = None,
) -> Dict:
    # Use provided sub_questions if given, otherwise generate them
    # This avoids regenerating sub-questions when called from test_all_files
    if sub_questions is None:
        sub_questions = decompose_question(question)

    docs = retrieve_for_sub_questions(sub_questions, retriever, max_docs_per_sub=5)
    docs = docs[:TOP_K * 2]  # cap to avoid context overflow

    llm = ChatOpenAI(model=OPENAI_MODEL, temperature=0.1)
    context = format_context(docs)
    sub_q_text = "\n".join(f"- {q}" for q in sub_questions)

    prompt = RAG_PROMPT.format_messages(
        question=question,
        sub_questions=sub_q_text,
        context=context,
    )
    answer = llm.invoke(prompt).content

    sources = _build_sources(docs)
    final_answer = f"{answer}\n\nSources:\n{sources}"

    citations = list({
        f'{d.metadata.get("source_file", "?")} • p.{d.metadata.get("page_index", "?")}'
        for d in docs
    })

    return {
        "answer": final_answer,
        "docs": docs,
        "citations": citations,
        "sub_questions": sub_questions,
    }


def make_retriever(vectorstore, k: int = TOP_K):
    return vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={"k": k, "fetch_k": max(20, k * 6), "lambda_mult": 0.3},
    )