"""
rag_engine.py
-------------
Core RAG logic with reranking support.
"""

from typing import Dict, List, Optional
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from .config import OPENAI_MODEL, TOP_K
from .query_decomposer import decompose_question, retrieve_for_sub_questions
from .reranker import rerank_documents


RAG_PROMPT = ChatPromptTemplate.from_template(
    """You are a Nepal business compliance assistant.

{memory_context}

User question:
{question}

Specific aspects researched:
{sub_questions}

Evidence passages (ranked by relevance):
{context}

Rules:
- Decide your own section headings based on what the question needs.
- Cover every aspect listed above — if evidence exists include it, \
  if not write "Not found in provided documents."
- Rewrite evidence in your own words. Do not copy sentences verbatim.
- Do not add any fact not present in the evidence.
- Note: If the text says a company may be incorporated "singly", this means the minimum number of shareholders is 1.
- Do NOT write any citations, page numbers, or filenames — \
  sources will be appended automatically.
- If this is a follow-up question, build on the previous conversation context above.
"""
)


def _format_context(docs: List[Document]) -> str:
    blocks = []
    for i, d in enumerate(docs, start=1):
        src   = d.metadata.get("source_file", "unknown")
        page  = d.metadata.get("page_index",  "?")
        score = d.metadata.get("rerank_score", "")
        score_str = f" | score {score}" if score else ""
        blocks.append(f"[passage {i} | {src} p.{page}{score_str}]\n{d.page_content}")
    return "\n\n".join(blocks)


def _build_sources(docs: List[Document]) -> str:
    seen:  set       = set()
    lines: List[str] = []
    for d in docs:
        src  = d.metadata.get("source_file", "unknown")
        page = d.metadata.get("page_index",  "?")
        label = f"{src} • p.{page}"
        if label not in seen:
            seen.add(label)
            lines.append(f"  {label}")
    return "\n".join(lines)


def make_retriever(vectorstore, k: int = TOP_K):
    return vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={"k": k, "fetch_k": max(20, k * 6), "lambda_mult": 0.3},
    )


def answer_with_retriever(
    question: str,
    retriever,
    sub_questions:  Optional[List[str]] = None,
    memory_context: str = "",
) -> Dict:
    if sub_questions is None:
        sub_questions = decompose_question(question)

                                                                 
    docs = retrieve_for_sub_questions(sub_questions, retriever, max_docs_per_sub=5)

                                                                                     
    print(f"    Reranking {len(docs)} chunks...")
    docs = rerank_documents(question, docs, top_n=TOP_K)

    llm        = ChatOpenAI(model=OPENAI_MODEL, temperature=0.1)
    context    = _format_context(docs)
    sub_q_text = "\n".join(f"- {q}" for q in sub_questions)

    answer = llm.invoke(
        RAG_PROMPT.format_messages(
            question=question,
            sub_questions=sub_q_text,
            context=context,
            memory_context=memory_context,
        )
    ).content

    sources      = _build_sources(docs)
    final_answer = f"{answer}\n\nSources:\n{sources}"

    citations = list({
        f'{d.metadata.get("source_file","?")} • p.{d.metadata.get("page_index","?")}'
        for d in docs
    })

    return {
        "answer":        final_answer,
        "docs":          docs,
        "passages": [
            {
                "source_file": d.metadata.get("source_file", "unknown"),
                "page_index":  d.metadata.get("page_index",  "?"),
                "rerank_score": d.metadata.get("rerank_score", None),
                "text":        d.page_content,
            }
            for d in docs
        ],
        "citations":     citations,
        "sub_questions": sub_questions,
    }