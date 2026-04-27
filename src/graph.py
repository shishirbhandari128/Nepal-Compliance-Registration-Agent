"""
graph.py
--------
LangGraph pipeline with 6 nodes:

  decompose → route → retrieve_and_answer → specialist_agents → merge → save → END
"""

import json
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import List, Dict

from langgraph.graph import StateGraph, END
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from .state import PipelineState
from .tools import (
    get_or_build_vectorstore,
    format_per_doc_block,
    build_sources_block,
    build_pdf_options_block,
)
from .pdf_loader import list_pdf_files, load_pdf
from .chunker import chunk_documents
from .rag_engine import make_retriever, answer_with_retriever
from .agents import run_all_specialists
from .query_decomposer import decompose_question
from .memory import get_session
from .config import OPENAI_MODEL


# ── Prompts ───────────────────────────────────────────────────────────────────

_RESOLVE_PROMPT = ChatPromptTemplate.from_template(
    """You are a question resolver. The user may have asked a follow-up question \
that references previous conversation. Rewrite it as a fully self-contained question.

Conversation history:
{history}

Current user question:
{question}

If the question is already self-contained, return it unchanged.
Return ONLY the rewritten question, nothing else.
"""
)

_ROUTER_PROMPT = ChatPromptTemplate.from_template(
    """You are a routing supervisor for a Nepal compliance document search system.
Select ONLY the PDF files directly relevant to the user question.

User question:
{question}

Available PDFs with content summaries:
{options}

Rules:
- Read each summary carefully before deciding.
- Select ONLY files whose summary directly matches the topic.
- Be precise — tax question → only tax documents.
- Do NOT select loosely related files.
- Select at most {max_files} files. If only one is relevant, select only that one.

Return ONLY valid JSON, no markdown:
{{
  "selected_files": ["file1.pdf"],
  "reason": "one sentence explaining the choice"
}}
"""
)

_MERGER_PROMPT = ChatPromptTemplate.from_template(
    """You are a senior Nepal compliance analyst. You have:
1. A base RAG answer drawn from official documents
2. Three specialist perspectives (Legal, Tax, Document)

Merge everything into one final authoritative answer for the user.

User question:
{question}

Base RAG answer:
{base_answer}

Specialist perspectives:
{specialist_outputs}

Instructions:
- Decide your own section headings based on what the question needs.
- Integrate specialist insights with the base answer — don't list them separately.
- Where specialists add new information, include it. Where they repeat, merge it.
- Rewrite everything in your own words.
- Do NOT write citations or page numbers — sources are appended automatically.
- Highlight critical steps or risks prominently.
- End with a short "What still needs clarification" note if specialists flagged gaps.
"""
)


# ── Node 1: Decompose (with memory-aware follow-up resolution) ────────────────

def node_decompose(state: PipelineState) -> dict:
    print("\n[node: decompose]")
    session  = get_session(state["session_id"])
    question = state["question"]

    # Resolve follow-up questions against conversation history
    resolved_question = question
    if session.is_followup:
        history = session.get_context_string(max_turns=3)
        llm = ChatOpenAI(model=OPENAI_MODEL, temperature=0)
        resolved_question = llm.invoke(
            _RESOLVE_PROMPT.format_messages(history=history, question=question)
        ).content.strip()
        if resolved_question != question:
            print(f"  Follow-up resolved: {resolved_question}")

    sub_questions = decompose_question(resolved_question)
    print(f"  Sub-questions ({len(sub_questions)}):")
    for sq in sub_questions:
        print(f"    - {sq}")

    # Log user turn to memory
    session.add_user_turn(question)

    return {
        "resolved_question": resolved_question,
        "sub_questions":     sub_questions,
    }


# ── Node 2: Route ─────────────────────────────────────────────────────────────

def node_route(state: PipelineState) -> dict:
    print("\n[node: route]")
    docs_dir = Path(state["docs_dir"]).resolve()
    pdfs     = list_pdf_files(docs_dir)

    if not pdfs:
        raise ValueError(f"No PDF files found in {docs_dir}")

    if state["mode"] == "all":
        print(f"  Mode=all — using all {len(pdfs)} PDFs")
        return {
            "selected_pdfs":  [str(p) for p in pdfs],
            "routing_reason": "mode=all",
            "per_doc_answers": [],
        }

    enriched = (
        f"{state['resolved_question']}\n\nRelated aspects:\n" +
        "\n".join(f"- {sq}" for sq in state["sub_questions"])
    )

    print("  Loading/building PDF summaries...")
    options = build_pdf_options_block(pdfs)

    llm = ChatOpenAI(model=OPENAI_MODEL, temperature=0)
    raw = llm.invoke(
        _ROUTER_PROMPT.format_messages(
            question=enriched,
            options=options,
            max_files=state["max_files"],
        )
    ).content.strip()

    raw = re.sub(r"^```[a-z]*\n?", "", raw, flags=re.IGNORECASE)
    raw = re.sub(r"\n?```$",       "", raw, flags=re.IGNORECASE)

    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        parsed = {"selected_files": [pdfs[0].name], "reason": "JSON parse failed — fallback"}

    valid          = {p.name for p in pdfs}
    selected_names = [n for n in parsed.get("selected_files", []) if n in valid]
    if not selected_names:
        selected_names = [pdfs[0].name]

    selected_pdfs  = [p for p in pdfs if p.name in selected_names][: state["max_files"]]
    routing_reason = parsed.get("reason", "")

    print(f"  Files selected : {', '.join(p.name for p in selected_pdfs)}")
    if routing_reason:
        print(f"  Routing reason : {routing_reason}")

    return {
        "selected_pdfs":  [str(p) for p in selected_pdfs],
        "routing_reason": routing_reason,
        "per_doc_answers": [],
    }


# ── Node 3: Retrieve & Answer (parallel per PDF + reranking) ──────────────────

def node_retrieve_and_answer(state: PipelineState) -> dict:
    print("\n[node: retrieve_and_answer]")
    selected_pdfs = [Path(p) for p in state["selected_pdfs"]]
    sub_questions = state["sub_questions"]
    question      = state["resolved_question"]
    memory_ctx    = get_session(state["session_id"]).get_context_string()

    def _process_pdf(pdf: Path) -> Dict:
        def _build_chunks():
            return chunk_documents(load_pdf(pdf))

        db        = get_or_build_vectorstore(pdf, _build_chunks)
        retriever = make_retriever(db)
        result    = answer_with_retriever(
            question,
            retriever,
            sub_questions=sub_questions,
            memory_context=memory_ctx,
        )
        return {
            "file":          pdf.name,
            "answer":        result["answer"],
            "citations":     result["citations"],
            "sub_questions": sub_questions,
        }

    print(f"  Processing {len(selected_pdfs)} PDF(s) in parallel...")
    results: List[Dict] = [None] * len(selected_pdfs)

    with ThreadPoolExecutor(max_workers=max(1, len(selected_pdfs))) as executor:
        future_to_idx = {
            executor.submit(_process_pdf, pdf): i
            for i, pdf in enumerate(selected_pdfs)
        }
        for future in as_completed(future_to_idx):
            idx = future_to_idx[future]
            pdf = selected_pdfs[idx]
            try:
                results[idx] = future.result()
                print(f"    ✓ Done: {pdf.name}")
            except Exception as e:
                print(f"    ✗ Failed: {pdf.name} — {e}")
                results[idx] = {
                    "file":          pdf.name,
                    "answer":        f"Error: {e}",
                    "citations":     [],
                    "sub_questions": sub_questions,
                }

    return {"per_doc_answers": [r for r in results if r is not None]}


# ── Node 4: Specialist Agents (parallel: Legal + Tax + Document) ───────────────

def node_specialist_agents(state: PipelineState) -> dict:
    print("\n[node: specialist_agents]")
    per_doc_answers = state["per_doc_answers"]

    # Combine per-doc answers into one base answer for specialists
    base_answer = "\n\n".join(
        f"--- {item['file']} ---\n{item['answer'].split(chr(10)+'Sources:')[0]}"
        for item in per_doc_answers
    )

    print("  Running Legal, Tax, Document specialists in parallel...")
    specialist_outputs = run_all_specialists(state["resolved_question"], base_answer)

    for role in specialist_outputs:
        print(f"    ✓ {role}")

    return {"specialist_outputs": specialist_outputs}


# ── Node 5: Merge ─────────────────────────────────────────────────────────────

def node_merge(state: PipelineState) -> dict:
    print("\n[node: merge]")
    per_doc_answers    = state["per_doc_answers"]
    specialist_outputs = state["specialist_outputs"]

    # Format specialist outputs
    spec_block = "\n\n".join(
        f"=== {role} ===\n{text}"
        for role, text in specialist_outputs.items()
    )

    # Base answer (combined from all PDFs)
    if len(per_doc_answers) == 1:
        base_answer = per_doc_answers[0]["answer"].split("\nSources:")[0].strip()
    else:
        base_answer = format_per_doc_block(per_doc_answers)

    llm = ChatOpenAI(model=OPENAI_MODEL, temperature=0.1)
    merged_text = llm.invoke(
        _MERGER_PROMPT.format_messages(
            question=state["resolved_question"],
            base_answer=base_answer,
            specialist_outputs=spec_block,
        )
    ).content.strip()

    sources      = build_sources_block(per_doc_answers)
    final_answer = f"{merged_text}\n\nSources:\n{sources}"

    all_citations = list({c for item in per_doc_answers for c in item["citations"]})

    # Save assistant turn to memory
    session = get_session(state["session_id"])
    session.add_assistant_turn(final_answer, state["sub_questions"], all_citations)

    return {"final_answer": final_answer, "all_citations": all_citations}


# ── Node 6: Save ──────────────────────────────────────────────────────────────

def node_save(state: PipelineState) -> dict:
    print("\n" + "=" * 70)
    print("FINAL COMBINED ANSWER")
    print("=" * 70)
    print(state["final_answer"])

    out_path = Path(state["save_json"])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "session_id":            state["session_id"],
        "mode":                  state["mode"],
        "routing_reason":        state["routing_reason"],
        "files_evaluated":       [Path(p).name for p in state["selected_pdfs"]],
        "question":              state["question"],
        "resolved_question":     state["resolved_question"],
        "sub_questions":         state["sub_questions"],
        "specialist_outputs":    state["specialist_outputs"],
        "per_document_answers":  state["per_doc_answers"],
        "final_combined_answer": state["final_answer"],
        "all_citations":         state["all_citations"],
    }
    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nSaved: {out_path}")
    return {}


# ── Assemble graph ────────────────────────────────────────────────────────────

def build_graph():
    g = StateGraph(PipelineState)

    g.add_node("decompose",           node_decompose)
    g.add_node("route",               node_route)
    g.add_node("retrieve_and_answer", node_retrieve_and_answer)
    g.add_node("specialist_agents",   node_specialist_agents)
    g.add_node("merge",               node_merge)
    g.add_node("save",                node_save)

    g.set_entry_point("decompose")
    g.add_edge("decompose",           "route")
    g.add_edge("route",               "retrieve_and_answer")
    g.add_edge("retrieve_and_answer", "specialist_agents")
    g.add_edge("specialist_agents",   "merge")
    g.add_edge("merge",               "save")
    g.add_edge("save",                END)

    return g.compile()