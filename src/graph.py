"""
graph.py
--------
LangGraph pipeline — 7 nodes:

  decompose → route → retrieve_and_answer → specialist_agents → merge → faithfulness → save → END
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
from .faithfulness import check_and_fix
from .config import OPENAI_MODEL


                                                                                

_RESOLVE_PROMPT = ChatPromptTemplate.from_template(
    """You are a question resolver. Rewrite the user's follow-up question as a
fully self-contained question using the conversation history.
If it is already self-contained, return it unchanged.
Return ONLY the rewritten question, nothing else.

Conversation history:
{history}

Current question:
{question}
"""
)

_ROUTER_PROMPT = ChatPromptTemplate.from_template(
    """You are a routing supervisor for a Nepal compliance document search system.
Select ONLY the PDF files directly relevant to the user question.

User question:
{question}

Available PDFs:
{options}

Rules:
- Select ONLY files whose summary directly matches the topic.
- Tax question → only tax documents. Registration → only registration documents.
- Do NOT select loosely related files.
- Select at most {max_files} files.

Return ONLY valid JSON, no markdown:
{{
  "selected_files": ["file1.pdf"],
  "reason": "one sentence explaining the choice"
}}
"""
)

_MERGER_PROMPT = ChatPromptTemplate.from_template(
    """You are a senior Nepal compliance analyst merging evidence into one answer.

STRICT RULES — violations cause real harm to users:
- You may ONLY include facts that are explicitly stated in the base RAG answer.
- You may ONLY include specialist insights that are directly supported by the base answer.
- If a specialist adds something NOT in the base answer, IGNORE it.
- Do NOT generalise, interpolate, or fill gaps with assumed legal knowledge.
- Numbers (shareholder counts, thresholds, fees, timelines) must be copied exactly
  as they appear in the base answer — never paraphrased or rounded.
- Do NOT write citations — sources are appended automatically.

User question:
{question}

Base RAG answer (this is ground truth — do not contradict it):
{base_answer}

Specialist perspectives (use only what is supported by the base answer above):
{specialist_outputs}

Write the final answer now:
"""
)


                                                                                

def node_decompose(state: PipelineState) -> dict:
    print("\n[node: decompose]")
    session  = get_session(state["session_id"])
    question = state["question"]

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

    session.add_user_turn(question)

    return {
        "resolved_question": resolved_question,
        "sub_questions":     sub_questions,
    }


                                                                                

def node_route(state: PipelineState) -> dict:
    print("\n[node: route]")
    docs_dir = Path(state["docs_dir"]).resolve()
    pdfs     = list_pdf_files(docs_dir)

    if not pdfs:
        raise ValueError(f"No PDF files found in {docs_dir}")

    if state["mode"] == "all":
        return {
            "selected_pdfs":   [str(p) for p in pdfs],
            "routing_reason":  "mode=all",
            "per_doc_answers": [],
            "all_docs":        [],
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
        parsed = {"selected_files": [pdfs[0].name], "reason": "fallback"}

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
        "selected_pdfs":   [str(p) for p in selected_pdfs],
        "routing_reason":  routing_reason,
        "per_doc_answers": [],
        "all_docs":        [],
    }


                                                                                

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
            "docs":          result["docs"],                                        
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
                    "docs":          [],
                }

    valid_results = [r for r in results if r is not None]

                                                                       
    all_docs = [doc for r in valid_results for doc in r.get("docs", [])]

                                                                                
    per_doc_answers = [
        {k: v for k, v in r.items() if k != "docs"}
        for r in valid_results
    ]

    return {
        "per_doc_answers": per_doc_answers,
        "all_docs":        all_docs,
    }


                                                                                

def node_specialist_agents(state: PipelineState) -> dict:
    print("\n[node: specialist_agents]")
    per_doc_answers = state["per_doc_answers"]

                                                                         
    base_answer = "\n\n".join(
        f"--- {item['file']} ---\n{item['answer'].split(chr(10)+'Sources:')[0]}"
        for item in per_doc_answers
    )

    print("  Running Legal, Tax, Document specialists in parallel...")
    specialist_outputs = run_all_specialists(state["resolved_question"], base_answer)

    for role in specialist_outputs:
        print(f"    ✓ {role}")

    return {"specialist_outputs": specialist_outputs}


                                                                                

def node_merge(state: PipelineState) -> dict:
    print("\n[node: merge]")
    per_doc_answers    = state["per_doc_answers"]
    specialist_outputs = state["specialist_outputs"]

    spec_block = "\n\n".join(
        f"=== {role} ===\n{text}"
        for role, text in specialist_outputs.items()
    )

    if len(per_doc_answers) == 1:
        base_answer = per_doc_answers[0]["answer"].split("\nSources:")[0].strip()
    else:
        base_answer = format_per_doc_block(per_doc_answers)

    llm = ChatOpenAI(model=OPENAI_MODEL, temperature=0)                           
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

    return {"final_answer": final_answer, "all_citations": all_citations}


                                                                                

def node_faithfulness(state: PipelineState) -> dict:
    print("\n[node: faithfulness]")
    result = check_and_fix(state["final_answer"], state["all_docs"])

    if result["was_corrected"]:
        print("  ⚠ Factual corrections applied to final answer")
    else:
        print("  ✓ Answer is faithful to source chunks")

                                                  
    session = get_session(state["session_id"])
    session.add_assistant_turn(
        result["answer"],
        state["sub_questions"],
        state["all_citations"],
    )

    return {"final_answer": result["answer"]}


                                                                                

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


                                                                                

def build_graph():
    g = StateGraph(PipelineState)

    g.add_node("decompose",           node_decompose)
    g.add_node("route",               node_route)
    g.add_node("retrieve_and_answer", node_retrieve_and_answer)
    g.add_node("specialist_agents",   node_specialist_agents)
    g.add_node("merge",               node_merge)
    g.add_node("faithfulness",        node_faithfulness)
    g.add_node("save",                node_save)

    g.set_entry_point("decompose")
    g.add_edge("decompose",           "route")
    g.add_edge("route",               "retrieve_and_answer")
    g.add_edge("retrieve_and_answer", "specialist_agents")
    g.add_edge("specialist_agents",   "merge")
    g.add_edge("merge",               "faithfulness")
    g.add_edge("faithfulness",        "save")
    g.add_edge("save",                END)

    return g.compile()