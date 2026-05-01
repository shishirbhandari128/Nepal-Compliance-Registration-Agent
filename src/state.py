"""
state.py
--------
Single shared state object flowing through every LangGraph node.
"""

import operator
from typing import List, Dict, Annotated, Any
from typing_extensions import TypedDict


class PipelineState(TypedDict):
    # ── Inputs ────────────────────────────────────────────────────────────────
    question:    str
    docs_dir:    str
    mode:        str
    max_files:   int
    save_json:   str
    session_id:  str

    # ── node_decompose ────────────────────────────────────────────────────────
    resolved_question: str
    sub_questions:     List[str]

    # ── node_route ────────────────────────────────────────────────────────────
    selected_pdfs:  List[str]
    routing_reason: str

    # ── node_retrieve_and_answer ──────────────────────────────────────────────
    per_doc_answers: Annotated[List[Dict], operator.add]
    all_docs:        Annotated[List[Any],  operator.add]  # raw chunks for faithfulness

    # ── node_specialist_agents ────────────────────────────────────────────────
    specialist_outputs: Dict[str, str]

    # ── node_merge ────────────────────────────────────────────────────────────
    final_answer:  str
    all_citations: List[str]