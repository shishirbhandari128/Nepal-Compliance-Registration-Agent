"""
state.py
--------
Single shared state object flowing through every LangGraph node.
"""

import operator
from typing import List, Dict, Annotated, Optional
from typing_extensions import TypedDict


class PipelineState(TypedDict):
    # ── Inputs ────────────────────────────────────────────────────────────────
    question:    str
    docs_dir:    str
    mode:        str        # "routed" | "all"
    max_files:   int
    save_json:   str
    session_id:  str        # conversation memory key

    # ── node_decompose ────────────────────────────────────────────────────────
    resolved_question: str          # follow-up resolved against history
    sub_questions:     List[str]

    # ── node_route ────────────────────────────────────────────────────────────
    selected_pdfs:  List[str]       # absolute path strings
    routing_reason: str

    # ── node_retrieve_and_answer — parallel branches append safely ────────────
    per_doc_answers: Annotated[List[Dict], operator.add]

    # ── node_specialist_agents ────────────────────────────────────────────────
    specialist_outputs: Dict[str, str]   # role → answer

    # ── node_merge ────────────────────────────────────────────────────────────
    final_answer:  str
    all_citations: List[str]