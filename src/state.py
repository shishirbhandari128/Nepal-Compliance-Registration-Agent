"""
state.py
--------
Single shared state object flowing through every LangGraph node.
"""

import operator
from typing import List, Dict, Annotated, Any
from typing_extensions import TypedDict


class PipelineState(TypedDict):
                                                                                
    question:    str
    docs_dir:    str
    mode:        str
    max_files:   int
    save_json:   str
    session_id:  str

                                                                                
    resolved_question: str
    sub_questions:     List[str]

                                                                                
    selected_pdfs:  List[str]
    routing_reason: str

                                                                                
    per_doc_answers: Annotated[List[Dict], operator.add]
    all_docs:        Annotated[List[Any],  operator.add]                               

                                                                                
    specialist_outputs: Dict[str, str]

                                                                                
    final_answer:  str
    all_citations: List[str]