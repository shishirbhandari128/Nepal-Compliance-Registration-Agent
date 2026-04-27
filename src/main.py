"""
main.py
-------
CLI entry point. For API usage run: uvicorn api.main:app --reload
"""

import argparse
import sys
import uuid
from pathlib import Path

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.graph import build_graph
from src.state import PipelineState


def main():
    parser = argparse.ArgumentParser(
        description="Nepal Compliance RAG — LangGraph pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m src.main --question "What are the tax rules for a restaurant?"
  python -m src.main --question "..." --mode all
  python -m src.main --question "..." --session-id abc123   # continue a session
        """,
    )
    parser.add_argument("--question",   "-q", required=True)
    parser.add_argument("--docs-dir",   "-d", default=".")
    parser.add_argument("--mode",       choices=["routed", "all"], default="routed")
    parser.add_argument("--max-files",  type=int, default=3)
    parser.add_argument("--save-json",  default="outputs/answer.json")
    parser.add_argument("--session-id", default=None, help="Reuse a session for follow-up questions")
    args = parser.parse_args()

    session_id = args.session_id or str(uuid.uuid4())
    print(f"Session ID: {session_id}")

    app = build_graph()

    initial_state: PipelineState = {
        "question":           args.question,
        "docs_dir":           args.docs_dir,
        "mode":               args.mode,
        "max_files":          args.max_files,
        "save_json":          args.save_json,
        "session_id":         session_id,
        "sub_questions":      [],
        "resolved_question":  "",
        "selected_pdfs":      [],
        "routing_reason":     "",
        "per_doc_answers":    [],
        "specialist_outputs": {},
        "final_answer":       "",
        "all_citations":      [],
    }

    app.invoke(initial_state)
    print(f"\nTo continue this conversation use: --session-id {session_id}")


if __name__ == "__main__":
    main()