import argparse
import shutil
import sys
from pathlib import Path

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.pdf_loader import list_pdf_files, load_selected_pdfs
from src.chunker import chunk_documents
from src.vector_store import build_chroma
from src.rag_engine import make_retriever, answer_with_retriever
from src.agents import run_all_specialists
from src.supervisor import supervisor_merge
from src.router import choose_relevant_pdfs


def main():
    parser = argparse.ArgumentParser(description="Run complete multi-agent compliance answer.")
    parser.add_argument("--docs-dir", default=".", help="Directory containing PDFs")
    parser.add_argument("--question", required=True, help="Question to ask")
    parser.add_argument("--max-files", type=int, default=3, help="Maximum PDFs supervisor can select")
    args = parser.parse_args()

    docs_dir = Path(args.docs_dir).resolve()
    pdfs = list_pdf_files(docs_dir)
    if not pdfs:
        raise ValueError(f"No PDF files found in {docs_dir}")

    route = choose_relevant_pdfs(args.question, pdfs, max_files=args.max_files)
    selected = route["selected_files"]
    pages = load_selected_pdfs(docs_dir, selected)
    if not pages:
        raise ValueError(f"No pages loaded from selected files: {selected}")
    chunks = chunk_documents(pages)

    persist = Path("vectorstore") / "global"
    if persist.exists():
        shutil.rmtree(persist)
    persist.mkdir(parents=True, exist_ok=True)
    db = build_chroma(chunks, persist, "global_nepal_compliance")

    retriever = make_retriever(db)
    base = answer_with_retriever(args.question, retriever)
    specialists = run_all_specialists(args.question, base["answer"])
    final = supervisor_merge(args.question, specialists)

    print("\n=== SUPERVISOR ROUTING ===")
    print(f'Selected PDFs: {", ".join(selected)}')
    if route.get("reason"):
        print(f'Reason: {route["reason"]}')
    print("\n=== BASE RAG ANSWER ===")
    print(base["answer"])
    print("\n=== SPECIALIST OUTPUTS ===")
    for role, text in specialists.items():
        print(f"\n[{role}]\n{text}")
    print("\n=== SUPERVISOR FINAL ANSWER ===")
    print(final)


if __name__ == "__main__":
    main()

