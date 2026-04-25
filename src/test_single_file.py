import argparse
import shutil
import sys
from pathlib import Path

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.pdf_loader import load_pdf
from src.chunker import chunk_documents
from src.vector_store import build_chroma
from src.rag_engine import make_retriever, answer_with_retriever


def main():
    parser = argparse.ArgumentParser(description="Test one PDF against one question.")
    parser.add_argument("--pdf", required=True, help="Path to a PDF file")
    parser.add_argument("--question", required=True, help="Question to ask")
    args = parser.parse_args()

    pdf_path = Path(args.pdf).resolve()
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    docs = load_pdf(pdf_path)
    chunks = chunk_documents(docs)

    persist = Path("vectorstore") / pdf_path.stem
    if persist.exists():
        shutil.rmtree(persist)
    persist.mkdir(parents=True, exist_ok=True)

    db = build_chroma(chunks, persist, f"single_{pdf_path.stem}")
    retriever = make_retriever(db)
    result = answer_with_retriever(args.question, retriever)

    print("\n=== ANSWER ===")
    print(result["answer"])
    print("\n=== TOP CONTEXT SOURCES ===")
    for d in result["docs"]:
        print(f'- {d.metadata.get("source_file")} page {d.metadata.get("page_index")}')


if __name__ == "__main__":
    main()

