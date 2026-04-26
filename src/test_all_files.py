import argparse
import hashlib
import json
import shutil
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import List, Dict

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.pdf_loader import list_pdf_files, load_pdf
from src.chunker import chunk_documents
from src.vector_store import build_chroma, load_chroma
from src.rag_engine import make_retriever, answer_with_retriever
from src.router import choose_relevant_pdfs
from src.query_decomposer import decompose_question

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

try:
    from src.config import OPENAI_MODEL
except ImportError:
    OPENAI_MODEL = "gpt-4o-mini"


MERGER_PROMPT = ChatPromptTemplate.from_template(
    """You are a senior Nepal compliance analyst. Multiple documents have been \
individually searched and each produced an answer to the same question. \
Your job is to merge them into one final, authoritative answer.

User question:
{question}

Per-document answers (each labelled with its source file):
{per_doc_answers}

How to write the merged answer:
- Read all per-document answers carefully before writing anything.
- Decide your own section headings based on what the question actually needs.
- Where multiple documents agree on a fact, state it once.
- Where documents differ or add complementary detail, explain the difference clearly.
- Rewrite everything in your own words — do NOT copy sentences verbatim.
- Do NOT add any fact not present in the per-document answers.
- If something is marked "Not found" in all documents, keep that note.
- Do NOT write any citations, page numbers, filenames, or source references \
anywhere in your answer — sources will be appended automatically.
"""
)


# ── Vectorstore caching helpers ───────────────────────────────────────────────

def _pdf_hash(pdf_path: Path) -> str:
    h = hashlib.md5()
    h.update(pdf_path.read_bytes())
    return h.hexdigest()

def _hash_file(persist_dir: Path) -> Path:
    return persist_dir / ".pdf_hash"

def _vectorstore_is_fresh(pdf_path: Path, persist_dir: Path) -> bool:
    hash_file = _hash_file(persist_dir)
    if not persist_dir.exists() or not hash_file.exists():
        return False
    return hash_file.read_text().strip() == _pdf_hash(pdf_path)

def _save_hash(pdf_path: Path, persist_dir: Path):
    _hash_file(persist_dir).write_text(_pdf_hash(pdf_path))

def get_or_build_vectorstore(pdf: Path):
    persist = Path("vectorstore") / pdf.stem
    collection = f"all_{pdf.stem}"
    if _vectorstore_is_fresh(pdf, persist):
        print(f"  [cache hit]  {pdf.name}")
        return load_chroma(persist, collection)
    print(f"  [rebuilding] {pdf.name}")
    if persist.exists():
        shutil.rmtree(persist)
    persist.mkdir(parents=True, exist_ok=True)
    pages = load_pdf(pdf)
    chunks = chunk_documents(pages)
    db = build_chroma(chunks, persist, collection)
    _save_hash(pdf, persist)
    return db


# ── Per-PDF worker (runs in its own thread) ───────────────────────────────────

def _process_pdf(pdf: Path, question: str, sub_questions: List[str]) -> Dict:
    """Load/build vectorstore, run parallel sub-question retrieval, generate answer."""
    db = get_or_build_vectorstore(pdf)
    retriever = make_retriever(db)
    result = answer_with_retriever(question, retriever, sub_questions=sub_questions)
    return {
        "file": pdf.name,
        "answer": result["answer"],
        "citations": result["citations"],
        "sub_questions": sub_questions,
    }


# ── Merger ────────────────────────────────────────────────────────────────────

def _format_per_doc_block(answers: List[Dict]) -> str:
    blocks = []
    for item in answers:
        answer_text = item["answer"].split("\nSources:")[0].strip()
        blocks.append(f"=== {item['file']} ===\n{answer_text}")
    return "\n\n".join(blocks)

def _build_sources_block(per_doc_answers: List[Dict]) -> str:
    seen = set()
    lines = []
    for item in per_doc_answers:
        for c in item["citations"]:
            if c not in seen:
                seen.add(c)
                lines.append(f"  {c}")
    return "\n".join(lines)

def merge_answers(question: str, per_doc_answers: List[Dict]) -> str:
    llm = ChatOpenAI(model=OPENAI_MODEL, temperature=0.1)
    per_doc_block = _format_per_doc_block(per_doc_answers)
    msg = MERGER_PROMPT.format_messages(question=question, per_doc_answers=per_doc_block)
    merged_text = llm.invoke(msg).content.strip()
    sources = _build_sources_block(per_doc_answers)
    return f"{merged_text}\n\nSources:\n{sources}"


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Answer a question per-PDF (parallel) then merge into one answer."
    )
    parser.add_argument("--docs-dir", default=".", help="Directory containing PDF files")
    parser.add_argument("--question", required=True, help="Question to ask")
    parser.add_argument("--save-json", default="outputs/all_file_answers.json", help="Output JSON path")
    parser.add_argument(
        "--mode",
        choices=["all", "routed"],
        default="routed",
        help="all = query every PDF, routed = supervisor selects relevant PDFs only (default)",
    )
    parser.add_argument("--max-files", type=int, default=3, help="Max PDFs the router can select")
    args = parser.parse_args()

    docs_dir = Path(args.docs_dir).resolve()
    pdfs = list_pdf_files(docs_dir)
    if not pdfs:
        raise ValueError(f"No PDF files found in {docs_dir}")

    # ── Step 1: Decompose question ONCE ───────────────────────────────────────
    print("\nDecomposing question into sub-questions...")
    sub_questions = decompose_question(args.question)
    print(f"  Sub-questions ({len(sub_questions)}):")
    for sq in sub_questions:
        print(f"    - {sq}")

    # ── Step 2: Router sees original question + all sub-questions ─────────────
    routing_reason = ""
    selected_pdfs = pdfs
    if args.mode == "routed":
        enriched_question = (
            f"{args.question}\n\nRelated aspects:\n" +
            "\n".join(f"- {sq}" for sq in sub_questions)
        )
        route = choose_relevant_pdfs(enriched_question, pdfs, max_files=args.max_files)
        selected_names = set(route["selected_files"])
        selected_pdfs = [p for p in pdfs if p.name in selected_names]
        routing_reason = route.get("reason", "")

    print(f"\nFiles selected : {', '.join(p.name for p in selected_pdfs)}")
    if routing_reason:
        print(f"Routing reason : {routing_reason}")

    # ── Step 3: Process all PDFs in PARALLEL ──────────────────────────────────
    print(f"\nProcessing {len(selected_pdfs)} PDF(s) in parallel...")
    per_doc_answers: List[Dict] = [None] * len(selected_pdfs)  # preserve order

    with ThreadPoolExecutor(max_workers=len(selected_pdfs)) as executor:
        future_to_idx = {
            executor.submit(_process_pdf, pdf, args.question, sub_questions): i
            for i, pdf in enumerate(selected_pdfs)
        }
        for future in as_completed(future_to_idx):
            idx = future_to_idx[future]
            pdf = selected_pdfs[idx]
            try:
                per_doc_answers[idx] = future.result()
                print(f"  ✓ Done: {pdf.name}")
            except Exception as e:
                print(f"  ✗ Failed: {pdf.name} — {e}")
                per_doc_answers[idx] = {
                    "file": pdf.name,
                    "answer": f"Error processing this file: {e}",
                    "citations": [],
                    "sub_questions": sub_questions,
                }

    # Remove any None entries (safety)
    per_doc_answers = [a for a in per_doc_answers if a is not None]

    # ── Step 4: Merge ─────────────────────────────────────────────────────────
    print("\nMerging answers across all files...")
    if len(per_doc_answers) == 1:
        final_answer = per_doc_answers[0]["answer"]
    else:
        final_answer = merge_answers(args.question, per_doc_answers)

    # ── Step 5: Print ─────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("FINAL COMBINED ANSWER")
    print("=" * 70)
    print(final_answer)

    # ── Step 6: Save JSON ─────────────────────────────────────────────────────
    all_citations = list({c for item in per_doc_answers for c in item["citations"]})
    out_path = Path(args.save_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "mode": args.mode,
        "routing_reason": routing_reason,
        "files_evaluated": [p.name for p in selected_pdfs],
        "question": args.question,
        "sub_questions": sub_questions,
        "per_document_answers": per_doc_answers,
        "final_combined_answer": final_answer,
        "all_citations": all_citations,
    }
    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()