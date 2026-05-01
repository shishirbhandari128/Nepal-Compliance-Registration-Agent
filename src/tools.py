"""
tools.py
--------
All shared utility functions used across graph nodes:
  - Vectorstore cache (build once, reload on hash match)
  - Embedding model singleton (load weights exactly once, thread-safe)
  - PDF summary cache (one-line summary per PDF for smart routing)
  - Merger helpers (format per-doc blocks, build sources)
"""

import hashlib
import shutil
import threading
from pathlib import Path
from typing import List, Dict

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from .config import EMBEDDING_MODEL, OPENAI_MODEL


# ── Embedding singleton — loads weights ONCE, thread-safe ────────────────────

_embeddings_instance = None
_embeddings_lock = threading.Lock()


def get_embeddings() -> HuggingFaceEmbeddings:
    """Return the shared embedding model, loading it only on the very first call."""
    global _embeddings_instance
    if _embeddings_instance is None:
        with _embeddings_lock:
            if _embeddings_instance is None:          # double-checked locking
                _embeddings_instance = HuggingFaceEmbeddings(
                    model_name=EMBEDDING_MODEL,
                    encode_kwargs={"normalize_embeddings": True},
                )
    return _embeddings_instance


# ── Vectorstore helpers ───────────────────────────────────────────────────────

def _pdf_hash(pdf_path: Path) -> str:
    h = hashlib.md5()
    h.update(pdf_path.read_bytes())
    return h.hexdigest()


def _hash_file(persist_dir: Path) -> Path:
    return persist_dir / ".pdf_hash"


def vectorstore_is_fresh(pdf_path: Path, persist_dir: Path) -> bool:
    """True if the vectorstore exists and was built from the exact same PDF bytes."""
    hf = _hash_file(persist_dir)
    if not persist_dir.exists() or not hf.exists():
        return False
    return hf.read_text().strip() == _pdf_hash(pdf_path)


def save_pdf_hash(pdf_path: Path, persist_dir: Path) -> None:
    _hash_file(persist_dir).write_text(_pdf_hash(pdf_path))


def build_vectorstore(chunks: List[Document], persist_dir: Path, collection: str) -> Chroma:
    return Chroma.from_documents(
        documents=chunks,
        embedding=get_embeddings(),
        persist_directory=str(persist_dir),
        collection_name=collection,
    )


def load_vectorstore(persist_dir: Path, collection: str) -> Chroma:
    return Chroma(
        persist_directory=str(persist_dir),
        collection_name=collection,
        embedding_function=get_embeddings(),
    )


def _safe_collection_count(db: Chroma) -> int:
    """
    Best-effort count of items in the underlying Chroma collection.
    Returns 0 if the count cannot be determined.
    """
    try:
        coll = getattr(db, "_collection", None)
        if coll is None:
            return 0
        return int(coll.count())
    except Exception:
        return 0


def get_or_build_vectorstore(pdf: Path, chunks_fn) -> Chroma:
    """
    Load the cached vectorstore if the PDF hasn't changed.
    Otherwise rebuild it from scratch and cache the new hash.
    chunks_fn: callable() -> List[Document]  (lazy — only called on cache miss)
    """
    persist = Path("vectorstore") / pdf.stem
    collection = f"vs_{pdf.stem}"

    if vectorstore_is_fresh(pdf, persist):
        print(f"    [cache hit]  {pdf.name}")
        db = load_vectorstore(persist, collection)
        # If a previous build wrote an empty / corrupt DB, force rebuild.
        if _safe_collection_count(db) > 0:
            return db
        print(f"    [cache invalid] {pdf.name} — empty vectorstore, rebuilding…")

    print(f"    [rebuilding] {pdf.name}")
    if persist.exists():
        shutil.rmtree(persist)
    persist.mkdir(parents=True, exist_ok=True)

    chunks = chunks_fn()
    db = build_vectorstore(chunks, persist, collection)
    # Only mark cache fresh if we actually stored embeddings.
    if _safe_collection_count(db) > 0:
        save_pdf_hash(pdf, persist)
    else:
        print(f"    [warning] {pdf.name} vectorstore built empty — not caching hash")
    return db


# ── PDF summary cache (used by router for smart PDF selection) ─────────────

_SUMMARY_PROMPT = ChatPromptTemplate.from_template(
    """Read the following text extracted from a legal/regulatory PDF document.
Write ONE sentence (max 30 words) describing what topics this document covers.
Be specific — mention the exact subject matter (e.g. income tax rates, VAT registration,
company incorporation, labour rights, tourism licensing).

Document text:
{text}

Return only the one-sentence summary, nothing else.
"""
)


def _summary_cache_path(pdf_path: Path) -> Path:
    persist = Path("vectorstore") / pdf_path.stem
    persist.mkdir(parents=True, exist_ok=True)
    return persist / ".pdf_summary"


def get_or_build_summary(pdf_path: Path) -> str:
    """One-line summary of a PDF — cached to disk after first generation."""
    cache = _summary_cache_path(pdf_path)
    if cache.exists():
        return cache.read_text(encoding="utf-8").strip()

    try:
        from langchain_community.document_loaders import PyPDFLoader
        pages = PyPDFLoader(str(pdf_path)).load()
        sample = " ".join(p.page_content for p in pages[:3])[:3000]
    except Exception:
        sample = pdf_path.stem.replace("_", " ").replace("-", " ")

    llm = ChatOpenAI(model=OPENAI_MODEL, temperature=0)
    summary = llm.invoke(_SUMMARY_PROMPT.format_messages(text=sample)).content.strip()
    cache.write_text(summary, encoding="utf-8")
    return summary


def build_pdf_options_block(pdf_paths: List[Path]) -> str:
    """Rich description block fed to the router LLM."""
    lines = []
    for p in pdf_paths:
        summary = get_or_build_summary(p)
        lines.append(f"- {p.name}\n    Summary: {summary}")
    return "\n".join(lines)


# ── Merger helpers ────────────────────────────────────────────────────────────

def format_per_doc_block(answers: List[Dict]) -> str:
    blocks = []
    for item in answers:
        # Strip the Sources block already appended by rag_engine
        answer_text = item["answer"].split("\nSources:")[0].strip()
        blocks.append(f"=== {item['file']} ===\n{answer_text}")
    return "\n\n".join(blocks)


def build_sources_block(per_doc_answers: List[Dict]) -> str:
    seen: set = set()
    lines: List[str] = []
    for item in per_doc_answers:
        for c in item["citations"]:
            if c not in seen:
                seen.add(c)
                lines.append(f"  {c}")
    return "\n".join(lines)