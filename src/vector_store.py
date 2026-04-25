import threading
from pathlib import Path
from typing import List
from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from .config import EMBEDDING_MODEL

_embeddings_instance = None
_embeddings_lock = threading.Lock()


def _get_embeddings() -> HuggingFaceEmbeddings:
    """Load the embedding model exactly once, even across parallel threads."""
    global _embeddings_instance
    if _embeddings_instance is None:
        with _embeddings_lock:
            # Double-check inside the lock in case another thread just built it
            if _embeddings_instance is None:
                _embeddings_instance = HuggingFaceEmbeddings(
                    model_name=EMBEDDING_MODEL,
                    encode_kwargs={"normalize_embeddings": True},
                )
    return _embeddings_instance


def build_chroma(chunks: List[Document], persist_dir: Path, collection_name: str) -> Chroma:
    return Chroma.from_documents(
        documents=chunks,
        embedding=_get_embeddings(),
        persist_directory=str(persist_dir),
        collection_name=collection_name,
    )


def load_chroma(persist_dir: Path, collection_name: str) -> Chroma:
    return Chroma(
        persist_directory=str(persist_dir),
        collection_name=collection_name,
        embedding_function=_get_embeddings(),
    )