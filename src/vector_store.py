"""
vector_store.py
---------------
Thin wrappers around Chroma that use the shared embedding singleton
from tools.py. Kept for backward compatibility with existing code
that imports build_chroma / load_chroma directly.
"""

from pathlib import Path
from typing import List
from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma
from .tools import get_embeddings


def build_chroma(chunks: List[Document], persist_dir: Path, collection_name: str) -> Chroma:
    return Chroma.from_documents(
        documents=chunks,
        embedding=get_embeddings(),
        persist_directory=str(persist_dir),
        collection_name=collection_name,
    )


def load_chroma(persist_dir: Path, collection_name: str) -> Chroma:
    return Chroma(
        persist_directory=str(persist_dir),
        collection_name=collection_name,
        embedding_function=get_embeddings(),
    )