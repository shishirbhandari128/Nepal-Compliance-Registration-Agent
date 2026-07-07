"""
reranker.py
-----------
Cross-encoder reranker using BAAI/bge-reranker-base.
Reranks retrieved chunks by actual relevance to the query
before sending to the LLM — significantly improves answer quality.
Singleton pattern: model loaded once, thread-safe.
"""

import threading
from typing import List, Tuple
from langchain_core.documents import Document

_reranker_instance = None
_reranker_lock = threading.Lock()

RERANKER_MODEL = "BAAI/bge-reranker-base"


def _get_reranker():
    global _reranker_instance
    if _reranker_instance is None:
        with _reranker_lock:
            if _reranker_instance is None:
                try:
                    from sentence_transformers import CrossEncoder
                    _reranker_instance = CrossEncoder(RERANKER_MODEL)
                except ImportError:
                    raise ImportError(
                        "sentence-transformers is required for reranking. "
                        "Run: pip install sentence-transformers"
                    )
    return _reranker_instance


def rerank_documents(
    query: str,
    docs: List[Document],
    top_n: int = 10,
) -> List[Document]:
    """
    Score every (query, chunk) pair with the cross-encoder and return
    the top_n chunks sorted by descending relevance score.
    Falls back to original order if reranker fails.
    """
    if not docs:
        return docs

    try:
        reranker = _get_reranker()
        pairs: List[Tuple[str, str]] = [
            (query, d.page_content) for d in docs
        ]
        scores: List[float] = reranker.predict(pairs)

        scored = sorted(zip(scores, docs), key=lambda x: x[0], reverse=True)
        reranked = [doc for _, doc in scored[:top_n]]

                                                  
        for score, doc in scored[:top_n]:
            doc.metadata["rerank_score"] = round(float(score), 4)

        return reranked

    except Exception as e:
        print(f"  Warning: reranking failed ({e}), using original order")
        return docs[:top_n]
