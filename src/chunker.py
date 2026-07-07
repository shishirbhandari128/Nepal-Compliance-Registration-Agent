import re
from typing import List
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from .config import CHUNK_OVERLAP, CHUNK_SIZE


def _normalize(text: str) -> str:
    """Collapse all whitespace variants to a single space for fuzzy matching."""
    return re.sub(r"\s+", " ", text).strip()


def _find_line_range(full_page: str, chunk_text: str):
    """
    Return (start_line, end_line) of chunk_text inside full_page.
    Falls back gracefully — never returns '?'.
    """
                           
    char_start = full_page.find(chunk_text[:80])

                                                             
    if char_start == -1:
        norm_page = _normalize(full_page)
        norm_anchor = _normalize(chunk_text[:120])
        norm_pos = norm_page.find(norm_anchor)
        if norm_pos != -1:
                                                                        
                                                                         
            raw_ratio = len(full_page) / max(len(norm_page), 1)
            char_start = int(norm_pos * raw_ratio)
        else:
            char_start = 0                                       

    lines_before = full_page[:char_start].count("\n")
    chunk_line_count = chunk_text.count("\n") + 1
    start_line = lines_before + 1
    end_line = lines_before + chunk_line_count
    return start_line, end_line


def chunk_documents(documents: List[Document]) -> List[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

                                                                    
    page_text_map = {}
    for doc in documents:
        key = (doc.metadata.get("source_file"), doc.metadata.get("page_index"))
        page_text_map[key] = doc.page_content

    chunks = splitter.split_documents(documents)

    for i, chunk in enumerate(chunks):
        chunk.metadata["chunk_id"] = i

        key = (chunk.metadata.get("source_file"), chunk.metadata.get("page_index"))
        full_page = page_text_map.get(key, "")

        if full_page:
            start_line, end_line = _find_line_range(full_page, chunk.page_content)
        else:
                                                                      
            start_line = 1
            end_line = chunk.page_content.count("\n") + 1

        chunk.metadata["start_line"] = start_line
        chunk.metadata["end_line"] = end_line

    return chunks