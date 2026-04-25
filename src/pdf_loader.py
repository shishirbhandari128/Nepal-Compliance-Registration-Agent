from pathlib import Path
from typing import List
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document


def list_pdf_files(docs_dir: Path) -> List[Path]:
    return sorted([p for p in docs_dir.glob("*.pdf") if p.is_file()])


def load_pdf(file_path: Path) -> List[Document]:
    loader = PyPDFLoader(str(file_path))
    pages = loader.load()
    for idx, page in enumerate(pages):
        page.metadata["source_file"] = file_path.name
        page.metadata["page_index"] = idx + 1
    return pages


def load_all_pdfs(docs_dir: Path) -> List[Document]:
    all_docs: List[Document] = []
    for pdf in list_pdf_files(docs_dir):
        all_docs.extend(load_pdf(pdf))
    return all_docs


def load_selected_pdfs(docs_dir: Path, selected_file_names: List[str]) -> List[Document]:
    selected = set(selected_file_names)
    all_docs: List[Document] = []
    for pdf in list_pdf_files(docs_dir):
        if pdf.name in selected:
            all_docs.extend(load_pdf(pdf))
    return all_docs

