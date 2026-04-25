import streamlit as st
from pathlib import Path
from src.pdf_loader import load_all_pdfs
from src.chunker import chunk_documents
from src.vector_store import build_chroma
from src.rag_engine import make_retriever, answer_with_retriever


st.set_page_config(page_title="Nepal Compliance Agent", layout="wide")
st.title("Nepal Compliance & Registration Agent")

docs_dir = st.text_input("PDF folder path", value=".")
question = st.text_area("Enter your compliance question")

if st.button("Ask"):
    with st.spinner("Building RAG index and answering..."):
        pages = load_all_pdfs(Path(docs_dir).resolve())
        chunks = chunk_documents(pages)
        persist = Path("vectorstore") / "streamlit_global"
        persist.mkdir(parents=True, exist_ok=True)
        db = build_chroma(chunks, persist, "streamlit_global")
        retriever = make_retriever(db)
        result = answer_with_retriever(question, retriever)

    st.subheader("Answer")
    st.write(result["answer"])

    st.subheader("Retrieved Sources")
    for d in result["docs"]:
        st.write(f'- {d.metadata.get("source_file")} page {d.metadata.get("page_index")}')

