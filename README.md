# Nepal Compliance & Registration Agent (Multi-Agent RAG)

This project helps test Nepal legal/tax PDF files against user questions using:
- RAG per file
- RAG across all files
- Multi-agent role-based reasoning (Legal/Tax/Document + Supervisor)

## 1) Setup

1. Create and activate virtual environment:

```powershell
python -m venv .venv
.venv\Scripts\activate
```

2. Install dependencies:

```powershell
pip install -r requirements.txt
```

3. Configure environment:

```powershell
copy .env.example .env
```

Then edit `.env` and set your OpenAI key.

## 2) Your Current PDF Files

Put all PDF files in the project root (already true in your case), or pass another folder with `--docs-dir`.

## 3) Test One File Against One Question

```powershell
python -m src.test_single_file --pdf "CompanyAct2006.pdf" --question "What are the requirements to register a private company in Nepal?"
```

## 4) Test All Files Against Same Question

```powershell
python -m src.test_all_files --docs-dir "." --question "What are the steps and tax registrations needed to start a tourism company in Nepal?"
```

This saves JSON output to `outputs/all_file_answers.json`.

### Routed Mode (Supervisor picks relevant PDFs first)

```powershell
python -m src.test_all_files --docs-dir "." --mode routed --max-files 3 --question "What are the legal and tax steps to start a tourism company in Nepal?"
```

## 5) Run Full Multi-Agent Pipeline

```powershell
python -m src.run_multi_agent --docs-dir "." --question "I want to open a partnership-based tourism business in Nepal. What legal and tax steps should I follow?"
```

This pipeline now includes a supervisor routing step that selects the most relevant PDFs before RAG.

## 6) Optional Streamlit UI

```powershell
streamlit run src/app.py
```

## 7) Project Structure

```
.
├─ README.md
├─ requirements.txt
├─ .env.example
├─ outputs/
├─ vectorstore/
├─ data/
│  ├─ raw/
│  └─ chunks/
└─ src/
   ├─ config.py
   ├─ pdf_loader.py
   ├─ chunker.py
   ├─ vector_store.py
   ├─ rag_engine.py
   ├─ router.py
   ├─ agents.py
   ├─ supervisor.py
   ├─ test_single_file.py
   ├─ test_all_files.py
   ├─ run_multi_agent.py
   └─ app.py
```

## 8) What Each Code File Contains

- `src/config.py`: global settings (model names, chunk size, paths).
- `src/pdf_loader.py`: list and load PDF pages with metadata.
- `src/chunker.py`: splits pages into retrievable chunks.
- `src/vector_store.py`: builds/loads Chroma vector database.
- `src/rag_engine.py`: retrieval + LLM answering with citations.
- `src/router.py`: supervisor routing agent that selects relevant PDFs per question.
- `src/agents.py`: specialist agent roles (Legal, Tax, Document).
- `src/supervisor.py`: merges specialist outputs into final guidance.
- `src/test_single_file.py`: run one question against one PDF.
- `src/test_all_files.py`: run same question across all PDFs and save JSON.
- `src/run_multi_agent.py`: full pipeline demonstration for proposal demo.
- `src/app.py`: simple Streamlit interface.

