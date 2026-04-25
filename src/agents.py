from dataclasses import dataclass
from typing import Dict
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from .config import OPENAI_MODEL


@dataclass
class AgentResult:
    role: str
    answer: str


def _run_role_prompt(role: str, lens: str, question: str, context_answer: str) -> AgentResult:
    llm = ChatOpenAI(model=OPENAI_MODEL, temperature=0.1)
    prompt = ChatPromptTemplate.from_template(
        """You are a specialist expert — specifically, the {role}.

Your lens: {lens}

The user asked:
{question}

A base RAG system already extracted the following evidence from official documents:
{context_answer}

Your task:
- Analyse the evidence strictly through your specialist lens.
- Organise your response in whatever sections make the most sense for this specific \
question and your domain — do NOT use a rigid template.
- Explain everything in plain, clear language as if briefing a business owner. \
Rewrite the evidence in your own words; do not copy sentences from it.
- Be direct and actionable. Omit anything that is not relevant to your lens.
- If the evidence does not cover something important in your domain, flag it as \
"Needs further verification" rather than guessing.
"""
    )
    msg = prompt.format_messages(
        role=role,
        lens=lens,
        question=question,
        context_answer=context_answer,
    )
    out = llm.invoke(msg)
    return AgentResult(role=role, answer=out.content)


def legal_strategist(question: str, context_answer: str) -> AgentResult:
    return _run_role_prompt(
        "Legal Strategist",
        "Legal entity structure, registration requirements, compliance obligations, "
        "and regulatory risk under Nepal's applicable laws and acts.",
        question,
        context_answer,
    )


def tax_consultant(question: str, context_answer: str) -> AgentResult:
    return _run_role_prompt(
        "Tax Consultant",
        "PAN registration, VAT thresholds and obligations, income tax filing, "
        "advance tax, withholding tax, and any other tax triggers relevant under Nepal's tax law.",
        question,
        context_answer,
    )


def document_auditor(question: str, context_answer: str) -> AgentResult:
    return _run_role_prompt(
        "Document Auditor",
        "All documents that must be prepared, submitted, or maintained — including "
        "what is missing, what sequence they are needed in, and any common pitfalls.",
        question,
        context_answer,
    )


def run_all_specialists(question: str, context_answer: str) -> Dict[str, str]:
    legal = legal_strategist(question, context_answer)
    tax = tax_consultant(question, context_answer)
    doc = document_auditor(question, context_answer)
    return {
        legal.role: legal.answer,
        tax.role: tax.answer,
        doc.role: doc.answer,
    }