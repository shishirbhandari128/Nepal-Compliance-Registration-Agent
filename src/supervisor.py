from typing import Dict
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from .config import OPENAI_MODEL


def supervisor_merge(question: str, specialist_outputs: Dict[str, str]) -> str:
    merged = "\n\n".join(
        [f"=== {role} ===\n{text}" for role, text in specialist_outputs.items()]
    )
    llm = ChatOpenAI(model=OPENAI_MODEL, temperature=0.1)
    prompt = ChatPromptTemplate.from_template(
        """You are the Supervisor Agent responsible for producing the final, unified \
compliance guidance for a user in Nepal.

Three domain specialists have analysed the question from their own angles. Your job \
is to synthesise their inputs into one coherent, well-reasoned answer.

User question:
{question}

Specialist inputs:
{merged}

How to write your response:
- Do NOT use a fixed numbered template. Instead, decide on the most natural and \
helpful structure based on what this specific question actually needs.
- Merge overlapping points from the specialists rather than listing each agent's \
output separately.
- Rewrite everything in your own words — clear, plain language that a business owner \
can act on immediately.
- Highlight the most critical steps or risks prominently.
- Where specialists flagged gaps or uncertainties, include a short "What still needs \
clarification" note at the end.
- Keep it focused: no padding, no repetition, no generic legal disclaimers.
"""
    )
    msg = prompt.format_messages(question=question, merged=merged)
    return llm.invoke(msg).content