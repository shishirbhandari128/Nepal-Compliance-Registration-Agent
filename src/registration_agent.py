"""
registration_agent.py
---------------------
Production-ready OCR registration agent (Nepal)

- Uses LLM for extraction + conversation
- Uses Python for validation (NO hallucination)
"""

import json
import re
from typing import Dict, List
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from .config import OPENAI_MODEL


                                                                               

REQUIRED_FIELDS = [
    "company_name_english",
    "company_name_nepali",
    "company_type",
    "objectives",
    "contact_phone",
    "contact_email",
    "district",
    "vdc_municipality",
    "ward_no",
    "capital_type",
    "authorized_capital",
    "quantity_of_shares",
    "issued_capital",
    "paid_up_capital",
    "shareholders",
]

OPTIONAL_FIELDS = [
    "fax_no",
    "street",
    "block_no",
    "authorized_rate",
    "document_files",
]

FIELD_LABELS = {
    "company_name_english": "Company Name (English)",
    "company_name_nepali": "Company Name (Nepali)",
    "company_type": "Company Type",
    "objectives": "Business Objectives",
    "contact_phone": "Phone Number",
    "contact_email": "Email",
    "district": "District",
    "vdc_municipality": "Municipality",
    "ward_no": "Ward Number",
    "capital_type": "Capital Type",
    "authorized_capital": "Authorized Capital",
    "quantity_of_shares": "Number of Shares",
    "issued_capital": "Issued Capital",
    "paid_up_capital": "Paid-up Capital",
    "shareholders": "Shareholders",
}


                                                                                

_EXTRACT_PROMPT = ChatPromptTemplate.from_template(
    """
Extract Nepal OCR company registration details from user input.

Rules:
- Do NOT guess
- Only extract if clearly provided
- Return valid JSON only

User message:
{message}

Return:
{{
  "company_name_english": null,
  "company_name_nepali": null,
  "company_type": null,
  "objectives": null,
  "contact_phone": null,
  "contact_email": null,
  "district": null,
  "vdc_municipality": null,
  "ward_no": null,
  "capital_type": null,
  "authorized_capital": null,
  "quantity_of_shares": null,
  "issued_capital": null,
  "paid_up_capital": null,
  "shareholders": null
}}
"""
)


                                                                                

QUESTION_FLOW = REQUIRED_FIELDS


def ask_question(field: str) -> str:
    questions = {
        "company_name_english": "What is your company name in English?",
        "company_name_nepali": "What is your company name in Nepali?",
        "company_type": "What type of company is it? (Private / Public / Partnership)",
        "objectives": "What does your company do? (You can list multiple objectives)",
        "contact_phone": "What is your phone number?",
        "contact_email": "What is your email?",
        "district": "Which district is your company located in?",
        "vdc_municipality": "Which municipality or VDC?",
        "ward_no": "What is the ward number?",
        "capital_type": "What is the capital type? (e.g. Private Multiple)",
        "authorized_capital": "What is the authorized capital (NPR)?",
        "quantity_of_shares": "How many shares?",
        "issued_capital": "What is the issued capital?",
        "paid_up_capital": "What is the paid-up capital?",
        "shareholders": "Provide shareholder details (Name, Citizenship No, Shares)",
    }
    return questions.get(field, f"Please provide {field}")


                                                                                

def extract_fields(message: str, collected: Dict) -> Dict:
    llm = ChatOpenAI(model=OPENAI_MODEL, temperature=0)

    raw = llm.invoke(
        _EXTRACT_PROMPT.format_messages(message=message)
    ).content.strip()

    raw = re.sub(r"^```[a-z]*\n?", "", raw, flags=re.IGNORECASE)
    raw = re.sub(r"\n?```$", "", raw, flags=re.IGNORECASE)

    try:
        extracted = json.loads(raw)
        for key, val in extracted.items():
            if val is not None:
                collected[key] = val
    except:
        pass

    return collected


                                                                                

def get_missing_fields(collected: Dict) -> List[str]:
    return [f for f in REQUIRED_FIELDS if not collected.get(f)]


                                                                                

def get_next_question(collected: Dict, history: List[Dict]) -> str:
    for field in QUESTION_FLOW:
        if not collected.get(field):
            return ask_question(field)
    return "all_collected"


                                                                                

def clean_number(value):
    if value is None:
        return 0
    return int(str(value).replace(",", "").strip())


                                                                                

def validate_form(collected: Dict) -> Dict:
    issues = []

                  
    name = str(collected.get("company_name_english", ""))
    if not name:
        issues.append("Company name is required")
    elif re.search(r"[^\w\s\.\-&]", name):
        issues.append("Invalid characters in company name")

                  
    if collected.get("company_type") not in ["Private", "Public", "Partnership"]:
        issues.append("Invalid company type")

           
    email = str(collected.get("contact_email", ""))
    if "@" not in email:
        issues.append("Invalid email")

           
    phone = str(collected.get("contact_phone", ""))
    if not re.match(r"^(98|97)\d{8}$", phone):
        issues.append("Invalid Nepal phone number")

             
    try:
        auth = clean_number(collected.get("authorized_capital"))
        issued = clean_number(collected.get("issued_capital"))
        paid = clean_number(collected.get("paid_up_capital"))

        if auth < issued:
            issues.append("Authorized capital must be >= issued capital")

        if issued < paid:
            issues.append("Issued capital must be >= paid-up capital")

    except:
        issues.append("Invalid capital values")

                  
    sh = collected.get("shareholders", [])
    if not sh:
        issues.append("At least one shareholder required")

    return {
        "valid": len(issues) == 0,
        "issues": issues
    }


                                                                                

def build_form_summary(collected: Dict) -> str:
    return f"""
🏢 COMPANY REGISTRATION SUMMARY

Name: {collected.get('company_name_english')}
Type: {collected.get('company_type')}

Location:
{collected.get('district')} - {collected.get('vdc_municipality')} (Ward {collected.get('ward_no')})

Capital:
Authorized: NPR {collected.get('authorized_capital')}
Issued: NPR {collected.get('issued_capital')}
Paid-up: NPR {collected.get('paid_up_capital')}

Contact:
Email: {collected.get('contact_email')}
Phone: {collected.get('contact_phone')}

Shareholders:
{collected.get('shareholders')}

⚠️ Please review before submission.
"""